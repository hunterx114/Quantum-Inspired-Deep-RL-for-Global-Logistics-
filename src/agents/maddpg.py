"""Classical multi-agent deep deterministic policy gradients (MADDPG)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import yaml


@dataclass(frozen=True)
class MaddpgConfig:
    """Network, optimization, replay, and exploration settings."""

    obs_dim: int = 16
    hidden_dim: int = 64
    critic_hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    replay_capacity: int = 100_000
    batch_size: int = 128
    noise_type: str | None = "ou"
    noise_scale: float = 0.2
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    seed: int = 7
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        if min(self.obs_dim, self.hidden_dim, self.critic_hidden_dim) < 1:
            raise ValueError("observation and hidden dimensions must be positive")
        if self.learning_rate <= 0 or self.replay_capacity < 1 or self.batch_size < 1:
            raise ValueError("learning rate, replay capacity, and batch size must be positive")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("gamma must be in [0, 1] and tau in (0, 1]")
        if self.noise_type not in (None, "gaussian", "ou"):
            raise ValueError("noise_type must be null, 'gaussian', or 'ou'")
        if self.noise_scale < 0 or self.ou_theta < 0 or self.ou_sigma < 0:
            raise ValueError("noise parameters must be non-negative")
        if self.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MaddpgConfig":
        """Load MADDPG settings from a YAML mapping."""
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "MaddpgConfig":
        """Create settings from a mapping and reject misspelled keys."""
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown MADDPG configuration keys: {sorted(unknown)}")
        return cls(**values)


class Actor(nn.Module):
    """Deterministic per-agent policy mapping observations to bounded actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class CentralizedCritic(nn.Module):
    """Joint-state/joint-action value function shared across all agents."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        joint_dim = n_agents * (obs_dim + action_dim)
        self.network = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        joint_input = torch.cat(
            (
                observations.reshape(observations.shape[0], -1),
                actions.reshape(actions.shape[0], -1),
            ),
            dim=1,
        )
        return self.network(joint_input)


@dataclass
class TransitionBatch:
    """A replay minibatch with a distinct observation/action per agent."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer that grows memory only as transitions arrive."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        if min(capacity, n_agents, obs_dim, action_dim) < 1:
            raise ValueError("replay dimensions and capacity must be positive")
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._storage: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ] = [None] * capacity
        self._next_index = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_observations: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        """Store one validated joint transition, overwriting the oldest if full."""
        obs = self._as_float_array(observations, (self.n_agents, self.obs_dim), "observations")
        act = self._as_float_array(actions, (self.n_agents, self.action_dim), "actions")
        next_obs = self._as_float_array(
            next_observations, (self.n_agents, self.obs_dim), "next_observations"
        )
        reward_array = np.broadcast_to(
            np.asarray(rewards, dtype=np.float32), (self.n_agents,)
        ).copy()
        done_array = np.broadcast_to(
            np.asarray(dones, dtype=np.float32), (self.n_agents,)
        ).copy()
        if not np.isfinite(reward_array).all() or not np.isfinite(done_array).all():
            raise ValueError("rewards and dones must be finite")
        self._storage[self._next_index] = (
            obs,
            act,
            reward_array,
            next_obs,
            done_array,
        )
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @staticmethod
    def _as_float_array(
        value: np.ndarray, expected_shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array.copy()

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        """Sample without replacement and return tensors on the requested device."""
        if not 1 <= batch_size <= self._size:
            raise ValueError(f"batch_size must be between 1 and replay size ({self._size})")
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        transitions = [self._storage[int(index)] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("replay buffer contains an uninitialized slot")
        fields = list(zip(*transitions))
        tensors = [
            torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)
            for values in fields
        ]
        return TransitionBatch(*tensors)


class GaussianNoise:
    """Independent, reproducible Gaussian action noise for each ship."""

    def __init__(self, n_agents: int, action_dim: int, seed: int) -> None:
        self.shape = (n_agents, action_dim)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.standard_normal(self.shape).astype(np.float32)

    def reset(self) -> None:
        """Stateless noise has no episode state to reset."""


class OUNoise:
    """Ornstein–Uhlenbeck noise with independent state per ship and action."""

    def __init__(
        self,
        n_agents: int,
        action_dim: int,
        theta: float,
        sigma: float,
        seed: int,
    ) -> None:
        self.shape = (n_agents, action_dim)
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(self.shape, dtype=np.float32)

    def sample(self) -> np.ndarray:
        increment = (
            -self.theta * self.state
            + self.sigma * self._rng.standard_normal(self.shape)
        )
        self.state += increment.astype(np.float32)
        return self.state.copy()

    def reset(self) -> None:
        self.state.fill(0.0)


class MADDPG:
    """MADDPG with independent actors and one centralized joint critic."""

    def __init__(
        self,
        n_agents: int,
        action_dim: int,
        config: MaddpgConfig | dict[str, Any] | str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if n_agents < 1 or action_dim < 1:
            raise ValueError("n_agents and action_dim must be positive")
        if config is None:
            self.config = MaddpgConfig()
        elif isinstance(config, MaddpgConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = MaddpgConfig.from_yaml(config)
        else:
            self.config = MaddpgConfig.from_mapping(config)
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.device = torch.device(device)
        torch.set_num_threads(self.config.torch_num_threads)
        torch.manual_seed(self.config.seed)

        self.actors = nn.ModuleList(
            [self._build_actor().to(self.device) for _ in range(n_agents)]
        )
        self.target_actors = copy.deepcopy(self.actors)
        self.critic = self._build_critic().to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=self.config.learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        self.replay_buffer = ReplayBuffer(
            self.config.replay_capacity,
            n_agents,
            self.config.obs_dim,
            action_dim,
            self.config.seed,
        )
        if self.config.noise_type == "ou":
            self.noise: OUNoise | GaussianNoise | None = OUNoise(
                n_agents,
                action_dim,
                self.config.ou_theta,
                self.config.ou_sigma,
                self.config.seed,
            )
        elif self.config.noise_type == "gaussian":
            self.noise = GaussianNoise(n_agents, action_dim, self.config.seed)
        else:
            self.noise = None

    def _build_actor(self) -> nn.Module:
        """Construct one actor; subclasses may supply another layer family."""
        return Actor(
            self.config.obs_dim,
            self.action_dim,
            self.config.hidden_dim,
        )

    def _build_critic(self) -> nn.Module:
        """Construct the centralized critic for the selected layer family."""
        return CentralizedCritic(
            self.n_agents,
            self.config.obs_dim,
            self.action_dim,
            self.config.critic_hidden_dim,
        )

    @property
    def parameter_count(self) -> int:
        """Count online and target model parameters."""
        modules = (self.actors, self.target_actors, self.critic, self.target_critic)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def act(self, observations: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return one bounded action vector per agent."""
        obs = np.asarray(observations, dtype=np.float32)
        expected_shape = (self.n_agents, self.config.obs_dim)
        if obs.shape != expected_shape:
            raise ValueError(f"observations must have shape {expected_shape}, got {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("observations must contain only finite values")
        observation_tensor = torch.as_tensor(obs, device=self.device)
        with torch.inference_mode():
            actions = torch.stack(
                [
                    actor(observation_tensor[index])
                    for index, actor in enumerate(self.actors)
                ]
            ).cpu().numpy()
        if explore and self.noise is not None:
            actions += self.config.noise_scale * self.noise.sample()
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def reset_noise(self) -> None:
        """Reset correlated exploration noise at an episode boundary."""
        if self.noise is not None:
            self.noise.reset()

    def store_transition(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        next_observations: np.ndarray,
        dones: np.ndarray | bool,
    ) -> None:
        """Add a joint transition to replay."""
        self.replay_buffer.add(observations, actions, rewards, next_observations, dones)

    def learn(self, batch_size: int | None = None) -> dict[str, float]:
        """Sample replay and perform one centralized-critic/actor update."""
        batch = self.replay_buffer.sample(
            batch_size or self.config.batch_size, self.device
        )
        return self.update(batch)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """Update critic, each actor, and all target networks from a minibatch."""
        with torch.no_grad():
            target_actions = torch.stack(
                [
                    actor(batch.next_observations[:, index, :])
                    for index, actor in enumerate(self.target_actors)
                ],
                dim=1,
            )
            target_values = self.target_critic(batch.next_observations, target_actions)
            team_rewards = batch.rewards.mean(dim=1, keepdim=True)
            team_dones = batch.dones.max(dim=1, keepdim=True).values
            targets = team_rewards + self.config.gamma * (1.0 - team_dones) * target_values

        predicted_values = self.critic(batch.observations, batch.actions)
        critic_loss = F.smooth_l1_loss(predicted_values, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_losses: list[float] = []
        try:
            for agent_index, (actor, optimizer) in enumerate(
                zip(self.actors, self.actor_optimizers)
            ):
                optimizer.zero_grad(set_to_none=True)
                joint_actions = []
                for other_index, other_actor in enumerate(self.actors):
                    agent_observations = batch.observations[:, other_index, :]
                    if other_index == agent_index:
                        joint_actions.append(actor(agent_observations))
                    else:
                        with torch.no_grad():
                            joint_actions.append(other_actor(agent_observations))
                actor_loss = -self.critic(
                    batch.observations, torch.stack(joint_actions, dim=1)
                ).mean()
                actor_loss.backward()
                optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

        self._soft_update(self.critic, self.target_critic)
        for actor, target_actor in zip(self.actors, self.target_actors):
            self._soft_update(actor, target_actor)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(np.mean(actor_losses)),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.lerp_(source_parameter, self.config.tau)
