"""Tensor-network MADDPG with MPS actor and centralized-critic layers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.tensor_nets import MPSLinear


@dataclass(frozen=True)
class TNMaddpgConfig(MaddpgConfig):
    """MADDPG settings plus the MPS bond dimension."""

    bond_dim: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TNMaddpgConfig":
        """Load TN-MADDPG settings from YAML."""
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TNMaddpgConfig":
        """Create TN settings and reject unknown keys."""
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown TN-MADDPG configuration keys: {sorted(unknown)}")
        return cls(**values)


class TensorNetworkActor(nn.Module):
    """Per-agent deterministic actor built from trainable MPSLinear layers."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, bond_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            MPSLinear(obs_dim, hidden_dim, bond_dim),
            nn.ReLU(),
            MPSLinear(hidden_dim, hidden_dim, bond_dim),
            nn.ReLU(),
            MPSLinear(hidden_dim, action_dim, bond_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class TensorNetworkCritic(nn.Module):
    """Centralized joint-state/action critic using MPSLinear throughout."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        bond_dim: int,
    ) -> None:
        super().__init__()
        joint_dim = n_agents * (obs_dim + action_dim)
        self.network = nn.Sequential(
            MPSLinear(joint_dim, hidden_dim, bond_dim),
            nn.ReLU(),
            MPSLinear(hidden_dim, hidden_dim, bond_dim),
            nn.ReLU(),
            MPSLinear(hidden_dim, 1, bond_dim),
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


class TNMADDPG(MADDPG):
    """MADDPG training loop with tensor-train factorized actors and critic."""

    def __init__(
        self,
        n_agents: int,
        action_dim: int,
        config: TNMaddpgConfig | MaddpgConfig | dict[str, Any] | str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if config is None:
            tn_config = TNMaddpgConfig()
        elif isinstance(config, TNMaddpgConfig):
            tn_config = config
        elif isinstance(config, MaddpgConfig):
            values = {
                field.name: getattr(config, field.name)
                for field in fields(MaddpgConfig)
            }
            tn_config = TNMaddpgConfig.from_mapping(values)
        elif isinstance(config, (str, Path)):
            tn_config = TNMaddpgConfig.from_yaml(config)
        else:
            tn_config = TNMaddpgConfig.from_mapping(config)
        self.bond_dim = tn_config.bond_dim
        self.tn_config = tn_config
        super().__init__(n_agents, action_dim, tn_config, device)

    def _build_actor(self) -> nn.Module:
        return TensorNetworkActor(
            self.config.obs_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.bond_dim,
        )

    def _build_critic(self) -> nn.Module:
        return TensorNetworkCritic(
            self.n_agents,
            self.config.obs_dim,
            self.action_dim,
            self.config.critic_hidden_dim,
            self.bond_dim,
        )
