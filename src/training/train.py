"""Config-driven MADDPG and TN-MADDPG training entry point."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.agents.tn_maddpg import TNMADDPG, TNMaddpgConfig
from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.observations import compact_ship_observations
from src.training.checkpoints import save_checkpoint
from src.training.curriculum import CurriculumTrainer


@dataclass(frozen=True)
class TrainingConfig:
    """Run control, curriculum, logging, and checkpoint settings."""

    episodes: int = 100
    max_steps_per_episode: int = 100
    learning_starts: int = 128
    updates_per_step: int = 1
    checkpoint_interval: int = 10
    curriculum_enabled: bool = True
    seed: int = 7
    log_dir: str = "outputs/tensorboard"
    checkpoint_dir: str = "outputs/checkpoints"

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.max_steps_per_episode < 1:
            raise ValueError("episodes and max_steps_per_episode must be positive")
        if self.learning_starts < 1 or self.updates_per_step < 0:
            raise ValueError("learning_starts must be positive and updates non-negative")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
        return cls(**values)


def _create_learner(
    learner_kind: str,
    n_agents: int,
    action_dim: int,
    agent_config: MaddpgConfig | TNMaddpgConfig,
) -> MADDPG:
    if learner_kind == "maddpg":
        if not isinstance(agent_config, MaddpgConfig):
            raise TypeError("MADDPG requires MaddpgConfig")
        return MADDPG(n_agents, action_dim, agent_config)
    if learner_kind == "tn_maddpg":
        if not isinstance(agent_config, TNMaddpgConfig):
            raise TypeError("TN-MADDPG requires TNMaddpgConfig")
        return TNMADDPG(n_agents, action_dim, agent_config)
    raise ValueError(f"Unsupported learner: {learner_kind}")


def _load_agent_config(path: Path, learner_kind: str) -> MaddpgConfig | TNMaddpgConfig:
    if learner_kind == "tn_maddpg":
        return TNMaddpgConfig.from_yaml(path)
    if learner_kind == "maddpg":
        return MaddpgConfig.from_yaml(path)
    raise ValueError(f"Unsupported learner: {learner_kind}")


def _dense_output(actor: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    output_layer = actor.network[4]
    if hasattr(output_layer, "to_dense"):
        weight = output_layer.to_dense().detach()
        bias = output_layer.bias.detach() if output_layer.bias is not None else None
        return weight, bias
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("actor output layer must be Linear or MPSLinear")
    bias = output_layer.bias.detach() if output_layer.bias is not None else None
    return output_layer.weight.detach(), bias


def _assign_dense_output(
    actor: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    output_layer = actor.network[4]
    if hasattr(output_layer, "to_dense"):
        from src.tensor_nets import MPSLinear

        actor.network[4] = MPSLinear.from_dense(
            output_layer.in_modes,
            output_layer.out_modes,
            output_layer.bond_dim,
            weight,
            bias,
        )
        return
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("actor output layer must be Linear or MPSLinear")
    with torch.no_grad():
        output_layer.weight.copy_(weight)
        if output_layer.bias is not None and bias is not None:
            output_layer.bias.copy_(bias)


def _transfer_actors(
    previous: MADDPG,
    current: MADDPG,
    previous_n_ports: int,
    current_n_ports: int,
) -> None:
    """Preserve shared actor features and port logits across curriculum growth."""
    for old_actor, new_actor in zip(previous.actors, current.actors):
        for layer_index in (0, 2):
            new_actor.network[layer_index].load_state_dict(
                old_actor.network[layer_index].state_dict()
            )
        old_weight, old_bias = _dense_output(old_actor)
        new_weight, new_bias = _dense_output(new_actor)
        shared_ports = min(previous_n_ports, current_n_ports)
        with torch.no_grad():
            new_weight[:shared_ports].copy_(old_weight[:shared_ports])
            new_weight[-1].copy_(old_weight[-1])
            if new_bias is not None and old_bias is not None:
                new_bias[:shared_ports].copy_(old_bias[:shared_ports])
                new_bias[-1].copy_(old_bias[-1])
        _assign_dense_output(new_actor, new_weight, new_bias)

    current.target_actors = copy.deepcopy(current.actors)
    current.actor_optimizers = [
        Adam(actor.parameters(), lr=current.config.learning_rate)
        for actor in current.actors
    ]


def run_training(
    training_config: TrainingConfig,
    env_config: LogisticsConfig,
    agent_config: MaddpgConfig | TNMaddpgConfig,
    learner_kind: str,
    curriculum_path: Path,
    reward_path: Path,
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
    learning_starts_override: int | None = None,
    batch_size_override: int | None = None,
    curriculum_enabled_override: bool | None = None,
) -> dict[str, float | int | str]:
    """Run episodes, updates, TensorBoard logging, and periodic checkpoints."""
    episode_count = (
        training_config.episodes if episodes_override is None else episodes_override
    )
    max_steps = (
        training_config.max_steps_per_episode
        if max_steps_override is None
        else max_steps_override
    )
    learning_starts = (
        training_config.learning_starts
        if learning_starts_override is None
        else learning_starts_override
    )
    if episode_count < 1 or max_steps < 1 or learning_starts < 1:
        raise ValueError("episodes, max steps, and learning-start threshold must be positive")
    if batch_size_override is not None:
        if isinstance(agent_config, TNMaddpgConfig):
            agent_config = TNMaddpgConfig(
                **{**asdict(agent_config), "batch_size": batch_size_override}
            )
        else:
            agent_config = MaddpgConfig(
                **{**asdict(agent_config), "batch_size": batch_size_override}
            )
    use_curriculum = (
        training_config.curriculum_enabled
        if curriculum_enabled_override is None
        else curriculum_enabled_override
    )
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    curriculum = (
        CurriculumTrainer(
            base_env_config=env_config,
            curriculum_config=curriculum_path,
            reward_config=reward_path,
        )
        if use_curriculum
        else None
    )
    log_dir = Path(training_config.log_dir) / learner_kind
    checkpoint_dir = Path(training_config.checkpoint_dir) / learner_kind
    writer = SummaryWriter(log_dir=str(log_dir))
    learner: MADDPG | None = None
    current_n_ports = 0
    final_episode_reward = 0.0
    final_stage_name = "fixed"
    total_updates = 0
    try:
        for episode_index in range(episode_count):
            if curriculum is not None:
                env = curriculum.make_env()
                stage_name = curriculum.current_stage.name
            else:
                env = GlobalLogisticsEnv(env_config, reward_config=reward_path)
                stage_name = "fixed"
            if (
                learner is None
                or learner.n_agents != env.config.n_ships
                or learner.action_dim != env.config.n_ports + 1
            ):
                next_learner = _create_learner(
                    learner_kind,
                    env.config.n_ships,
                    env.config.n_ports + 1,
                    agent_config,
                )
                if learner is not None:
                    _transfer_actors(
                        learner,
                        next_learner,
                        current_n_ports,
                        env.config.n_ports,
                    )
                learner = next_learner
                current_n_ports = env.config.n_ports

            observations, _ = env.reset(seed=training_config.seed + episode_index)
            learner.reset_noise()
            episode_reward = 0.0
            episode_components = {
                name: 0.0
                for name in (
                    "fuel_cost",
                    "on_time_delivery",
                    "sla_violation",
                    "catastrophic_risk",
                    "reward_fuel_cost",
                    "reward_on_time_delivery",
                    "reward_sla_violation",
                    "reward_catastrophic_risk",
                )
            }
            critic_losses: list[float] = []
            actor_losses: list[float] = []
            episode_steps = 0
            for step_index in range(max_steps):
                state = compact_ship_observations(env, observations)
                actions = learner.act(state)
                next_observations, rewards, terminations, truncations, infos = (
                    env.step_vectorized(actions)
                )
                next_state = compact_ship_observations(env, next_observations)
                reward_array = np.array(
                    [rewards[agent] for agent in env.possible_agents],
                    dtype=np.float32,
                )
                episode_reward += float(reward_array.sum())
                for name in episode_components:
                    episode_components[name] += sum(
                        float(infos[agent][name]) for agent in env.possible_agents
                    )
                done = np.array(
                    [
                        terminations[agent] or truncations[agent]
                        for agent in env.possible_agents
                    ],
                    dtype=np.float32,
                )
                if step_index + 1 == max_steps:
                    done.fill(1.0)
                learner.store_transition(state, actions, reward_array, next_state, done)
                if (
                    len(learner.replay_buffer) >= learning_starts
                    and len(learner.replay_buffer) >= learner.config.batch_size
                ):
                    for _ in range(training_config.updates_per_step):
                        losses = learner.learn()
                        critic_losses.append(losses["critic_loss"])
                        actor_losses.append(losses["actor_loss"])
                        total_updates += 1
                observations = next_observations
                episode_steps += 1
                if any(terminations.values()) or any(truncations.values()):
                    break

            mean_reward_per_ship = episode_reward / env.config.n_ships
            final_episode_reward = mean_reward_per_ship
            final_stage_name = stage_name
            writer.add_scalar("episode/reward_per_ship", mean_reward_per_ship, episode_index)
            writer.add_scalar("episode/steps", episode_steps, episode_index)
            writer.add_scalar("episode/ships", env.config.n_ships, episode_index)
            writer.add_scalar("episode/ports", env.config.n_ports, episode_index)
            writer.add_scalar("episode/stage", curriculum.stage_index if curriculum else 0, episode_index)
            for name, value in episode_components.items():
                writer.add_scalar(
                    f"reward_components/{name}",
                    value / max(episode_steps * env.config.n_ships, 1),
                    episode_index,
                )
            if critic_losses:
                writer.add_scalar("loss/critic", float(np.mean(critic_losses)), episode_index)
                writer.add_scalar("loss/actor", float(np.mean(actor_losses)), episode_index)
            moving_average = (
                curriculum.moving_average_reward if curriculum is not None else None
            )
            if curriculum is not None:
                promoted = curriculum.record_episode(mean_reward_per_ship)
                if moving_average is not None:
                    writer.add_scalar(
                        "curriculum/moving_average_reward",
                        moving_average,
                        episode_index,
                    )
                if promoted:
                    writer.add_scalar("curriculum/promoted", 1.0, episode_index)
            writer.flush()
            completed_episodes = episode_index + 1
            if completed_episodes % training_config.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir
                    / f"{learner_kind}_stage-{stage_name}_episode-{completed_episodes:06d}.pt",
                    learner,
                    learner_kind,
                    asdict(agent_config),
                    completed_episodes,
                    stage_name,
                )
            env.close()
            print(
                f"episode={completed_episodes} stage={stage_name} ships={env.config.n_ships} "
                f"reward_per_ship={mean_reward_per_ship:.4f} steps={episode_steps} "
                f"updates={total_updates}"
            )
    finally:
        writer.close()

    if learner is None:
        raise RuntimeError("training did not create a learner")
    final_checkpoint = checkpoint_dir / f"{learner_kind}_final.pt"
    save_checkpoint(
        final_checkpoint,
        learner,
        learner_kind,
        asdict(agent_config),
        episode_count,
        final_stage_name,
    )
    result: dict[str, float | int | str] = {
        "learner": learner_kind,
        "episodes": episode_count,
        "final_reward_per_ship": final_episode_reward,
        "final_stage": final_stage_name,
        "updates": total_updates,
        "checkpoint": str(final_checkpoint),
    }
    print(f"TRAIN_RESULT={json.dumps(result, sort_keys=True)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--curriculum-config", type=Path, default=Path("configs/curriculum.yaml"))
    parser.add_argument("--reward-config", type=Path, default=Path("configs/reward.yaml"))
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--learner", choices=("maddpg", "tn_maddpg"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()

    training_config = TrainingConfig.from_yaml(args.training_config)
    learner_kind = args.learner or "maddpg"
    agent_config_path = args.agent_config or Path(
        "configs/tn_maddpg.yaml" if learner_kind == "tn_maddpg" else "configs/maddpg.yaml"
    )
    result = run_training(
        training_config,
        LogisticsConfig.from_yaml(args.env_config),
        _load_agent_config(agent_config_path, learner_kind),
        learner_kind,
        args.curriculum_config,
        args.reward_config,
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
        learning_starts_override=args.learning_starts,
        batch_size_override=args.batch_size,
        curriculum_enabled_override=False if args.no_curriculum else None,
    )
    print(f"Training finished: {result}")


if __name__ == "__main__":
    main()
