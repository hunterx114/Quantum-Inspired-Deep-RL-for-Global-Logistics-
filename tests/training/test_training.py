"""Tests for checkpoint persistence and curriculum policy expansion."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.agents.tn_maddpg import TNMADDPG, TNMaddpgConfig
from src.envs import LogisticsConfig
from src.tensor_nets import MPSLinear
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.train import (
    TrainingConfig,
    _assign_dense_output,
    _dense_output,
    _transfer_actors,
    run_training,
)


def _learner(learner_kind: str, n_agents: int, action_dim: int):
    if learner_kind == "maddpg":
        config = MaddpgConfig(
            obs_dim=4,
            hidden_dim=8,
            critic_hidden_dim=10,
            replay_capacity=4,
            batch_size=2,
            noise_type=None,
            seed=14,
        )
        return MADDPG(n_agents, action_dim, config)
    config = TNMaddpgConfig(
        obs_dim=4,
        hidden_dim=8,
        critic_hidden_dim=10,
        bond_dim=16,
        replay_capacity=4,
        batch_size=2,
        noise_type=None,
        seed=14,
    )
    return TNMADDPG(n_agents, action_dim, config)


def test_curriculum_transfer_preserves_existing_ports_and_speed() -> None:
    for learner_kind in ("maddpg", "tn_maddpg"):
        previous = _learner(learner_kind, n_agents=2, action_dim=4)
        current = _learner(learner_kind, n_agents=3, action_dim=6)
        for index, actor in enumerate(previous.actors):
            weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
            weight = weight / 32.0 + index
            bias = torch.arange(4, dtype=torch.float32) + index
            _assign_dense_output(actor, weight, bias)

        old_weights = [_dense_output(actor) for actor in previous.actors]
        _transfer_actors(previous, current, previous_n_ports=3, current_n_ports=5)

        for index, old_actor in enumerate(previous.actors):
            for old_parameter, new_parameter in zip(
                old_actor.network[0].parameters(),
                current.actors[index].network[0].parameters(),
                strict=True,
            ):
                torch.testing.assert_close(new_parameter, old_parameter)
            new_weight, new_bias = _dense_output(current.actors[index])
            old_weight, old_bias = old_weights[index]
            torch.testing.assert_close(new_weight[:3], old_weight[:3])
            torch.testing.assert_close(new_weight[-1], old_weight[-1])
            assert new_bias is not None and old_bias is not None
            torch.testing.assert_close(new_bias[:3], old_bias[:3])
            torch.testing.assert_close(new_bias[-1], old_bias[-1])
            if learner_kind == "tn_maddpg":
                assert isinstance(current.actors[index].network[4], MPSLinear)


def test_checkpoint_round_trip_restores_actor_and_critic_state(tmp_path) -> None:
    config = MaddpgConfig(
        obs_dim=4,
        hidden_dim=8,
        critic_hidden_dim=10,
        replay_capacity=4,
        batch_size=2,
        noise_type=None,
        seed=18,
    )
    trained = MADDPG(n_agents=2, action_dim=3, config=config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained, "maddpg", asdict(config), 3, "test")
    restored = MADDPG(n_agents=2, action_dim=3, config=config)
    checkpoint = load_checkpoint(path, restored, "maddpg")

    assert checkpoint["episode"] == 3
    for original, loaded in zip(trained.actors, restored.actors, strict=True):
        for original_parameter, loaded_parameter in zip(
            original.parameters(), loaded.parameters(), strict=True
        ):
            torch.testing.assert_close(original_parameter, loaded_parameter)


def test_training_promotes_and_resizes_policy_for_next_stage(tmp_path) -> None:
    curriculum_path = tmp_path / "curriculum.yaml"
    curriculum_path.write_text(
        yaml.safe_dump(
            {
                "moving_average_window": 1,
                "stages": [
                    {
                        "name": "small",
                        "n_ports": 100,
                        "n_ships": 10,
                        "port_closure_probability": 0.0,
                        "fuel_spike_probability": 0.0,
                        "promotion_threshold": -1_000_000.0,
                    },
                    {
                        "name": "large",
                        "n_ports": 101,
                        "n_ships": 20,
                        "port_closure_probability": 0.01,
                        "fuel_spike_probability": 0.01,
                        "promotion_threshold": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    run_config = TrainingConfig(
        episodes=2,
        max_steps_per_episode=2,
        learning_starts=2,
        updates_per_step=0,
        checkpoint_interval=2,
        curriculum_enabled=True,
        seed=6,
        log_dir=str(tmp_path / "tensorboard"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    result = run_training(
        run_config,
        LogisticsConfig(
            n_ports=100,
            n_ships=10,
            grid_height=4,
            grid_width=8,
            graph_neighbors=2,
            episode_steps=2,
            storm_cells=0,
        ),
        MaddpgConfig(
            obs_dim=16,
            hidden_dim=8,
            critic_hidden_dim=8,
            replay_capacity=16,
            batch_size=2,
            noise_type=None,
            seed=6,
        ),
        "maddpg",
        curriculum_path,
        Path("configs/reward.yaml"),
    )

    assert result["episodes"] == 2
    assert result["final_stage"] == "large"
    assert Path(str(result["checkpoint"])).is_file()
