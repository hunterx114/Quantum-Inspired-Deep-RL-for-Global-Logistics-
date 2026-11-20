"""Tests for reward-based curriculum promotion."""

from __future__ import annotations

import pytest

from src.envs import LogisticsConfig
from src.training import CurriculumConfig, CurriculumStage, CurriculumTrainer


def _curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        moving_average_window=2,
        stages=(
            CurriculumStage(
                name="small",
                n_ports=100,
                n_ships=10,
                port_closure_probability=0.001,
                fuel_spike_probability=0.001,
                promotion_threshold=1.0,
            ),
            CurriculumStage(
                name="large",
                n_ports=150,
                n_ships=20,
                port_closure_probability=0.01,
                fuel_spike_probability=0.01,
                promotion_threshold=None,
            ),
        ),
    )


def test_curriculum_promotes_after_full_reward_window_and_builds_new_env() -> None:
    trainer = CurriculumTrainer(
        base_env_config=LogisticsConfig(
            n_ports=100,
            n_ships=10,
            grid_height=4,
            grid_width=8,
            graph_neighbors=2,
            storm_cells=0,
        ),
        curriculum_config=_curriculum(),
    )
    assert trainer.make_env().config.n_ships == 10
    assert not trainer.record_episode(2.0)
    assert trainer.moving_average_reward == 2.0
    assert trainer.record_episode(0.0)
    assert trainer.current_stage.name == "large"
    assert trainer.episodes_in_stage == 0
    stage_env = trainer.make_env()
    assert stage_env.config.n_ports == 150
    assert stage_env.config.n_ships == 20
    assert stage_env.config.port_closure_probability == 0.01


def test_curriculum_rejects_non_increasing_stages_and_non_finite_reward() -> None:
    with pytest.raises(ValueError, match="n_ports must increase"):
        CurriculumConfig(
            moving_average_window=1,
            stages=(
                _curriculum().stages[0],
                CurriculumStage(
                    name="bad",
                    n_ports=100,
                    n_ships=20,
                    port_closure_probability=0.01,
                    fuel_spike_probability=0.01,
                    promotion_threshold=None,
                ),
            ),
        )

    trainer = CurriculumTrainer(curriculum_config=_curriculum())
    with pytest.raises(ValueError, match="finite"):
        trainer.record_episode(float("nan"))
