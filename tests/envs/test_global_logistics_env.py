"""Tests for the global logistics multi-agent simulator."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.global_logistics import haversine_km


@pytest.fixture
def config() -> LogisticsConfig:
    return LogisticsConfig(
        n_ports=100,
        n_ships=10,
        grid_height=6,
        grid_width=12,
        graph_neighbors=4,
        episode_steps=3,
        storm_cells=2,
    )


def test_reset_is_deterministic_and_observations_match_spaces(
    config: LogisticsConfig,
) -> None:
    env = GlobalLogisticsEnv(config)
    observations, infos = env.reset(seed=1234)
    second_env = GlobalLogisticsEnv(config)
    second_observations, second_infos = second_env.reset(seed=1234)

    assert observations.keys() == second_observations.keys()
    assert all(
        np.array_equal(
            observations[agent]["ship_state"],
            second_observations[agent]["ship_state"],
        )
        for agent in env.agents
    )
    assert infos["ship_0"]["seed"] == second_infos["ship_0"]["seed"]
    for agent, observation in observations.items():
        assert env.observation_space(agent).contains(observation)
        assert env.action_space(agent).shape == (config.n_ports + 1,)


def test_vectorized_step_matches_mapping_api(config: LogisticsConfig) -> None:
    env = GlobalLogisticsEnv(config)
    env.reset(seed=23)
    action_matrix = np.zeros((config.n_ships, config.n_ports + 1), dtype=np.float32)
    action_matrix[:, 5] = 1.0
    vectorized_result = env.step_vectorized(action_matrix)

    second_env = GlobalLogisticsEnv(config)
    second_env.reset(seed=23)
    mapped_result = second_env.step(
        {
            agent: action_matrix[index]
            for index, agent in enumerate(second_env.agents)
        }
    )
    assert vectorized_result[1] == mapped_result[1]
    assert vectorized_result[2:4] == mapped_result[2:4]
    for agent in env.possible_agents:
        np.testing.assert_array_equal(
            vectorized_result[0][agent]["ship_state"],
            mapped_result[0][agent]["ship_state"],
        )


def test_events_metrics_and_episode_horizon(config: LogisticsConfig) -> None:
    env = GlobalLogisticsEnv(config)
    env.reset(seed=19)
    env.port_closed_steps[:] = 1
    env.storm_centers[:] = env.position[0]
    actions = np.zeros((config.n_ships, config.n_ports + 1), dtype=np.float32)

    observations, rewards, terminations, truncations, infos = env.step_vectorized(
        actions
    )
    assert all(np.isfinite(list(rewards.values())))
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert all("catastrophic_risk" in info for info in infos.values())
    assert observations["ship_0"]["weather_grid"].shape == (
        config.grid_height,
        config.grid_width,
        2,
    )
    env.step_vectorized(actions)
    _, _, _, truncations, _ = env.step_vectorized(actions)
    assert all(truncations.values())
    assert env.agents == []


def test_invalid_configuration_and_action_shape_are_reported(
    config: LogisticsConfig,
) -> None:
    with pytest.raises(ValueError, match="n_ports"):
        LogisticsConfig(n_ports=99)

    env = GlobalLogisticsEnv(config)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="Expected action shape"):
        env.step_vectorized(np.zeros((config.n_ships, config.n_ports)))


def test_haversine_distance_is_zero_for_same_coordinate() -> None:
    assert float(haversine_km(0.0, 0.0, 0.0, 0.0)) == 0.0


def test_pettingzoo_parallel_api(config: LogisticsConfig) -> None:
    parallel_api_test(GlobalLogisticsEnv(config), num_cycles=5)
