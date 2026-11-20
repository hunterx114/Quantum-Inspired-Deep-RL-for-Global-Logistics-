"""Tests for weighted logistics reward calculation and environment logging."""

from __future__ import annotations

import numpy as np
import pytest

from src.envs import GlobalLogisticsEnv, LogisticsConfig, RewardConfig, compute_reward


def test_reward_formula_and_signed_components() -> None:
    config = RewardConfig(
        fuel_cost_weight=2.0,
        on_time_delivery_weight=3.0,
        sla_violation_weight=4.0,
        catastrophic_risk_weight=5.0,
    )
    components = compute_reward(
        fuel_cost=np.array([1.0, 2.0]),
        on_time_delivery=np.array([4.0, 1.0]),
        sla_violation=np.array([1.0, 0.0]),
        catastrophic_risk=np.array([0.5, 2.0]),
        config=config,
    )

    np.testing.assert_allclose(components.reward, [3.5, -11.0])
    np.testing.assert_allclose(components.reward_fuel_cost, [-2.0, -4.0])
    np.testing.assert_allclose(components.reward_on_time_delivery, [12.0, 3.0])
    np.testing.assert_allclose(components.reward_sla_violation, [-4.0, 0.0])
    np.testing.assert_allclose(components.reward_catastrophic_risk, [-2.5, -10.0])


def test_reward_rejects_non_finite_and_negative_components() -> None:
    config = RewardConfig()
    with pytest.raises(ValueError, match="finite"):
        compute_reward(np.nan, 0.0, 0.0, 0.0, config)
    with pytest.raises(ValueError, match="non-negative"):
        compute_reward(0.0, -1.0, 0.0, 0.0, config)


def test_environment_info_logs_every_reward_component() -> None:
    config = LogisticsConfig(
        n_ports=100,
        n_ships=10,
        grid_height=4,
        grid_width=8,
        graph_neighbors=2,
        episode_steps=2,
        storm_cells=0,
    )
    env = GlobalLogisticsEnv(
        config,
        reward_config={
            "fuel_cost_weight": 0.2,
            "on_time_delivery_weight": 2.0,
            "sla_violation_weight": 3.0,
            "catastrophic_risk_weight": 4.0,
        },
    )
    env.reset(seed=52)
    actions = np.zeros((config.n_ships, config.n_ports + 1), dtype=np.float32)
    _, rewards, _, _, infos = env.step_vectorized(actions)

    for agent, info in infos.items():
        component_names = (
            "reward_fuel_cost",
            "reward_on_time_delivery",
            "reward_sla_violation",
            "reward_catastrophic_risk",
        )
        assert all(name in info for name in component_names)
        assert info["reward"] == pytest.approx(rewards[agent])
        assert sum(info[name] for name in component_names) == pytest.approx(
            rewards[agent]
        )


def test_sla_expirations_are_shared_and_late_deliveries_are_not_on_time() -> None:
    config = LogisticsConfig(
        n_ports=100,
        n_ships=10,
        grid_height=4,
        grid_width=8,
        graph_neighbors=2,
        episode_steps=2,
        mean_demand=0.0,
        sla_steps=1,
        storm_cells=0,
    )
    env = GlobalLogisticsEnv(config, reward_config=RewardConfig())
    env.reset(seed=77)
    env.current_port[:] = 0
    env.position[:] = env.port_coordinates[0]
    env.port_demand[0] = 10.0
    env.demand_deadline[0] = 1
    actions = np.full((config.n_ships, config.n_ports + 1), -1.0, dtype=np.float32)
    actions[:, 0] = 1.0

    _, _, _, _, infos = env.step_vectorized(actions)

    assert sum(info["sla_violation"] for info in infos.values()) == pytest.approx(10.0)
    assert sum(info["on_time_delivery"] for info in infos.values()) == pytest.approx(0.0)
