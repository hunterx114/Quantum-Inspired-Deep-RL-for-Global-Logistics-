"""Focused tests for the classical MADDPG implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.maddpg import (
    MADDPG,
    GaussianNoise,
    MaddpgConfig,
    OUNoise,
    ReplayBuffer,
)


@pytest.fixture
def agent() -> MADDPG:
    config = MaddpgConfig(
        obs_dim=6,
        hidden_dim=16,
        critic_hidden_dim=24,
        batch_size=4,
        replay_capacity=8,
        seed=11,
    )
    return MADDPG(n_agents=3, action_dim=2, config=config)


def test_actor_outputs_are_bounded_and_noise_is_resettable(agent: MADDPG) -> None:
    observations = np.ones((agent.n_agents, agent.config.obs_dim), dtype=np.float32)
    actions = agent.act(observations)
    assert actions.shape == (agent.n_agents, agent.action_dim)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)
    agent.reset_noise()
    assert isinstance(agent.noise, OUNoise)
    np.testing.assert_array_equal(agent.noise.state, np.zeros_like(agent.noise.state))


def test_replay_ring_buffer_overwrites_oldest_transition() -> None:
    buffer = ReplayBuffer(capacity=2, n_agents=1, obs_dim=1, action_dim=1, seed=3)
    for value in (1.0, 2.0, 3.0):
        buffer.add(
            np.array([[value]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            0.0,
            np.array([[value + 1.0]], dtype=np.float32),
            False,
        )
    batch = buffer.sample(2, torch.device("cpu"))
    assert sorted(batch.observations[:, 0, 0].tolist()) == [2.0, 3.0]


def test_gaussian_noise_is_reproducible() -> None:
    first = GaussianNoise(n_agents=2, action_dim=3, seed=5)
    second = GaussianNoise(n_agents=2, action_dim=3, seed=5)
    np.testing.assert_array_equal(first.sample(), second.sample())


def test_learning_updates_parameters_and_returns_finite_losses(agent: MADDPG) -> None:
    rng = np.random.default_rng(4)
    for _ in range(6):
        observations = rng.normal(size=(agent.n_agents, agent.config.obs_dim)).astype(
            np.float32
        )
        actions = rng.uniform(-1.0, 1.0, size=(agent.n_agents, agent.action_dim)).astype(
            np.float32
        )
        next_observations = rng.normal(
            size=(agent.n_agents, agent.config.obs_dim)
        ).astype(np.float32)
        agent.store_transition(observations, actions, rng.normal(size=agent.n_agents), next_observations, False)

    previous = next(agent.actors[0].parameters()).detach().clone()
    losses = agent.learn()
    assert np.isfinite(list(losses.values())).all()
    assert not torch.equal(previous, next(agent.actors[0].parameters()))


def test_scaling_parameter_count_increases_with_agent_count() -> None:
    config = MaddpgConfig(obs_dim=4, hidden_dim=8, critic_hidden_dim=12)
    small = MADDPG(n_agents=2, action_dim=3, config=config)
    large = MADDPG(n_agents=4, action_dim=3, config=config)
    assert large.parameter_count > small.parameter_count


def test_invalid_action_shape_and_configuration_fail_explicitly(agent: MADDPG) -> None:
    with pytest.raises(ValueError, match="observations must have shape"):
        agent.act(np.zeros((agent.n_agents, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="noise_type"):
        MaddpgConfig(noise_type="invalid")
