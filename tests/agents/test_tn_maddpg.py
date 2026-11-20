"""Integration tests for tensor-network MADDPG."""

from __future__ import annotations

import numpy as np

from src.agents.tn_maddpg import TNMADDPG, TNMaddpgConfig, TensorNetworkActor, TensorNetworkCritic
from src.tensor_nets import MPSLinear


def test_tn_maddpg_uses_mps_in_actor_and_centralized_critic() -> None:
    config = TNMaddpgConfig(
        obs_dim=4,
        hidden_dim=8,
        critic_hidden_dim=12,
        bond_dim=2,
        replay_capacity=8,
        batch_size=2,
        noise_type=None,
        seed=3,
    )
    learner = TNMADDPG(n_agents=2, action_dim=3, config=config)
    assert isinstance(learner.actors[0], TensorNetworkActor)
    assert isinstance(learner.critic, TensorNetworkCritic)
    assert any(isinstance(module, MPSLinear) for module in learner.actors[0].modules())
    assert any(isinstance(module, MPSLinear) for module in learner.critic.modules())

    actions = learner.act(np.zeros((2, 4), dtype=np.float32), explore=False)
    assert actions.shape == (2, 3)
    assert np.isfinite(actions).all()


def test_tn_maddpg_supports_replay_updates() -> None:
    learner = TNMADDPG(
        n_agents=2,
        action_dim=2,
        config=TNMaddpgConfig(
            obs_dim=3,
            hidden_dim=6,
            critic_hidden_dim=8,
            bond_dim=2,
            replay_capacity=4,
            batch_size=2,
            noise_type=None,
            seed=9,
        ),
    )
    rng = np.random.default_rng(12)
    for _ in range(2):
        state = rng.normal(size=(2, 3)).astype(np.float32)
        action = rng.normal(size=(2, 2)).astype(np.float32)
        next_state = rng.normal(size=(2, 3)).astype(np.float32)
        learner.store_transition(state, action, 1.0, next_state, False)

    losses = learner.learn()
    assert np.isfinite(list(losses.values())).all()
