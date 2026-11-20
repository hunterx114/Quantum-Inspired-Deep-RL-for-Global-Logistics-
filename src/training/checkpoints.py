"""Portable state-dict checkpoint helpers for MADDPG variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.agents.maddpg import MADDPG


def save_checkpoint(
    path: str | Path,
    learner: MADDPG,
    learner_kind: str,
    agent_config: dict[str, Any],
    episode: int,
    stage_name: str,
) -> None:
    """Save model and optimizer state without pickling Python model objects."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "learner_kind": learner_kind,
        "n_agents": learner.n_agents,
        "action_dim": learner.action_dim,
        "agent_config": agent_config,
        "actors": [actor.state_dict() for actor in learner.actors],
        "target_actors": [actor.state_dict() for actor in learner.target_actors],
        "critic": learner.critic.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "actor_optimizers": [
            optimizer.state_dict() for optimizer in learner.actor_optimizers
        ],
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "episode": episode,
        "stage_name": stage_name,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    learner: MADDPG,
    expected_learner_kind: str,
) -> dict[str, Any]:
    """Load a compatible checkpoint into an already constructed learner."""
    checkpoint = torch.load(
        Path(path),
        map_location=learner.device,
        weights_only=True,
    )
    if checkpoint["learner_kind"] != expected_learner_kind:
        raise ValueError(
            f"Checkpoint learner is {checkpoint['learner_kind']!r}, "
            f"expected {expected_learner_kind!r}"
        )
    if checkpoint["n_agents"] != learner.n_agents:
        raise ValueError("checkpoint agent count does not match the learner")
    if checkpoint["action_dim"] != learner.action_dim:
        raise ValueError("checkpoint action dimension does not match the learner")
    for actor, state in zip(learner.actors, checkpoint["actors"], strict=True):
        actor.load_state_dict(state)
    for actor, state in zip(
        learner.target_actors, checkpoint["target_actors"], strict=True
    ):
        actor.load_state_dict(state)
    learner.critic.load_state_dict(checkpoint["critic"])
    learner.target_critic.load_state_dict(checkpoint["target_critic"])
    for optimizer, state in zip(
        learner.actor_optimizers, checkpoint["actor_optimizers"], strict=True
    ):
        optimizer.load_state_dict(state)
    learner.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return checkpoint
