"""Run a reproducible random-policy episode for the logistics environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.envs import GlobalLogisticsEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    env = GlobalLogisticsEnv(args.config)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0
    steps = 0
    while env.agents and steps < args.steps:
        actions = rng.uniform(
            -1.0,
            1.0,
            size=(env.config.n_ships, env.config.n_ports + 1),
        ).astype(np.float32)
        _, rewards, _, truncated, _ = env.step_vectorized(actions)
        total_reward += sum(rewards.values())
        steps += 1
        if any(truncated.values()):
            break
    print(
        f"seed={args.seed} steps={steps} ships={env.config.n_ships} "
        f"ports={env.config.n_ports} reward={total_reward:.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
