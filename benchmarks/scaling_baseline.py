"""Measure classical MADDPG parameter, memory, and runtime scaling."""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.observations import compact_ship_observations


def _run_case(
    n_agents: int,
    logistics_values: dict[str, object],
    maddpg_values: dict[str, object],
    steps: int,
    seed: int,
) -> dict[str, float | int]:
    """Run one scale case in a fresh worker process for isolated memory stats."""
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    env_config = LogisticsConfig.from_mapping(
        {**logistics_values, "n_ships": n_agents, "episode_steps": steps}
    )
    agent_config = MaddpgConfig.from_mapping(
        {**maddpg_values, "seed": seed}
    )
    env = GlobalLogisticsEnv(env_config)
    learner = MADDPG(
        n_agents=n_agents,
        action_dim=env_config.n_ports + 1,
        config=agent_config,
    )
    peak_rss = max(peak_rss, process.memory_info().rss)
    observations, _ = env.reset(seed=seed)
    learner.reset_noise()
    elapsed_ms: list[float] = []
    total_reward = 0.0
    for _ in range(steps):
        started = time.perf_counter()
        policy_observations = compact_ship_observations(env, observations)
        actions = learner.act(policy_observations)
        observations, rewards, _, _, _ = env.step_vectorized(actions)
        elapsed_ms.append((time.perf_counter() - started) * 1_000.0)
        total_reward += sum(rewards.values())
        peak_rss = max(peak_rss, process.memory_info().rss)
    mean_step_ms = float(np.mean(elapsed_ms))
    return {
        "n_agents": n_agents,
        "n_ports": env_config.n_ports,
        "steps": steps,
        "parameters": learner.parameter_count,
        "peak_cpu_memory_mb": (peak_rss - baseline_rss) / (1024.0**2),
        "mean_step_time_ms": mean_step_ms,
        "p95_step_time_ms": float(np.percentile(elapsed_ms, 95)),
        "decisions_per_second": n_agents / (mean_step_ms / 1_000.0),
        "total_reward": total_reward,
        "mean_reward_per_agent_step": total_reward / (n_agents * steps),
    }


def _save_results(
    results: list[dict[str, float | int]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scaling_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    counts = [int(result["n_agents"]) for result in results]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, key, title, ylabel in (
        (axes[0], "parameters", "Model parameters", "parameters"),
        (axes[1], "peak_cpu_memory_mb", "Peak additional CPU memory", "MiB"),
        (axes[2], "mean_step_time_ms", "Policy + environment step time", "ms"),
    ):
        axis.plot(counts, [float(result[key]) for result in results], marker="o")
        axis.set_title(title)
        axis.set_xlabel("ships / agents")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "scaling_baseline.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--agent-config", type=Path, default=Path("configs/maddpg.yaml"))
    parser.add_argument(
        "--agent-counts", type=int, nargs="+", default=[10, 50, 200, 1_000]
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    logistics_values = LogisticsConfig.from_yaml(args.config).__dict__
    maddpg_values = MaddpgConfig.from_yaml(args.agent_config).__dict__
    results: list[dict[str, float | int]] = []
    for n_agents in args.agent_counts:
        if not 10 <= n_agents <= 2_000:
            parser.error("agent counts must be between 10 and 2000")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            result = executor.submit(
                _run_case,
                n_agents,
                logistics_values,
                maddpg_values,
                args.steps,
                args.seed,
            ).result()
        results.append(result)
        print(
            f"agents={n_agents} params={result['parameters']} "
            f"peak_cpu_mib={result['peak_cpu_memory_mb']:.1f} "
            f"step_ms={result['mean_step_time_ms']:.3f} "
            f"decisions_per_sec={result['decisions_per_second']:.1f} "
            f"reward={result['total_reward']:.3f}"
        )
        gc.collect()

    _save_results(results, args.output_dir)
    print(f"Saved results and plot to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
