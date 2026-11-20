"""Compare MADDPG and TN-MADDPG inference on real logistics rollouts."""

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
import torch

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.agents.tn_maddpg import TNMADDPG, TNMaddpgConfig
from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.observations import compact_ship_observations
from src.training.checkpoints import load_checkpoint


def _run_case(
    learner_kind: str,
    n_agents: int,
    logistics_values: dict[str, object],
    agent_values: dict[str, object],
    steps: int,
    warmup_calls: int,
    seed: int,
    device_name: str,
    checkpoint_path: str | None,
) -> dict[str, float | int | str]:
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    env_config = LogisticsConfig.from_mapping(
        {**logistics_values, "n_ships": n_agents, "episode_steps": steps}
    )
    env = GlobalLogisticsEnv(env_config)
    checkpoint: dict[str, object] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        agent_values = checkpoint["agent_config"]
        if checkpoint["n_agents"] != n_agents:
            raise ValueError(f"{checkpoint_path} has a different number of agents")
        if checkpoint["action_dim"] != env_config.n_ports + 1:
            raise ValueError(f"{checkpoint_path} has a different action dimension")
    if learner_kind == "maddpg":
        agent_config = MaddpgConfig.from_mapping(agent_values)
        learner: MADDPG = MADDPG(
            n_agents,
            env_config.n_ports + 1,
            agent_config,
            device=device,
        )
    else:
        agent_config = TNMaddpgConfig.from_mapping(agent_values)
        learner = TNMADDPG(
            n_agents,
            env_config.n_ports + 1,
            agent_config,
            device=device,
        )
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, learner, learner_kind)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_rss = max(peak_rss, process.memory_info().rss)

    observations, _ = env.reset(seed=seed)
    if warmup_calls:
        warmup_state = compact_ship_observations(env, observations)
        for _ in range(warmup_calls):
            learner.act(warmup_state, explore=False)
    inference_times: list[float] = []
    total_reward = 0.0
    for _ in range(steps):
        state = compact_ship_observations(env, observations)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        actions = learner.act(state, explore=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times.append((time.perf_counter() - start) * 1_000.0)
        observations, rewards, _, _, _ = env.step_vectorized(actions)
        total_reward += sum(rewards.values())
        peak_rss = max(peak_rss, process.memory_info().rss)

    p50_ms = float(np.percentile(inference_times, 50))
    p95_ms = float(np.percentile(inference_times, 95))
    return {
        "learner": learner_kind,
        "n_agents": n_agents,
        "n_ports": env_config.n_ports,
        "steps": steps,
        "warmup_calls": warmup_calls,
        "parameters": learner.parameter_count,
        "p50_inference_ms": p50_ms,
        "p95_inference_ms": p95_ms,
        "peak_cpu_memory_mb": (peak_rss - baseline_rss) / (1024.0**2),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "decisions_per_second": n_agents / (float(np.mean(inference_times)) / 1_000.0),
        "total_reward": total_reward,
        "mean_reward_per_ship": total_reward / n_agents,
        "checkpoint": checkpoint_path or "fresh_initialization",
    }


def _save_results(results: list[dict[str, float | int | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tn_vs_maddpg.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    counts = sorted({int(row["n_agents"]) for row in results})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    charts = (
        (axes[0, 0], "parameters", "Parameters", "count"),
        (axes[0, 1], "p50_inference_ms", "p50 inference latency", "ms"),
        (axes[1, 0], "peak_cpu_memory_mb", "Peak additional CPU memory", "MiB"),
        (axes[1, 1], "mean_reward_per_ship", "Mean rollout reward per ship", "reward"),
    )
    for axis, key, title, ylabel in charts:
        for learner_kind in ("maddpg", "tn_maddpg"):
            selected = {
                int(row["n_agents"]): row
                for row in results
                if row["learner"] == learner_kind
            }
            axis.plot(
                [count for count in counts if count in selected],
                [float(selected[count][key]) for count in counts if count in selected],
                marker="o",
                label=learner_kind,
            )
            if key == "p50_inference_ms":
                axis.plot(
                    [count for count in counts if count in selected],
                    [
                        float(selected[count]["p95_inference_ms"])
                        for count in counts
                        if count in selected
                    ],
                    marker="x",
                    linestyle="--",
                    label=f"{learner_kind} p95",
                )
        axis.set_title(title)
        axis.set_xlabel("ships / agents")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "tn_vs_maddpg.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--maddpg-config", type=Path, default=Path("configs/maddpg.yaml"))
    parser.add_argument("--tn-config", type=Path, default=Path("configs/tn_maddpg.yaml"))
    parser.add_argument("--maddpg-checkpoint", type=Path)
    parser.add_argument("--tn-checkpoint", type=Path)
    parser.add_argument(
        "--agent-counts", type=int, nargs="+", default=[10, 50, 200, 1_000]
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_calls < 0:
        parser.error("--steps must be positive and warmup calls non-negative")
    if any(not 10 <= count <= 2_000 for count in args.agent_counts):
        parser.error("agent counts must be between 10 and 2000")

    logistics_values = LogisticsConfig.from_yaml(args.env_config).__dict__
    maddpg_values = MaddpgConfig.from_yaml(args.maddpg_config).__dict__
    tn_values = TNMaddpgConfig.from_yaml(args.tn_config).__dict__
    results: list[dict[str, float | int | str]] = []
    for learner_kind, agent_values, checkpoint_path in (
        ("maddpg", maddpg_values, args.maddpg_checkpoint),
        ("tn_maddpg", tn_values, args.tn_checkpoint),
    ):
        for n_agents in args.agent_counts:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                row = executor.submit(
                    _run_case,
                    learner_kind,
                    n_agents,
                    logistics_values,
                    agent_values,
                    args.steps,
                    args.warmup_calls,
                    args.seed,
                    args.device,
                    str(checkpoint_path) if checkpoint_path else None,
                ).result()
            results.append(row)
            print(
                f"{learner_kind} agents={n_agents} params={row['parameters']} "
                f"p50_ms={row['p50_inference_ms']:.3f} "
                f"p95_ms={row['p95_inference_ms']:.3f} "
                f"cpu_mib={row['peak_cpu_memory_mb']:.1f} "
                f"reward_per_ship={row['mean_reward_per_ship']:.3f}"
            )
            gc.collect()
    _save_results(results, args.output_dir)
    print(f"Saved comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
