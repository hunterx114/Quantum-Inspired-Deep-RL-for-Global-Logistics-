"""Evaluate logistics response to deterministic disruption scenarios."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from src.agents.maddpg import MADDPG, MaddpgConfig
from src.agents.tn_maddpg import TNMADDPG, TNMaddpgConfig
from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.observations import compact_ship_observations
from src.training.checkpoints import load_checkpoint


SCENARIOS = ("port_closure", "fuel_spike", "cascade", "storm_front")


def _policy(
    policy_name: str,
    env: GlobalLogisticsEnv,
    seed: int,
    device: str,
    checkpoint_path: Path | None,
    maddpg_config: MaddpgConfig,
    tn_config: TNMaddpgConfig,
) -> tuple[MADDPG | None, np.random.Generator]:
    rng = np.random.default_rng(seed)
    if policy_name == "random":
        if checkpoint_path is not None:
            raise ValueError("random policy cannot load a checkpoint")
        return None, rng
    learner_kind = policy_name
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        values = checkpoint["agent_config"]
        if checkpoint["n_agents"] != env.config.n_ships:
            raise ValueError("checkpoint agent count does not match stress environment")
        if checkpoint["action_dim"] != env.config.n_ports + 1:
            raise ValueError("checkpoint action size does not match stress environment")
        if learner_kind == "maddpg":
            learner = MADDPG(
                env.config.n_ships,
                env.config.n_ports + 1,
                MaddpgConfig.from_mapping(values),
                device=device,
            )
        else:
            learner = TNMADDPG(
                env.config.n_ships,
                env.config.n_ports + 1,
                TNMaddpgConfig.from_mapping(values),
                device=device,
            )
        load_checkpoint(checkpoint_path, learner, learner_kind)
        return learner, rng
    if policy_name == "maddpg":
        return (
            MADDPG(
                env.config.n_ships,
                env.config.n_ports + 1,
                maddpg_config,
                device=device,
            ),
            rng,
        )
    if policy_name == "tn_maddpg":
        return (
            TNMADDPG(
                env.config.n_ships,
                env.config.n_ports + 1,
                tn_config,
                device=device,
            ),
            rng,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def _inject_event(
    env: GlobalLogisticsEnv,
    scenario: str,
    destinations: np.ndarray,
    duration: int,
) -> None:
    if scenario == "port_closure":
        impacted = np.unique(destinations)
    elif scenario == "cascade":
        hub_count = max(1, env.config.n_ports // 20)
        hubs = np.argsort(env.port_capacity)[-hub_count:]
        impacted = np.unique(
            np.concatenate((hubs, env.edge_indices[hubs].reshape(-1), destinations))
        )
        env.fuel_spike_steps = duration + 1
    else:
        impacted = np.empty(0, dtype=np.int32)

    if scenario in ("port_closure", "cascade") and impacted.size:
        env.port_closed_steps[impacted] = duration + 1
    if scenario == "fuel_spike":
        env.fuel_spike_steps = duration + 1
    if scenario == "storm_front":
        origin = env.position
        target = env.port_coordinates[destinations]
        center_latitude = float(np.mean((origin[:, 0] + target[:, 0]) / 2.0))
        target_angles = np.radians((origin[:, 1] + target[:, 1]) / 2.0)
        center_longitude = math.degrees(
            math.atan2(np.sin(target_angles).mean(), np.cos(target_angles).mean())
        )
        env.storm_centers[0] = (center_latitude, center_longitude)
        env.storm_intensity[0] = 1.0
        env.storm_velocity[0] = 0.0


def _rollout(
    env_values: dict[str, object],
    policy_name: str,
    seed: int,
    steps: int,
    event_step: int,
    event_duration: int,
    scenario: str | None,
    device: str,
    checkpoint_path: Path | None,
    maddpg_config: MaddpgConfig,
    tn_config: TNMaddpgConfig,
) -> tuple[list[float], list[float]]:
    config_values = dict(env_values)
    if scenario == "storm_front":
        config_values["storm_cells"] = max(int(config_values["storm_cells"]), 1)
    env_config = LogisticsConfig.from_mapping(
        {**config_values, "episode_steps": steps}
    )
    env = GlobalLogisticsEnv(env_config)
    learner, rng = _policy(
        policy_name,
        env,
        seed,
        device,
        checkpoint_path,
        maddpg_config,
        tn_config,
    )
    observations, _ = env.reset(seed=seed)
    if learner is not None:
        learner.reset_noise()
    step_rewards: list[float] = []
    sla_rates: list[float] = []
    for step_index in range(steps):
        state = compact_ship_observations(env, observations)
        if learner is None:
            actions = rng.uniform(
                -1.0,
                1.0,
                size=(env.config.n_ships, env.config.n_ports + 1),
            ).astype(np.float32)
        else:
            actions = learner.act(state, explore=False)
        if scenario is not None and step_index == event_step:
            destinations = np.argmax(actions[:, : env.config.n_ports], axis=1)
            _inject_event(env, scenario, destinations, event_duration)
        observations, rewards, _, _, infos = env.step_vectorized(actions)
        step_rewards.append(sum(rewards.values()) / env.config.n_ships)
        on_time = sum(info["on_time_delivery"] for info in infos.values())
        violations = sum(info["sla_violation_quantity"] for info in infos.values())
        sla_rates.append(
            on_time / (on_time + violations) if on_time + violations > 0 else 1.0
        )
    env.close()
    return step_rewards, sla_rates


def _recovery_time(
    rewards: list[float],
    baseline_rewards: list[float],
    event_step: int,
    window: int,
    tolerance: float,
    consecutive_windows: int,
) -> int | None:
    consecutive = 0
    for end in range(event_step + window, len(rewards) + 1):
        average = float(np.mean(rewards[end - window : end]))
        baseline = float(np.mean(baseline_rewards[end - window : end]))
        threshold = tolerance * max(abs(baseline), 1.0)
        if abs(average - baseline) <= threshold:
            consecutive += 1
            if consecutive >= consecutive_windows:
                return end - event_step
        else:
            consecutive = 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--maddpg-config", type=Path, default=Path("configs/maddpg.yaml"))
    parser.add_argument("--tn-config", type=Path, default=Path("configs/tn_maddpg.yaml"))
    parser.add_argument("--policy", choices=("random", "maddpg", "tn_maddpg"), default="random")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--event-step", type=int, default=20)
    parser.add_argument("--event-duration", type=int, default=10)
    parser.add_argument("--recovery-window", type=int, default=5)
    parser.add_argument("--consecutive-windows", type=int, default=3)
    parser.add_argument("--recovery-tolerance", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/stress_eval.csv"))
    args = parser.parse_args()
    if (
        args.steps < 1
        or not args.recovery_window <= args.event_step < args.steps
        or args.event_duration < 1
        or args.recovery_window < 1
        or args.consecutive_windows < 1
        or not np.isfinite(args.recovery_tolerance)
        or args.recovery_tolerance < 0
    ):
        parser.error(
            "stress timing must leave a pre-event baseline; tolerance must be finite and non-negative"
        )
    if args.checkpoint is not None and args.policy == "random":
        parser.error("--checkpoint requires --policy maddpg or tn_maddpg")

    env_values = LogisticsConfig.from_yaml(args.env_config).__dict__
    maddpg_config = MaddpgConfig.from_yaml(args.maddpg_config)
    tn_config = TNMaddpgConfig.from_yaml(args.tn_config)
    baseline_rewards, baseline_sla = _rollout(
        env_values,
        args.policy,
        args.seed,
        args.steps,
        args.event_step,
        args.event_duration,
        None,
        args.device,
        args.checkpoint,
        maddpg_config,
        tn_config,
    )
    window_stop = min(args.steps, args.event_step + args.event_duration)
    baseline_event_reward = float(
        np.mean(baseline_rewards[args.event_step:window_stop])
    )
    baseline_event_sla = float(np.mean(baseline_sla[args.event_step:window_stop]))
    results: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        rewards, sla = _rollout(
            env_values,
            args.policy,
            args.seed,
            args.steps,
            args.event_step,
            args.event_duration,
            scenario,
            args.device,
            args.checkpoint,
            maddpg_config,
            tn_config,
        )
        scenario_reward = float(np.mean(rewards[args.event_step:window_stop]))
        scenario_sla = float(np.mean(sla[args.event_step:window_stop]))
        recovery = _recovery_time(
            rewards,
            baseline_rewards,
            args.event_step,
            args.recovery_window,
            args.recovery_tolerance,
            args.consecutive_windows,
        )
        row: dict[str, float | int | str] = {
            "scenario": scenario,
            "policy": args.policy,
            "seed": args.seed,
            "event_step": args.event_step,
            "event_duration": args.event_duration,
            "baseline_reward_per_ship_step": baseline_event_reward,
            "scenario_reward_per_ship_step": scenario_reward,
            "reward_drop_per_ship_step": baseline_event_reward - scenario_reward,
            "baseline_sla_percent": baseline_event_sla * 100.0,
            "scenario_sla_percent": scenario_sla * 100.0,
            "sla_drop_percentage_points": (baseline_event_sla - scenario_sla) * 100.0,
            "reward_recovery_steps": recovery if recovery is not None else -1,
            "checkpoint": str(args.checkpoint) if args.checkpoint else "random_policy",
        }
        results.append(row)
        recovery_text = str(recovery) if recovery is not None else "not recovered"
        print(
            f"{scenario} reward_drop={row['reward_drop_per_ship_step']:.4f} "
            f"sla_drop_pp={row['sla_drop_percentage_points']:.2f} "
            f"recovery_steps={recovery_text}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved stress results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
