"""Grid search over TN bond dimension and MADDPG hyperparameters."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from src.training.train import TrainingConfig
from src.agents.tn_maddpg import TNMaddpgConfig


def _write_yaml(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(values, yaml_file, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", type=Path, default=Path("configs/tn_maddpg.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/logistics.yaml"))
    parser.add_argument("--reward-config", type=Path, default=Path("configs/reward.yaml"))
    parser.add_argument("--bond-dims", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001]
    )
    parser.add_argument(
        "--noise-types", nargs="+", choices=("ou", "gaussian", "none"), default=["ou", "gaussian"]
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-starts", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweeps"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.learning_starts < 1:
        parser.error("--episodes, --max-steps, and --learning-starts must be positive")

    base_agent = asdict(TNMaddpgConfig.from_yaml(args.agent_config))
    base_training = asdict(TrainingConfig.from_yaml(args.training_config))
    run_values = itertools.product(
        args.bond_dims,
        args.learning_rates,
        args.noise_types,
        args.batch_sizes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tn-maddpg-sweep-") as temp_name:
        temp_dir = Path(temp_name)
        for index, (bond_dim, learning_rate, noise_name, batch_size) in enumerate(
            run_values
        ):
            if bond_dim < 1 or learning_rate <= 0 or batch_size < 1:
                parser.error("bond dimensions, learning rates, and batch sizes must be positive")
            label = (
                f"bond-{bond_dim}_lr-{learning_rate:g}_noise-{noise_name}_"
                f"batch-{batch_size}"
            )
            agent_values = {
                **base_agent,
                "bond_dim": bond_dim,
                "learning_rate": learning_rate,
                "noise_type": None if noise_name == "none" else noise_name,
                "batch_size": batch_size,
            }
            training_values = {
                **base_training,
                "episodes": args.episodes,
                "max_steps_per_episode": args.max_steps,
                "learning_starts": args.learning_starts,
                "curriculum_enabled": False,
                "log_dir": str(args.output_dir / "tensorboard" / label),
                "checkpoint_dir": str(args.output_dir / "checkpoints" / label),
            }
            agent_path = temp_dir / f"agent-{index}.yaml"
            training_path = temp_dir / f"training-{index}.yaml"
            _write_yaml(agent_path, agent_values)
            _write_yaml(training_path, training_values)
            command = [
                sys.executable,
                "-m",
                "src.training.train",
                "--learner",
                "tn_maddpg",
                "--agent-config",
                str(agent_path),
                "--training-config",
                str(training_path),
                "--env-config",
                str(args.env_config),
                "--reward-config",
                str(args.reward_config),
                "--no-curriculum",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"training subprocess failed for {label} "
                    f"(exit {completed.returncode})\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            result_lines = [
                line.removeprefix("TRAIN_RESULT=")
                for line in completed.stdout.splitlines()
                if line.startswith("TRAIN_RESULT=")
            ]
            if len(result_lines) != 1:
                raise RuntimeError(
                    f"training subprocess produced {len(result_lines)} result records "
                    f"for {label}"
                )
            result = json.loads(result_lines[0])
            records.append(
                {
                    "bond_dim": bond_dim,
                    "learning_rate": learning_rate,
                    "noise_type": noise_name,
                    "batch_size": batch_size,
                    "episodes": args.episodes,
                    "final_reward_per_ship": result["final_reward_per_ship"],
                    "updates": result["updates"],
                    "checkpoint": result["checkpoint"],
                }
            )
            print(f"{label} reward_per_ship={result['final_reward_per_ship']:.4f}")

    output_path = args.output_dir / "hyperparameter_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved sweep results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
