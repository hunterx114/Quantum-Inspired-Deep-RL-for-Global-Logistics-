"""Configurable curriculum stages and reward-based promotion control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import yaml

from src.envs import GlobalLogisticsEnv, LogisticsConfig
from src.envs.reward import RewardConfig


@dataclass(frozen=True)
class CurriculumStage:
    """Fleet size, disruption rates, and threshold for the next stage."""

    name: str
    n_ports: int
    n_ships: int
    port_closure_probability: float
    fuel_spike_probability: float
    promotion_threshold: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("curriculum stage name must not be empty")
        if not 100 <= self.n_ports <= 5_000:
            raise ValueError("stage n_ports must be between 100 and 5000")
        if not 10 <= self.n_ships <= 2_000:
            raise ValueError("stage n_ships must be between 10 and 2000")
        for field_name in ("port_closure_probability", "fuel_spike_probability"):
            probability = getattr(self, field_name)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.promotion_threshold is not None and not np.isfinite(
            self.promotion_threshold
        ):
            raise ValueError("promotion_threshold must be finite or null")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "CurriculumStage":
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown curriculum stage keys: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered training stages and the reward-window length."""

    moving_average_window: int
    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if self.moving_average_window < 1:
            raise ValueError("moving_average_window must be positive")
        if not self.stages:
            raise ValueError("curriculum must contain at least one stage")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("curriculum stage names must be unique")
        for current, following in zip(self.stages, self.stages[1:]):
            if following.n_ports <= current.n_ports:
                raise ValueError("n_ports must increase at every curriculum stage")
            if following.n_ships <= current.n_ships:
                raise ValueError("n_ships must increase at every curriculum stage")
            if (
                following.port_closure_probability
                < current.port_closure_probability
                or following.fuel_spike_probability
                < current.fuel_spike_probability
            ):
                raise ValueError("disruption probabilities must not decrease by stage")
        if any(stage.promotion_threshold is None for stage in self.stages[:-1]):
            raise ValueError("every non-final stage needs a promotion threshold")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CurriculumConfig":
        """Load a staged curriculum from YAML."""
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "CurriculumConfig":
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown curriculum configuration keys: {sorted(unknown)}")
        stage_values = values.get("stages", [])
        stages = tuple(
            value
            if isinstance(value, CurriculumStage)
            else CurriculumStage.from_mapping(value)
            for value in stage_values
        )
        return cls(
            moving_average_window=values.get("moving_average_window", 1),
            stages=stages,
        )


class CurriculumTrainer:
    """Create stage-specific environments and promote on moving-average reward.

    Call ``record_episode`` with the mean episode reward per ship. When it
    returns ``True``, construct a fresh policy sized for ``current_stage``.
    """

    def __init__(
        self,
        base_env_config: LogisticsConfig | None = None,
        curriculum_config: CurriculumConfig | dict[str, Any] | str | Path | None = None,
        reward_config: RewardConfig | dict[str, Any] | str | Path | None = None,
    ) -> None:
        self.base_env_config = base_env_config or LogisticsConfig()
        if curriculum_config is None:
            default_path = (
                Path(__file__).resolve().parents[2] / "configs" / "curriculum.yaml"
            )
            self.curriculum_config = CurriculumConfig.from_yaml(default_path)
        elif isinstance(curriculum_config, CurriculumConfig):
            self.curriculum_config = curriculum_config
        elif isinstance(curriculum_config, (str, Path)):
            self.curriculum_config = CurriculumConfig.from_yaml(curriculum_config)
        else:
            self.curriculum_config = CurriculumConfig.from_mapping(curriculum_config)

        if reward_config is None:
            reward_path = Path(__file__).resolve().parents[2] / "configs" / "reward.yaml"
            self.reward_config = RewardConfig.from_yaml(reward_path)
        elif isinstance(reward_config, RewardConfig):
            self.reward_config = reward_config
        elif isinstance(reward_config, (str, Path)):
            self.reward_config = RewardConfig.from_yaml(reward_config)
        else:
            self.reward_config = RewardConfig.from_mapping(reward_config)

        self.stage_index = 0
        self.episodes_in_stage = 0
        self._recent_rewards: deque[float] = deque(
            maxlen=self.curriculum_config.moving_average_window
        )

    @property
    def current_stage(self) -> CurriculumStage:
        return self.curriculum_config.stages[self.stage_index]

    @property
    def moving_average_reward(self) -> float | None:
        if not self._recent_rewards:
            return None
        return fmean(self._recent_rewards)

    @property
    def is_complete(self) -> bool:
        return self.stage_index == len(self.curriculum_config.stages) - 1

    def make_env(self) -> GlobalLogisticsEnv:
        """Build an environment configured for the current stage."""
        stage = self.current_stage
        stage_config = replace(
            self.base_env_config,
            n_ports=stage.n_ports,
            n_ships=stage.n_ships,
            port_closure_probability=stage.port_closure_probability,
            fuel_spike_probability=stage.fuel_spike_probability,
        )
        return GlobalLogisticsEnv(stage_config, reward_config=self.reward_config)

    def record_episode(self, mean_reward_per_ship: float) -> bool:
        """Record one episode and promote once the full window reaches threshold."""
        if not np.isfinite(mean_reward_per_ship):
            raise ValueError("episode reward must be finite")
        self._recent_rewards.append(float(mean_reward_per_ship))
        self.episodes_in_stage += 1
        threshold = self.current_stage.promotion_threshold
        if (
            self.is_complete
            or threshold is None
            or len(self._recent_rewards) < self.curriculum_config.moving_average_window
            or self.moving_average_reward is None
            or self.moving_average_reward < threshold
        ):
            return False
        self.stage_index += 1
        self.episodes_in_stage = 0
        self._recent_rewards.clear()
        return True
