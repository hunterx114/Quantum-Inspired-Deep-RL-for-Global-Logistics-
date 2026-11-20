"""Configurable, decomposed logistics reward calculation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class RewardConfig:
    """Non-negative weights for each reward component."""

    fuel_cost_weight: float = 0.01
    on_time_delivery_weight: float = 1.0
    sla_violation_weight: float = 1.0
    catastrophic_risk_weight: float = 10.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be a finite non-negative weight")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardConfig":
        """Load reward weights from a YAML mapping."""
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RewardConfig":
        """Create reward weights from a mapping, rejecting unknown keys."""
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown reward configuration keys: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class RewardComponents:
    """Raw and signed weighted reward arrays for a batch of ships."""

    reward: np.ndarray
    fuel_cost: np.ndarray
    on_time_delivery: np.ndarray
    sla_violation: np.ndarray
    catastrophic_risk: np.ndarray
    reward_fuel_cost: np.ndarray
    reward_on_time_delivery: np.ndarray
    reward_sla_violation: np.ndarray
    reward_catastrophic_risk: np.ndarray

    def for_agent(self, index: int) -> dict[str, float]:
        """Return scalar reward metrics for one ship's info dictionary."""
        return {
            field.name: float(getattr(self, field.name)[index])
            for field in fields(self)
        }


def compute_reward(
    fuel_cost: np.ndarray | float,
    on_time_delivery: np.ndarray | float,
    sla_violation: np.ndarray | float,
    catastrophic_risk: np.ndarray | float,
    config: RewardConfig,
) -> RewardComponents:
    """Compute ``-w1*fuel + w2*delivery - w3*SLA - w4*risk`` elementwise."""
    raw_components = np.broadcast_arrays(
        np.asarray(fuel_cost, dtype=np.float32),
        np.asarray(on_time_delivery, dtype=np.float32),
        np.asarray(sla_violation, dtype=np.float32),
        np.asarray(catastrophic_risk, dtype=np.float32),
    )
    names = ("fuel_cost", "on_time_delivery", "sla_violation", "catastrophic_risk")
    for name, component in zip(names, raw_components):
        if not np.isfinite(component).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(component < 0):
            raise ValueError(f"{name} must be non-negative")
    fuel, delivery, sla, risk = raw_components
    weighted_fuel = -config.fuel_cost_weight * fuel
    weighted_delivery = config.on_time_delivery_weight * delivery
    weighted_sla = -config.sla_violation_weight * sla
    weighted_risk = -config.catastrophic_risk_weight * risk
    total = weighted_fuel + weighted_delivery + weighted_sla + weighted_risk
    return RewardComponents(
        reward=total.astype(np.float32),
        fuel_cost=fuel.astype(np.float32),
        on_time_delivery=delivery.astype(np.float32),
        sla_violation=sla.astype(np.float32),
        catastrophic_risk=risk.astype(np.float32),
        reward_fuel_cost=weighted_fuel.astype(np.float32),
        reward_on_time_delivery=weighted_delivery.astype(np.float32),
        reward_sla_violation=weighted_sla.astype(np.float32),
        reward_catastrophic_risk=weighted_risk.astype(np.float32),
    )
