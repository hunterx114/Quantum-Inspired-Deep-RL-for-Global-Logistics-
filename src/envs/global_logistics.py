"""A configurable, vectorized multi-agent environment for global shipping."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from src.envs.reward import RewardConfig, compute_reward


@dataclass(frozen=True)
class LogisticsConfig:
    """Parameters controlling the logistics simulator."""

    n_ports: int = 100
    n_ships: int = 10
    grid_height: int = 24
    grid_width: int = 48
    graph_neighbors: int = 8
    seed: int = 7
    episode_steps: int = 365
    max_fuel: float = 10_000.0
    cargo_capacity: float = 100.0
    min_speed_km_per_step: float = 100.0
    max_speed_km_per_step: float = 900.0
    fuel_per_km: float = 0.02
    mean_demand: float = 3.0
    demand_seasonality: float = 0.25
    sla_steps: int = 14
    storm_cells: int = 4
    storm_radius_km: float = 600.0
    storm_speed_penalty: float = 0.5
    port_closure_probability: float = 0.002
    closure_duration_steps: int = 5
    fuel_spike_probability: float = 0.005
    fuel_spike_duration_steps: int = 7
    fuel_spike_multiplier: float = 2.5
    max_port_capacity: int = 200

    def __post_init__(self) -> None:
        """Reject invalid or unsupported simulation sizes and probabilities."""
        if not 100 <= self.n_ports <= 5_000:
            raise ValueError("n_ports must be between 100 and 5000")
        if not 10 <= self.n_ships <= 2_000:
            raise ValueError("n_ships must be between 10 and 2000")
        if self.grid_height < 2 or self.grid_width < 2:
            raise ValueError("weather grid dimensions must be at least 2")
        if not 1 <= self.graph_neighbors < self.n_ports:
            raise ValueError("graph_neighbors must be in [1, n_ports)")
        if self.episode_steps < 1 or self.sla_steps < 1:
            raise ValueError("episode_steps and sla_steps must be positive")
        if self.storm_cells < 0 or self.max_port_capacity < 1:
            raise ValueError("storm_cells must be non-negative and capacity positive")
        if self.max_fuel <= 0 or self.cargo_capacity <= 0 or self.fuel_per_km <= 0:
            raise ValueError("fuel and cargo parameters must be positive")
        if self.min_speed_km_per_step < 0 or self.max_speed_km_per_step <= 0:
            raise ValueError("speed parameters must be non-negative and non-zero")
        if self.min_speed_km_per_step > self.max_speed_km_per_step:
            raise ValueError("minimum speed cannot exceed maximum speed")
        for name in ("port_closure_probability", "fuel_spike_probability"):
            probability = getattr(self, name)
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.demand_seasonality <= 1:
            raise ValueError("demand_seasonality must be between 0 and 1")
        if self.storm_speed_penalty < 0 or self.storm_speed_penalty > 1:
            raise ValueError("storm_speed_penalty must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LogisticsConfig":
        """Load validated configuration values from a YAML mapping."""
        with Path(path).open("r", encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "LogisticsConfig":
        """Build configuration from a mapping, rejecting unknown keys."""
        valid_keys = {field.name for field in fields(cls)}
        unknown = set(values) - valid_keys
        if unknown:
            raise ValueError(f"Unknown logistics configuration keys: {sorted(unknown)}")
        return cls(**values)


def haversine_km(
    latitude_a: np.ndarray | float,
    longitude_a: np.ndarray | float,
    latitude_b: np.ndarray | float,
    longitude_b: np.ndarray | float,
) -> np.ndarray:
    """Compute great-circle distance in kilometers with broadcasting."""
    lat_a = np.radians(latitude_a)
    lat_b = np.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = np.radians(longitude_b) - np.radians(longitude_a)
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6_371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


class GlobalLogisticsEnv(ParallelEnv):
    """Parallel ship-routing simulator with dict observations and continuous actions.

    Each ship chooses a destination from per-port logits and a speed fraction.
    All ships are advanced together using NumPy arrays; ``step_vectorized`` also
    accepts actions as an ``(n_ships, n_ports + 1)`` matrix.
    """

    metadata = {"name": "global_logistics_v0", "render_modes": []}

    def __init__(
        self,
        config: LogisticsConfig | dict[str, Any] | str | Path | None = None,
        reward_config: RewardConfig | dict[str, Any] | str | Path | None = None,
    ) -> None:
        if config is None:
            self.config = LogisticsConfig()
        elif isinstance(config, LogisticsConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = LogisticsConfig.from_yaml(config)
        else:
            self.config = LogisticsConfig.from_mapping(config)
        if reward_config is None:
            default_reward_path = (
                Path(__file__).resolve().parents[2] / "configs" / "reward.yaml"
            )
            self.reward_config = RewardConfig.from_yaml(default_reward_path)
        elif isinstance(reward_config, RewardConfig):
            self.reward_config = reward_config
        elif isinstance(reward_config, (str, Path)):
            self.reward_config = RewardConfig.from_yaml(reward_config)
        else:
            self.reward_config = RewardConfig.from_mapping(reward_config)

        self.possible_agents = [f"ship_{index}" for index in range(self.config.n_ships)]
        self.agents: list[str] = []
        self._agent_indices = np.arange(self.config.n_ships)
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.n_ports + 1,),
            dtype=np.float32,
        )
        self._observation_space = spaces.Dict(
            {
                "ship_state": spaces.Box(
                    low=-1.0, high=1.0, shape=(9,), dtype=np.float32
                ),
                "port_state": spaces.Box(
                    low=0.0, high=1.0, shape=(self.config.n_ports, 3), dtype=np.float32
                ),
                "weather_grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.grid_height, self.config.grid_width, 2),
                    dtype=np.float32,
                ),
                "demand": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.n_ports, 2),
                    dtype=np.float32,
                ),
            }
        )
        self.np_random, self.np_random_seed = seeding.np_random(self.config.seed)
        self._initialize_state()

    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the shared observation schema for an active ship."""
        self._check_agent(agent)
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Box:
        """Return the destination-logits-plus-speed action schema."""
        self._check_agent(agent)
        return self._action_space

    def _check_agent(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent}")

    def _initialize_state(self) -> None:
        """Allocate initial fleet, port, weather, and demand arrays."""
        config = self.config
        self.time = 0
        self.episode_reward = 0.0
        self.port_coordinates = np.column_stack(
            (
                self.np_random.uniform(-65.0, 70.0, config.n_ports),
                self.np_random.uniform(-180.0, 180.0, config.n_ports),
            )
        ).astype(np.float32)
        self.port_capacity = self.np_random.integers(
            max(1, config.max_port_capacity // 4),
            config.max_port_capacity + 1,
            size=config.n_ports,
        )
        self.edge_indices, self.edge_distances_km = self._build_port_graph()
        self.current_port = self.np_random.integers(0, config.n_ports, config.n_ships)
        self.position = self.port_coordinates[self.current_port].copy()
        self.target_port = self.current_port.copy()
        self.fuel = np.full(config.n_ships, config.max_fuel, dtype=np.float32)
        self.cargo = np.full(config.n_ships, config.cargo_capacity, dtype=np.float32)
        self.ship_speed = np.zeros(config.n_ships, dtype=np.float32)
        self.delivered_total = np.zeros(config.n_ships, dtype=np.float32)
        self.destination_queues: list[list[int]] = [[] for _ in self.possible_agents]
        self.port_demand = np.zeros(config.n_ports, dtype=np.float32)
        self.demand_deadline = np.full(config.n_ports, -1, dtype=np.int32)
        self.port_phase = self.np_random.uniform(0.0, 2.0 * np.pi, config.n_ports)
        self.port_closed_steps = np.zeros(config.n_ports, dtype=np.int32)
        self.fuel_spike_steps = 0
        self.fuel_price = 1.0
        self.storm_centers = np.column_stack(
            (
                self.np_random.uniform(-65.0, 70.0, config.storm_cells),
                self.np_random.uniform(-180.0, 180.0, config.storm_cells),
            )
        ).astype(np.float32)
        self.storm_intensity = self.np_random.uniform(
            0.4, 1.0, config.storm_cells
        ).astype(np.float32)
        self.storm_velocity = self.np_random.normal(
            0.0, 1.0, size=(config.storm_cells, 2)
        ).astype(np.float32)
        self._weather_grid = np.zeros(
            (config.grid_height, config.grid_width, 2), dtype=np.float32
        )
        self._port_occupancy = np.zeros(config.n_ports, dtype=np.int32)
        zeros = np.zeros(config.n_ships, dtype=np.float32)
        self._last_components = compute_reward(zeros, zeros, zeros, zeros, self.reward_config)
        self._update_weather_grid()
        self._update_occupancy()

    def _build_port_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a directed k-nearest-port graph using bounded temporary memory."""
        n_ports = self.config.n_ports
        neighbors = np.empty((n_ports, self.config.graph_neighbors), dtype=np.int32)
        distances = np.empty(neighbors.shape, dtype=np.float32)
        for port_index in range(n_ports):
            distance = haversine_km(
                self.port_coordinates[port_index, 0],
                self.port_coordinates[port_index, 1],
                self.port_coordinates[:, 0],
                self.port_coordinates[:, 1],
            )
            distance[port_index] = np.inf
            nearest = np.argpartition(distance, self.config.graph_neighbors)[
                : self.config.graph_neighbors
            ]
            order = np.argsort(distance[nearest])
            nearest = nearest[order]
            neighbors[port_index] = nearest
            distances[port_index] = distance[nearest]
        return neighbors, distances

    def _update_occupancy(self) -> None:
        self._port_occupancy = np.bincount(
            self.current_port, minlength=self.config.n_ports
        ).astype(np.int32)

    def _move_storms(self) -> None:
        if self.config.storm_cells == 0:
            return
        self.storm_centers += self.storm_velocity
        self.storm_centers[:, 0] = np.clip(self.storm_centers[:, 0], -70.0, 75.0)
        self.storm_centers[:, 1] = (self.storm_centers[:, 1] + 180.0) % 360.0 - 180.0

    def _storm_exposure(self) -> np.ndarray:
        if self.config.storm_cells == 0:
            return np.zeros(self.config.n_ships, dtype=np.float32)
        distance = haversine_km(
            self.position[:, 0, None],
            self.position[:, 1, None],
            self.storm_centers[None, :, 0],
            self.storm_centers[None, :, 1],
        )
        proximity = np.clip(1.0 - distance / self.config.storm_radius_km, 0.0, 1.0)
        return np.max(proximity * self.storm_intensity[None, :], axis=1).astype(
            np.float32
        )

    def _update_weather_grid(self) -> None:
        config = self.config
        latitudes = np.linspace(-90.0, 90.0, config.grid_height)
        longitudes = np.linspace(-180.0, 180.0, config.grid_width, endpoint=False)
        grid_lat, grid_lon = np.meshgrid(latitudes, longitudes, indexing="ij")
        if config.storm_cells == 0:
            self._weather_grid.fill(0.0)
            return
        distance = haversine_km(
            grid_lat[..., None],
            grid_lon[..., None],
            self.storm_centers[None, None, :, 0],
            self.storm_centers[None, None, :, 1],
        )
        proximity = np.clip(1.0 - distance / config.storm_radius_km, 0.0, 1.0)
        intensity = np.max(proximity * self.storm_intensity, axis=-1)
        self._weather_grid[..., 0] = intensity
        self._weather_grid[..., 1] = intensity * proximity.max(axis=-1)

    def _advance_events(self) -> None:
        config = self.config
        self.port_closed_steps = np.maximum(self.port_closed_steps - 1, 0)
        closures = (
            self.np_random.random(config.n_ports) < config.port_closure_probability
        )
        self.port_closed_steps[closures] = config.closure_duration_steps
        self.fuel_spike_steps = max(self.fuel_spike_steps - 1, 0)
        if self.np_random.random() < config.fuel_spike_probability:
            self.fuel_spike_steps = config.fuel_spike_duration_steps
        self.fuel_price = (
            config.fuel_spike_multiplier if self.fuel_spike_steps > 0 else 1.0
        )

    def _generate_demand(self) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        seasonal_factor = 1.0 + config.demand_seasonality * np.sin(
            2.0 * np.pi * self.time / config.episode_steps + self.port_phase
        )
        arrivals = self.np_random.poisson(config.mean_demand * seasonal_factor).astype(
            np.float32
        )
        empty_before = self.port_demand <= 0
        self.port_demand += arrivals
        newly_active = empty_before & (self.port_demand > 0)
        self.demand_deadline[newly_active] = self.time + config.sla_steps
        expired = (self.port_demand > 0) & (self.time >= self.demand_deadline)
        violations = np.where(expired, self.port_demand, 0.0).astype(np.float32)
        self.demand_deadline[expired] = self.time + config.sla_steps
        return violations, expired

    def _deliver(self, expired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = np.maximum(self._port_occupancy[self.current_port], 1)
        allocation = np.minimum(self.cargo, self.port_demand[self.current_port] / counts)
        delivered = allocation.astype(np.float32)
        np.add.at(self.port_demand, self.current_port, -delivered)
        self.port_demand = np.maximum(self.port_demand, 0.0)
        current_ports_expired = expired[self.current_port]
        on_time = delivered * (
            (self.time <= self.demand_deadline[self.current_port])
            & ~current_ports_expired
        )
        self.cargo = np.maximum(self.cargo - delivered, 0.0)
        self.delivered_total += delivered
        return delivered, on_time.astype(np.float32)

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        config = self.config
        port_state = np.column_stack(
            (
                self.port_capacity / config.max_port_capacity,
                self._port_occupancy / np.maximum(self.port_capacity, 1),
                (self.port_closed_steps > 0).astype(np.float32),
            )
        ).clip(0.0, 1.0).astype(np.float32)
        deadline_remaining = np.where(
            self.port_demand > 0,
            np.maximum(self.demand_deadline - self.time, 0),
            0,
        )
        demand = np.column_stack(
            (
                self.port_demand / (config.mean_demand * config.sla_steps + 1.0),
                deadline_remaining / config.sla_steps,
            )
        ).clip(0.0, 1.0).astype(np.float32)
        port_state.setflags(write=False)
        demand.setflags(write=False)
        weather_grid = self._weather_grid.copy()
        weather_grid.setflags(write=False)
        observations: dict[str, dict[str, np.ndarray]] = {}
        for ship_index, agent in enumerate(self.possible_agents):
            remaining_km = haversine_km(
                self.position[ship_index, 0],
                self.position[ship_index, 1],
                *self.port_coordinates[self.target_port[ship_index]],
            )
            ship_state = np.array(
                [
                    self.position[ship_index, 0] / 90.0,
                    self.position[ship_index, 1] / 180.0,
                    self.fuel[ship_index] / config.max_fuel,
                    self.cargo[ship_index] / config.cargo_capacity,
                    self.target_port[ship_index] / max(config.n_ports - 1, 1),
                    self.ship_speed[ship_index] / config.max_speed_km_per_step,
                    float(remaining_km > 1.0),
                    min(float(remaining_km / 20_000.0), 1.0),
                    min(float(self.delivered_total[ship_index] / 1_000.0), 1.0),
                ],
                dtype=np.float32,
            )
            observations[agent] = {
                "ship_state": ship_state,
                "port_state": port_state,
                "weather_grid": weather_grid,
                "demand": demand,
            }
        return observations

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        """Reset the fleet and return per-ship observations and seed metadata."""
        del options
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents.copy()
        self._initialize_state()
        infos = {
            agent: {"seed": self.np_random_seed}
            for agent in self.agents
        }
        return self._observations(), infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance all ships from an agent-to-action mapping."""
        if not self.agents:
            raise RuntimeError("step() called before reset() or after episode end")
        missing = set(self.agents) - set(actions)
        if missing:
            raise ValueError(f"Missing actions for agents: {sorted(missing)}")
        action_matrix = np.stack([np.asarray(actions[a]) for a in self.agents])
        return self._step_array(action_matrix)

    def step_vectorized(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance all ships from a dense action matrix in stable agent order."""
        if not self.agents:
            raise RuntimeError("step_vectorized() called before reset() or after end")
        return self._step_array(np.asarray(actions))

    def _step_array(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        config = self.config
        expected_shape = (len(self.agents), config.n_ports + 1)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Actions must contain only finite values")
        actions = np.clip(actions, -1.0, 1.0)
        self.time += 1
        self._advance_events()
        self._move_storms()
        self._update_weather_grid()
        sla_violations, expired_ports = self._generate_demand()
        active_demand_quantity = float(self.port_demand.sum())
        exposure = self._storm_exposure()

        destination_logits = actions[:, : config.n_ports].copy()
        closed = self.port_closed_steps > 0
        destination_logits[:, closed] = -np.inf
        no_open_port = np.all(~np.isfinite(destination_logits), axis=1)
        destination_logits[no_open_port] = actions[no_open_port, : config.n_ports]
        self.target_port = np.argmax(destination_logits, axis=1).astype(np.int32)
        self.destination_queues = [
            [] if target == current else [int(target)]
            for target, current in zip(self.target_port, self.current_port)
        ]

        speed_fraction = (actions[:, -1] + 1.0) / 2.0
        requested_speed = config.min_speed_km_per_step + speed_fraction * (
            config.max_speed_km_per_step - config.min_speed_km_per_step
        )
        weather_factor = 1.0 - config.storm_speed_penalty * exposure
        distance_to_target = haversine_km(
            self.position[:, 0],
            self.position[:, 1],
            self.port_coordinates[self.target_port, 0],
            self.port_coordinates[self.target_port, 1],
        )
        fuel_limited_distance = self.fuel / config.fuel_per_km
        travel_distance = np.minimum(
            np.minimum(distance_to_target, requested_speed * weather_factor),
            fuel_limited_distance,
        )
        grounding = (
            (distance_to_target > 1.0)
            & (fuel_limited_distance < np.minimum(distance_to_target, requested_speed * weather_factor))
        )
        travel_fraction = np.divide(
            travel_distance,
            distance_to_target,
            out=np.zeros_like(travel_distance),
            where=distance_to_target > 0,
        )
        self.position += (self.port_coordinates[self.target_port] - self.position) * (
            travel_fraction[:, None]
        )
        arrived = distance_to_target <= travel_distance + 1e-6
        self.position[arrived] = self.port_coordinates[self.target_port[arrived]]
        self.current_port[arrived] = self.target_port[arrived]
        self.ship_speed = requested_speed.astype(np.float32)
        fuel_used = (travel_distance * config.fuel_per_km).astype(np.float32)
        self.fuel = np.maximum(self.fuel - fuel_used, 0.0)
        self._update_occupancy()
        delivered, on_time = self._deliver(expired_ports)

        port_load = self._port_occupancy / np.maximum(self.port_capacity, 1)
        overload = np.maximum(port_load[self.current_port] - 1.0, 0.0)
        risk = exposure + grounding.astype(np.float32) + overload.astype(np.float32)
        expired_quantity = float(sla_violations.sum())
        sla_violation_rate = expired_quantity / max(active_demand_quantity, 1.0)
        violation_share = np.full(
            config.n_ships,
            sla_violation_rate,
            dtype=np.float32,
        )
        violation_quantity_share = expired_quantity / config.n_ships
        fuel_cost = fuel_used * self.fuel_price
        reward_components = compute_reward(
            fuel_cost,
            on_time,
            violation_share,
            risk,
            self.reward_config,
        )
        reward = reward_components.reward
        self.episode_reward += float(reward.sum())
        self._last_components = reward_components
        observations = self._observations()
        truncated_value = self.time >= config.episode_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated_value for agent in self.agents}
        rewards = {agent: float(reward[i]) for i, agent in enumerate(self.agents)}
        infos: dict[str, dict[str, Any]] = {}
        for ship_index, agent in enumerate(self.agents):
            infos[agent] = {
                "fuel_used": float(fuel_used[ship_index]),
                "fuel_price": self.fuel_price,
                "delivered": float(delivered[ship_index]),
                "storm_exposure": float(exposure[ship_index]),
                "grounding": bool(grounding[ship_index]),
                "port_overload": float(overload[ship_index]),
                "closed_ports": int(np.count_nonzero(closed)),
                "episode_reward": self.episode_reward,
            }
            infos[agent].update(reward_components.for_agent(ship_index))
            infos[agent]["sla_violation_quantity"] = violation_quantity_share
        if truncated_value:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return a centralized state for centralized-critic algorithms."""
        return np.concatenate(
            (
                self.position.reshape(-1) / np.array([90.0, 180.0] * self.config.n_ships),
                self.fuel / self.config.max_fuel,
                self.cargo / self.config.cargo_capacity,
                self.port_demand / (self.config.mean_demand * self.config.sla_steps + 1.0),
                self.port_closed_steps.astype(np.float32)
                / self.config.closure_duration_steps,
                self._weather_grid.reshape(-1),
            )
        ).astype(np.float32)

    def render(self) -> None:
        """Rendering is provided by the separate streaming dashboard milestone."""

    def close(self) -> None:
        """Release environment resources (none are held in this simulator)."""
