"""Compact observation views shared by training and evaluation programs."""

from __future__ import annotations

import numpy as np

from src.envs.global_logistics import GlobalLogisticsEnv


def compact_ship_observations(
    env: GlobalLogisticsEnv,
    observations: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Build a 16-feature local view for each ship from the full env output.

    The features are ship state, selected target-port state and demand, and the
    weather channels at the ship's current grid cell.
    """
    ship_features = np.stack(
        [observations[agent]["ship_state"] for agent in env.possible_agents]
    )
    shared = observations[env.possible_agents[0]]
    targets = env.target_port
    port_features = shared["port_state"][targets]
    demand_features = shared["demand"][targets]
    weather = shared["weather_grid"]
    latitudes = np.clip(
        (
            (env.position[:, 0] + 90.0)
            / 180.0
            * (env.config.grid_height - 1)
        ).astype(int),
        0,
        env.config.grid_height - 1,
    )
    longitudes = (
        ((env.position[:, 1] + 180.0) / 360.0 * env.config.grid_width).astype(int)
        % env.config.grid_width
    )
    weather_features = weather[latitudes, longitudes]
    return np.concatenate(
        (ship_features, port_features, demand_features, weather_features), axis=1
    ).astype(np.float32)
