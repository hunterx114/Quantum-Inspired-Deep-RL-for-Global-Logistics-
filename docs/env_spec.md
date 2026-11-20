# Global logistics environment

`GlobalLogisticsEnv` is a PettingZoo `ParallelEnv` in which each ship is an
agent. The YAML configuration in `configs/logistics.yaml` controls fleet size,
port count, weather resolution, travel parameters, demand seasonality, and
disruption rates. Supported scaling limits are 100–5,000 ports and 10–2,000
ships.

## State and actions

Each ship receives a dictionary with its normalized `ship_state`, shared
`port_state` (capacity, occupancy, closure), `weather_grid` (storm intensity
and risk channels), and per-port `demand` (backlog and remaining SLA time).
The continuous action is a vector of `n_ports + 1` values: one destination
logit per port followed by a speed control in `[-1, 1]`. The highest logit
selects the destination; closed ports are masked when an open alternative
exists. `step_vectorized` accepts an `(n_ships, n_ports + 1)` NumPy array.

Ports are sampled with latitude/longitude coordinates. A directed k-nearest
port graph stores haversine distances without retaining an all-pairs matrix.
Ships travel along great-circle distance estimates, consume fuel, and encounter
moving storm cells. Demand follows a seasonally modulated Poisson process;
port closures, fuel-price spikes, congestion, storm exposure, grounding, and
SLA expiration are represented in transition information.

Randomness is controlled by `reset(seed=...)`; environment-level arrays and
actions are advanced in batches. The reward components are included separately
in each agent's `info` alongside their signed weighted contributions. Weights
are loaded from `configs/reward.yaml`; the formula is
`-w1*fuel_cost + w2*on_time_delivery - w3*SLA_violation - w4*catastrophic_risk`.
The reward's SLA term is the fleet-wide share of active demand past its
deadline; expired cargo quantity is logged separately. Catastrophic risk
combines storm exposure, grounding, and port overload.

The staged curriculum is configured in `configs/curriculum.yaml`. Its controller
promotes after a full moving-average window meets the stage threshold and
constructs an environment for the new port/fleet/disruption scale. Episode
rewards supplied to the controller are normalized per ship.

## Smoke rollout

```powershell
python -m scripts.random_logistics_rollout --config configs/logistics.yaml --seed 7 --steps 100
```
