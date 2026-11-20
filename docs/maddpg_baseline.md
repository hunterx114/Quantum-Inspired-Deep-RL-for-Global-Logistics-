# MADDPG scaling baseline

The classical baseline uses one deterministic actor per ship and one
centralized critic over the concatenated joint observation/action. Target
networks, a bounded ring replay buffer, Gaussian or Ornstein–Uhlenbeck action
noise, and soft target updates are implemented in `src/agents/maddpg.py`.

The scaling benchmark runs policy inference and real `GlobalLogisticsEnv` steps
for 10, 50, 200, and 1,000 ships in separate worker processes. Each actor uses
a 16-feature local view: ship state, its target port state, target-port demand,
and weather at the ship's grid cell. This compact input is explicit for the
baseline; the environment still provides its full observation dictionary.
Measurements are generated at runtime and saved as CSV plus a PNG plot. They
are hardware- and step-count-dependent, not precomputed claims.

```powershell
python -m benchmarks.scaling_baseline --steps 100
```
