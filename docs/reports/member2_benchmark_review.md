# Member 2 Benchmark Review

## Scope and evidence reviewed

This is an initial review of benchmark artifacts already present in the repository. It is a preliminary evidence note, not the final project comparison. The reviewed files are `outputs/benchmarks/trained-smoke/tn_vs_maddpg.csv`, `outputs/benchmarks/scaling-comparison/tn_vs_maddpg.csv`, and `outputs/stress_eval.csv`.

## Existing checkpoint comparison

The `trained-smoke` CSV records both learners with 10 agents, 100 ports, 50 rollout steps, five warm-up calls, and checkpoint paths under `outputs/checkpoints`. The table below copies the recorded measurements.

| Measure | MADDPG | TN-MADDPG |
| --- | ---: | ---: |
| Parameters (online plus target) | 569,318 | 185,702 |
| p50 inference latency | 0.724 ms | 4.514 ms |
| p95 inference latency | 1.676 ms | 6.153 ms |
| Additional CPU memory | 82.59 MiB | 75.84 MiB |
| GPU memory | 0 MiB | 0 MiB |
| Decisions per second | 11,129 | 2,140 |
| Total rollout reward | -23,144.65 | -23,433.58 |
| Mean reward per ship | -2,314.47 | -2,343.36 |

In this recorded run, TN-MADDPG has fewer parameters and lower measured additional CPU memory, while MADDPG has lower measured inference latency and higher decisions per second. The reward values are close in this single run; they do not establish a reliable performance difference.

## Important limitations

- The checkpoint comparison contains one seed and 50 rollout steps. It is a smoke result, not a statistically robust evaluation.
- The CSV names checkpoint paths but does not include checkpoint hashes, training provenance, machine details, software versions, or the complete command line. Confirm checkpoint provenance with Member 1 and record the missing metadata on reruns.
- GPU memory is zero in these rows, consistent with a CPU run; the file does not explicitly record the selected device.
- The `scaling-comparison` CSV labels every row `fresh_initialization`. Its reward values must not be presented as trained-policy performance. It can inform a preliminary architecture/runtime discussion only.
- `outputs/stress_eval.csv` labels its policy `random` and checkpoint `random_policy`. It is not evidence about trained MADDPG or TN-MADDPG resilience.
- No result here demonstrates convergence, statistical significance, or that one learner is generally superior.

## Next evaluation steps

1. Confirm checkpoint provenance, compatible configuration, and inference schema with Member 1.
2. Rerun checkpoint comparisons with multiple shared seeds and documented hardware, software, command, warm-up count, and rollout length.
3. Run each trained checkpoint through all four stress scenarios with identical seeds and conditions; preserve each raw CSV.
4. Summarize repeated-run variability and limitations before drawing conclusions.

## Reproduction command template

Run from the repository root after confirming the checkpoints match the configured 10-agent, 100-port environment:

```powershell
python -m benchmarks.compare_agents `
  --maddpg-checkpoint outputs/checkpoints/maddpg/maddpg_final.pt `
  --tn-checkpoint outputs/checkpoints/tn_maddpg/tn_maddpg_final.pt `
  --agent-counts 10 `
  --steps 50 `
  --warmup-calls 5 `
  --seed 7 `
  --device cpu `
  --output-dir outputs/benchmarks/member2-seed-7
```

The repository environment in this workspace did not expose a `python` or `py` command during this review, so no new benchmark was run here. The numbers above are transcribed from the checked-in CSV rather than newly generated results.
