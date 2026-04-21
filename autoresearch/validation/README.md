# Validation Chain

Correctness gate between `original_python/` (Mesa reference) and the mojo ports. Every autoresearch branch that modifies `mojo_cpu.mojo` or `mojo_gpu.mojo` must pass this before its benchmark numbers are considered.

## Canonical scripts

| Script | Purpose |
|--------|---------|
| `seed_picker.py` | Scans N seeds, keeps those where activation dynamics are non-trivial (peak active >= min_peak). Writes `picked_seeds.json`. |
| `run_python_trace.py` | Runs the Mesa model for each picked seed, dumps per-agent per-step state to `python_trace.parquet`. |
| `run_mojo_cpu_trace.py` | (todo) Runs the mojo CPU binary with tracing enabled, dumps `mojo_cpu_trace.parquet`. |
| `compare_traces.py` | Diffs two parquet traces by (seed, step, agent_id). Exit 0 on agreement within tolerance. |

## Legacy scripts (reference)

These predate the reorg and targeted the mesa-removed Python port, not the original Mesa code. Useful as historical cross-validation logic reference:

- `cross_validate.py` -- per-step metric agreement (older setup)
- `cross_validate_full.py` -- full replay in pure Python mirroring mojo order-of-ops
- `cross_validate_inject.py` -- inject Python RNG values into pure-Python replay to isolate math bugs
- `cross_validate_step_metrics.py` -- aggregate step-level comparison
- `gpu_algo_python.py` -- Python reference of the GPU kernel algorithm

## Workflow

```bash
uv sync
uv run autoresearch/validation/seed_picker.py --n-seeds 30 --n-keep 12 --steps 500
uv run autoresearch/validation/run_python_trace.py
# (future) pixi run build-cpu-trace && uv run run_mojo_cpu_trace.py
uv run autoresearch/validation/compare_traces.py \
    --ref autoresearch/validation/python_trace.parquet \
    --cand autoresearch/validation/mojo_cpu_trace.parquet
```

## Seed selection criterion

Seeds are picked so that trajectories *actually move* -- at some step, at least `min_peak` agents are in the Active condition. Validating against seeds stuck at Support-forever is vacuous.

## Why this is not bit-exact

Mesa's RNG (Python `random.Random`, Mersenne Twister) and mojo's RNG (LCG) are different. The trajectories will diverge after the first random draw. For bit-exact validation we need **state injection**: capture Mesa's initial agent positions, thresholds, private preferences, etc., feed them into mojo as a starting state, then verify per-agent per-step agreement on the shared deterministic evolution. That injection path is planned but not yet implemented -- see the follow-up in the `autoresearch/` PLAN.
