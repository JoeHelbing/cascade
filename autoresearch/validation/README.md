# Validation Chain

Correctness gate between `original_python/` (Mesa reference) and the Mojo ports. Every autoresearch branch that modifies `mojo_cpu.mojo` or `mojo_gpu.mojo` should pass the relevant checks before benchmark numbers are considered.

For the paper-facing overview, see [`../../docs/verification/`](../../docs/verification/).

## Canonical CPU scripts and artifacts

| Path | Purpose |
|---|---|
| `picked_seeds.json` | Fixed seed/config set chosen for non-trivial dynamics. |
| `run_python_trace.py` | Runs Mesa reference traces for the picked seeds. |
| `python_trace.parquet` | Mesa per-agent trace artifact; ignored because it is regenerable. |
| `python_model_trace.parquet` | Mesa per-step/model trace artifact; ignored because it is regenerable. |
| `mojo_cpu_bitexact.csv` | Mojo CPU per-agent trace artifact emitted by `mojo_cpu.mojo`; ignored because it is regenerable. |
| `mojo_cpu_model_trace.csv` | Mojo CPU per-step/model trace artifact; ignored because it is regenerable. |
| `compare_bitexact.py` | Bit-exact per-agent comparison. |
| `compare_mojo_cpu.py` | Aggregate/model-trace comparison. |

## CPU validation workflow

```bash
uv run autoresearch/validation/run_python_trace.py
pixi run build-cpu
pixi run run-cpu > autoresearch/validation/mojo_cpu_bitexact.csv
uv run autoresearch/validation/compare_bitexact.py \
  --python autoresearch/validation/python_trace.parquet \
  --mojo autoresearch/validation/mojo_cpu_bitexact.csv
```

The current project memory records the CPU port as bit-exact for the picked-seed dataset: `96,320 / 96,320` citizen rows matched Mesa bit-for-bit.

## GPU validation status

GPU validation currently uses aggregate output comparison and benchmark fingerprinting rather than a clean per-agent trace dataset. Relevant files:

- `../../mojo_gpu.mojo` — hardcoded 45-run correctness parameter set.
- `../analysis/compare_outputs.py` — Python comparison script with matching hardcoded values.
- `../benchmark.py` and `../benchmark_baseline.json` — benchmark and sorted-output fingerprint path.

A paper-grade GPU verification pass should add a documented GPU trace artifact analogous to the CPU bit-exact path.

## Historical and exploratory scripts

These scripts are retained because they encode useful debugging history, but they are not the primary CPU trace gate:

- `cross_validate.py`, `cross_validate_full.py`, `cross_validate_inject.py`, `cross_validate_step_metrics.py`
- `gpu_algo_python.py`
- `capture_mesa.py`, `replay_python.py`, `capture_oscillating_trace.py`
- `sweep_oscillating.py`, `sweep_oscillating_with_sec.py`, `probe_oscillation.py`
- `build_animation.py`, `build_showboat.py`

## Seed selection criterion

Seeds are picked so that trajectories actually move: at some step, at least the configured minimum number of agents are Active. Validating against seeds stuck at Support forever is vacuous.
