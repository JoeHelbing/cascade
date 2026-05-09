# Validation Pipeline

This directory has one live validation-chain script:

```text
run_pipeline.py
```

Everything else in this directory is config (`picked_seeds.json`) or prose
(`README.md`). Historical helper scripts are preserved under `archive/`.

## Chain

```text
python-core-simulation/cascade_core.py
    ↓ canonical Parquet trace + SHA256 equality
mojo_cpu.mojo --rng python
    ↓ GPU-compatible RNG mode for no-security aggregate cases
mojo_cpu.mojo --rng gpu
    ↓ Parquet aggregate comparison plus full GPU smoke/fingerprint gate
mojo_gpu.mojo
```

## Quick commands

```bash
pixi run validate-cpu   # python-core Parquet -> mojo_cpu --rng python Parquet -> SHA256 match
pixi run validate-gpu   # mojo_cpu --rng gpu Parquet artifacts + mojo_gpu aggregate Parquet gate
pixi run validate       # run both boundaries in order
```

Direct form:

```bash
pixi run python autoresearch/validation/run_pipeline.py --stage cpu
pixi run python autoresearch/validation/run_pipeline.py --stage gpu
pixi run python autoresearch/validation/run_pipeline.py --stage all
```

## Live files

| Path | Purpose |
|---|---|
| `run_pipeline.py` | The single validation-chain script. It generates Parquet artifacts, runs Mojo binaries, writes SHA256 manifests, and performs the current comparisons. |
| `picked_seeds.json` | Canonical CPU validation seed/config set. |
| `README.md` | This guide. |
| `../../mojo_cpu.mojo` | CPU simulation/validation bridge; emits per-agent CSV and supports `--rng python|gpu`. |
| `../../mojo_gpu.mojo` | GPU throughput kernel; emits aggregate `Sim ...` lines. |

## CPU validation: Python core -> Mojo CPU

`run_pipeline.py --stage cpu` validates the Python-core reference basis in two layers:

1. Run `tests.test_python_core_simulation` for core mechanics.
2. Run `tests.test_mojo_cpu_cli` for a small row-by-row CLI regression with raw Float64 bit checks.
3. Generate `python_core_trace.parquet` for all picked seeds.
4. Run `mojo_cpu --rng python` once per picked seed, append the simulations into `mojo_cpu_python_rng_trace.parquet`, and compare the two Parquet files by SHA256.

The Parquet trace includes simulation metadata columns (`sim_id`, `seed`, `epsilon_config`, `security_density_config`) plus the trace schema and raw `*_bits` columns for every Float64 field. A matching SHA therefore proves identical row order, metadata, scalar values, nulls, booleans, and IEEE-754 bit patterns in the canonical artifact.

Expected success text:

```text
CPU validation PASS: Parquet SHA256 match (...).
```

## GPU validation: Mojo CPU boundary -> Mojo GPU

GPU validation uses `mojo_cpu.mojo --rng gpu` as the CPU-side bridge for the GPU-compatible LCG stream. The GPU binary still runs its full hardcoded 45-case grid and must emit 45 `Sim ...` aggregate lines.

Because `mojo_cpu.mojo` intentionally does not implement the security arrest path in GPU RNG mode, the CPU-vs-GPU aggregate comparison is limited to the 15 no-security cases (`security_density == 0.0`) that both implementations support. For those cases the gate checks citizen totals, revolution status, and active-count drift within the documented Float64-vs-Float32/kernel-layout tolerance.

`run_pipeline.py --stage gpu` writes:

- `python_core_state_trace.parquet` — a 500-step per-agent state trace for the GPU comparison seed under Python-core/Python RNG.
- `mojo_cpu_python_rng_state_trace.parquet` — the same state trace from `mojo_cpu --rng python`.
- `mojo_cpu_mojo_rng_state_trace.parquet` — the 500-step per-agent state trace from `mojo_cpu --rng mojo`.
- `mojo_gpu_state_trace.parquet` — the 500-step per-agent state trace emitted by `mojo_gpu --trace-validation`.
- `mojo_cpu_gpu_rng_trace.parquet` — per-agent CPU/GPU-RNG traces for the 15 no-security aggregate comparison cases.
- `mojo_cpu_gpu_rng_aggregate.parquet` — final aggregates derived from those traces.
- `mojo_gpu_aggregate.parquet` — the 45 aggregate `Sim ...` lines emitted by the GPU binary.
- `validation_sha256.json` — SHA256 digests for the artifacts generated in that run.

The GPU trace gate now requires `mojo_cpu --rng mojo` and `mojo_gpu --trace-validation` to produce identical state-trace SHA256 values for the no-security GPU comparison seed. The broader 45-case GPU sweep remains an aggregate tolerance gate: it checks citizen totals, revolution status, and active-count drift within ±35 for supported no-security cases.

Expected success text:

```text
GPU validation PASS: mojo_cpu --rng gpu and mojo_gpu agree on no-security aggregate outcomes
```

## Archive

Historical scripts and generated artifacts live under [`archive/`](archive/).
Nothing was deleted. In particular, older split helper scripts such as
`run_python_trace.py`, `compare_bitexact.py`, and `compare_mojo_cpu.py` are now
archived; their live logic has been folded into `run_pipeline.py`.

Generated local outputs such as `*.parquet`, `*.csv`, `*.html`, captures, and
caches live under `archive/generated/` when preserved from old runs. New
validation runs may write fresh artifacts in this root, but they are ignored and
can be moved back under `archive/generated/` when reviewing the tree.
