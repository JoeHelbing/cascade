# Validation Pipeline

This directory is the canonical correctness gate for the implementation chain:

```text
original_python/ Mesa reference
    ↓ bit-exact per-agent trace comparison
mojo_cpu.mojo --rng python
    ↓ same CPU simulation, RNG provider switch only
mojo_cpu.mojo --rng gpu
    ↓ aggregate GPU smoke/fingerprint gate
mojo_gpu.mojo
```

## Quick commands

```bash
pixi run validate-cpu   # Mesa trace -> mojo_cpu trace -> bit-exact compare
pixi run validate-gpu   # build/run mojo_gpu and check 45 aggregate Sim lines
pixi run validate       # run both boundaries in order
```

The direct orchestrator is:

```bash
uv run autoresearch/validation/run_pipeline.py --stage cpu
uv run autoresearch/validation/run_pipeline.py --stage gpu
uv run autoresearch/validation/run_pipeline.py --stage all
```

## Live files

| Path | Purpose |
|---|---|
| `picked_seeds.json` | Canonical CPU validation seed/config set chosen for non-trivial dynamics. |
| `run_python_trace.py` | Runs the original Mesa model and writes `python_trace.parquet` plus `python_model_trace.parquet`. |
| `compare_bitexact.py` | Bit-exact per-agent gate from Mesa parquet to `mojo_cpu.mojo` CSV output. |
| `compare_mojo_cpu.py` | Secondary aggregate/model-trace helper; useful for summaries, not the primary bit-exact gate. |
| `run_pipeline.py` | One public CLI for `cpu`, `gpu`, or `all` validation stages. |
| `../../mojo_cpu.mojo` | CPU validation bridge; emits per-agent CSV to stdout and supports `--rng python|gpu`. |
| `../../mojo_gpu.mojo` | GPU throughput kernel; emits aggregate `Sim ...` lines. |

## CPU validation: Python -> Mojo CPU

`mojo_cpu.mojo` is expected to be bit-exact against Mesa for the picked seed set.
The orchestrator runs these steps:

```bash
uv run autoresearch/validation/run_python_trace.py
pixi run build-cpu
build/mojo_cpu --rng python > autoresearch/validation/mojo_cpu_bitexact.csv
uv run autoresearch/validation/compare_bitexact.py \
  --mesa autoresearch/validation/python_trace.parquet \
  --mojo autoresearch/validation/mojo_cpu_bitexact.csv
```

Expected success text from the comparer:

```text
PASS: mojo_cpu is bit-for-bit identical to Mesa on every tracked column.
```

## GPU validation: Mojo CPU boundary -> Mojo GPU

GPU validation is currently less formal than CPU validation. The intended bridge
is `mojo_cpu.mojo --rng gpu`: the same CPU simulation implementation runs with
only its RNG provider switched from Python/Mesa RNG to the GPU-compatible LCG
provider. The current GPU gate then build/runs the GPU binary, captures
`mojo_gpu_output.txt`, and asserts the hardcoded 45-run correctness grid
produces 45 `Sim ...` lines. The next hardening step is exact aggregate
comparison of `mojo_cpu --rng gpu` against `mojo_gpu` for the same cases.

The hardcoded GPU grid is:

- seeds: `42, 123, 456, 789, 1001`
- epsilons: `0.2, 0.5, 1.0`
- security densities: `0.0, 0.02, 0.05`
- steps: `50`

## Archive

Historical scripts and generated artifacts were moved to
[`archive/`](archive/) so this directory only shows the live validation chain.
Nothing was deleted.

Examples in `archive/historical/` include old `cross_validate*.py`, replay,
capture, seed-picking, oscillation, animation, and showboat helpers. They are
kept for provenance and debugging and may contain stale embedded paths.

Generated local outputs such as `*.parquet`, `*.csv`, `*.html`, captures, and
caches live under `archive/generated/` when preserved from old runs. New
canonical validation runs write fresh artifacts back to this directory root.

## Gap to close later

A paper-grade GPU gate should emit a clean GPU trace dataset or stable aggregate
fixture analogous to the CPU bit-exact trace path. Until then, preserve the
boundary discipline: validate Mesa -> CPU first, then CPU -> GPU separately.
