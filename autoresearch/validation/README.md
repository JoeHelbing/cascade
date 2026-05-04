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
    ↓ CSV trace comparison inside run_pipeline.py
mojo_cpu.mojo --rng python
    ↓ same CLI, GPU-compatible RNG mode for no-security aggregate cases
mojo_cpu.mojo --rng gpu
    ↓ aggregate comparison plus full GPU smoke/fingerprint gate
mojo_gpu.mojo
```

## Quick commands

```bash
pixi run validate-cpu   # Mesa trace -> mojo_cpu --rng python -> bit-exact compare
pixi run validate-gpu   # build/run mojo_gpu and check 45 aggregate Sim lines
pixi run validate       # run both boundaries in order
```

Direct form:

```bash
uv run autoresearch/validation/run_pipeline.py --stage cpu
uv run autoresearch/validation/run_pipeline.py --stage gpu
uv run autoresearch/validation/run_pipeline.py --stage all
```

## Live files

| Path | Purpose |
|---|---|
| `run_pipeline.py` | The single validation-chain script. It generates Mesa traces, runs Mojo binaries, and performs the current comparisons. |
| `picked_seeds.json` | Canonical CPU validation seed/config set. |
| `README.md` | This guide. |
| `../../mojo_cpu.mojo` | CPU simulation/validation bridge; emits per-agent CSV and supports `--rng python|gpu`. |
| `../../mojo_gpu.mojo` | GPU throughput kernel; emits aggregate `Sim ...` lines. |

## CPU validation: Python core -> Mojo CPU

`run_pipeline.py --stage cpu` now uses the checked-in regression tests for the Python core reference basis:

1. Run `tests.test_python_core_simulation` for core mechanics.
2. Run `tests.test_mojo_cpu_cli`, which builds `mojo_cpu.mojo` and compares `--rng python` output against `cascade_core.ResistanceCascade` row by row with raw Float64 bit checks.

Expected success text:

```text
CPU validation PASS: python-core and mojo_cpu CLI regression tests passed.
```

## GPU validation: Mojo CPU boundary -> Mojo GPU

GPU validation uses `mojo_cpu.mojo --rng gpu` as the CPU-side bridge for the GPU-compatible LCG stream. The GPU binary still runs its full hardcoded 45-case grid and must emit 45 `Sim ...` aggregate lines.

Because `mojo_cpu.mojo` intentionally does not implement the security arrest path in GPU RNG mode, the CPU-vs-GPU aggregate comparison is limited to the 15 no-security cases (`security_density == 0.0`) that both implementations support. For those cases the gate checks citizen totals, revolution status, and active-count drift within the documented Float64-vs-Float32/kernel-layout tolerance.

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
