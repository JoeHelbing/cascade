# Validation Pipeline

This directory has one live validation-chain script:

```text
run_pipeline.py
```

Everything else in this directory is config (`picked_seeds.json`) or prose
(`README.md`). Historical helper scripts are preserved under `archive/`.

## Chain

```text
original_python/ Mesa reference
    ↓ bit-exact per-agent trace comparison inside run_pipeline.py
mojo_cpu.mojo --rng python
    ↓ same CPU simulation, RNG provider switch only
mojo_cpu.mojo --rng gpu
    ↓ aggregate GPU smoke/fingerprint gate inside run_pipeline.py
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

## CPU validation: Python -> Mojo CPU

`run_pipeline.py --stage cpu` performs the former helper-script steps inline:

1. Run `original_python/resistance_cascade` for `picked_seeds.json`.
2. Write `python_trace.parquet` and `python_model_trace.parquet`.
3. Build `mojo_cpu.mojo`.
4. Run `build/mojo_cpu --rng python > mojo_cpu_bitexact.csv`.
5. Compare Mesa vs Mojo rows bit-for-bit on the tracked columns.

Expected success text:

```text
PASS: mojo_cpu is bit-for-bit identical to Mesa on every tracked column.
```

## GPU validation: Mojo CPU boundary -> Mojo GPU

GPU validation is currently less formal than CPU validation. The intended bridge
is `mojo_cpu.mojo --rng gpu`: the same CPU simulation implementation runs with
only its RNG provider switched from Python/Mesa RNG to the GPU-compatible LCG
provider. The current GPU gate build/runs the GPU binary, captures
`mojo_gpu_output.txt`, and asserts the hardcoded 45-run correctness grid
produces 45 `Sim ...` lines.

The next hardening step is exact aggregate comparison of `mojo_cpu --rng gpu`
against `mojo_gpu` for the same cases.

## Archive

Historical scripts and generated artifacts live under [`archive/`](archive/).
Nothing was deleted. In particular, older split helper scripts such as
`run_python_trace.py`, `compare_bitexact.py`, and `compare_mojo_cpu.py` are now
archived; their live logic has been folded into `run_pipeline.py`.

Generated local outputs such as `*.parquet`, `*.csv`, `*.html`, captures, and
caches live under `archive/generated/` when preserved from old runs. New
validation runs may write fresh artifacts in this root, but they are ignored and
can be moved back under `archive/generated/` when reviewing the tree.
