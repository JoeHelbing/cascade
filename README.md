# Resistance Cascade

Three progressively-optimized implementations of the same agent-based model of resistance cascades in authoritarian regimes, plus an autoresearch harness for systematic optimization.

Paper: [Mentos Regimes: How Individual Uncertainty Affects the Explosive Strength of Resistance Movements](./docs/paper/Resistance%20Cascade%20Thesis.pdf)

## Layout

```
cascade/
├── original_python/        Original Mesa-based reference implementation (unchanged thesis code)
├── mojo_cpu.mojo           Single-file Mojo CPU validation bridge
├── mojo_gpu.mojo           Single-file Mojo GPU kernel (block-per-sim throughput engine)
├── autoresearch/           Benchmarking, sweeps, validation, and analysis workflow
├── empirical_regressions/  Empirical protest data analysis (R)
├── docs/                   Repository guide, verification guide, and paper material
└── build/                  Compiled mojo binaries (gitignored)
```

## The Three-File Contract

Each implementation is a single, self-contained file (or directory, for the original):

- **Reference** — `original_python/` — the authoritative semantics. Never edit.
- **CPU** — `mojo_cpu.mojo` — single file. All changes target this one file.
- **GPU** — `mojo_gpu.mojo` — single file. All changes target this one file.

This is the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) contract: the autoresearch agent proposes edits to exactly one file, a new branch is spawned, the file is validated against the reference, and the branch either survives (merged) or dies (deleted).

## Validation chain

`autoresearch/validation/` is the canonical cross-check harness. Start there for commands, or read [`docs/verification/`](docs/verification/) for the human-facing map.

```bash
pixi run validate-cpu   # python-core -> mojo_cpu --rng python Parquet SHA256 gate
pixi run validate-gpu   # mojo_cpu --rng gpu + mojo_gpu Parquet aggregate gate
pixi run validate       # both boundaries in order
```

1. Run `python-core-simulation/cascade_core.py` with the picked seeds — write a canonical per-agent Parquet trace with seed metadata and Float64 bit columns.
2. Run `mojo_cpu.mojo --rng python` once per picked seed — write the same canonical Parquet trace layout.
3. `autoresearch/validation/run_pipeline.py` compares the Python-core and Mojo CPU Parquet SHA256 digests. A match proves bit-level parity for that CPU boundary.
4. Run `mojo_cpu.mojo --rng mojo` and `mojo_gpu.mojo --trace-validation` through a per-agent state-trace SHA gate, then run the broader 45-case GPU aggregate tolerance gate.

## Quick start

```bash
pixi install
pixi run build-cpu           # compile mojo_cpu.mojo -> build/mojo_cpu
pixi run build-core-cpu      # compile core_cpu_mojo.mojo -> build/core_cpu_mojo
pixi run visualize-core-cpu  # browser UI for changing core CPU params and replaying traces
pixi run build-gpu           # compile mojo_gpu.mojo -> build/mojo_gpu
pixi run validate-cpu        # Python-core -> Mojo CPU Parquet SHA256 correctness
pixi run validate-gpu        # Mojo CPU/GPU-RNG -> Mojo GPU aggregate validation
```

## Branch policy

- `main` — the clean canonical three files + autoresearch harness. Merges only after validation.
- `autoresearch/<name>` — one branch per experiment. Edits exactly one target file (declared in the branch config). Spawned by the autoresearch orchestrator; merged or dropped based on benchmark + validation results.
- `archive/*` — historical branches kept for provenance.
