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
pixi run validate-cpu   # original_python -> mojo_cpu bit-exact gate
pixi run validate-gpu   # mojo_gpu aggregate smoke/fingerprint gate
pixi run validate       # both boundaries in order
```

1. Run `original_python/` with the picked seeds — dump Mesa per-agent and per-step traces.
2. Run `mojo_cpu.mojo` with the same seeds — emit per-agent CSV.
3. `autoresearch/validation/run_pipeline.py` compares Mesa and Mojo rows bit-for-bit. Any divergence is a correctness bug in the CPU port.
4. Run `mojo_gpu.mojo` through the GPU aggregate gate.

## Quick start

```bash
pixi install
pixi run build-cpu      # compile mojo_cpu.mojo -> build/mojo_cpu
pixi run build-gpu      # compile mojo_gpu.mojo -> build/mojo_gpu
pixi run validate-cpu   # Mesa -> Mojo CPU correctness
pixi run validate-gpu   # Mojo GPU aggregate validation
```

## Branch policy

- `main` — the clean canonical three files + autoresearch harness. Merges only after validation.
- `autoresearch/<name>` — one branch per experiment. Edits exactly one target file (declared in the branch config). Spawned by the autoresearch orchestrator; merged or dropped based on benchmark + validation results.
- `archive/*` — historical branches kept for provenance.
