# Resistance Cascade

Three progressively-optimized implementations of the same agent-based model of resistance cascades in authoritarian regimes, plus an autoresearch harness for systematic optimization.

Paper: [Mentos Regimes: How Individual Uncertainty Affects the Explosive Strength of Resistance Movements](./Resistance%20Cascade%20Thesis.pdf)

## Layout

```
cascade/
├── original_python/    Original Mesa-based reference implementation (unchanged thesis code)
├── mojo_cpu.mojo       Single-file Mojo CPU port (struct-of-arrays, ~3.8x faster than Python)
├── mojo_gpu.mojo       Single-file Mojo GPU kernel (block-per-sim, spatial grid, ~17x wall clock)
├── autoresearch/       Karpathy-style autoresearch orchestrator + sweeps + validation
├── R_regressions/      Empirical protest data analysis (R)
└── build/              Compiled mojo binaries (gitignored)
```

## The Three-File Contract

Each implementation is a single, self-contained file (or directory, for the original):

- **Reference** — `original_python/` — the authoritative semantics. Never edit.
- **CPU** — `mojo_cpu.mojo` — single file. All changes target this one file.
- **GPU** — `mojo_gpu.mojo` — single file. All changes target this one file.

This is the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) contract: the autoresearch agent proposes edits to exactly one file, a new branch is spawned, the file is validated against the reference, and the branch either survives (merged) or dies (deleted).

## Validation chain

`autoresearch/validation/` contains the cross-check harness:

1. Run `original_python/` with a fixed set of seeds (chosen so activations actually move) — dump per-agent per-step state to parquet.
2. Run `mojo_cpu.mojo` with the same seeds — dump per-agent per-step state to parquet.
3. Diff the two parquets. Any divergence is a correctness bug in the mojo port.
4. Run `mojo_gpu.mojo` against `mojo_cpu.mojo` for GPU-side validation.

## Quick start

```bash
pixi install
pixi run build-cpu      # compile mojo_cpu.mojo -> build/mojo_cpu
pixi run build-gpu      # compile mojo_gpu.mojo -> build/mojo_gpu
pixi run validate       # run the CPU <-> Python correctness chain
```

## Branch policy

- `main` — the clean canonical three files + autoresearch harness. Merges only after validation.
- `autoresearch/<name>` — one branch per experiment. Edits exactly one target file (declared in the branch config). Spawned by the autoresearch orchestrator; merged or dropped based on benchmark + validation results.
- `archive/*` — historical branches kept for provenance.
