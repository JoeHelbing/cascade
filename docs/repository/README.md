# Repository Guide

This repository contains the simulation model, optimization harness, validation artifacts, analysis scripts, and paper support material for the Resistance Cascade project.

## Top-level map

| Path | Purpose | Notes |
|---|---|---|
| `original_python/` | Mesa reference implementation from the thesis-era model. | Treat as the semantic source of truth when validating ports. |
| `mojo_cpu.mojo` | Single-file Mojo CPU validation bridge. | Current work has made this bit-exact against Mesa for the picked validation seeds. |
| `mojo_gpu.mojo` | Single-file Mojo GPU throughput engine. | Optimized block-per-simulation kernel; GPU validation is less formal than the CPU trace path. |
| `autoresearch/` | Benchmarking, validation, sweep orchestration, and analysis scripts. | This is the main computational workflow area. |
| `empirical_regressions/` | R scripts, input CSV, and regression output tables/figures for empirical protest analysis. | Moved out of the root-level jumble from `R_regressions/`. |
| `docs/verification/` | Human-facing guide to the verification pipeline. | Start here before changing validation logic. |
| `docs/paper/` | Thesis/paper source material currently stored in the repo. | Contains the original thesis PDF. |

## Conceptual organization

The project is best read as a validated computational experiment pipeline:

1. **Reference semantics** live in `original_python/resistance_cascade/`.
2. **CPU validation bridge** lives in `mojo_cpu.mojo` and `autoresearch/validation/`.
3. **GPU throughput implementation** lives in `mojo_gpu.mojo`.
4. **Large parameter sweeps** live in `autoresearch/sweeps/`.
5. **Post-hoc paper analysis** lives in `autoresearch/analysis/` and `empirical_regressions/`.
6. **Paper context and writing material** lives in `docs/paper/` plus the generated reports/figures from analysis scripts.

## Regenerable local clutter

The repository intentionally ignores generated environments, worktrees, compiled binaries, sweep outputs, and validation run products. Important ignored paths include:

- `.venv/`, `.pixi/`, `.serena/`, `.claude/`, `.codex/`, `.worktrees/`
- `build/`
- `original_python/data/`
- `autoresearch/validation/*.parquet`, `*.csv`, `*.html`, and `captures/`
- large sweep outputs such as `autoresearch/**/*_output/` and `autoresearch/manifold_results/`

If a future file is large, reproducible, or machine-local, prefer documenting how to regenerate it rather than committing it.
