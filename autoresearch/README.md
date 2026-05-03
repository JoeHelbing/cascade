# Autoresearch Workspace

This folder is the computational workflow layer for the Cascade project. It is organized by stage: benchmark, validate, sweep, analyze, and document.

## Layout

| Path | Purpose | Start here when... |
|---|---|---|
| `benchmarks/` | Fixed benchmark and fingerprint harnesses. | Comparing speed/correctness against a baseline. |
| `configs/` | Small checked-in configuration sets. | Looking for sweep or agent-level parameter lists. |
| `validation/` | Mesa ↔ Mojo CPU correctness checks and GPU comparison support. | Changing model semantics, CPU port behavior, or verification. |
| `sweeps/` | Parameter-space execution scripts. | Running 1D, 1E, 7D, or Phase 1F simulations. |
| `analysis/` | Post-hoc report, figure, and manifold analysis scripts. | Turning sweep outputs into paper-facing results. |
| `plans/` | Historical and active computational plans. | Understanding why a sweep or analysis phase exists. |
| `cleanup-report-2026-05-03.md` | Report for this folder cleanup. | Reviewing what moved and what remains to improve. |

## Workflow mental model

```text
validation/  proves implementation correctness
benchmarks/  measures speed and checks fingerprints
sweeps/      generates large simulation result sets
analysis/    turns result sets into figures/reports
configs/     stores small reusable parameter selections
plans/       records why major computational phases were run
```

## Verification chain

The correctness gate before any performance claim is documented in [`validation/README.md`](validation/README.md) and [`../docs/verification/`](../docs/verification/).

In short: generate Mesa traces with `validation/run_python_trace.py`, run `mojo_cpu.mojo` for the same picked seeds, compare with `validation/compare_bitexact.py` / `validation/compare_mojo_cpu.py`, then use the CPU/GPU comparison and benchmark fingerprinting path for `mojo_gpu.mojo`.

## Benchmarking

Benchmark scripts live in [`benchmarks/`](benchmarks/):

- `gpu_kernel_benchmark.py` compares GPU output fingerprints and throughput against `benchmark_baseline.json`.
- `python_reference_benchmark.py` is the fixed reference benchmark for the original Python model.

## Notes

- Generated sweep outputs and validation artifacts are intentionally ignored by git.
- `configs/agent_sim_params.json` is checked in because it is small and documents the current Phase 1F target set.
- Several validation scripts are intentionally still in one directory because their docstrings and visualization helpers use stable `autoresearch/validation/...` paths.
