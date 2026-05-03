# Autoresearch Harness

Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) orchestration against the Resistance Cascade Mojo kernels.

## Contract

An autoresearch run targets **exactly one file** — either `mojo_cpu.mojo` or `mojo_gpu.mojo` at the repo root. The agent proposes an edit, the orchestrator spawns a branch `autoresearch/<name>`, compiles, runs validation + benchmark, and keeps or discards based on the result.

## Layout

```
autoresearch/
├── README.md                 this file
├── agent_sim_params.json     params used across sweeps (vision, grid, thresholds)
├── benchmark.py              wall-clock + throughput measurements against benchmark_baseline.json
├── benchmark_baseline.json   baseline numbers from the current main
├── validation/               correctness harness (Mesa vs Mojo CPU; GPU comparison support)
├── sweeps/                   parameter sweeps (1D, 1E paired, 7D, manifold)
├── analysis/                 post-hoc charts and reports for sweep outputs
└── PLAN-manifold-grid-search.md   plan doc for the 3D manifold exploration
```

## Validation chain

The correctness gate before any performance claim is documented in [`validation/README.md`](validation/README.md) and [`../docs/verification/`](../docs/verification/).

In short: generate Mesa traces with `validation/run_python_trace.py`, run `mojo_cpu.mojo` for the same picked seeds, compare with `validation/compare_bitexact.py` / `validation/compare_mojo_cpu.py`, then use the CPU/GPU comparison and benchmark fingerprinting path for `mojo_gpu.mojo`.

Seeds are pre-selected for *non-trivial dynamics* — activation counts that actually move above zero, not stuck-at-support trivial runs.

## Benchmark

`benchmark.py` records wall time per sim, sims/sec, and total steps across a fixed batch size. Comparison is always against `benchmark_baseline.json` on `main`, not self-consistency.
