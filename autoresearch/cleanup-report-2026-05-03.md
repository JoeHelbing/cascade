# Autoresearch Cleanup Report — 2026-05-03

## Summary

The `autoresearch/` folder is now organized around workflow stages instead of mixing benchmark harnesses, configs, plans, validation, sweeps, and analysis at the same level.

## Physical changes

Moved root-level files into clearer homes:

- `benchmark.py` → `benchmarks/gpu_kernel_benchmark.py`
- `benchmark_python.py` → `benchmarks/python_reference_benchmark.py`
- `benchmark_baseline.json` → `benchmarks/benchmark_baseline.json`
- `agent_sim_params.json` → `configs/agent_sim_params.json`
- `PLAN-manifold-grid-search.md` → `plans/manifold-grid-search.md`

Added README guides for:

- `autoresearch/`
- `autoresearch/benchmarks/`
- `autoresearch/configs/`
- `autoresearch/sweeps/`
- `autoresearch/analysis/`
- `autoresearch/plans/`

## Current organization

```text
autoresearch/
├── benchmarks/   fixed benchmark and fingerprint harnesses
├── configs/      small checked-in parameter/config files
├── validation/   Mesa ↔ Mojo correctness and comparison scripts
├── sweeps/       parameter-space execution scripts
├── analysis/     post-hoc paper/report analysis scripts
├── plans/        planning/rationale docs for major computational phases
└── README.md     folder map and workflow overview
```

## Verification pipeline section

The verification path remains centered in `validation/` and the repo-level guide at `../docs/verification/`:

```text
Mesa reference (`original_python/`)
    ↓ trace/bit-exact comparison
Mojo CPU (`mojo_cpu.mojo`)
    ↓ aggregate/fingerprint comparison
Mojo GPU (`mojo_gpu.mojo`)
```

`validation/` is still relatively flat on purpose. Many scripts and docs use stable `autoresearch/validation/...` artifact paths, and some visualization helpers load neighboring validation scripts directly. Reorganizing those scripts into nested folders should be a separate verification-focused refactor with tests around script entry points.

## Follow-up opportunities

1. Give every validation script an explicit CLI entry point and move exploratory tools under `validation/exploratory/`.
2. Move hardcoded paths in analysis scripts into a shared config/helper.
3. Add expected-output snippets for benchmark and validation commands.
4. Decide whether untracked validation helper scripts should be committed, archived, or regenerated from another source.
