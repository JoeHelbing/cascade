# Cleanup Report — 2026-05-03

## Summary

The repository is now organized around the actual research workflow rather than a flat collection of code, generated outputs, empirical scripts, and paper material.

Physical cleanup performed:

- Moved the thesis PDF from the root into `docs/paper/`.
- Renamed/moved `R_regressions/` to `empirical_regressions/` so its purpose is obvious from the top-level tree.
- Added `docs/` as the central reader-facing entry point.
- Added `docs/repository/README.md` as a repository map.
- Added `docs/verification/README.md` as the verification pipeline guide.
- Added `empirical_regressions/README.md` for the R workflow.
- Updated root, autoresearch, validation, and original Python READMEs to point at the new structure.

## Cleaned top-level layout

```text
cascade/
├── original_python/        Mesa reference implementation and thesis-era model analysis
├── mojo_cpu.mojo           Mojo CPU validation bridge
├── mojo_gpu.mojo           Mojo GPU throughput implementation
├── autoresearch/           Benchmarks, sweeps, validation, and analysis scripts
├── empirical_regressions/  R empirical protest-data workflow and output tables/figures
├── docs/                   Reader-facing repository, verification, and paper guides
├── build/                  Ignored compiled binaries
├── pixi.toml               Mojo build task environment
├── pyproject.toml          Python project metadata
└── README.md               High-level project entry point
```

## What relates to what

- `original_python/` defines the semantic baseline.
- `mojo_cpu.mojo` should be read as a correctness bridge from Mesa into Mojo.
- `mojo_gpu.mojo` should be read as the high-throughput simulation engine.
- `autoresearch/validation/` contains the scripts and local/generated artifacts that prove the CPU bridge and support GPU comparison.
- `autoresearch/sweeps/` runs the large parameter sweeps.
- `autoresearch/analysis/` turns sweep outputs into paper-facing figures and reports.
- `empirical_regressions/` is separate from the ABM verification pipeline; it supports the empirical protest-data section.
- `docs/paper/` keeps paper/thesis artifacts out of the code root.

## Verification pipeline

See `docs/verification/README.md` for the full guide. The short version:

```text
original_python/ Mesa reference
    ↓ CPU bit-exact trace comparison
mojo_cpu.mojo validation bridge
    ↓ aggregate/fingerprint comparison
mojo_gpu.mojo throughput kernel
```

### CPU verification state

From Basic Memory project context:

- The CPU port has reached bit-exact validation for the picked-seed dataset.
- Recorded result: `96,320 / 96,320` citizen rows matched Mesa bit-for-bit.
- Key files:
  - `autoresearch/validation/picked_seeds.json`
  - `autoresearch/validation/run_python_trace.py`
  - `autoresearch/validation/compare_bitexact.py`
  - `autoresearch/validation/compare_mojo_cpu.py`
  - `mojo_cpu.mojo`

### GPU verification state

GPU validation is still less formal than CPU validation:

- `mojo_gpu.mojo` uses a hardcoded 45-run correctness parameter set.
- `autoresearch/analysis/compare_outputs.py` mirrors that parameter set for comparison.
- `autoresearch/benchmark.py` and `autoresearch/benchmark_baseline.json` provide benchmark/fingerprint validation.
- There is not yet a clean GPU per-agent trace dataset analogous to the CPU bit-exact path.

### Paper-facing verification gaps

Before writing the academic verification section, consider tightening these points:

1. Create a documented GPU trace artifact and comparer.
2. Move hardcoded GPU validation parameters into a config file.
3. Add exact command examples and expected success output for every validation gate.
4. Preserve the two-link validation logic: Mesa → CPU first, CPU → GPU second.
5. Extend GPU/runner output for Phase 1F per-agent deep dives.

## Notes about pre-existing local changes

Before this cleanup, the worktree already had modified `mojo_cpu.mojo`, `pixi.toml`, `pixi.lock`, `.gitignore`, and several untracked validation scripts. This cleanup updated documentation and paths around those files, but did not attempt to rewrite or resolve the existing Mojo/Pixi changes.
