# Verification Pipeline

This page is the repo-local map for understanding and extending Cascade verification.

## Chain of trust

The model has three implementation surfaces:

```text
original_python/ Mesa reference
    ↓ CPU trace / bit-exact comparison
mojo_cpu.mojo validation bridge
    ↓ aggregate/fingerprint comparison
mojo_gpu.mojo throughput kernel
```

The important idea from project memory is to validate one change boundary at a time. Do not use a new Python reimplementation of the GPU algorithm as the source of truth; that only validates the reimplementation. The original Mesa model remains the semantic reference.

## CPU verification: Mesa → Mojo CPU

Current canonical artifacts and scripts:

| Path | Role |
|---|---|
| `autoresearch/validation/picked_seeds.json` | Seed/config set used for non-trivial validation runs. |
| `autoresearch/validation/run_python_trace.py` | Generates Mesa reference traces. |
| `autoresearch/validation/python_trace.parquet` | Mesa per-agent trace artifact; ignored as regenerable but present locally when captured. |
| `autoresearch/validation/python_model_trace.parquet` | Mesa per-step/model trace artifact. |
| `mojo_cpu.mojo` | Emits per-step, per-agent CSV for the same seed set. |
| `autoresearch/validation/compare_bitexact.py` | Bit-exact per-agent comparer. |
| `autoresearch/validation/compare_mojo_cpu.py` | Aggregate/model-trace comparer. |

Known current result from Basic Memory: the CPU port reached bit-exact validation for the picked-seed set: `96,320 / 96,320` citizen rows matched Mesa bit-for-bit.

### Typical CPU validation flow

```bash
uv run autoresearch/validation/run_python_trace.py
pixi run build-cpu
pixi run run-cpu > autoresearch/validation/mojo_cpu_bitexact.csv
uv run autoresearch/validation/compare_bitexact.py \
  --python autoresearch/validation/python_trace.parquet \
  --mojo autoresearch/validation/mojo_cpu_bitexact.csv
```

Use the exact command-line flags accepted by the scripts if they diverge from this sketch; the files above are the important contract.

## GPU verification: Mojo CPU → Mojo GPU

GPU validation is currently less formal than CPU validation. The root `mojo_gpu.mojo` hardcodes a 45-run correctness parameter set:

- seeds: `42, 123, 456, 789, 1001`
- epsilons: `0.2, 0.5, 1.0`
- security densities: `0.0, 0.02, 0.05`
- `num_steps = 50`
- `citizen_density = 0.7`
- `pp_mean = 0.0`
- `threshold = 2.94444`
- `max_jail = 100`
- `vision = 7`

Related files:

| Path | Role |
|---|---|
| `autoresearch/analysis/compare_outputs.py` | Python comparison script with matching hardcoded values. |
| `autoresearch/benchmarks/gpu_kernel_benchmark.py` | Benchmark/fingerprint comparison for sorted `Sim ...` output lines. |
| `autoresearch/benchmarks/benchmark_baseline.json` | Older benchmark fingerprint baseline. |
| `autoresearch/analysis/visualize_comparison.py` | Historical embedded cross-validation table; useful as an analysis artifact, not the live canonical validation dataset. |

## Legacy and exploratory validation scripts

`autoresearch/validation/` also contains older and exploratory scripts. Keep them close to validation, but distinguish them from the canonical CPU trace path:

- `cross_validate*.py` and `gpu_algo_python.py`: historical cross-validation logic.
- `capture_mesa.py`, `replay_python.py`, `capture_oscillating_trace.py`: trace capture/replay support.
- `sweep_oscillating*.py`, `probe_oscillation.py`: targeted oscillation/seed search utilities.
- `build_animation.py`, `build_showboat.py`: reader-facing validation visualization output.

## Gaps to resolve before paper-grade verification

1. Add a clean GPU trace dataset analogous to the CPU bit-exact validation outputs.
2. Replace or wrap hardcoded GPU comparison parameters with a documented config file.
3. Document the exact accepted commands and expected success text for each comparer.
4. Preserve the two-link validation chain in future work: Mesa → CPU first, then CPU → GPU.
5. Add targeted Phase 1F per-agent state output for thesis-style individual-agent analysis.
