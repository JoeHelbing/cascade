# Benchmarks

Fixed evaluation harnesses for performance checks. Correctness validation lives in `../validation/`; start with `pixi run validate-cpu`, `pixi run validate-gpu`, or `pixi run validate`.

| File | Purpose |
|---|---|
| `gpu_kernel_benchmark.py` | Historical GPU benchmark/fingerprint helper for old binary names; use `../validation/run_pipeline.py` for the live GPU validation smoke gate. |
| `benchmark_baseline.json` | Current GPU benchmark/fingerprint baseline. |
| `python_reference_benchmark.py` | Fixed benchmark for the original Python/Mesa implementation. |

## Typical commands

```bash
uv run autoresearch/benchmarks/gpu_kernel_benchmark.py --compare
uv run autoresearch/benchmarks/python_reference_benchmark.py --compare
```

Treat these as fixed harnesses: when optimizing implementation files, do not tune benchmark parameters to improve reported performance.
