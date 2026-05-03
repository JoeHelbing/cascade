# Benchmarks

Fixed evaluation harnesses for performance and fingerprint checks.

| File | Purpose |
|---|---|
| `gpu_kernel_benchmark.py` | Runs GPU binaries, hashes sorted `Sim ...` output lines, and compares throughput/fingerprint against `benchmark_baseline.json`. |
| `benchmark_baseline.json` | Current GPU benchmark/fingerprint baseline. |
| `python_reference_benchmark.py` | Fixed benchmark for the original Python/Mesa implementation. |

## Typical commands

```bash
uv run autoresearch/benchmarks/gpu_kernel_benchmark.py --compare
uv run autoresearch/benchmarks/python_reference_benchmark.py --compare
```

Treat these as fixed harnesses: when optimizing implementation files, do not tune benchmark parameters to improve reported performance.
