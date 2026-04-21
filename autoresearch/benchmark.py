"""
Benchmark harness for block_per_sim GPU kernel.
Compares correctness (SHA-256 fingerprint) and throughput against baseline.

Usage:
    uv run benchmark_block_per_sim.py --baseline   # Save baseline fingerprint + timing
    uv run benchmark_block_per_sim.py --compare    # Compare against saved baseline
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASELINE_FILE = Path("benchmark_baseline.json")
BLOCK_PER_SIM_BIN = "./block_per_sim_gpu"
BASELINE_BIN = "./cascade_gpu_batch"


def run_binary(binary: str) -> tuple[str, list[str], float]:
    """Run a binary and return (stdout, sim_lines, wall_time)."""
    start = time.perf_counter()
    result = subprocess.run(
        [binary], capture_output=True, text=True, timeout=300
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"ERROR: {binary} failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(1)
    stdout = result.stdout
    sim_lines = sorted(
        line.strip() for line in stdout.splitlines() if line.startswith("Sim ")
    )
    return stdout, sim_lines, elapsed


def fingerprint(sim_lines: list[str]) -> str:
    """SHA-256 of sorted simulation output lines."""
    content = "\n".join(sim_lines)
    return hashlib.sha256(content.encode()).hexdigest()


def extract_kernel_time(stdout: str) -> float:
    """Extract GPU kernel time from output."""
    for line in stdout.splitlines():
        if "simulations in" in line and "seconds" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "in":
                    return float(parts[i + 1])
    return 0.0


def extract_throughput(stdout: str) -> float:
    """Extract throughput from output."""
    for line in stdout.splitlines():
        if "Throughput:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "Throughput:":
                    return float(parts[i + 1])
    return 0.0


def save_baseline():
    print("=== Saving Baseline ===")
    print(f"Running: {BASELINE_BIN}")
    stdout, sim_lines, wall_time = run_binary(BASELINE_BIN)
    fp = fingerprint(sim_lines)
    kernel_time = extract_kernel_time(stdout)
    throughput = extract_throughput(stdout)

    baseline = {
        "fingerprint": fp,
        "num_sims": len(sim_lines),
        "kernel_time_s": kernel_time,
        "throughput_sims_sec": throughput,
        "wall_time_s": wall_time,
        "binary": BASELINE_BIN,
    }
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
    print(f"Fingerprint: {fp[:16]}...")
    print(f"Sims: {len(sim_lines)}")
    print(f"Kernel time: {kernel_time:.4f}s")
    print(f"Throughput: {throughput:.1f} sims/sec")
    print(f"Saved to {BASELINE_FILE}")


def compare():
    if not BASELINE_FILE.exists():
        print("ERROR: No baseline found. Run with --baseline first.")
        sys.exit(1)

    baseline = json.loads(BASELINE_FILE.read_text())
    print("=== Comparing Against Baseline ===")
    print(f"Baseline: {baseline['binary']} ({baseline['num_sims']} sims, "
          f"{baseline['throughput_sims_sec']:.1f} sims/sec)")
    print()

    print(f"Running: {BLOCK_PER_SIM_BIN}")
    stdout, sim_lines, wall_time = run_binary(BLOCK_PER_SIM_BIN)
    fp = fingerprint(sim_lines)
    kernel_time = extract_kernel_time(stdout)
    throughput = extract_throughput(stdout)

    # Correctness check
    if fp == baseline["fingerprint"]:
        print(f"Correctness: PASS (fingerprint match: {fp[:16]}...)")
    else:
        print(f"Correctness: FAIL")
        print(f"  Expected: {baseline['fingerprint'][:16]}...")
        print(f"  Got:      {fp[:16]}...")
        # Show first difference
        _, baseline_lines, _ = run_binary(baseline["binary"])
        for i, (a, b) in enumerate(zip(baseline_lines, sim_lines)):
            if a != b:
                print(f"  First diff at line {i}:")
                print(f"    Baseline: {a}")
                print(f"    Current:  {b}")
                break
        sys.exit(1)

    # Performance
    speedup = throughput / baseline["throughput_sims_sec"] if baseline["throughput_sims_sec"] > 0 else 0
    print(f"Throughput: {throughput:.1f} sims/sec (baseline: {baseline['throughput_sims_sec']:.1f})")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Kernel time: {kernel_time:.4f}s (baseline: {baseline['kernel_time_s']:.4f}s)")


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        save_baseline()
    elif "--compare" in sys.argv:
        compare()
    else:
        print("Usage: uv run benchmark_block_per_sim.py [--baseline|--compare]")
        sys.exit(1)
