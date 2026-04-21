#!/usr/bin/env python3
# /// script
# dependencies = ["numpy>=1.24", "pandas>=2.0"]
# ///
"""
Benchmark harness for the Resistance Cascade simulation.

Runs the model with fixed parameters and seed, captures step-by-step metrics,
and measures wall clock time. This is the evaluation harness — DO NOT MODIFY
during optimization work.

Usage:
    uv run benchmark.py                  # Run benchmark, print results
    uv run benchmark.py --baseline       # Save current results as baseline
    uv run benchmark.py --compare        # Compare against saved baseline
    uv run benchmark.py --steps 100      # Override step count
    uv run benchmark.py --json           # Output as JSON
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Fixed benchmark parameters — never change these
BENCH_PARAMS = dict(
    width=40,
    height=40,
    citizen_vision=7,
    citizen_density=0.7,
    security_density=0.02,
    security_vision=7,
    max_jail_term=100,
    movement=True,
    multiple_agents_per_cell=True,
    private_preference_distribution_mean=-0.7,
    standard_deviation=1,
    epsilon=0.5,
    max_iters=1000,
    threshold=3.66356,
    seed=42,
    random_seed=False,
)

BASELINE_FILE = Path(__file__).parent / "benchmark_baseline.json"


def run_benchmark(steps: int = 200) -> dict:
    """Run the model and collect metrics."""
    # Import here so we measure import time too
    t_import = time.perf_counter()
    from resistance_cascade.model import ResistanceCascade
    t_import = time.perf_counter() - t_import

    # Initialize
    t_init = time.perf_counter()
    model = ResistanceCascade(**BENCH_PARAMS)
    t_init = time.perf_counter() - t_init

    # Run steps and collect key metrics per step
    step_metrics = []
    t_run = time.perf_counter()
    for i in range(steps):
        model.step()
        if not model.running:
            break
        step_metrics.append({
            "step": i + 1,
            "active": model.count_active(model),
            "support": model.count_support(model),
            "oppose": model.count_oppose(model),
            "jail": model.count_jail(model),
            "spread": round(model.speed_of_spread(model), 6),
            "revolution": model.revolution,
        })
    t_run = time.perf_counter() - t_run
    actual_steps = len(step_metrics)

    # Compute a fingerprint of the metrics for correctness checking
    metrics_str = json.dumps(step_metrics, sort_keys=True)
    fingerprint = hashlib.sha256(metrics_str.encode()).hexdigest()[:16]

    return {
        "steps_completed": actual_steps,
        "revolution": model.revolution,
        "import_time_s": round(t_import, 4),
        "init_time_s": round(t_init, 4),
        "run_time_s": round(t_run, 4),
        "total_time_s": round(t_import + t_init + t_run, 4),
        "steps_per_second": round(actual_steps / t_run, 2) if t_run > 0 else 0,
        "fingerprint": fingerprint,
        "final_active": step_metrics[-1]["active"] if step_metrics else 0,
        "final_support": step_metrics[-1]["support"] if step_metrics else 0,
        "final_oppose": step_metrics[-1]["oppose"] if step_metrics else 0,
        "final_jail": step_metrics[-1]["jail"] if step_metrics else 0,
        "final_spread": step_metrics[-1]["spread"] if step_metrics else 0,
    }


def print_results(results: dict, as_json: bool = False):
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print("---")
    print(f"steps_completed:  {results['steps_completed']}")
    print(f"revolution:       {results['revolution']}")
    print(f"fingerprint:      {results['fingerprint']}")
    print(f"import_time_s:    {results['import_time_s']}")
    print(f"init_time_s:      {results['init_time_s']}")
    print(f"run_time_s:       {results['run_time_s']}")
    print(f"total_time_s:     {results['total_time_s']}")
    print(f"steps_per_second: {results['steps_per_second']}")
    print(f"final_active:     {results['final_active']}")
    print(f"final_support:    {results['final_support']}")
    print(f"final_oppose:     {results['final_oppose']}")
    print(f"final_jail:       {results['final_jail']}")
    print(f"final_spread:     {results['final_spread']}")
    print("---")


def save_baseline(results: dict):
    BASELINE_FILE.write_text(json.dumps(results, indent=2))
    print(f"Baseline saved to {BASELINE_FILE}")


def compare_baseline(results: dict):
    if not BASELINE_FILE.exists():
        print("ERROR: No baseline found. Run with --baseline first.", file=sys.stderr)
        sys.exit(1)

    baseline = json.loads(BASELINE_FILE.read_text())

    # Check correctness
    correct = results["fingerprint"] == baseline["fingerprint"]
    speedup = baseline["run_time_s"] / results["run_time_s"] if results["run_time_s"] > 0 else float("inf")

    print(f"Correctness:  {'PASS' if correct else 'FAIL'}")
    print(f"  baseline fingerprint: {baseline['fingerprint']}")
    print(f"  current fingerprint:  {results['fingerprint']}")
    print(f"Performance:")
    print(f"  baseline run_time:    {baseline['run_time_s']}s")
    print(f"  current run_time:     {results['run_time_s']}s")
    print(f"  speedup:              {speedup:.2f}x")
    print(f"  baseline steps/s:     {baseline['steps_per_second']}")
    print(f"  current steps/s:      {results['steps_per_second']}")

    if not correct:
        print("\nWARNING: Output metrics differ from baseline!", file=sys.stderr)
        print("This means the simulation math has changed.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cascade simulation benchmark")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare against baseline")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps (default: 200)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_benchmark(args.steps)
    print_results(results, as_json=args.json)

    if args.baseline:
        save_baseline(results)
    if args.compare:
        compare_baseline(results)


if __name__ == "__main__":
    main()
