#!/usr/bin/env python3
# /// script
# dependencies = ["pandas", "pyarrow"]
# ///
"""
Parallel batch runner for ResistanceCascade.

Replaces the MPI-based batch_run.py with ProcessPoolExecutor for single-node
parallel execution. Runs all parameter combinations across available CPU cores.

Usage:
    uv run batch_run_parallel.py                          # Run with defaults
    uv run batch_run_parallel.py --params custom.json     # Custom parameters
    uv run batch_run_parallel.py --workers 8              # Limit workers
    uv run batch_run_parallel.py --steps 500 --workers 16 # Full sweep
    uv run batch_run_parallel.py --dry-run                # Show combos only
"""
import argparse
import json
import logging as log
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path


def run_single(params, fixed_params, max_steps, output_dir):
    """Run a single simulation with the given parameters. Runs in a worker process."""
    import pandas as pd
    from resistance_cascade.model import ResistanceCascade

    all_params = {**fixed_params, **params}
    seed = all_params.get("seed")

    try:
        model = ResistanceCascade(**all_params)

        for step in range(max_steps):
            model.step()
            if not model.running:
                break

        # Collect results
        model_df = model.datacollector.get_model_dataframe()
        agent_df = model.datacollector.get_agent_dataframe()

        # Build filename from parameters
        tag = (
            f"seed_{params.get('seed', seed)}"
            f"_pp_{params.get('private_preference_distribution_mean', 0)}"
            f"_sd{params.get('security_density', 0)}"
            f"_ep_{params.get('epsilon', 0)}"
            f"_th{fixed_params.get('threshold', 0)}"
        )

        model_path = output_dir / "model" / f"model_{tag}.parquet"
        agent_path = output_dir / "agent" / f"agent_{tag}.parquet"

        model_df.to_parquet(model_path)
        agent_df.to_parquet(agent_path)

        return {
            "params": params,
            "steps": step + 1,
            "revolution": model.revolution,
            "status": "ok",
            "model_path": str(model_path),
        }

    except Exception as e:
        return {
            "params": params,
            "status": "error",
            "error": str(e),
        }


def generate_combinations(variable_params):
    """Generate all parameter combinations from a dict of lists."""
    keys = list(variable_params.keys())
    values = list(variable_params.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch runner for ResistanceCascade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--params", default="resistance_cascade/batch_run_params.json",
        help="JSON file with parameter lists to sweep",
    )
    parser.add_argument("--steps", type=int, default=500, help="Max steps per simulation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--output", default=None, help="Output directory (default: data/YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print parameter combos without running")
    parser.add_argument("--resume", action="store_true", help="Skip combos with existing output files")
    args = parser.parse_args()

    # Load variable parameters
    with open(args.params) as f:
        variable_params = json.load(f)

    # Fixed parameters (not swept)
    fixed_params = {
        "multiple_agents_per_cell": True,
        "threshold": variable_params.pop("threshold", [2.94444])[0] if "threshold" in variable_params else 2.94444,
    }
    # If threshold was in variable params as a list with multiple values, put it back
    # For now, keep it simple - threshold is fixed

    # Generate all combinations
    all_combos = list(generate_combinations(variable_params))

    if args.dry_run:
        print(f"Fixed parameters: {fixed_params}")
        print(f"Total combinations: {len(all_combos)}")
        for i, combo in enumerate(all_combos):
            print(f"  [{i+1}] {combo}")
        return

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path("data") / current_date

    (output_dir / "model").mkdir(parents=True, exist_ok=True)
    (output_dir / "agent").mkdir(parents=True, exist_ok=True)

    # Resume: filter out already-completed runs
    if args.resume:
        original_count = len(all_combos)
        filtered = []
        for combo in all_combos:
            tag = (
                f"seed_{combo.get('seed', 0)}"
                f"_pp_{combo.get('private_preference_distribution_mean', 0)}"
                f"_sd{combo.get('security_density', 0)}"
                f"_ep_{combo.get('epsilon', 0)}"
                f"_th{fixed_params.get('threshold', 0)}"
            )
            if not (output_dir / "model" / f"model_{tag}.parquet").exists():
                filtered.append(combo)
        all_combos = filtered
        skipped = original_count - len(all_combos)
        if skipped:
            print(f"Resuming: skipping {skipped} completed runs, {len(all_combos)} remaining")

    if not all_combos:
        print("Nothing to run.")
        return

    workers = args.workers or os.cpu_count()
    print(f"Running {len(all_combos)} parameter combinations")
    print(f"  Workers: {workers}")
    print(f"  Max steps: {args.steps}")
    print(f"  Output: {output_dir}")
    print(f"  Fixed params: {fixed_params}")
    print()

    log.basicConfig(
        filename=str(output_dir / "batch.log"),
        level=log.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t_start = time.perf_counter()
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single, combo, fixed_params, args.steps, output_dir): combo
            for combo in all_combos
        }

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["status"] == "ok":
                rev = " REVOLUTION!" if result.get("revolution") else ""
                print(
                    f"  [{completed}/{len(all_combos)}] "
                    f"seed={result['params'].get('seed', '?')} "
                    f"eps={result['params'].get('epsilon', '?')} "
                    f"steps={result['steps']}{rev}"
                )
                log.info(f"Completed: {result['params']} steps={result['steps']}")
            else:
                errors += 1
                print(f"  [{completed}/{len(all_combos)}] ERROR: {result['error']}")
                log.error(f"Failed: {result['params']} error={result['error']}")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {completed} runs in {elapsed:.1f}s ({errors} errors)")
    print(f"  Throughput: {completed / elapsed:.2f} runs/sec")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
