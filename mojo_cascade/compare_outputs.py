#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
Run the Python cascade simulation with the same parameters and seeds as
the Mojo version, to compare outputs.
"""
import sys
import os
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resistance_cascade.model import ResistanceCascade


def run_comparison():
    seeds = [42, 123, 456, 789, 1001]
    epsilons = [0.2, 0.5, 1.0]
    sec_densities = [0.0, 0.02, 0.05]
    num_steps = 50

    total = len(seeds) * len(epsilons) * len(sec_densities)
    print(f"Running {total} Python simulations x {num_steps} steps")
    print()

    idx = 0
    t0 = time.perf_counter()
    for seed in seeds:
        for eps in epsilons:
            for sd in sec_densities:
                model = ResistanceCascade(
                    seed=seed,
                    citizen_density=0.7,
                    security_density=sd,
                    epsilon=eps,
                    threshold=2.94444,
                    multiple_agents_per_cell=True,
                    max_iters=1000,
                )
                for _ in range(num_steps):
                    model.step()

                active = model.count_active(model)
                support = model.count_support(model)
                oppose = model.count_oppose(model)
                jail = model.count_jail(model)
                rev = model.revolution

                print(
                    f"Sim {idx} seed={seed} eps={eps} sd={sd} "
                    f"active={active} support={support} "
                    f"oppose={oppose} jail={jail} rev={rev}"
                )
                idx += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {idx} simulations in {elapsed:.2f}s")
    print(f"Throughput: {idx / elapsed:.2f} sims/sec")
    print(f"Per-sim: {elapsed / idx:.3f}s")


if __name__ == "__main__":
    run_comparison()
