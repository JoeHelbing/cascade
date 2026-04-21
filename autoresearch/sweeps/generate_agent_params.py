"""Generate Phase 1F agent-level simulation parameter configurations.

Produces agent_sim_params.json with ~459 carefully chosen parameter
coordinates for per-agent state capture across four analysis groups.
"""

import json
from pathlib import Path


def main():
    params = []

    # Fixed defaults
    citizen_density = 0.7
    max_jail = 100
    num_steps = 500

    # Group 1: Security density phase transition (sec_density x threshold)
    # Primary phase transition sweep -- the cliff is at sd 0.02-0.04
    group1_seeds = [42, 123, 456, 789, 1001]
    group1_sd = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]
    group1_th = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    for sd in group1_sd:
        for th in group1_th:
            for seed in group1_seeds:
                params.append({
                    "seed": seed,
                    "pp_mean": -0.5,
                    "sec_density": sd,
                    "epsilon": 0.5,
                    "threshold": th,
                    "vision": 7,
                    "num_steps": num_steps,
                    "citizen_density": citizen_density,
                    "max_jail": max_jail,
                    "group": "phase_transition_sd",
                    "note": f"SD={sd} TH={th} -- security density phase transition",
                })

    # Group 2: Vision cascade control (vision x sec_density)
    group2_seeds = [42, 123, 456]
    group2_vision = [1, 2, 3, 4, 5, 7, 10]
    group2_sd = [0.02, 0.03, 0.04]
    for v in group2_vision:
        for sd in group2_sd:
            for seed in group2_seeds:
                params.append({
                    "seed": seed,
                    "pp_mean": -0.5,
                    "sec_density": sd,
                    "epsilon": 0.5,
                    "threshold": 2.94,
                    "vision": v,
                    "num_steps": num_steps,
                    "citizen_density": citizen_density,
                    "max_jail": max_jail,
                    "group": "vision_cascade",
                    "note": f"V={v} SD={sd} -- vision controls cascade periodicity",
                })

    # Group 3: Epsilon micro-level comparison
    group3_seeds = [42, 123, 456]
    group3_eps = [0.1, 0.2, 0.5, 1.0, 1.5]
    group3_sd = [0.0, 0.02, 0.04]
    for eps in group3_eps:
        for sd in group3_sd:
            for seed in group3_seeds:
                params.append({
                    "seed": seed,
                    "pp_mean": -0.5,
                    "sec_density": sd,
                    "epsilon": eps,
                    "threshold": 2.94,
                    "vision": 7,
                    "num_steps": num_steps,
                    "citizen_density": citizen_density,
                    "max_jail": max_jail,
                    "group": "epsilon_micro",
                    "note": f"EPS={eps} SD={sd} -- epsilon micro-level behavior",
                })

    # Group 4: PP mean at the cliff edge
    group4_seeds = [42, 123, 456]
    group4_pp = [-1.0, -0.8, -0.5, -0.3, 0.0, 0.3]
    group4_sd = [0.02, 0.03]
    for pp in group4_pp:
        for sd in group4_sd:
            for seed in group4_seeds:
                params.append({
                    "seed": seed,
                    "pp_mean": pp,
                    "sec_density": sd,
                    "epsilon": 0.5,
                    "threshold": 2.94,
                    "vision": 7,
                    "num_steps": num_steps,
                    "citizen_density": citizen_density,
                    "max_jail": max_jail,
                    "group": "pp_mean_cliff",
                    "note": f"PP={pp} SD={sd} -- perceived legitimacy at cliff edge",
                })

    # Write JSON
    out_path = Path(__file__).parent / "agent_sim_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    # Summary
    from collections import Counter
    counts = Counter(p["group"] for p in params)
    print(f"Total simulations: {len(params)}")
    print(f"Estimated agent data: ~{len(params) * 9:.0f} MB ({len(params) * 9 / 1024:.1f} GB)")
    print()
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")
    print()
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
