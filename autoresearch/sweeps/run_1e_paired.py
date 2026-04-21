"""
Phase 1E: Paired Model Comparisons.

Reads phase transition boundaries from Phase 1D analysis and runs paired
simulations at those boundaries, varying one parameter at a time.

Each pair shares the same seed so the only difference is the parameter change.

Usage:
    cd mojo_cascade
    pixi run python run_1e_paired.py
"""

import json
import struct
import subprocess
import time
from pathlib import Path

import apsw
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GRID_SIZE = 33 * 33
N_STEP_FIELDS = 5
RUNNER_BIN = "./cascade_gpu_runner"
NUM_STEPS = 500
N_SEEDS = 20  # Same seeds as Phase 1D for reproducibility

OUT_DIR = Path("analysis_1e_output")
FIGURES_DIR = OUT_DIR / "figures"
DB_PATH = "manifold_results/manifold.db"

# Fixed parameters (same as Phase 1D)
CITIZEN_DENSITY = 0.7
MAX_JAIL = 100

# Default parameter values (center of Phase 1D ranges)
DEFAULTS = {
    "pp_mean": 0.0,
    "sec_density": 0.04,
    "epsilon": 1.0,
    "threshold": 3.5,
    "vision": 5,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = FIGURES_DIR / f"{name}.json"
    html_path = FIGURES_DIR / f"{name}.html"
    with open(json_path, "w") as f:
        json.dump(fig.to_dict(), f, cls=NumpyEncoder, separators=(",", ":"))
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {name}")


def build_paired_configs():
    """Build paired simulation configs based on Phase 1D transition analysis."""
    transitions_path = Path("analysis_1d_output/phase_transitions_1d.json")

    # Pre-defined comparisons from the plan + auto-detected transitions
    comparisons = []

    # 1. Security density cliff: vary sec_density across the transition
    #    at several threshold values
    for th in [2.0, 3.0, 4.0]:
        for sd_lo, sd_hi in [(0.02, 0.04), (0.03, 0.05), (0.04, 0.06)]:
            comparisons.append({
                "name": f"sec_density_{sd_lo}_vs_{sd_hi}_th{th}",
                "description": f"Security density {sd_lo} vs {sd_hi} at threshold={th}",
                "params_a": {**DEFAULTS, "sec_density": sd_lo, "threshold": th},
                "params_b": {**DEFAULTS, "sec_density": sd_hi, "threshold": th},
                "varied_param": "sec_density",
            })

    # 2. Vision cascade effect: vision 1 vs 7 at various security densities
    for sd in [0.02, 0.03, 0.04, 0.05]:
        comparisons.append({
            "name": f"vision_1_vs_7_sd{sd}",
            "description": f"Vision 1 vs 7 at sec_density={sd}",
            "params_a": {**DEFAULTS, "vision": 1, "sec_density": sd},
            "params_b": {**DEFAULTS, "vision": 7, "sec_density": sd},
            "varied_param": "vision",
        })

    # 3. Vision at threshold boundary
    for th in [2.0, 3.0, 4.0]:
        comparisons.append({
            "name": f"vision_2_vs_8_th{th}",
            "description": f"Vision 2 vs 8 at threshold={th}",
            "params_a": {**DEFAULTS, "vision": 2, "threshold": th},
            "params_b": {**DEFAULTS, "vision": 8, "threshold": th},
            "varied_param": "vision",
        })

    # 4. PP mean at the security density cliff
    for sd in [0.03, 0.04, 0.05]:
        comparisons.append({
            "name": f"pp_neg05_vs_pos05_sd{sd}",
            "description": f"PP mean -0.5 vs 0.5 at sec_density={sd}",
            "params_a": {**DEFAULTS, "pp_mean": -0.5, "sec_density": sd},
            "params_b": {**DEFAULTS, "pp_mean": 0.5, "sec_density": sd},
            "varied_param": "pp_mean",
        })

    # If Phase 1D transitions are available, add auto-detected pairs
    if transitions_path.exists():
        with open(transitions_path) as f:
            transitions = json.load(f)
        for t in transitions[:5]:
            pa, pb = t["param_a"], t["param_b"]
            a_coord, b_coord = t["a_coord"], t["b_coord"]
            # Create pairs straddling the transition
            if t["axis"] == "a":
                # Vary param_a around the transition point
                delta = a_coord * 0.2 if a_coord != 0 else 0.01
                params_base = {**DEFAULTS, pa: a_coord, pb: b_coord}
                comparisons.append({
                    "name": f"auto_{pa}_{a_coord:.4f}_at_{pb}_{b_coord:.4f}",
                    "description": f"Auto-detected: {pa} transition at {pb}={b_coord:.3f}",
                    "params_a": {**params_base, pa: a_coord - delta},
                    "params_b": {**params_base, pa: a_coord + delta},
                    "varied_param": pa,
                })

    return comparisons


def run_paired_sims(comparisons):
    """Run all paired simulations and collect time series data."""
    seeds = [42 + i * 7919 for i in range(N_SEEDS)]
    outdir = Path("manifold_results")
    params_path = outdir / "params.bin"
    metrics_path = outdir / "metrics.bin"
    steps_path = outdir / "step_metrics.bin"

    # Build all sim configs (2 * n_comparisons * n_seeds)
    all_params = []
    all_labels = []
    for comp in comparisons:
        for side in ["a", "b"]:
            p = comp[f"params_{side}"]
            for seed in seeds:
                row = np.zeros(9, dtype=np.float32)
                row[0] = seed
                row[1] = CITIZEN_DENSITY
                row[2] = p["sec_density"]
                row[3] = p["pp_mean"]
                row[4] = p["epsilon"]
                row[5] = p["threshold"]
                row[6] = MAX_JAIL
                row[7] = NUM_STEPS
                row[8] = p["vision"]
                all_params.append(row)
                all_labels.append((comp["name"], side, seed, p))

    n_sims = len(all_params)
    print(f"  {len(comparisons)} comparisons x 2 sides x {N_SEEDS} seeds = {n_sims:,} sims")

    params_arr = np.array(all_params, dtype=np.float32)

    # Write params.bin
    with open(params_path, "wb") as f:
        f.write(struct.pack("ii", n_sims, NUM_STEPS))
        f.write(params_arr.tobytes())

    # Run GPU
    print(f"  Launching GPU runner for {n_sims:,} sims...")
    t0 = time.time()
    subprocess.run(
        [RUNNER_BIN, str(params_path), str(metrics_path), str(steps_path)],
        check=True,
    )
    gpu_time = time.time() - t0
    print(f"  GPU completed in {gpu_time:.1f}s ({n_sims / gpu_time:.0f} sims/sec)")

    # Read results
    metrics = np.fromfile(str(metrics_path), dtype=np.int32).reshape(n_sims, 6)
    step_data = np.fromfile(str(steps_path), dtype=np.int32).reshape(n_sims, NUM_STEPS, N_STEP_FIELDS)

    # Clean up binary files
    params_path.unlink(missing_ok=True)
    metrics_path.unlink()
    steps_path.unlink()

    # Organize results by comparison
    results = {}
    idx = 0
    for comp in comparisons:
        comp_result = {"a": [], "b": []}
        for side in ["a", "b"]:
            for seed_idx in range(N_SEEDS):
                comp_result[side].append({
                    "seed": seeds[seed_idx],
                    "active": step_data[idx, :, 0].tolist(),
                    "jail": step_data[idx, :, 3].tolist(),
                    "revolution": int(metrics[idx, 3]),  # revolution_step from metrics
                    "n_citizens": int(metrics[idx, 5]),
                })
                idx += 1
        results[comp["name"]] = {
            "comparison": comp,
            "results": comp_result,
        }

    return results


def plot_paired_comparison(comp_name, comp_data, save_dir):
    """Plot overlay time series for a paired comparison."""
    comp = comp_data["comparison"]
    results = comp_data["results"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'{comp["varied_param"]}={comp["params_a"][comp["varied_param"]]}',
            f'{comp["varied_param"]}={comp["params_b"][comp["varied_param"]]}',
        ],
    )

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for side_idx, side in enumerate(["a", "b"]):
        col = side_idx + 1
        for i, run in enumerate(results[side][:5]):  # Show 5 seeds
            fig.add_trace(
                go.Scatter(
                    x=list(range(NUM_STEPS)),
                    y=run["active"],
                    mode="lines",
                    line=dict(color=colors[i % 5], width=1),
                    name=f"seed {run['seed']}" if side_idx == 0 else None,
                    showlegend=(side_idx == 0 and i < 5),
                    legendgroup=f"seed{i}",
                ),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="Step", row=1, col=col)
        fig.update_yaxes(title_text="Active Citizens", row=1, col=col)

    fig.update_layout(
        title=comp["description"],
        height=400, width=1000,
    )
    save_figure(fig, f"paired_{comp_name}")


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1E: Paired Model Comparisons ===\n")

    # Build comparison configs
    comparisons = build_paired_configs()
    print(f"Built {len(comparisons)} paired comparisons")

    # Run all paired sims
    results = run_paired_sims(comparisons)

    # Save raw results
    with open(OUT_DIR / "paired_results_1e.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

    # Generate figures
    print("\n--- Generating paired comparison figures ---")
    for comp_name, comp_data in results.items():
        plot_paired_comparison(comp_name, comp_data, FIGURES_DIR)

    # Summary statistics
    print("\n--- Summary ---")
    for comp_name, comp_data in results.items():
        results_a = comp_data["results"]["a"]
        results_b = comp_data["results"]["b"]
        rev_a = sum(1 for r in results_a if r["revolution"] >= 0) / len(results_a)
        rev_b = sum(1 for r in results_b if r["revolution"] >= 0) / len(results_b)
        desc = comp_data["comparison"]["description"]
        print(f"  {desc}: rev_prob {rev_a:.1%} vs {rev_b:.1%}")

    with open(OUT_DIR / "summary_1e.json", "w") as f:
        summary = {}
        for comp_name, comp_data in results.items():
            results_a = comp_data["results"]["a"]
            results_b = comp_data["results"]["b"]
            summary[comp_name] = {
                "description": comp_data["comparison"]["description"],
                "rev_prob_a": sum(1 for r in results_a if r["revolution"] >= 0) / len(results_a),
                "rev_prob_b": sum(1 for r in results_b if r["revolution"] >= 0) / len(results_b),
                "mean_max_active_a": np.mean([max(r["active"]) for r in results_a]),
                "mean_max_active_b": np.mean([max(r["active"]) for r in results_b]),
            }
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t_start
    print(f"\nPhase 1E complete in {elapsed:.1f}s")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
