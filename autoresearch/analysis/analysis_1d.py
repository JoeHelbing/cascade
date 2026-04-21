"""
Phase 1D: High-Resolution Sweep Analysis.

Analyzes the Phase 1D targeted sweep results:
1. High-resolution pairwise manifold surfaces (sec_density x threshold at 25x25)
2. Vision's cascade effect at high resolution
3. Phase transition boundary identification
4. Parameter importance at higher resolution
5. Archetype maps at high resolution

Outputs interactive Plotly figures for reports-web.

Usage:
    cd mojo_cascade
    pixi run python analysis_1d.py
"""

import json
import time
from itertools import combinations
from pathlib import Path

import apsw
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DB_PATH = "manifold_results/manifold.db"
OUT_DIR = Path("analysis_1d_output")
FIGURES_DIR = OUT_DIR / "figures"

PARAMS = ["pp_mean", "sec_density", "epsilon", "threshold", "vision"]
PARAM_LABELS = {
    "pp_mean": "PP Mean",
    "sec_density": "Security Density",
    "epsilon": "Epsilon",
    "threshold": "Threshold",
    "vision": "Vision",
}

Z_METRICS = {
    "revolution_prob": "Revolution Probability",
    "max_active_pct": "Max Active %",
    "mean_active_pct": "Mean Active %",
    "cascade_rate": "Cascade Rate (2+ peaks)",
    "periodic_rate": "Periodic Rate",
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


def save_figure(fig, save_path, name):
    save_path.mkdir(parents=True, exist_ok=True)
    json_path = save_path / f"{name}.json"
    html_path = save_path / f"{name}.html"
    fig_dict = fig.to_dict()
    with open(json_path, "w") as f:
        json.dump(fig_dict, f, cls=NumpyEncoder, separators=(",", ":"))
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {name}")


def get_conn():
    return apsw.Connection(DB_PATH, apsw.SQLITE_OPEN_READONLY)


def load_1d_data(conn):
    """Load Phase 1D simulation data (citizen_density=0.7, max_jail=100, num_steps=500)."""
    # Phase 1D sims are distinguished by fixed cd=0.7 and mj=100 at 500 steps
    # Also by sim_id range if sweep_metadata exists
    try:
        meta = list(conn.execute(
            "SELECT start_sim_id, end_sim_id FROM sweep_metadata WHERE sweep_id='phase_1d'"
        ))
        if meta:
            start_id, end_id = meta[0]
            where = f"WHERE sim_id BETWEEN {start_id} AND {end_id}"
        else:
            where = "WHERE num_steps = 500 AND citizen_density = 0.7 AND max_jail = 100"
    except apsw.SQLError:
        where = "WHERE num_steps = 500 AND citizen_density = 0.7 AND max_jail = 100"

    query = f"""
        SELECT pp_mean, sec_density, epsilon, threshold, vision, seed,
               n_citizens, max_active, revolution_step,
               n_cascades, cascade_periodic, sum_active, num_steps, sim_id
        FROM simulations {where}
        ORDER BY pp_mean, sec_density, epsilon, threshold, vision, seed
    """
    rows = list(conn.execute(query))
    if not rows:
        return None
    data = np.array(rows)
    return {
        "pp_mean": data[:, 0],
        "sec_density": data[:, 1],
        "epsilon": data[:, 2],
        "threshold": data[:, 3],
        "vision": data[:, 4],
        "seed": data[:, 5],
        "n_citizens": data[:, 6],
        "max_active": data[:, 7],
        "revolution_step": data[:, 8],
        "n_cascades": data[:, 9],
        "cascade_periodic": data[:, 10],
        "sum_active": data[:, 11],
        "num_steps": data[:, 12],
        "sim_id": data[:, 13],
    }


def compute_z_metrics(data):
    n_cit = data["n_citizens"].astype(float)
    n_cit = np.where(n_cit > 0, n_cit, 1.0)
    num_steps = data["num_steps"].astype(float)
    return {
        "revolution_prob": (data["revolution_step"] >= 0).astype(float),
        "max_active_pct": data["max_active"] / n_cit,
        "mean_active_pct": data["sum_active"] / (n_cit * num_steps),
        "cascade_rate": (data["n_cascades"] >= 2).astype(float),
        "periodic_rate": data["cascade_periodic"].astype(float),
    }


def compute_parameter_importance(data, z_metrics):
    results = {}
    for z_name, z_vals in z_metrics.items():
        total_var = np.var(z_vals)
        if total_var < 1e-12:
            results[z_name] = {p: 0.0 for p in PARAMS}
            continue
        param_importance = {}
        for param in PARAMS:
            p_vals = data[param]
            unique_vals = np.unique(p_vals)
            group_means = np.array([z_vals[p_vals == v].mean() for v in unique_vals])
            between_var = np.var(group_means)
            param_importance[param] = float(between_var / total_var)
        results[z_name] = param_importance
    return results


def compute_pairwise_manifold(data, z_metrics, z_name, param_a, param_b):
    z_vals = z_metrics[z_name]
    a_vals = data[param_a]
    b_vals = data[param_b]
    a_unique = np.unique(a_vals)
    b_unique = np.unique(b_vals)
    z_grid = np.full((len(a_unique), len(b_unique)), np.nan)
    for i, av in enumerate(a_unique):
        for j, bv in enumerate(b_unique):
            mask = (a_vals == av) & (b_vals == bv)
            if mask.sum() > 0:
                z_grid[i, j] = z_vals[mask].mean()
    return a_unique, b_unique, z_grid


def plot_hires_manifold(data, z_metrics, z_name, z_label, param_a, param_b, save_path, extra_title=""):
    a_unique, b_unique, z_grid = compute_pairwise_manifold(data, z_metrics, z_name, param_a, param_b)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_grid.T,
        x=np.round(a_unique, 4),
        y=np.round(b_unique, 4),
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(title=z_label),
    ))
    fig.update_layout(
        title=f"{z_label}: {PARAM_LABELS[param_a]} vs {PARAM_LABELS[param_b]}{extra_title}",
        xaxis_title=PARAM_LABELS[param_a],
        yaxis_title=PARAM_LABELS[param_b],
        height=550,
        width=700,
    )
    slug = f"hires_{z_name}_{param_a}_vs_{param_b}"
    save_figure(fig, save_path, slug)
    return fig


def plot_all_pairwise(data, z_metrics, z_name, z_label, save_path):
    """Plot all 10 pairwise manifolds for 5 parameters."""
    pairs = list(combinations(PARAMS, 2))
    n_pairs = len(pairs)
    n_cols = 5
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{PARAM_LABELS[pa]} vs {PARAM_LABELS[pb]}" for pa, pb in pairs],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for idx, (pa, pb) in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        a_unique, b_unique, z_grid = compute_pairwise_manifold(data, z_metrics, z_name, pa, pb)
        fig.add_trace(
            go.Heatmap(
                z=z_grid.T,
                x=np.round(a_unique, 4),
                y=np.round(b_unique, 4),
                colorscale="Viridis",
                zmin=0, zmax=1,
                showscale=(idx == 0),
                colorbar=dict(title=z_label) if idx == 0 else None,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=PARAM_LABELS[pa], row=row, col=col)
        fig.update_yaxes(title_text=PARAM_LABELS[pb], row=row, col=col)

    fig.update_layout(
        title=f"High-Resolution Manifolds: {z_label} (Phase 1D, 500 steps)",
        height=350 * n_rows,
        width=1200,
    )
    save_figure(fig, save_path, f"hires_all_{z_name}")
    return fig


def find_phase_transitions(data, z_metrics):
    """Identify parameter coordinates where sharp transitions occur."""
    transitions = []

    for z_name in ["revolution_prob", "cascade_rate"]:
        z_vals = z_metrics[z_name]

        for pa, pb in [("sec_density", "threshold"), ("vision", "sec_density"),
                       ("vision", "threshold")]:
            a_vals = data[pa]
            b_vals = data[pb]
            a_unique = np.unique(a_vals)
            b_unique = np.unique(b_vals)

            z_grid = np.full((len(a_unique), len(b_unique)), np.nan)
            for i, av in enumerate(a_unique):
                for j, bv in enumerate(b_unique):
                    mask = (a_vals == av) & (b_vals == bv)
                    if mask.sum() > 0:
                        z_grid[i, j] = z_vals[mask].mean()

            # Find max gradient along each axis
            for axis in [0, 1]:
                grad = np.abs(np.diff(z_grid, axis=axis))
                grad = np.nan_to_num(grad)
                if grad.size == 0:
                    continue
                max_idx = np.unravel_index(grad.argmax(), grad.shape)
                max_grad = grad[max_idx]
                if max_grad > 0.3:  # >30% change in one step
                    if axis == 0:
                        a_coord = 0.5 * (a_unique[max_idx[0]] + a_unique[max_idx[0] + 1])
                        b_coord = b_unique[max_idx[1]]
                    else:
                        a_coord = a_unique[max_idx[0]]
                        b_coord = 0.5 * (b_unique[max_idx[1]] + b_unique[max_idx[1] + 1])

                    transitions.append({
                        "z_metric": z_name,
                        "param_a": pa,
                        "param_b": pb,
                        "a_coord": float(a_coord),
                        "b_coord": float(b_coord),
                        "gradient": float(max_grad),
                        "axis": ["a", "b"][axis],
                    })

    transitions.sort(key=lambda x: x["gradient"], reverse=True)
    return transitions


def classify_trajectories(data, z_metrics):
    """Classify Phase 1D sims into archetypes."""
    n = len(data["sim_id"])
    n_cit = data["n_citizens"].astype(float)
    n_cit = np.where(n_cit > 0, n_cit, 1.0)
    max_active_pct = data["max_active"] / n_cit
    rev_step = data["revolution_step"]
    n_cascades = data["n_cascades"]
    periodic = data["cascade_periodic"]
    num_steps = NUM_STEPS_VAL = 500

    archetypes = np.full(n, "unknown", dtype=object)
    archetypes[(rev_step >= 0) & (rev_step < num_steps * 0.2)] = "fast_revolution"
    archetypes[(rev_step >= num_steps * 0.5) & (rev_step >= 0)] = "slow_burn"
    archetypes[(rev_step >= num_steps * 0.2) & (rev_step < num_steps * 0.5) & (rev_step >= 0)] = "mid_revolution"
    archetypes[(periodic == 1) & (rev_step < 0)] = "oscillating"
    archetypes[(n_cascades >= 1) & (rev_step < 0) & (periodic == 0)] = "abortive_spike"
    archetypes[(n_cascades == 0) & (rev_step < 0) & (max_active_pct < 0.1)] = "stable_suppression"
    archetypes[(n_cascades == 0) & (rev_step < 0) & (max_active_pct >= 0.1)] = "simmering"

    unique, counts = np.unique(archetypes, return_counts=True)
    return archetypes, dict(zip(unique, counts.astype(int).tolist()))


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_conn()

    print("=== Phase 1D Analysis: High-Resolution Targeted Sweep ===\n")
    print("Loading Phase 1D data...")
    data = load_1d_data(conn)
    if data is None:
        print("ERROR: No Phase 1D data found in database!")
        return
    n = len(data["sim_id"])
    print(f"  Loaded {n:,} simulations")

    z_metrics = compute_z_metrics(data)

    # 1. Parameter importance
    print("\n--- 1. Parameter Importance ---")
    importance = compute_parameter_importance(data, z_metrics)
    for z_name in Z_METRICS:
        ranked = sorted(importance[z_name].items(), key=lambda x: x[1], reverse=True)
        top3 = ", ".join(f"{PARAM_LABELS[p]}={v:.3f}" for p, v in ranked[:3])
        print(f"  {Z_METRICS[z_name]}: {top3}")

    with open(OUT_DIR / "parameter_importance_1d.json", "w") as f:
        json.dump(importance, f, indent=2, cls=NumpyEncoder)

    # Plot importance
    fig = go.Figure()
    for z_name in Z_METRICS:
        vals = importance[z_name]
        fig.add_trace(go.Bar(
            x=[PARAM_LABELS[p] for p in PARAMS],
            y=[vals[p] for p in PARAMS],
            name=Z_METRICS[z_name],
        ))
    fig.update_layout(
        title="Phase 1D Parameter Importance (500 steps, high-resolution)",
        yaxis_title="Fraction of Variance Explained",
        barmode="group",
        height=500, width=900,
    )
    save_figure(fig, FIGURES_DIR, "parameter_importance_1d")

    # 2. Key pairwise manifolds (high-res)
    print("\n--- 2. High-Resolution Pairwise Manifolds ---")
    key_pairs = [
        ("sec_density", "threshold"),
        ("vision", "threshold"),
        ("vision", "sec_density"),
        ("pp_mean", "sec_density"),
        ("pp_mean", "threshold"),
    ]
    for z_name in Z_METRICS:
        print(f"\n  {Z_METRICS[z_name]}:")
        # Individual high-res plots for key pairs
        for pa, pb in key_pairs:
            plot_hires_manifold(data, z_metrics, z_name, Z_METRICS[z_name], pa, pb, FIGURES_DIR)
        # All-pairs grid
        plot_all_pairwise(data, z_metrics, z_name, Z_METRICS[z_name], FIGURES_DIR)

    # 3. Phase transition detection
    print("\n--- 3. Phase Transition Boundaries ---")
    transitions = find_phase_transitions(data, z_metrics)
    for t in transitions[:10]:
        print(f"  {Z_METRICS[t['z_metric']]}: "
              f"{PARAM_LABELS[t['param_a']]}~{t['a_coord']:.4f} x "
              f"{PARAM_LABELS[t['param_b']]}~{t['b_coord']:.4f} "
              f"(gradient={t['gradient']:.3f})")

    with open(OUT_DIR / "phase_transitions_1d.json", "w") as f:
        json.dump(transitions, f, indent=2, cls=NumpyEncoder)

    # 4. Archetype distribution
    print("\n--- 4. Trajectory Archetypes ---")
    archetypes, arch_counts = classify_trajectories(data, z_metrics)
    total = sum(arch_counts.values())
    for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {arch}: {count:,} ({count/total:.1%})")

    with open(OUT_DIR / "archetype_counts_1d.json", "w") as f:
        json.dump(arch_counts, f, indent=2, cls=NumpyEncoder)

    # Archetype maps for key pairs
    print("\n  Computing archetype maps...")
    for pa, pb in [("sec_density", "threshold"), ("vision", "threshold"), ("vision", "sec_density")]:
        a_vals = data[pa]
        b_vals = data[pb]
        a_unique = np.unique(a_vals)
        b_unique = np.unique(b_vals)

        arch_types_plot = ["fast_revolution", "oscillating", "stable_suppression", "abortive_spike"]
        present = [a for a in arch_types_plot if a in set(archetypes)]

        fig = make_subplots(
            rows=1, cols=len(present),
            subplot_titles=[t.replace("_", " ").title() for t in present],
        )
        for col_i, arch in enumerate(present, 1):
            z_grid = np.full((len(a_unique), len(b_unique)), 0.0)
            for i, av in enumerate(a_unique):
                for j, bv in enumerate(b_unique):
                    mask = (a_vals == av) & (b_vals == bv)
                    if mask.sum() > 0:
                        z_grid[i, j] = (archetypes[mask] == arch).mean()
            fig.add_trace(
                go.Heatmap(
                    z=z_grid.T, x=np.round(a_unique, 4), y=np.round(b_unique, 4),
                    colorscale="Viridis", zmin=0, zmax=1,
                    showscale=(col_i == len(present)),
                ),
                row=1, col=col_i,
            )
            fig.update_xaxes(title_text=PARAM_LABELS[pa], row=1, col=col_i)
            fig.update_yaxes(title_text=PARAM_LABELS[pb], row=1, col=col_i)

        fig.update_layout(
            title=f"Archetype Map: {PARAM_LABELS[pa]} vs {PARAM_LABELS[pb]} (Phase 1D, 500 steps)",
            height=450, width=280 * len(present),
        )
        save_figure(fig, FIGURES_DIR, f"hires_archetype_{pa}_vs_{pb}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Phase 1D Analysis Complete in {elapsed:.1f}s")
    print(f"Output: {OUT_DIR}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('*.html')))} HTML, {len(list(FIGURES_DIR.glob('*.json')))} JSON")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
