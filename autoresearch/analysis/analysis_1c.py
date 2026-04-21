"""
Phase 1C: Coarse 7D Sweep Analysis.

Analyzes the coarse 7D parameter sweep to identify:
1. Parameter importance ranking (variance decomposition)
2. Step count comparison (100 vs 500 vs 1000)
3. Pairwise manifold surfaces for all 21 parameter pairs
4. Surprising findings with new parameters
5. Temporal dynamics / trajectory archetypes

Outputs:
- Interactive Plotly figures as JSON for reports-web
- Summary statistics as JSON
- Console report

Usage:
    cd mojo_cascade
    pixi run python analysis_1c.py
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
OUT_DIR = Path("analysis_1c_output")
FIGURES_DIR = OUT_DIR / "figures"

PARAMS = ["pp_mean", "sec_density", "epsilon", "threshold",
          "citizen_density", "max_jail", "vision"]
PARAM_LABELS = {
    "pp_mean": "PP Mean",
    "sec_density": "Security Density",
    "epsilon": "Epsilon",
    "threshold": "Threshold",
    "citizen_density": "Citizen Density",
    "max_jail": "Max Jail",
    "vision": "Vision",
}

# Z-metrics computed from simulation summaries
Z_METRICS = {
    "revolution_prob": "Revolution Probability",
    "max_active_pct": "Max Active %",
    "mean_active_pct": "Mean Active %",
    "cascade_rate": "Cascade Rate (2+ peaks)",
    "periodic_rate": "Periodic Rate",
}


def get_conn():
    conn = apsw.Connection(DB_PATH, apsw.SQLITE_OPEN_READONLY)
    return conn


def load_simulation_data(conn, num_steps=None):
    """Load simulation summary data into numpy arrays."""
    where = f"WHERE num_steps = {num_steps}" if num_steps else ""
    query = f"""
        SELECT pp_mean, sec_density, epsilon, threshold,
               citizen_density, max_jail, vision, seed,
               n_citizens, max_active, revolution_step,
               n_cascades, cascade_periodic, sum_active, num_steps
        FROM simulations {where}
        ORDER BY pp_mean, sec_density, epsilon, threshold,
                 citizen_density, max_jail, vision, seed
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
        "citizen_density": data[:, 4],
        "max_jail": data[:, 5],
        "vision": data[:, 6],
        "seed": data[:, 7],
        "n_citizens": data[:, 8],
        "max_active": data[:, 9],
        "revolution_step": data[:, 10],
        "n_cascades": data[:, 11],
        "cascade_periodic": data[:, 12],
        "sum_active": data[:, 13],
        "num_steps": data[:, 14],
    }


def compute_z_metrics(data):
    """Compute z-metrics from raw simulation data."""
    n = len(data["n_citizens"])
    n_cit = data["n_citizens"].astype(float)
    n_cit = np.where(n_cit > 0, n_cit, 1.0)  # avoid div by zero
    num_steps = data["num_steps"].astype(float)

    return {
        "revolution_prob": (data["revolution_step"] >= 0).astype(float),
        "max_active_pct": data["max_active"] / n_cit,
        "mean_active_pct": data["sum_active"] / (n_cit * num_steps),
        "cascade_rate": (data["n_cascades"] >= 2).astype(float),
        "periodic_rate": data["cascade_periodic"].astype(float),
    }


# ============================================================
# 1. Parameter Importance (Variance Decomposition)
# ============================================================

def compute_parameter_importance(data, z_metrics):
    """
    For each parameter, compute the fraction of total variance in each z-metric
    explained by that parameter (averaging over all other parameters).

    Uses the "main effect" approach: group by each parameter's unique values,
    compute the mean z-metric per group, then the variance of those group means
    divided by total variance.
    """
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
            group_means = np.array([
                z_vals[p_vals == v].mean() for v in unique_vals
            ])
            # Variance of group means = between-group variance
            between_var = np.var(group_means)
            param_importance[param] = float(between_var / total_var)

        results[z_name] = param_importance

    return results


def plot_parameter_importance(importance_by_step, save_path):
    """Create grouped bar chart of parameter importance across step counts."""
    fig = make_subplots(
        rows=len(Z_METRICS), cols=1,
        subplot_titles=[Z_METRICS[z] for z in Z_METRICS],
        vertical_spacing=0.06,
    )

    colors = {100: "#636EFA", 500: "#EF553B", 1000: "#00CC96"}

    for row_idx, z_name in enumerate(Z_METRICS, 1):
        for steps, importance in sorted(importance_by_step.items()):
            if z_name not in importance:
                continue
            vals = importance[z_name]
            fig.add_trace(
                go.Bar(
                    x=[PARAM_LABELS[p] for p in PARAMS],
                    y=[vals[p] for p in PARAMS],
                    name=f"{steps} steps",
                    marker_color=colors.get(steps, "#AB63FA"),
                    showlegend=(row_idx == 1),
                    legendgroup=str(steps),
                ),
                row=row_idx, col=1,
            )
        fig.update_yaxes(title_text="Var. Explained", row=row_idx, col=1)

    fig.update_layout(
        title="Parameter Importance: Fraction of Variance Explained",
        barmode="group",
        height=300 * len(Z_METRICS),
        width=1000,
    )
    save_figure(fig, save_path, "parameter_importance")
    return fig


# ============================================================
# 2. Step Count Comparison
# ============================================================

def get_comparable_prefix_len(data1, data2):
    """Return aligned prefix length for step-count comparison, or None if order differs."""
    key_fields = PARAMS + ["seed"]
    n1 = len(data1[key_fields[0]])
    n2 = len(data2[key_fields[0]])
    n = min(n1, n2)
    if n == 0:
        return 0

    for field in key_fields:
        if not np.array_equal(data1[field][:n], data2[field][:n]):
            return None
    return n



def step_count_comparison(data_by_steps, z_by_steps):
    """Compare z-metrics across step counts at each parameter config."""
    results = {}

    # For each z-metric, compute correlation between step counts
    step_counts = sorted(data_by_steps.keys())
    if len(step_counts) < 2:
        return {"note": "Need multiple step counts for comparison"}

    for z_name in Z_METRICS:
        results[z_name] = {}

        for i, s1 in enumerate(step_counts):
            for s2 in step_counts[i + 1:]:
                z1 = z_by_steps[s1][z_name]
                z2 = z_by_steps[s2][z_name]
                comparable_n = get_comparable_prefix_len(data_by_steps[s1], data_by_steps[s2])

                if comparable_n is None:
                    results[z_name][f"{s1}_vs_{s2}"] = "parameter ordering mismatch"
                    continue
                if comparable_n == 0:
                    results[z_name][f"{s1}_vs_{s2}"] = "no overlapping simulations"
                    continue

                z1 = z1[:comparable_n]
                z2 = z2[:comparable_n]

                corr = np.corrcoef(z1, z2)[0, 1] if np.std(z1) > 0 and np.std(z2) > 0 else 1.0
                mean_diff = float(np.mean(z2 - z1))
                comparison = {
                    "correlation": float(corr),
                    "mean_diff": mean_diff,
                    "n_compared": int(comparable_n),
                    "truncated_to_overlap": bool(comparable_n != len(data_by_steps[s1]["seed"]) or comparable_n != len(data_by_steps[s2]["seed"])),
                }
                # How many sims flip from no-revolution to revolution?
                if z_name == "revolution_prob":
                    flips = int(np.sum((z1 == 0) & (z2 == 1)))
                    total_no_rev = int(np.sum(z1 == 0))
                    comparison.update({
                        "new_revolutions": flips,
                        "pct_new_revolutions": float(flips / max(total_no_rev, 1)),
                    })

                results[z_name][f"{s1}_vs_{s2}"] = comparison

    return results


def plot_step_count_scatter(data_by_steps, z_by_steps, save_path):
    """Scatter plots comparing z-metrics between step counts."""
    step_counts = sorted(z_by_steps.keys())
    if len(step_counts) < 2:
        return None

    pairs = [(step_counts[0], step_counts[-1])]  # 100 vs 1000
    if len(step_counts) == 3:
        pairs.append((step_counts[0], step_counts[1]))  # 100 vs 500

    for s1, s2 in pairs:
        comparable_n = get_comparable_prefix_len(data_by_steps[s1], data_by_steps[s2])
        if comparable_n is None:
            print(f"  Skipping {s1} vs {s2} scatter: parameter ordering mismatch")
            continue
        if comparable_n == 0:
            print(f"  Skipping {s1} vs {s2} scatter: no overlapping simulations")
            continue
        if comparable_n != len(data_by_steps[s1]["seed"]) or comparable_n != len(data_by_steps[s2]["seed"]):
            print(
                f"  Plotting {s1} vs {s2} scatter on overlapping prefix only "
                f"({comparable_n:,} shared rows; total {len(data_by_steps[s1]['seed']):,} vs {len(data_by_steps[s2]['seed']):,})"
            )

        fig = make_subplots(
            rows=1, cols=len(Z_METRICS),
            subplot_titles=[Z_METRICS[z] for z in Z_METRICS],
        )
        for col_idx, z_name in enumerate(Z_METRICS, 1):
            z1 = z_by_steps[s1][z_name][:comparable_n]
            z2 = z_by_steps[s2][z_name][:comparable_n]
            # Subsample for plotting performance
            n = len(z1)
            if n > 50000:
                idx = np.random.default_rng(42).choice(n, 50000, replace=False)
                z1_plot, z2_plot = z1[idx], z2[idx]
            else:
                z1_plot, z2_plot = z1, z2

            fig.add_trace(
                go.Scattergl(
                    x=z1_plot, y=z2_plot,
                    mode="markers",
                    marker=dict(size=2, opacity=0.3),
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=1, col=col_idx,
            )
            fig.update_xaxes(title_text=f"{s1} steps", row=1, col=col_idx)
            fig.update_yaxes(title_text=f"{s2} steps", row=1, col=col_idx)

        fig.update_layout(
            title=f"Z-Metric Comparison: {s1} vs {s2} Steps",
            height=400, width=1200,
        )
        save_figure(fig, save_path, f"step_comparison_{s1}_vs_{s2}")


# ============================================================
# 3. Pairwise Manifold Surfaces (21 pairs)
# ============================================================

def compute_pairwise_manifold(data, z_metrics, z_name):
    """
    For each pair of parameters, compute the mean z-metric on a 2D grid
    (averaging over all other parameters and seeds).
    Returns dict of (param_a, param_b) -> (x_vals, y_vals, z_grid).
    """
    results = {}
    z_vals = z_metrics[z_name]

    for pa, pb in combinations(PARAMS, 2):
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

        results[(pa, pb)] = (a_unique, b_unique, z_grid)

    return results


def plot_manifold_grid(manifold_data, z_name, z_label, save_path, step_label=""):
    """Plot all 21 pairwise manifold surfaces as a grid of heatmaps."""
    pairs = list(combinations(PARAMS, 2))
    n_pairs = len(pairs)
    n_cols = 4
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[
            f"{PARAM_LABELS[pa]} vs {PARAM_LABELS[pb]}"
            for pa, pb in pairs
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
    )

    for idx, (pa, pb) in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        a_unique, b_unique, z_grid = manifold_data[(pa, pb)]

        fig.add_trace(
            go.Heatmap(
                z=z_grid.T,
                x=np.round(a_unique, 3),
                y=np.round(b_unique, 3),
                colorscale="Viridis",
                zmin=0, zmax=1,
                showscale=(idx == 0),
                colorbar=dict(title=z_label) if idx == 0 else None,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=PARAM_LABELS[pa], row=row, col=col)
        fig.update_yaxes(title_text=PARAM_LABELS[pb], row=row, col=col)

    suffix = f" ({step_label})" if step_label else ""
    fig.update_layout(
        title=f"Pairwise Manifold Surfaces: {z_label}{suffix}",
        height=350 * n_rows,
        width=1200,
    )
    slug = f"manifold_{z_name}_{step_label}" if step_label else f"manifold_{z_name}"
    save_figure(fig, save_path, slug)
    return fig


# ============================================================
# 4. Surprising Findings Detection
# ============================================================

def detect_surprising_findings(importance_by_step):
    """Flag cases where new parameters show non-trivial importance."""
    new_params = ["citizen_density", "max_jail", "vision"]
    findings = []

    for steps, importance in importance_by_step.items():
        for z_name, param_imp in importance.items():
            for p in new_params:
                imp = param_imp.get(p, 0.0)
                if imp > 0.01:  # >1% variance explained
                    findings.append({
                        "parameter": p,
                        "z_metric": z_name,
                        "steps": steps,
                        "importance": imp,
                        "note": f"{PARAM_LABELS[p]} explains {imp:.1%} of {Z_METRICS[z_name]} variance at {steps} steps",
                    })

    # Sort by importance descending
    findings.sort(key=lambda x: x["importance"], reverse=True)
    return findings


# ============================================================
# 5. Temporal Dynamics / Trajectory Archetypes
# ============================================================

def classify_trajectories(conn, num_steps):
    """
    Classify simulation trajectories into behavioral archetypes.
    Uses summary metrics from simulations table (fast) rather than
    reading all step data (would be too slow for 8M sims).
    """
    query = f"""
        SELECT sim_id, n_citizens, max_active, revolution_step,
               n_cascades, cascade_periodic, sum_active
        FROM simulations
        WHERE num_steps = {num_steps}
    """
    rows = list(conn.execute(query))
    data = np.array(rows)

    n_cit = data[:, 1].astype(float)
    n_cit = np.where(n_cit > 0, n_cit, 1.0)
    max_active_pct = data[:, 2] / n_cit
    rev_step = data[:, 3]
    n_cascades = data[:, 4]
    periodic = data[:, 5]

    n = len(data)
    archetypes = np.full(n, "unknown", dtype=object)

    # Fast revolution: revolution in first 20% of steps
    fast_rev = (rev_step >= 0) & (rev_step < num_steps * 0.2)
    archetypes[fast_rev] = "fast_revolution"

    # Slow burn: revolution in last 50% of steps
    slow_burn = (rev_step >= num_steps * 0.5) & (rev_step >= 0)
    archetypes[slow_burn] = "slow_burn"

    # Mid revolution: between fast and slow
    mid_rev = (rev_step >= num_steps * 0.2) & (rev_step < num_steps * 0.5) & (rev_step >= 0)
    archetypes[mid_rev] = "mid_revolution"

    # Periodic/oscillating: multiple cascades but no revolution
    oscillating = (periodic == 1) & (rev_step < 0)
    archetypes[oscillating] = "oscillating"

    # Abortive spike: cascades but no revolution and not periodic
    abortive = (n_cascades >= 1) & (rev_step < 0) & (periodic == 0)
    archetypes[abortive] = "abortive_spike"

    # Stable suppression: no cascades, no revolution, low active
    suppressed = (n_cascades == 0) & (rev_step < 0) & (max_active_pct < 0.1)
    archetypes[suppressed] = "stable_suppression"

    # Simmering: no cascades, no revolution, but moderate active
    simmering = (n_cascades == 0) & (rev_step < 0) & (max_active_pct >= 0.1)
    archetypes[simmering] = "simmering"

    # Count archetypes
    unique, counts = np.unique(archetypes, return_counts=True)
    archetype_counts = dict(zip(unique, counts.astype(int).tolist()))

    return archetypes, archetype_counts, data


def plot_archetype_distribution(archetype_counts_by_step, save_path):
    """Bar chart of archetype distribution across step counts."""
    all_types = set()
    for counts in archetype_counts_by_step.values():
        all_types.update(counts.keys())
    all_types = sorted(all_types)

    colors = {
        "fast_revolution": "#d62728",
        "mid_revolution": "#ff7f0e",
        "slow_burn": "#e377c2",
        "oscillating": "#2ca02c",
        "abortive_spike": "#9467bd",
        "stable_suppression": "#1f77b4",
        "simmering": "#8c564b",
        "unknown": "#7f7f7f",
    }

    fig = go.Figure()
    for steps in sorted(archetype_counts_by_step.keys()):
        counts = archetype_counts_by_step[steps]
        total = sum(counts.values())
        fig.add_trace(go.Bar(
            x=[t.replace("_", " ").title() for t in all_types],
            y=[counts.get(t, 0) / total for t in all_types],
            name=f"{steps} steps",
        ))

    fig.update_layout(
        title="Trajectory Archetype Distribution by Step Count",
        xaxis_title="Archetype",
        yaxis_title="Fraction of Simulations",
        barmode="group",
        height=500, width=900,
    )
    save_figure(fig, save_path, "archetype_distribution")
    return fig


def plot_archetype_parameter_map(data_full, archetypes, param_a, param_b, save_path, step_label):
    """
    For each archetype, show where in parameter space it occurs
    as a heatmap (fraction of sims with that archetype at each param combo).
    """
    a_vals = data_full[:, col_idx(param_a)]
    b_vals = data_full[:, col_idx(param_b)]
    a_unique = np.unique(a_vals)
    b_unique = np.unique(b_vals)

    arch_types = ["fast_revolution", "slow_burn", "oscillating",
                  "abortive_spike", "stable_suppression"]

    fig = make_subplots(
        rows=1, cols=len(arch_types),
        subplot_titles=[t.replace("_", " ").title() for t in arch_types],
    )

    for col, arch in enumerate(arch_types, 1):
        z_grid = np.full((len(a_unique), len(b_unique)), 0.0)
        for i, av in enumerate(a_unique):
            for j, bv in enumerate(b_unique):
                mask = (a_vals == av) & (b_vals == bv)
                if mask.sum() > 0:
                    z_grid[i, j] = (archetypes[mask] == arch).mean()

        fig.add_trace(
            go.Heatmap(
                z=z_grid.T, x=np.round(a_unique, 3), y=np.round(b_unique, 3),
                colorscale="Viridis", zmin=0, zmax=1,
                showscale=(col == len(arch_types)),
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text=PARAM_LABELS[param_a], row=1, col=col)
        fig.update_yaxes(title_text=PARAM_LABELS[param_b], row=1, col=col)

    fig.update_layout(
        title=f"Archetype Map: {PARAM_LABELS[param_a]} vs {PARAM_LABELS[param_b]} ({step_label})",
        height=400, width=1400,
    )
    save_figure(fig, save_path, f"archetype_map_{param_a}_{param_b}_{step_label}")


def col_idx(param):
    """Get column index for parameter in simulation data loaded via load_simulation_data."""
    return PARAMS.index(param)


def sample_time_series(conn, sim_ids):
    """Load step-level time series for selected simulations."""
    if not sim_ids:
        return {}
    placeholders = ",".join(str(int(s)) for s in sim_ids[:100])
    query = f"""
        SELECT sim_id, step, active_count, jail_count, revolution
        FROM model_steps
        WHERE sim_id IN ({placeholders})
        ORDER BY sim_id, step
    """
    rows = list(conn.execute(query))
    series = {}
    for sim_id, step, active, jail, rev in rows:
        if sim_id not in series:
            series[sim_id] = {"steps": [], "active": [], "jail": [], "revolution": []}
        series[sim_id]["steps"].append(step)
        series[sim_id]["active"].append(active)
        series[sim_id]["jail"].append(jail)
        series[sim_id]["revolution"].append(rev)
    return series


def plot_archetype_exemplars(conn, archetypes_data, archetypes, save_path, step_label):
    """Plot example time series for each archetype."""
    arch_types = ["fast_revolution", "slow_burn", "oscillating",
                  "abortive_spike", "stable_suppression", "simmering"]
    present = [a for a in arch_types if a in set(archetypes)]
    if not present:
        return

    # Pick 3 exemplars per archetype
    exemplar_ids = {}
    for arch in present:
        mask = archetypes == arch
        candidates = archetypes_data[mask, 0].astype(int)
        rng = np.random.default_rng(42)
        n_pick = min(3, len(candidates))
        exemplar_ids[arch] = rng.choice(candidates, n_pick, replace=False).tolist()

    all_ids = [sid for ids in exemplar_ids.values() for sid in ids]
    series = sample_time_series(conn, all_ids)

    n_cols = len(present)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[t.replace("_", " ").title() for t in present],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for col_i, arch in enumerate(present, 1):
        for j, sid in enumerate(exemplar_ids[arch]):
            if sid in series:
                s = series[sid]
                fig.add_trace(
                    go.Scatter(
                        x=s["steps"], y=s["active"],
                        mode="lines",
                        line=dict(color=colors[j % 3], width=1),
                        name=f"sim {sid}",
                        showlegend=False,
                    ),
                    row=1, col=col_i,
                )
        fig.update_xaxes(title_text="Step", row=1, col=col_i)
        fig.update_yaxes(title_text="Active Count", row=1, col=col_i)

    fig.update_layout(
        title=f"Archetype Exemplar Time Series ({step_label})",
        height=400, width=300 * n_cols,
    )
    save_figure(fig, save_path, f"archetype_exemplars_{step_label}")


# ============================================================
# Utility
# ============================================================

def save_figure(fig, base_path, name):
    """Save Plotly figure as JSON (for reports-web) and static HTML."""
    base_path.mkdir(parents=True, exist_ok=True)
    json_path = base_path / f"{name}.json"
    html_path = base_path / f"{name}.html"

    fig_dict = fig.to_dict()
    # Convert numpy types for JSON serialization
    with open(json_path, "w") as f:
        json.dump(fig_dict, f, cls=NumpyEncoder, separators=(",", ":"))
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {json_path.name}, {html_path.name}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_conn()

    # Check what step counts are available
    step_counts = [r[0] for r in conn.execute(
        "SELECT DISTINCT num_steps FROM simulations ORDER BY num_steps"
    )]
    print(f"Available step counts: {step_counts}")
    total_sims = list(conn.execute("SELECT COUNT(*) FROM simulations"))[0][0]
    print(f"Total simulations: {total_sims:,}")

    # ---- Load data per step count ----
    print("\n=== Loading simulation data ===")
    data_by_steps = {}
    z_by_steps = {}
    for steps in step_counts:
        print(f"  Loading {steps}-step data...")
        data = load_simulation_data(conn, steps)
        if data is not None:
            data_by_steps[steps] = data
            z_by_steps[steps] = compute_z_metrics(data)
            n = len(data["pp_mean"])
            print(f"    {n:,} simulations loaded")

    # ---- 1. Parameter Importance ----
    print("\n=== 1. Parameter Importance (Variance Decomposition) ===")
    importance_by_step = {}
    for steps in step_counts:
        if steps in data_by_steps:
            imp = compute_parameter_importance(data_by_steps[steps], z_by_steps[steps])
            importance_by_step[steps] = imp
            print(f"\n  {steps} steps:")
            for z_name in Z_METRICS:
                ranked = sorted(imp[z_name].items(), key=lambda x: x[1], reverse=True)
                top3 = ", ".join(f"{PARAM_LABELS[p]}={v:.3f}" for p, v in ranked[:3])
                print(f"    {Z_METRICS[z_name]}: {top3}")

    plot_parameter_importance(importance_by_step, FIGURES_DIR)

    # Save importance data
    with open(OUT_DIR / "parameter_importance.json", "w") as f:
        json.dump(importance_by_step, f, indent=2, cls=NumpyEncoder)

    # ---- 2. Step Count Comparison ----
    if len(step_counts) >= 2:
        print("\n=== 2. Step Count Comparison ===")
        step_comp = step_count_comparison(data_by_steps, z_by_steps)
        for z_name, comparisons in step_comp.items():
            print(f"\n  {Z_METRICS[z_name]}:")
            for pair, stats in comparisons.items():
                if isinstance(stats, dict) and "correlation" in stats:
                    print(f"    {pair}: corr={stats['correlation']:.4f}, mean_diff={stats['mean_diff']:.4f}")
                    if "new_revolutions" in stats:
                        print(f"      new revolutions: {stats['new_revolutions']:,} ({stats['pct_new_revolutions']:.1%})")

        with open(OUT_DIR / "step_count_comparison.json", "w") as f:
            json.dump(step_comp, f, indent=2, cls=NumpyEncoder)

        plot_step_count_scatter(data_by_steps, z_by_steps, FIGURES_DIR)

    # ---- 3. Pairwise Manifold Surfaces ----
    print("\n=== 3. Pairwise Manifold Surfaces ===")
    # Use the longest step count for primary manifolds
    primary_steps = max(step_counts)
    for z_name in Z_METRICS:
        print(f"  Computing manifolds for {Z_METRICS[z_name]} ({primary_steps} steps)...")
        manifold = compute_pairwise_manifold(
            data_by_steps[primary_steps], z_by_steps[primary_steps], z_name
        )
        plot_manifold_grid(manifold, z_name, Z_METRICS[z_name], FIGURES_DIR, f"{primary_steps}steps")

    # Also do 100-step manifolds for comparison
    if 100 in data_by_steps and primary_steps != 100:
        for z_name in ["revolution_prob", "max_active_pct"]:
            print(f"  Computing 100-step manifolds for {Z_METRICS[z_name]}...")
            manifold = compute_pairwise_manifold(
                data_by_steps[100], z_by_steps[100], z_name
            )
            plot_manifold_grid(manifold, z_name, Z_METRICS[z_name], FIGURES_DIR, "100steps")

    # ---- 4. Surprising Findings ----
    print("\n=== 4. Surprising Findings (New Parameters) ===")
    findings = detect_surprising_findings(importance_by_step)
    if findings:
        for f in findings[:15]:
            print(f"  {f['note']}")
    else:
        print("  No new parameters explain >1% of variance in any z-metric.")

    with open(OUT_DIR / "surprising_findings.json", "w") as f:
        json.dump(findings, f, indent=2, cls=NumpyEncoder)

    # ---- 5. Temporal Dynamics / Trajectory Archetypes ----
    print("\n=== 5. Trajectory Archetypes ===")
    archetype_counts_by_step = {}
    for steps in step_counts:
        print(f"\n  Classifying {steps}-step trajectories...")
        archetypes, counts, raw_data = classify_trajectories(conn, steps)
        archetype_counts_by_step[steps] = counts
        total = sum(counts.values())
        for arch, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {arch}: {count:,} ({count/total:.1%})")

        # Plot exemplar time series for longest step count
        if steps == primary_steps:
            print(f"  Plotting exemplar time series...")
            plot_archetype_exemplars(conn, raw_data, archetypes, FIGURES_DIR, f"{steps}steps")

            # Archetype parameter maps for key pairs
            # Need to load full param data aligned with archetype array
            print(f"  Computing archetype parameter maps...")
            data = data_by_steps[steps]
            # Build array matching classify_trajectories output
            for pa, pb in [("sec_density", "threshold"), ("pp_mean", "epsilon"),
                           ("citizen_density", "sec_density"), ("vision", "threshold"),
                           ("max_jail", "sec_density")]:
                a_vals = data[pa]
                b_vals = data[pb]
                a_unique = np.unique(a_vals)
                b_unique = np.unique(b_vals)

                arch_types_plot = ["fast_revolution", "slow_burn", "oscillating",
                                   "abortive_spike", "stable_suppression"]
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
                            z=z_grid.T, x=np.round(a_unique, 3), y=np.round(b_unique, 3),
                            colorscale="Viridis", zmin=0, zmax=1,
                            showscale=(col_i == len(present)),
                        ),
                        row=1, col=col_i,
                    )
                    fig.update_xaxes(title_text=PARAM_LABELS[pa], row=1, col=col_i)
                    fig.update_yaxes(title_text=PARAM_LABELS[pb], row=1, col=col_i)

                fig.update_layout(
                    title=f"Archetype Map: {PARAM_LABELS[pa]} vs {PARAM_LABELS[pb]} ({steps} steps)",
                    height=400, width=280 * len(present),
                )
                save_figure(fig, FIGURES_DIR, f"archetype_map_{pa}_{pb}_{steps}steps")

    plot_archetype_distribution(archetype_counts_by_step, FIGURES_DIR)

    with open(OUT_DIR / "archetype_counts.json", "w") as f:
        json.dump({str(k): v for k, v in archetype_counts_by_step.items()}, f, indent=2, cls=NumpyEncoder)

    # ---- Summary ----
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Phase 1C Analysis Complete in {elapsed:.1f}s")
    print(f"Output directory: {OUT_DIR}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('*.html')))} HTML, {len(list(FIGURES_DIR.glob('*.json')))} JSON")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
