"""Regenerate cascade manifold figures using newly computed metrics from cascade_metrics.db."""

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import apsw
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
MANIFOLD_DB = "manifold_results/manifold.db"
METRICS_DB = "manifold_results/cascade_metrics.db"
FIGURES_DIR = Path(
    "/home/joehe/git/reports-web/data/cascade/2026-04-05-cascade-comprehensive-analysis/figures"
)
PAPER_FIGURES_DIR = Path(
    "/home/joehe/git/reports-web/data/cascade/2026-04-08-cascade-research-paper/figures"
)

# Dark theme colors
BG_COLOR = "#1F2430"
TEXT_COLOR = "#CCCAC2"
ACCENT_COLOR = "#FFCC66"

# Viridis colorscale
VIRIDIS = [
    [0.0, "#440154"],
    [0.1111111111111111, "#482878"],
    [0.2222222222222222, "#3e4989"],
    [0.3333333333333333, "#31688e"],
    [0.4444444444444444, "#26828e"],
    [0.5555555555555556, "#1f9e89"],
    [0.6666666666666666, "#35b779"],
    [0.7777777777777778, "#6ece58"],
    [0.8888888888888888, "#b5de2b"],
    [1.0, "#fde725"],
]

PARAM_NAMES = {
    "pp_mean": "PP Mean",
    "sec_density": "Security Density",
    "epsilon": "Epsilon",
    "threshold": "Threshold",
    "vision": "Vision",
}

METRIC_NAMES = {
    "revolution_prob": "Revolution Probability",
    "max_active_pct": "Max Active %",
    "mean_active_pct": "Mean Active %",
    "cascade_rate": "Cascade Rate",
    "oscillation_power": "Oscillation Power",
}

METRIC_RANGES = {
    "revolution_prob": (0, 1),
    "max_active_pct": (0, 1),
    "mean_active_pct": (0, 1),
    "cascade_rate": (0, 0.35),
    "oscillation_power": (0, 1),
}

PARAM_PAIRS = [
    ("sec_density", "threshold"),
    ("vision", "threshold"),
    ("vision", "sec_density"),
    ("pp_mean", "sec_density"),
    ("pp_mean", "threshold"),
]


def sanitize_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    return obj


def save_json(path, data):
    """Save Plotly figure data as JSON with NaN handling."""
    clean = sanitize_json(data)
    with open(path, "w") as f:
        json.dump(clean, f, separators=(",", ":"))
    print(f"  Saved: {path.name} ({path.stat().st_size / 1024:.1f} KB)")


def load_data():
    """Load simulation data and metrics, merge on sim_id."""
    print("Loading manifold.db (500-step sims)...")
    conn = apsw.Connection(MANIFOLD_DB, flags=apsw.SQLITE_OPEN_READONLY)
    cur = conn.cursor()

    rows = cur.execute(
        """SELECT sim_id, pp_mean, sec_density, epsilon, threshold, vision,
                  n_citizens, max_active, revolution_step, n_cascades,
                  cascade_periodic, sum_active, num_steps
           FROM simulations
           WHERE num_steps=500 AND citizen_density=0.7 AND max_jail=100"""
    ).fetchall()
    print(f"  Loaded {len(rows)} simulations")

    # Build dict keyed by sim_id
    sims = {}
    for r in rows:
        sid = r[0]
        n_cit = r[6]
        rev_step = r[8]
        sims[sid] = {
            "pp_mean": r[1],
            "sec_density": r[2],
            "epsilon": r[3],
            "threshold": r[4],
            "vision": r[5],
            "n_citizens": n_cit,
            "revolution_prob": 1.0 if rev_step is not None and rev_step >= 0 else 0.0,
            "max_active_pct": r[7] / n_cit if n_cit > 0 else 0.0,
            "mean_active_pct": r[11] / (n_cit * r[12]) if n_cit > 0 and r[12] > 0 else 0.0,
        }
    conn.close()

    print("Loading cascade_metrics.db...")
    mconn = sqlite3.connect(METRICS_DB)
    mcur = mconn.cursor()
    mcur.execute("SELECT sim_id, max_speed, oscillation_power FROM metrics")
    metrics_rows = mcur.fetchall()
    mconn.close()
    print(f"  Loaded {len(metrics_rows)} metric rows")

    # Merge
    merged = 0
    for sid, max_speed, osc_power in metrics_rows:
        if sid in sims:
            sims[sid]["cascade_rate"] = max_speed if max_speed is not None else 0.0
            sims[sid]["oscillation_power"] = osc_power if osc_power is not None else 0.0
            merged += 1

    # Filter to only sims that have metrics
    result = {sid: s for sid, s in sims.items() if "cascade_rate" in s}
    print(f"  Merged: {merged}, final dataset: {len(result)} sims")
    return result


def compute_eta_squared(sims, param, metric):
    """Compute eta-squared (variance decomposition) for a parameter-metric pair."""
    groups = defaultdict(list)
    all_vals = []
    for s in sims.values():
        v = s[metric]
        if v is not None and not math.isnan(v):
            groups[s[param]].append(v)
            all_vals.append(v)

    if not all_vals:
        return 0.0

    grand_mean = np.mean(all_vals)
    total_var = np.var(all_vals)
    if total_var == 0:
        return 0.0

    group_means = [np.mean(vals) for vals in groups.values()]
    group_sizes = [len(vals) for vals in groups.values()]

    ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(group_sizes, group_means))
    ss_total = total_var * len(all_vals)

    return ss_between / ss_total if ss_total > 0 else 0.0


def generate_parameter_importance(sims):
    """Generate grouped bar chart of parameter importance (eta-squared)."""
    print("\n=== 1. Parameter Importance Bar Chart ===")

    params = ["pp_mean", "sec_density", "epsilon", "threshold", "vision"]
    metrics = [
        "revolution_prob",
        "max_active_pct",
        "mean_active_pct",
        "cascade_rate",
        "oscillation_power",
    ]

    # Compute eta-squared for each param x metric
    eta_sq = {}
    for p in params:
        for m in metrics:
            eta_sq[(p, m)] = compute_eta_squared(sims, p, m)
            print(f"  eta^2({PARAM_NAMES[p]}, {METRIC_NAMES[m]}): {eta_sq[(p, m)]:.4f}")

    # Build grouped bar chart
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    traces = []
    for i, m in enumerate(metrics):
        traces.append(
            {
                "type": "bar",
                "name": METRIC_NAMES[m],
                "x": [PARAM_NAMES[p] for p in params],
                "y": [eta_sq[(p, m)] for p in params],
                "marker": {"color": bar_colors[i], "opacity": 0.85},
            }
        )

    layout = {
        "barmode": "group",
        "title": {
            "text": "Parameter Importance (Main-Effect Eta-Squared)",
            "font": {"color": TEXT_COLOR, "size": 18},
        },
        "xaxis": {
            "title": {"text": "Parameter", "font": {"color": TEXT_COLOR}},
            "tickfont": {"color": TEXT_COLOR},
        },
        "yaxis": {
            "title": {"text": "Eta-Squared (Variance Explained)", "font": {"color": TEXT_COLOR}},
            "tickfont": {"color": TEXT_COLOR},
            "range": [0, 1],
        },
        "paper_bgcolor": BG_COLOR,
        "plot_bgcolor": BG_COLOR,
        "font": {"color": TEXT_COLOR},
        "legend": {"font": {"color": TEXT_COLOR}},
        "width": 900,
        "height": 550,
    }

    save_json(FIGURES_DIR / "parameter_importance_1d.json", {"data": traces, "layout": layout})


def compute_surface_data(sims, param_a, param_b, metric):
    """Compute a 2D grid of mean metric values for a parameter pair."""
    # Get unique sorted values
    vals_a = sorted(set(s[param_a] for s in sims.values()))
    vals_b = sorted(set(s[param_b] for s in sims.values()))

    # Build lookup
    grid = defaultdict(list)
    for s in sims.values():
        key = (s[param_a], s[param_b])
        v = s[metric]
        if v is not None and not math.isnan(v):
            grid[key].append(v)

    # Build z matrix
    z = []
    for vb in vals_b:
        row = []
        for va in vals_a:
            vals = grid[(va, vb)]
            row.append(np.mean(vals) if vals else None)
        z.append(row)

    return vals_a, vals_b, z


def make_surface_trace(x, y, z, name, metric, visible=True):
    """Create a Plotly surface trace."""
    zmin, zmax = METRIC_RANGES.get(metric, (0, 1))
    return {
        "type": "surface",
        "x": x,
        "y": y,
        "z": z,
        "name": name,
        "visible": visible,
        "colorscale": VIRIDIS,
        "cmin": zmin,
        "cmax": zmax,
        "colorbar": {"title": {"text": name}, "tickfont": {"color": TEXT_COLOR}},
        "contours": {
            "z": {"show": True, "usecolormap": True, "highlightcolor": "white", "project": {"z": True}}
        },
        "lighting": {"ambient": 0.6, "diffuse": 0.5, "roughness": 0.5, "specular": 0.2},
    }


def generate_3d_surfaces(sims, metric, metric_label, prefix):
    """Generate 3D surface plots for a single metric across all parameter pairs."""
    print(f"\n=== 3D Surfaces: {metric_label} ===")

    for param_a, param_b in PARAM_PAIRS:
        x, y, z = compute_surface_data(sims, param_a, param_b, metric)
        trace = make_surface_trace(x, y, z, metric_label, metric, visible=True)

        zmin, zmax = METRIC_RANGES.get(metric, (0, 1))
        layout = {
            "title": {
                "text": f"{metric_label}: {PARAM_NAMES[param_a]} vs {PARAM_NAMES[param_b]}",
                "font": {"color": TEXT_COLOR},
            },
            "scene": {
                "xaxis": {"title": {"text": PARAM_NAMES[param_a]}},
                "yaxis": {"title": {"text": PARAM_NAMES[param_b]}},
                "zaxis": {"title": {"text": metric_label}, "range": [zmin, zmax]},
                "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.0}, "up": {"x": 0, "y": 0, "z": 1}},
                "aspectratio": {"x": 1, "y": 1, "z": 0.7},
                "aspectmode": "manual",
            },
            "paper_bgcolor": BG_COLOR,
            "plot_bgcolor": BG_COLOR,
            "font": {"color": TEXT_COLOR},
            "width": 900,
            "height": 750,
        }

        fname = f"{prefix}_{param_a}_vs_{param_b}.json"
        save_json(FIGURES_DIR / fname, {"data": [trace], "layout": layout})


def generate_multi_dropdown_surfaces(sims):
    """Update the 5 multi-metric dropdown figures with new metrics."""
    print("\n=== 4. Multi-Metric Dropdown Surfaces ===")

    metrics_order = [
        ("revolution_prob", "Revolution Probability"),
        ("max_active_pct", "Max Active %"),
        ("mean_active_pct", "Mean Active %"),
        ("cascade_rate", "Cascade Rate"),
        ("oscillation_power", "Oscillation Power"),
    ]

    for param_a, param_b in PARAM_PAIRS:
        traces = []
        buttons = []

        for i, (metric, label) in enumerate(metrics_order):
            x, y, z = compute_surface_data(sims, param_a, param_b, metric)
            visible = i == 0
            trace = make_surface_trace(x, y, z, label, metric, visible=visible)
            traces.append(trace)

            # Button: show only this trace
            vis = [j == i for j in range(len(metrics_order))]
            zmin, zmax = METRIC_RANGES.get(metric, (0, 1))
            buttons.append(
                {
                    "label": label,
                    "method": "update",
                    "args": [
                        {"visible": vis},
                        {"scene.zaxis.title": label, "scene.zaxis.range": [zmin, zmax]},
                    ],
                    "font": {"color": "#FFFFFF"},
                }
            )

        first_metric = metrics_order[0][0]
        first_zmin, first_zmax = METRIC_RANGES[first_metric]

        layout = {
            "title": {
                "text": (
                    f"3D Manifold: {PARAM_NAMES[param_a]} vs {PARAM_NAMES[param_b]}"
                    "<br><sub>Use dropdown to switch metrics. Drag to rotate, scroll to zoom.</sub>"
                ),
                "x": 0.5,
            },
            "scene": {
                "xaxis": {"title": {"text": PARAM_NAMES[param_a]}},
                "yaxis": {"title": {"text": PARAM_NAMES[param_b]}},
                "zaxis": {
                    "title": {"text": metrics_order[0][1]},
                    "range": [first_zmin, first_zmax],
                },
                "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.0}, "up": {"x": 0, "y": 0, "z": 1}},
                "aspectratio": {"x": 1, "y": 1, "z": 0.7},
                "aspectmode": "manual",
            },
            "updatemenus": [
                {
                    "type": "dropdown",
                    "direction": "down",
                    "x": 0.02,
                    "xanchor": "left",
                    "y": 0.98,
                    "yanchor": "top",
                    "bgcolor": "#1a1f2e",
                    "bordercolor": ACCENT_COLOR,
                    "borderwidth": 2,
                    "font": {"color": "#FFFFFF", "size": 14, "family": "Inter, sans-serif"},
                    "pad": {"r": 10, "t": 10},
                    "buttons": buttons,
                }
            ],
            "margin": {"t": 100},
            "width": 900,
            "height": 750,
        }

        fname = f"surface3d_multi_{param_a}_vs_{param_b}.json"
        save_json(FIGURES_DIR / fname, {"data": traces, "layout": layout})


def make_heatmap_grid(sims, metric, metric_label, zmin, zmax, title_suffix="", font_size=16, width=1200, height=700):
    """Create a 10-pairwise heatmap subplot grid for a metric."""
    params = ["pp_mean", "sec_density", "epsilon", "threshold", "vision"]

    # Generate all 10 pairs
    all_pairs = []
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            all_pairs.append((params[i], params[j]))

    # Layout: 2 rows x 5 cols
    fig = make_subplots(
        rows=2,
        cols=5,
        subplot_titles=[f"{PARAM_NAMES[a]} vs {PARAM_NAMES[b]}" for a, b in all_pairs],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    for idx, (pa, pb) in enumerate(all_pairs):
        row = idx // 5 + 1
        col = idx % 5 + 1

        x, y, z = compute_surface_data(sims, pa, pb, metric)

        trace = go.Heatmap(
            x=x,
            y=y,
            z=z,
            colorscale=VIRIDIS,
            zmin=zmin,
            zmax=zmax,
            showscale=idx == 0,
            colorbar={"title": {"text": metric_label}} if idx == 0 else None,
        )
        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title={
            "text": f"High-Resolution Manifolds: {metric_label}{title_suffix}"
        },
        width=width,
        height=height,
    )

    # Update annotation font sizes
    for ann in fig.layout.annotations:
        ann.font.size = font_size

    return fig


def generate_heatmap_grids(sims):
    """Generate pairwise heatmap grids for cascade_rate and oscillation_power."""
    print("\n=== 5. Pairwise Heatmap Grids ===")

    metrics = [
        ("cascade_rate", "Cascade Rate", 0, 0.35),
        ("oscillation_power", "Oscillation Power", 0, 1),
    ]

    for metric, label, zmin, zmax in metrics:
        # Standard version (comprehensive analysis)
        fig = make_heatmap_grid(sims, metric, label, zmin, zmax, " (Phase 1D, 500 steps)")
        fig_dict = json.loads(fig.to_json())
        save_json(FIGURES_DIR / f"hires_all_{metric}.json", fig_dict)

        # Fixed version (research paper) - better font sizes and spacing
        fig2 = make_heatmap_grid(
            sims, metric, label, zmin, zmax, " (Phase 1D, 500 steps)",
            font_size=10, width=1400, height=800,
        )
        fig2_dict = json.loads(fig2.to_json())
        save_json(PAPER_FIGURES_DIR / f"hires_{metric}_fixed.json", fig2_dict)


def main():
    sims = load_data()

    # 1. Parameter importance
    generate_parameter_importance(sims)

    # 2. 3D surfaces for cascade_rate
    generate_3d_surfaces(sims, "cascade_rate", "Cascade Rate", "surface3d_cascade_rate")

    # 3. 3D surfaces for oscillation_power
    generate_3d_surfaces(sims, "oscillation_power", "Oscillation Power", "surface3d_oscillation_power")

    # 4. Multi-metric dropdown surfaces
    generate_multi_dropdown_surfaces(sims)

    # 5. Heatmap grids
    generate_heatmap_grids(sims)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
