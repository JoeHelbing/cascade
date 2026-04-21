"""
Generate interactive 3D surface manifolds from Phase 1D data.

Creates rotatable, zoomable Plotly Surface plots where:
- X axis: parameter A
- Y axis: parameter B
- Z axis: z-metric value (revolution prob, cascade rate, etc.)

Usage:
    cd mojo_cascade
    pixi run python generate_3d_manifolds.py
"""

import json
import time
from pathlib import Path

import apsw
import numpy as np
import plotly.graph_objects as go

DB_PATH = "manifold_results/manifold.db"
OUT_DIR = Path("manifold_3d_output/figures")

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
    "cascade_rate": "Cascade Rate",
    "periodic_rate": "Periodic Rate",
}

# Key parameter pairs to generate 3D surfaces for
KEY_PAIRS = [
    ("sec_density", "threshold"),
    ("vision", "threshold"),
    ("vision", "sec_density"),
    ("pp_mean", "sec_density"),
    ("pp_mean", "threshold"),
]


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{name}.json"
    html_path = OUT_DIR / f"{name}.html"
    with open(json_path, "w") as f:
        json.dump(fig.to_dict(), f, cls=NumpyEncoder, separators=(",", ":"))
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {name}")


def load_1d_data(conn):
    """Load Phase 1D data from DB."""
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
        SELECT pp_mean, sec_density, epsilon, threshold, vision,
               n_citizens, max_active, revolution_step,
               n_cascades, cascade_periodic, sum_active, num_steps
        FROM simulations {where}
    """
    print("  Loading data from DB...")
    t0 = time.time()
    rows = list(conn.execute(query))
    data = np.array(rows)
    print(f"  Loaded {len(data):,} rows in {time.time() - t0:.1f}s")

    return {
        "pp_mean": data[:, 0],
        "sec_density": data[:, 1],
        "epsilon": data[:, 2],
        "threshold": data[:, 3],
        "vision": data[:, 4],
        "n_citizens": data[:, 5],
        "max_active": data[:, 6],
        "revolution_step": data[:, 7],
        "n_cascades": data[:, 8],
        "cascade_periodic": data[:, 9],
        "sum_active": data[:, 10],
        "num_steps": data[:, 11],
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


def compute_surface(data, z_metrics, z_name, param_a, param_b):
    """Compute a 2D grid of mean z-metric values for the parameter pair."""
    z_vals = z_metrics[z_name]
    a_vals = data[param_a]
    b_vals = data[param_b]
    a_unique = np.sort(np.unique(a_vals))
    b_unique = np.sort(np.unique(b_vals))

    z_grid = np.full((len(a_unique), len(b_unique)), np.nan)
    for i, av in enumerate(a_unique):
        for j, bv in enumerate(b_unique):
            mask = (a_vals == av) & (b_vals == bv)
            if mask.sum() > 0:
                z_grid[i, j] = z_vals[mask].mean()

    return a_unique, b_unique, z_grid


def make_3d_surface(a_unique, b_unique, z_grid, param_a, param_b, z_name, z_label):
    """Create an interactive 3D surface plot."""
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=np.round(a_unique, 4),
        y=np.round(b_unique, 4),
        z=z_grid.T,
        colorscale="Viridis",
        cmin=0, cmax=1,
        colorbar=dict(title=z_label, len=0.6),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
        ),
        lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2, roughness=0.5),
    ))

    fig.update_layout(
        title=dict(
            text=f"{z_label}<br><sub>{PARAM_LABELS[param_a]} vs {PARAM_LABELS[param_b]} (Phase 1D, 500 steps)</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title=PARAM_LABELS[param_a]),
            yaxis=dict(title=PARAM_LABELS[param_b]),
            zaxis=dict(title=z_label, range=[0, 1]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7),
        ),
        width=800,
        height=700,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def make_3d_multi_metric(a_unique, b_unique, surfaces_by_metric, param_a, param_b):
    """Create a 3D surface with dropdown to switch between z-metrics."""
    fig = go.Figure()

    metric_names = list(surfaces_by_metric.keys())
    for i, (z_name, z_grid) in enumerate(surfaces_by_metric.items()):
        fig.add_trace(go.Surface(
            x=np.round(a_unique, 4),
            y=np.round(b_unique, 4),
            z=z_grid.T,
            colorscale="Viridis",
            cmin=0, cmax=1,
            colorbar=dict(title=Z_METRICS[z_name], len=0.6),
            visible=(i == 0),
            name=Z_METRICS[z_name],
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            ),
            lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2, roughness=0.5),
        ))

    # Dropdown buttons to switch metrics
    buttons = []
    for i, z_name in enumerate(metric_names):
        visibility = [j == i for j in range(len(metric_names))]
        buttons.append(dict(
            label=Z_METRICS[z_name],
            method="update",
            args=[
                {"visible": visibility},
                {"scene.zaxis.title": Z_METRICS[z_name]},
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"3D Manifold: {PARAM_LABELS[param_a]} vs {PARAM_LABELS[param_b]}<br>"
                 f"<sub>Use dropdown to switch metrics. Drag to rotate, scroll to zoom.</sub>",
            x=0.5,
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            buttons=buttons,
            bgcolor="rgba(255,255,255,0.8)",
        )],
        scene=dict(
            xaxis=dict(title=PARAM_LABELS[param_a]),
            yaxis=dict(title=PARAM_LABELS[param_b]),
            zaxis=dict(title=Z_METRICS[metric_names[0]], range=[0, 1]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7),
        ),
        width=900,
        height=750,
        margin=dict(l=20, r=20, t=100, b=20),
    )

    return fig


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = apsw.Connection(DB_PATH, apsw.SQLITE_OPEN_READONLY)

    print("=== Generating 3D Surface Manifolds ===\n")
    data = load_1d_data(conn)
    z_metrics = compute_z_metrics(data)

    # Generate individual 3D surfaces for each pair x metric
    for param_a, param_b in KEY_PAIRS:
        print(f"\n{PARAM_LABELS[param_a]} vs {PARAM_LABELS[param_b]}:")

        surfaces = {}
        for z_name in Z_METRICS:
            a_unique, b_unique, z_grid = compute_surface(data, z_metrics, z_name, param_a, param_b)
            surfaces[z_name] = z_grid

            # Individual surface
            fig = make_3d_surface(a_unique, b_unique, z_grid, param_a, param_b, z_name, Z_METRICS[z_name])
            save_figure(fig, f"surface3d_{z_name}_{param_a}_vs_{param_b}")

        # Multi-metric surface with dropdown
        fig = make_3d_multi_metric(a_unique, b_unique, surfaces, param_a, param_b)
        save_figure(fig, f"surface3d_multi_{param_a}_vs_{param_b}")

    conn.close()

    elapsed = time.time() - t_start
    n_files = len(list(OUT_DIR.glob("*.html")))
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {OUT_DIR} ({n_files} HTML, {n_files} JSON)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
