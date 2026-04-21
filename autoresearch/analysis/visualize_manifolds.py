"""
Cascade ABM Manifold Visualization.

Reads manifold.db (SQLite) and generates 3D surface plots for all 6 pairwise
parameter combinations. Style matches Joe's Schelling model 3D visualization.

Usage:
    cd mojo_cascade
    pixi run python visualize_manifolds.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


DB_PATH = Path("manifold_results/manifold.db")
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

# The 4 swept parameters
PARAMS = {
    "pp_mean": {"col": "pp_mean", "label": "PP Mean", "fmt": ".2f"},
    "sec_density": {"col": "sec_density", "label": "Security Density", "fmt": ".3f"},
    "epsilon": {"col": "epsilon", "label": "Epsilon", "fmt": ".2f"},
    "threshold": {"col": "threshold", "label": "Threshold", "fmt": ".2f"},
}

PARAM_KEYS = list(PARAMS.keys())


def load_data(db_path: Path) -> dict:
    """Load simulation results grouped by parameter values."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all simulation summary data
    rows = conn.execute("""
        SELECT pp_mean, sec_density, epsilon, threshold,
               revolution_step, n_citizens
        FROM simulations
    """).fetchall()
    conn.close()

    data = {
        "pp_mean": np.array([r["pp_mean"] for r in rows]),
        "sec_density": np.array([r["sec_density"] for r in rows]),
        "epsilon": np.array([r["epsilon"] for r in rows]),
        "threshold": np.array([r["threshold"] for r in rows]),
        "revolution": np.array([1 if r["revolution_step"] >= 0 else 0 for r in rows]),
    }
    return data


def compute_pairwise_surface(
    data: dict, x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute revolution probability surface for a pair of parameters.

    Averages over the other 2 parameters and all seeds.
    Returns (x_vals, y_vals, z_grid) where z_grid[i,j] = P(revolution).
    """
    x_vals = np.unique(data[x_key])
    y_vals = np.unique(data[y_key])

    z_grid = np.zeros((len(y_vals), len(x_vals)))

    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            mask = (np.abs(data[x_key] - xv) < 1e-6) & (
                np.abs(data[y_key] - yv) < 1e-6
            )
            if mask.sum() > 0:
                z_grid[i, j] = data["revolution"][mask].mean()

    return x_vals, y_vals, z_grid


def make_surface_plot(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_grid: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str = "Revolution Probability",
    title: str = "",
) -> go.Figure:
    """Create a 3D surface plot matching the Schelling model style."""
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_vals,
                y=y_vals,
                z=z_grid,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text=z_label, font=dict(size=14)),
                    tickfont=dict(size=12),
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.5,
                    specular=0.1,
                    roughness=0.5,
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, project_z=True),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title=x_label, tickfont=dict(size=11)),
            yaxis=dict(title=y_label, tickfont=dict(size=11)),
            zaxis=dict(
                title=z_label,
                tickfont=dict(size=11),
                range=[0, 1],
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        width=900,
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def make_conditional_surface(
    data: dict,
    x_key: str,
    y_key: str,
    fix_key: str,
    fix_val: float,
    tol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface with one parameter fixed (not averaged), averaging over the 4th + seeds."""
    other_keys = [k for k in PARAM_KEYS if k not in (x_key, y_key, fix_key)]

    fix_mask = np.abs(data[fix_key] - fix_val) < tol
    sub = {k: data[k][fix_mask] for k in data}

    return compute_pairwise_surface(sub, x_key, y_key)


def main():
    print(f"Loading data from {DB_PATH}...")
    data = load_data(DB_PATH)
    n_sims = len(data["revolution"])
    rev_rate = data["revolution"].mean()
    print(f"  {n_sims:,} simulations, overall revolution rate: {rev_rate:.3f}")

    # --- 6 pairwise manifolds ---
    print("\nGenerating 6 pairwise manifold surfaces...")
    for i, x_key in enumerate(PARAM_KEYS):
        for y_key in PARAM_KEYS[i + 1 :]:
            x_info = PARAMS[x_key]
            y_info = PARAMS[y_key]

            x_vals, y_vals, z_grid = compute_pairwise_surface(data, x_key, y_key)

            title = f"Resistance Cascade - {x_info['label']} vs {y_info['label']}"
            fig = make_surface_plot(
                x_vals, y_vals, z_grid,
                x_label=x_info["label"],
                y_label=y_info["label"],
                title=title,
            )

            slug = f"manifold_{x_key}_vs_{y_key}"
            fig.write_html(str(CHART_DIR / f"{slug}.html"))
            fig.write_image(str(CHART_DIR / f"{slug}.png"), scale=2)
            print(f"  {slug} -> z range [{z_grid.min():.3f}, {z_grid.max():.3f}]")

    # --- Conditional manifolds: fix sec_density at key values ---
    print("\nGenerating conditional manifolds (sec_density fixed)...")
    sec_values = [0.0, 0.01, 0.02, 0.04]
    unique_sd = np.unique(data["sec_density"])

    for sd_target in sec_values:
        # Find closest grid point
        sd_val = unique_sd[np.argmin(np.abs(unique_sd - sd_target))]

        x_vals, y_vals, z_grid = make_conditional_surface(
            data, "pp_mean", "threshold", "sec_density", sd_val, tol=0.001
        )

        title = f"PP Mean vs Threshold (sec_density={sd_val:.3f})"
        fig = make_surface_plot(
            x_vals, y_vals, z_grid,
            x_label="PP Mean",
            y_label="Threshold",
            title=title,
        )

        slug = f"manifold_ppmean_vs_threshold_sd{sd_val:.3f}"
        fig.write_html(str(CHART_DIR / f"{slug}.html"))
        fig.write_image(str(CHART_DIR / f"{slug}.png"), scale=2)
        print(f"  {slug} -> z range [{z_grid.min():.3f}, {z_grid.max():.3f}]")

    # --- Conditional manifolds: fix threshold at key values ---
    print("\nGenerating conditional manifolds (threshold fixed)...")
    th_values = [2.0, 3.0, 4.0, 5.0]
    unique_th = np.unique(data["threshold"])

    for th_target in th_values:
        th_val = unique_th[np.argmin(np.abs(unique_th - th_target))]

        x_vals, y_vals, z_grid = make_conditional_surface(
            data, "sec_density", "epsilon", "threshold", th_val, tol=0.01
        )

        title = f"Security Density vs Epsilon (threshold={th_val:.2f})"
        fig = make_surface_plot(
            x_vals, y_vals, z_grid,
            x_label="Security Density",
            y_label="Epsilon",
            title=title,
        )

        slug = f"manifold_secdensity_vs_epsilon_th{th_val:.2f}"
        fig.write_html(str(CHART_DIR / f"{slug}.html"))
        fig.write_image(str(CHART_DIR / f"{slug}.png"), scale=2)
        print(f"  {slug} -> z range [{z_grid.min():.3f}, {z_grid.max():.3f}]")

    print(f"\nDone. Charts saved to {CHART_DIR}/")


if __name__ == "__main__":
    main()
