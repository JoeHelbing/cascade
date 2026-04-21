"""
Generate improved figures for the Cascade Methodology Explainer report.

Addresses annotation feedback on the 2026-04-02 report:
- Improved archetype exemplar layout (2x3 grid instead of 1x5 row)
- Annotated parameter importance example
- Annotated manifold example with reading guide
- Annotated Z-metric scatter example
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SRC_FIGURES = Path("analysis_1c_output/figures")
OUT_DIR = Path("explainer_output/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    json_path = OUT_DIR / f"{name}.json"
    html_path = OUT_DIR / f"{name}.html"
    fig_dict = fig.to_dict()
    with open(json_path, "w") as f:
        json.dump(fig_dict, f, cls=NumpyEncoder, separators=(",", ":"))
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {name}")


def improved_archetype_exemplars():
    """Rebuild archetype exemplars as 2x3 grid with proper sizing."""
    with open(SRC_FIGURES / "archetype_exemplars_1000steps.json") as f:
        orig = json.load(f)

    traces = orig["data"]
    # Original has 5 subplots (3 traces each) in 1 row
    # Rebuild as 2 rows x 3 cols with proper dimensions
    subplot_titles = [a["text"] for a in orig["layout"]["annotations"]]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=subplot_titles + [""],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    archetype_colors = {
        "Fast Revolution": ["#d62728", "#e74c3c", "#c0392b"],
        "Slow Burn": ["#e377c2", "#f1948a", "#d35400"],
        "Oscillating": ["#2ca02c", "#27ae60", "#1abc9c"],
        "Abortive Spike": ["#9467bd", "#8e44ad", "#6c3483"],
        "Stable Suppression": ["#1f77b4", "#2980b9", "#3498db"],
    }

    for subplot_idx, title in enumerate(subplot_titles):
        row = subplot_idx // 3 + 1
        col = subplot_idx % 3 + 1
        base_trace = subplot_idx * 3

        colors = archetype_colors.get(title, ["#1f77b4", "#ff7f0e", "#2ca02c"])

        for j in range(3):
            trace_idx = base_trace + j
            if trace_idx >= len(traces):
                break
            t = traces[trace_idx]
            fig.add_trace(
                go.Scatter(
                    x=t["x"], y=t["y"],
                    mode="lines",
                    line=dict(color=colors[j], width=1.5),
                    name=f"Example {j+1}",
                    showlegend=(subplot_idx == 0),
                    legendgroup=f"ex{j}",
                ),
                row=row, col=col,
            )
        fig.update_xaxes(title_text="Simulation Step", row=row, col=col)
        fig.update_yaxes(title_text="Active Citizens", row=row, col=col)

    fig.update_layout(
        title="Trajectory Archetype Exemplars (1000 Steps)<br>"
              "<sub>Each panel shows 3 randomly selected simulations of that behavioral type</sub>",
        height=700,
        width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    save_figure(fig, "archetype_exemplars_improved")


def annotated_parameter_importance():
    """Create a focused, annotated parameter importance figure for revolution probability only."""
    with open(SRC_FIGURES / "parameter_importance.json") as f:
        orig = json.load(f)

    # Extract just revolution probability data (first 3 traces)
    # The original figure has 5 z-metrics x 3 step counts = 15 traces
    # Revolution probability is the first row -> traces 0, 1, 2
    params = ["PP Mean", "Security Density", "Epsilon", "Threshold",
              "Citizen Density", "Max Jail", "Vision"]

    # Get data from traces for revolution prob (first subplot)
    step_colors = {"100 steps": "#636EFA", "500 steps": "#EF553B", "1000 steps": "#00CC96"}

    fig = go.Figure()

    for trace in orig["data"][:3]:  # First 3 traces = rev prob at 100, 500, 1000
        fig.add_trace(go.Bar(
            x=trace["x"],
            y=trace["y"],
            name=trace["name"],
            marker_color=step_colors.get(trace["name"], "#AB63FA"),
        ))

    # Add annotation arrows pointing to key findings
    fig.add_annotation(
        x="Security Density", y=0.41,
        text="Security Density explains<br>~40% of the total variance<br>in revolution probability",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        ax=120, ay=-80,
        font=dict(size=12, color="#d62728"),
        bordercolor="#d62728", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.add_annotation(
        x="Epsilon", y=0.005,
        text="Epsilon explains <0.1%<br>-- nearly irrelevant<br>at the macro level",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        ax=-80, ay=-100,
        font=dict(size=11, color="#7f7f7f"),
        bordercolor="#7f7f7f", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.add_annotation(
        x="Max Jail", y=0.003,
        text="Bars are nearly identical<br>across step counts --<br>result is stable",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        ax=80, ay=-90,
        font=dict(size=11, color="#2ca02c"),
        bordercolor="#2ca02c", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title="How to Read: Parameter Importance (Revolution Probability)<br>"
              "<sub>Bar height = fraction of total outcome variance explained by that parameter alone</sub>",
        xaxis_title="Model Parameter",
        yaxis_title="Fraction of Variance Explained",
        yaxis=dict(tickformat=".0%", range=[0, 0.5]),
        barmode="group",
        height=550,
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    save_figure(fig, "parameter_importance_annotated")


def annotated_manifold_example():
    """Create an annotated single manifold heatmap as a reading guide."""
    with open(SRC_FIGURES / "manifold_revolution_prob_1000steps.json") as f:
        orig = json.load(f)

    # Extract the sec_density vs threshold heatmap (the most important pair)
    # In the 21-pair grid, sec_density vs threshold is at a specific position
    # sec_density is index 1, threshold is index 3 in PARAMS
    # combinations order: (0,1),(0,2),(0,3),...,(1,2),(1,3),...
    # sec_density(1) vs threshold(3) -> need to find which trace index
    # Pairs in order: (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3, (0,5)=4, (0,6)=5,
    #                 (1,2)=6, (1,3)=7, ...
    # So sec_density vs threshold = index 7

    trace = orig["data"][7]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=trace["z"],
        x=trace["x"],
        y=trace["y"],
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(
            title="Revolution<br>Probability",
            tickformat=".0%",
        ),
    ))

    # Add reading guide annotations
    fig.add_annotation(
        x=0.01, y=1.5,
        text="Dark purple = low revolution<br>probability (strong suppression)",
        showarrow=True, arrowhead=2,
        ax=-120, ay=-40,
        font=dict(size=11, color="white"),
        bordercolor="white", borderwidth=1, borderpad=4,
        bgcolor="rgba(68,1,84,0.85)",
    )

    fig.add_annotation(
        x=0.09, y=5.5,
        text="Bright yellow = high revolution<br>probability (system tips over)",
        showarrow=True, arrowhead=2,
        ax=90, ay=40,
        font=dict(size=11, color="black"),
        bordercolor="black", borderwidth=1, borderpad=4,
        bgcolor="rgba(253,231,37,0.85)",
    )

    fig.add_annotation(
        x=0.04, y=3.0,
        text="Sharp color boundary =<br>phase transition<br>(small parameter change,<br>big outcome change)",
        showarrow=True, arrowhead=2,
        ax=130, ay=0,
        font=dict(size=12, color="#d62728"),
        bordercolor="#d62728", borderwidth=2, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title="How to Read: Pairwise Manifold Heatmap<br>"
              "<sub>Each cell averages over all other parameters and seeds -- shows the 'main effect' of this pair</sub>",
        xaxis_title="Security Density (fraction of population that are cops)",
        yaxis_title="Threshold (how much grievance needed to rebel)",
        height=550,
        width=700,
    )
    save_figure(fig, "manifold_reading_guide")


def annotated_zmetric_scatter():
    """Create an annotated Z-metric scatter plot reading guide."""
    with open(SRC_FIGURES / "step_comparison_100_vs_1000.json") as f:
        orig = json.load(f)

    # Extract just the revolution probability scatter (first trace pair)
    scatter_trace = orig["data"][0]
    line_trace = orig["data"][1]

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=scatter_trace["x"],
        y=scatter_trace["y"],
        mode="markers",
        marker=dict(size=3, opacity=0.3, color="#636EFA"),
        name="Simulations",
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Perfect agreement line",
    ))

    # Annotations
    fig.add_annotation(
        x=0.5, y=0.5,
        text="Points ON the red line = identical<br>result at both step counts",
        showarrow=True, arrowhead=2,
        ax=-140, ay=-50,
        font=dict(size=12, color="#d62728"),
        bordercolor="#d62728", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.add_annotation(
        x=0.0, y=1.0,
        text="Points ABOVE the line = higher<br>value at 1000 steps than 100",
        showarrow=True, arrowhead=2,
        ax=120, ay=30,
        font=dict(size=11),
        bordercolor="#636EFA", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.add_annotation(
        x=0.85, y=0.85,
        text="Dense cloud hugging the line =<br>strong correlation, step count<br>doesn't change the answer much",
        showarrow=True, arrowhead=2,
        ax=-120, ay=60,
        font=dict(size=11, color="#2ca02c"),
        bordercolor="#2ca02c", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title="How to Read: Step Count Comparison Scatter<br>"
              "<sub>Each dot is one simulation -- its revolution probability at 100 steps (x) vs 1000 steps (y)</sub>",
        xaxis_title="Revolution Probability at 100 Steps",
        yaxis_title="Revolution Probability at 1000 Steps",
        height=550,
        width=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    save_figure(fig, "zmetric_scatter_reading_guide")


if __name__ == "__main__":
    print("Generating explainer figures...")
    print("\n1. Improved archetype exemplars (2x3 grid)")
    improved_archetype_exemplars()
    print("\n2. Annotated parameter importance")
    annotated_parameter_importance()
    print("\n3. Annotated manifold reading guide")
    annotated_manifold_example()
    print("\n4. Annotated Z-metric scatter reading guide")
    annotated_zmetric_scatter()
    print(f"\nDone. Output in {OUT_DIR}")
