"""
Phase 1F: Agent-Level Microstructure Analysis for Cascade Research Paper.

Generates Plotly JSON figures for reports-web:
1. Per-agent activation scatterplots (thesis Figs 18-19 style) -- 4 sims + 2x2 grid
2. Epsilon coherence comparison (std dev of activation over time)
3. Epsilon speed-of-cascade comparison (active/jail counts over time)
4. Spatial grid snapshots during revolution (sim 74)
5. Arrest dynamics (arrest prob vs active fraction)
6. Epsilon macro vs micro (revolution rate + activation std dev)
7. Fixed pairwise heatmaps (from 2026-04-05 analysis)

Usage:
    cd mojo_cascade
    pixi run python analysis_1f.py
"""

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DB_PATH = Path("manifold_results/agent_data.db")
FIG_DIR = Path.home() / "git/reports-web/data/cascade/2026-04-08-cascade-research-paper/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Dark theme
THEME = dict(
    bg="#1F2430",
    text="#CCCAC2",
    accent="#FFCC66",
    grid="rgba(204,202,194,0.15)",
)

CONDITION_COLORS = {
    0: "rgba(100,149,237,0.12)",  # support - blue
    1: "rgba(255,80,80,0.12)",    # active - red
    2: "rgba(180,100,255,0.12)",  # oppose - purple
    3: "rgba(150,150,150,0.12)",  # jailed - gray
}
CONDITION_NAMES = {0: "Support", 1: "Active", 2: "Oppose", 3: "Jailed"}

# Sims to plot
SIM_SCATTER = {
    678: "eps=0.5, sd=0.02, th=2.0 (low-eps cascade)",
    708: "eps=1.5, sd=0.02, th=2.0 (high-eps cascade)",
    74:  "sd=0.02, th=1.5 (fast revolution)",
    226: "sd=0.04, th=3.0 (suppressed)",
}

# Standard layout defaults
def base_layout(**overrides):
    layout = dict(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"], size=12),
        xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    )
    layout.update(overrides)
    return layout


def safe_json_dumps(obj):
    """Serialize to JSON, replacing NaN/Inf with 0."""
    def default(x):
        if isinstance(x, float) and (x != x or abs(x) == float("inf")):
            return 0
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            v = float(x)
            if v != v or abs(v) == float("inf"):
                return 0
            return v
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(f"Not serializable: {type(x)}")
    return json.dumps(obj, default=default)


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.json"
    fig_dict = fig.to_dict()
    path.write_text(safe_json_dumps(fig_dict))
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {name}.json ({size_kb:.0f} KB)")
    return path


def get_conn():
    return sqlite3.connect(str(DB_PATH))


# -------------------------------------------------------------------
# 1. Per-agent activation scatterplots
# -------------------------------------------------------------------
def make_scatter_single(conn, sim_id, title_extra, subsample_step=3):
    """Create a single per-agent activation scatterplot for one sim."""
    t0 = time.time()
    cur = conn.cursor()

    # Get all agent data, subsampled by step
    cur.execute(
        """SELECT step, agent_id, activation_val, condition, arrest_prob
           FROM agent_steps
           WHERE sim_id = ? AND step % ? = 0
           ORDER BY step""",
        (sim_id, subsample_step),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"  WARNING: No data for sim {sim_id}")
        return None

    steps = np.array([r[0] for r in rows])
    agent_ids = np.array([r[1] for r in rows])
    act_vals = np.array([r[2] for r in rows])
    conditions = np.array([r[3] for r in rows])
    arrest_probs = np.array([r[4] for r in rows])

    print(f"  sim {sim_id}: {len(rows)} points loaded in {time.time()-t0:.1f}s")

    fig = go.Figure()

    # Scatter dots by condition using Scattergl
    for cond in [0, 1, 2, 3]:
        mask = conditions == cond
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=steps[mask].tolist(),
            y=act_vals[mask].tolist(),
            mode="markers",
            marker=dict(size=2, color=CONDITION_COLORS[cond]),
            name=CONDITION_NAMES[cond],
            showlegend=True,
            hoverinfo="skip",
        ))

    # Compute per-step aggregates
    unique_steps = np.unique(steps)
    mean_active_level = []
    mean_oppose_level = []
    mean_arrest = []
    for s in unique_steps:
        smask = steps == s
        non_jailed = smask & (conditions != 3)
        if non_jailed.sum() > 0:
            # Active level: mean activation of active agents
            active_mask = smask & (conditions == 1)
            oppose_mask = smask & (conditions == 2)
            mean_active_level.append(np.mean(act_vals[active_mask]) if active_mask.sum() > 0 else 0)
            mean_oppose_level.append(np.mean(act_vals[oppose_mask]) if oppose_mask.sum() > 0 else 0)
            mean_arrest.append(np.mean(arrest_probs[non_jailed]))
        else:
            mean_active_level.append(0)
            mean_oppose_level.append(0)
            mean_arrest.append(0)

    # Overlay lines
    fig.add_trace(go.Scatter(
        x=unique_steps.tolist(), y=mean_active_level,
        mode="lines", line=dict(color="red", width=2),
        name="Mean Active Level",
    ))
    fig.add_trace(go.Scatter(
        x=unique_steps.tolist(), y=mean_oppose_level,
        mode="lines", line=dict(color="cornflowerblue", width=2),
        name="Mean Oppose Level",
    ))
    fig.add_trace(go.Scatter(
        x=unique_steps.tolist(), y=mean_arrest,
        mode="lines", line=dict(color="limegreen", width=2),
        name="Mean Arrest Prob",
    ))

    fig.update_layout(**base_layout(
        title=dict(text=f"Sim {sim_id}: {title_extra}", font=dict(size=14)),
        width=900, height=600,
        xaxis_title="Step",
        yaxis_title="Activation Value",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    ))

    return fig


def fig1_agent_scatters():
    """Generate individual scatter plots and 2x2 grid."""
    print("Figure 1: Per-agent activation scatterplots")
    conn = get_conn()

    figs = {}
    for sim_id, label in SIM_SCATTER.items():
        fig = make_scatter_single(conn, sim_id, label)
        if fig:
            save_fig(fig, f"agent_scatter_{sim_id}")
            figs[sim_id] = fig

    # 2x2 grid -- rebuild from scratch (copying traces is fragile)
    sim_ids = list(SIM_SCATTER.keys())
    grid_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Sim {sid}: {SIM_SCATTER[sid]}" for sid in sim_ids],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    for idx, sim_id in enumerate(sim_ids):
        row = idx // 2 + 1
        col = idx % 2 + 1
        if sim_id not in figs:
            continue
        for trace in figs[sim_id].data:
            new_trace = trace
            new_trace.showlegend = (idx == 0)
            grid_fig.add_trace(new_trace, row=row, col=col)

    grid_fig.update_layout(**base_layout(
        title=dict(text="Agent Activation Scatterplots", font=dict(size=16)),
        width=1000, height=800,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    ))
    for i in range(1, 5):
        xkey = f"xaxis{i}" if i > 1 else "xaxis"
        ykey = f"yaxis{i}" if i > 1 else "yaxis"
        grid_fig.update_layout(**{
            xkey: dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"], title="Step"),
            ykey: dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"], title="Activation"),
        })
    for ann in grid_fig.layout.annotations:
        ann.font.size = 11

    save_fig(grid_fig, "agent_scatter_grid")
    conn.close()


# -------------------------------------------------------------------
# 2. Epsilon coherence comparison
# -------------------------------------------------------------------
def fig2_epsilon_coherence():
    """Activation std dev over time for low vs high epsilon."""
    print("Figure 2: Epsilon coherence comparison")
    conn = get_conn()
    cur = conn.cursor()

    fig = go.Figure()

    for sim_id, eps_label, color in [(678, "eps=0.5", "#FFCC66"), (708, "eps=1.5", "#73D0FF")]:
        # Bulk load all data for this sim
        cur.execute(
            "SELECT step, activation_val, condition FROM agent_steps WHERE sim_id=?",
            (sim_id,),
        )
        rows = cur.fetchall()
        data = np.array(rows)
        all_steps = data[:, 0].astype(int)
        all_act = data[:, 1]
        all_cond = data[:, 2].astype(int)

        unique_steps = np.unique(all_steps)
        std_devs = []
        active_counts = []
        n_agents = 0

        for s in unique_steps:
            smask = all_steps == s
            non_jailed = smask & (all_cond != 3)
            if non_jailed.sum() > 0:
                std_devs.append(np.std(all_act[non_jailed]))
            else:
                std_devs.append(0)
            active_counts.append(int((smask & (all_cond == 1)).sum()))
            if n_agents == 0:
                n_agents = int(smask.sum())

        fig.add_trace(go.Scatter(
            x=unique_steps.tolist(), y=std_devs,
            mode="lines", line=dict(color=color, width=2),
            name=f"Std Dev ({eps_label})",
        ))
        fig.add_trace(go.Scatter(
            x=unique_steps.tolist(), y=[c / n_agents for c in active_counts],
            mode="lines", line=dict(color=color, width=1, dash="dash"),
            name=f"Active Frac ({eps_label})",
            opacity=0.6,
        ))

    fig.update_layout(**base_layout(
        title=dict(text="Activation Coherence: Low vs High Epsilon (sd=0.02, th=2.0, seed=123)", font=dict(size=14)),
        width=900, height=600,
        xaxis_title="Step",
        yaxis_title="Std Dev / Active Fraction",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    ))
    save_fig(fig, "epsilon_coherence_comparison")
    conn.close()


# -------------------------------------------------------------------
# 3. Epsilon speed-of-cascade comparison
# -------------------------------------------------------------------
def fig3_epsilon_cascade_speed():
    """Active and jail counts over time for paired epsilon sims."""
    print("Figure 3: Epsilon cascade speed comparison")
    conn = get_conn()
    cur = conn.cursor()

    fig = go.Figure()

    for sim_id, eps_label, color in [(678, "eps=0.5", "#FFCC66"), (708, "eps=1.5", "#73D0FF")]:
        # Get per-step condition counts
        cur.execute(
            """SELECT step,
                      SUM(CASE WHEN condition=1 THEN 1 ELSE 0 END) as actives,
                      SUM(CASE WHEN condition=3 THEN 1 ELSE 0 END) as jailed
               FROM agent_steps
               WHERE sim_id = ?
               GROUP BY step ORDER BY step""",
            (sim_id,),
        )
        rows = cur.fetchall()
        steps = [r[0] for r in rows]
        actives = [r[1] for r in rows]
        jailed = [r[2] for r in rows]

        fig.add_trace(go.Scatter(
            x=steps, y=actives,
            mode="lines", line=dict(color=color, width=2.5),
            name=f"Active ({eps_label})",
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=jailed,
            mode="lines", line=dict(color=color, width=1.5, dash="dash"),
            name=f"Jailed ({eps_label})",
        ))

    fig.update_layout(**base_layout(
        title=dict(text="Cascade Speed: Low vs High Epsilon (sd=0.02, th=2.0, seed=123)", font=dict(size=14)),
        width=900, height=600,
        xaxis_title="Step",
        yaxis_title="Agent Count",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    ))
    save_fig(fig, "epsilon_cascade_speed")
    conn.close()


# -------------------------------------------------------------------
# 4. Spatial grid snapshots (sim 74, revolution)
# -------------------------------------------------------------------
def fig4_spatial_snapshots():
    """2x3 grid of spatial heatmaps at cascade moments for sim 74."""
    print("Figure 4: Spatial grid snapshots (sim 74)")
    conn = get_conn()
    cur = conn.cursor()

    target_steps = [10, 25, 30, 35, 40, 50]

    # Discrete colorscale: support=light blue, active=red, oppose=purple, jailed=dark gray, empty=black
    # Map: -1=empty, 0=support, 1=active, 2=oppose, 3=jailed
    # Normalize to 0-1 for colorscale: (-1 -> 0, 0 -> 0.25, 1 -> 0.5, 2 -> 0.75, 3 -> 1.0)
    colorscale = [
        [0.0, "#111111"],    # empty (no agent)
        [0.2, "#111111"],
        [0.2, "#4A6FA5"],    # support (0)
        [0.4, "#4A6FA5"],
        [0.4, "#FF4444"],    # active (1)
        [0.6, "#FF4444"],
        [0.6, "#9966CC"],    # oppose (2)
        [0.8, "#9966CC"],
        [0.8, "#555555"],    # jailed (3)
        [1.0, "#555555"],
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"Step {s}" for s in target_steps],
        horizontal_spacing=0.05, vertical_spacing=0.08,
    )

    for idx, step in enumerate(target_steps):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Get agent positions and conditions
        cur.execute(
            "SELECT pos_x, pos_y, condition FROM agent_steps WHERE sim_id=74 AND step=?",
            (step,),
        )
        rows_data = cur.fetchall()

        # Build 33x33 grid
        grid = np.full((33, 33), -1.0)  # empty
        for px, py, cond in rows_data:
            if 0 <= px < 33 and 0 <= py < 33:
                grid[py][px] = float(cond)

        # Normalize to 0-1 range for colorscale
        # -1 -> 0.1, 0 -> 0.3, 1 -> 0.5, 2 -> 0.7, 3 -> 0.9
        display = np.where(grid == -1, 0.1,
                  np.where(grid == 0, 0.3,
                  np.where(grid == 1, 0.5,
                  np.where(grid == 2, 0.7, 0.9))))

        fig.add_trace(go.Heatmap(
            z=display.tolist(),
            colorscale=colorscale,
            zmin=0, zmax=1,
            showscale=False,
            hoverinfo="skip",
        ), row=row, col=col)

    fig.update_layout(**base_layout(
        title=dict(text="Spatial Cascade Evolution -- Sim 74 (sd=0.02, th=1.5, revolution)", font=dict(size=14)),
        width=1000, height=700,
    ))
    # Remove axis ticks from heatmaps
    for i in range(1, 7):
        xkey = f"xaxis{i}" if i > 1 else "xaxis"
        ykey = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(**{
            xkey: dict(showticklabels=False, gridcolor=THEME["grid"]),
            ykey: dict(showticklabels=False, gridcolor=THEME["grid"], scaleanchor=f"x{i}" if i > 1 else "x"),
        })
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = THEME["text"]

    # Add a manual legend as annotations
    legend_items = [
        ("Support", "#4A6FA5"), ("Active", "#FF4444"),
        ("Oppose", "#9966CC"), ("Jailed", "#555555"), ("Empty", "#111111"),
    ]
    for li, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=1.02, y=0.9 - li * 0.08,
            xref="paper", yref="paper",
            text=f"<b style='color:{color}'>\u25A0</b> {label}",
            showarrow=False,
            font=dict(size=11, color=THEME["text"]),
            xanchor="left",
        )

    save_fig(fig, "spatial_cascade_evolution")
    conn.close()


# -------------------------------------------------------------------
# 5. Arrest dynamics (sim 74)
# -------------------------------------------------------------------
def fig5_arrest_dynamics():
    """Mean arrest prob vs active fraction for sim 74."""
    print("Figure 5: Arrest dynamics (sim 74)")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """SELECT step,
                  AVG(CASE WHEN condition != 3 THEN arrest_prob END) as mean_arrest,
                  SUM(CASE WHEN condition = 1 THEN 1 ELSE 0 END) * 1.0 /
                  COUNT(*) as active_frac
           FROM agent_steps
           WHERE sim_id = 74
           GROUP BY step ORDER BY step""",
        (),
    )
    rows = cur.fetchall()
    steps = [r[0] for r in rows]
    mean_arrest = [r[1] or 0 for r in rows]
    active_frac = [r[2] for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=mean_arrest,
        mode="lines", line=dict(color="limegreen", width=2.5),
        name="Mean Arrest Prob (non-jailed)",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=active_frac,
        mode="lines", line=dict(color="#FF4444", width=2.5),
        name="Active Fraction",
    ))

    fig.update_layout(**base_layout(
        title=dict(text="Arrest Dynamics -- Sim 74 (sd=0.02, th=1.5, revolution)", font=dict(size=14)),
        width=900, height=600,
        xaxis_title="Step",
        yaxis_title="Probability / Fraction (0-1)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    ))
    save_fig(fig, "arrest_dynamics_clean")
    conn.close()


# -------------------------------------------------------------------
# 6. Epsilon macro vs micro
# -------------------------------------------------------------------
def fig6_epsilon_macro_micro():
    """Revolution rate and activation std dev by epsilon at th=2.0, sd=0.02."""
    print("Figure 6: Epsilon macro vs micro")
    conn = get_conn()
    cur = conn.cursor()

    # Get sims: epsilon_threshold_interaction at th=2.0, sd=0.02
    cur.execute(
        """SELECT sim_id, epsilon, seed FROM sim_params
           WHERE sim_group='epsilon_threshold_interaction'
             AND threshold=2.0 AND sec_density=0.02
           ORDER BY epsilon, seed""",
    )
    sims = cur.fetchall()

    # Group by epsilon
    eps_groups = {}
    for sim_id, eps, seed in sims:
        eps_groups.setdefault(eps, []).append(sim_id)

    epsilons = sorted(eps_groups.keys())
    max_active_fracs = []  # mean of max active fraction across seeds
    mean_std_devs = []

    for eps in epsilons:
        sim_ids = eps_groups[eps]
        peak_fracs = []
        std_devs_at_max = []

        for sid in sim_ids:
            # Bulk load
            cur.execute(
                "SELECT step, activation_val, condition FROM agent_steps WHERE sim_id=?",
                (sid,),
            )
            rows = cur.fetchall()
            data = np.array(rows)
            all_steps = data[:, 0].astype(int)
            all_act = data[:, 1]
            all_cond = data[:, 2].astype(int)

            unique_steps = np.sort(np.unique(all_steps))
            best_active_frac = 0
            transition_step = unique_steps[0]
            for s in unique_steps:
                smask = all_steps == s
                n_total = smask.sum()
                n_active = ((smask) & (all_cond == 1)).sum()
                frac = n_active / n_total if n_total > 0 else 0
                if frac > best_active_frac:
                    best_active_frac = frac
                # Find the transition step: first step with >25% active
                if frac > 0.25 and transition_step == unique_steps[0]:
                    transition_step = s
            peak_fracs.append(best_active_frac)

            # Std dev at transition step (more informative than at peak)
            smask = (all_steps == transition_step) & (all_cond != 3)
            if smask.sum() > 0:
                std_devs_at_max.append(np.std(all_act[smask]))

        max_active_fracs.append(np.mean(peak_fracs) if peak_fracs else 0)
        mean_std_devs.append(np.mean(std_devs_at_max) if std_devs_at_max else 0)

    # Two stacked subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Peak Active Fraction by Epsilon", "Activation Std Dev at Cascade Onset by Epsilon"],
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(
        x=epsilons, y=max_active_fracs,
        mode="lines+markers",
        line=dict(color=THEME["accent"], width=2.5),
        marker=dict(size=10, color=THEME["accent"]),
        name="Peak Active Frac",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epsilons, y=mean_std_devs,
        mode="lines+markers",
        line=dict(color="#73D0FF", width=2.5),
        marker=dict(size=10, color="#73D0FF"),
        name="Mean Std Dev",
    ), row=2, col=1)

    fig.update_layout(**base_layout(
        title=dict(text="Epsilon: Macro Outcome vs Micro Coherence (th=2.0, sd=0.02)", font=dict(size=14)),
        width=900, height=700,
        showlegend=False,
    ))
    fig.update_xaxes(title_text="Epsilon", gridcolor=THEME["grid"], row=1, col=1)
    fig.update_xaxes(title_text="Epsilon", gridcolor=THEME["grid"], row=2, col=1)
    fig.update_yaxes(title_text="Peak Active Fraction", gridcolor=THEME["grid"], row=1, col=1)
    fig.update_yaxes(title_text="Std Dev of Activation", gridcolor=THEME["grid"], row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = THEME["text"]

    save_fig(fig, "epsilon_macro_micro_clean")
    conn.close()


# -------------------------------------------------------------------
# 7. Fix pairwise heatmaps
# -------------------------------------------------------------------
def fig7_fix_heatmaps():
    """Fix overlapping text in the 10-pairwise manifold heatmaps."""
    print("Figure 7: Fix pairwise heatmaps")
    src_dir = Path.home() / "git/reports-web/data/cascade/2026-04-05-cascade-comprehensive-analysis/figures"

    for fname in ["hires_all_revolution_prob", "hires_all_cascade_rate"]:
        src = src_dir / f"{fname}.json"
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue

        with open(src) as f:
            fig_dict = json.load(f)

        layout = fig_dict.get("layout", {})

        # Fix annotations (subplot titles) -- reduce font size
        for ann in layout.get("annotations", []):
            ann["font"] = ann.get("font", {})
            ann["font"]["size"] = 10

        # Fix axis labels -- shorter titles, smaller font, remove inner labels
        # Identify all axes
        axis_keys = [k for k in layout.keys() if k.startswith(("xaxis", "yaxis"))]
        for ak in axis_keys:
            ax = layout[ak]
            # Reduce title font
            if "title" in ax:
                if isinstance(ax["title"], dict):
                    ax["title"]["font"] = ax["title"].get("font", {})
                    ax["title"]["font"]["size"] = 9
                    # Abbreviate
                    text = ax["title"].get("text", "")
                    text = text.replace("Security Density", "Sec Den")
                    text = text.replace("PP Mean", "PP")
                    text = text.replace("Epsilon", "Eps")
                    text = text.replace("Threshold", "Thr")
                    text = text.replace("Vision", "Vis")
                    ax["title"]["text"] = text
                elif isinstance(ax["title"], str):
                    text = ax["title"]
                    text = text.replace("Security Density", "Sec Den")
                    text = text.replace("PP Mean", "PP")
                    text = text.replace("Epsilon", "Eps")
                    text = text.replace("Threshold", "Thr")
                    text = text.replace("Vision", "Vis")
                    ax["title"] = dict(text=text, font=dict(size=9))
            # Reduce tick font
            ax["tickfont"] = ax.get("tickfont", {})
            ax["tickfont"]["size"] = 8

        # Increase spacing -- we need to adjust domains
        # For a 2x5 grid (2 rows, 5 cols), increase gaps
        # Recalculate with more spacing
        n_cols = 5
        n_rows = 2
        h_gap = 0.06
        v_gap = 0.12

        # Sort x-axes by their domain start position
        x_axes = sorted(
            [(k, v) for k, v in layout.items() if k.startswith("xaxis")],
            key=lambda kv: kv[1].get("domain", [0])[0],
        )
        y_axes = sorted(
            [(k, v) for k, v in layout.items() if k.startswith("yaxis")],
            key=lambda kv: kv[1].get("domain", [0])[0],
        )

        # Recalculate domains for 5 columns
        col_width = (1.0 - h_gap * (n_cols - 1)) / n_cols
        row_height = (1.0 - v_gap * (n_rows - 1)) / n_rows

        # Map each axis to its grid position based on current domain
        for i, (ak, ax) in enumerate(x_axes):
            col = i % n_cols
            x0 = col * (col_width + h_gap)
            x1 = x0 + col_width
            ax["domain"] = [x0, x1]

        for i, (ak, ax) in enumerate(y_axes):
            row_from_bottom = i // n_cols  # bottom row = 0
            # Actually need to figure out row assignment from current domain
            pass

        # Simpler approach: just adjust existing domains to add more spacing
        # by scaling them down slightly
        for ak in axis_keys:
            ax = layout[ak]
            domain = ax.get("domain", [0, 1])
            if ak.startswith("xaxis"):
                # Shrink each x domain slightly to add gaps
                center = (domain[0] + domain[1]) / 2
                half_w = (domain[1] - domain[0]) / 2 * 0.88
                ax["domain"] = [center - half_w, center + half_w]

        # Update overall size
        layout["width"] = 1400
        layout["height"] = 800
        layout["margin"] = {"l": 40, "r": 40, "t": 60, "b": 40}

        # Apply dark theme
        layout["paper_bgcolor"] = THEME["bg"]
        layout["plot_bgcolor"] = THEME["bg"]
        layout["font"] = {"color": THEME["text"], "size": 10}

        out_name = fname.replace("hires_all_", "hires_") + "_fixed"
        out_path = FIG_DIR / f"{out_name}.json"
        out_path.write_text(safe_json_dumps(fig_dict))
        size_kb = out_path.stat().st_size / 1024
        print(f"  Saved {out_name}.json ({size_kb:.0f} KB)")


# -------------------------------------------------------------------
# Cleanup old figures
# -------------------------------------------------------------------
def cleanup_old():
    """Delete old broken figures."""
    old_files = [
        "agent_activation_scatter_310.json",
        "agent_activation_scatter_318.json",
        "agent_activation_scatter_363.json",
        "agent_activation_scatter_459.json",
        "agent_activation_scatter_644.json",
        "agent_activation_scatter_80.json",
        "agent_activation_grid.json",
        "epsilon_coherence_low.json",
        "epsilon_coherence_high.json",
        "epsilon_macro_vs_micro.json",
        "activation_distribution_by_epsilon.json",
        "spatial_grid_evolution.json",
        "spatial_grid_80_step10.json",
        "spatial_grid_80_step100.json",
        "spatial_grid_80_step250.json",
        "spatial_grid_80_step50.json",
        "arrest_prob_dynamics.json",
        "vision_cascade_periodicity.json",
    ]
    deleted = 0
    for fname in old_files:
        p = FIG_DIR / fname
        if p.exists():
            p.unlink()
            deleted += 1
    print(f"Cleanup: deleted {deleted} old figure files")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"Phase 1F Analysis -- Output: {FIG_DIR}\n")

    cleanup_old()
    print()

    fig1_agent_scatters()
    print()
    fig2_epsilon_coherence()
    print()
    fig3_epsilon_cascade_speed()
    print()
    fig4_spatial_snapshots()
    print()
    fig5_arrest_dynamics()
    print()
    fig6_epsilon_macro_micro()
    print()
    fig7_fix_heatmaps()

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
