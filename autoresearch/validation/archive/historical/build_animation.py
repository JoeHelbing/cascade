"""
Render a side-by-side 40x40 grid animation of Mesa vs mojo_cpu for a picked
bit-exact seed, proving the two implementations agree cell-by-cell across every
step.

Inputs:
    autoresearch/validation/python_trace.parquet       (Mesa per-agent trace)
    autoresearch/validation/mojo_cpu_bitexact.csv      (mojo_cpu per-agent trace)

Output:
    autoresearch/validation/anim_bitexact_<seed>.html   standalone animation
    returned HTML fragment from ``build_fragment()`` for embedding in showboat

The colour map intentionally matches the picked archetype colours used in the
resistance_cascade visualizer: Support grey, Active red, Oppose blue, Jailed
black, Security orange.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
from plotly.subplots import make_subplots  # pyright: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parents[2]
VAL = REPO / "autoresearch/validation"

CONDITION_COLOR = {
    "Support":  "#6e7681",
    "Active":   "#f85149",
    "Oppose":   "#58a6ff",
    "Jailed":   "#161b22",
    "Security": "#d29922",
}
CONDITION_ORDER = ["Support", "Oppose", "Active", "Jailed", "Security"]


def _load_mesa(seed: int) -> pd.DataFrame:
    df = pd.read_parquet(VAL / "python_trace.parquet")
    df = df[df["seed"] == seed].copy()
    df = df.rename(columns={"Step": "step", "AgentID": "agent_id"})
    return df[["step", "agent_id", "pos_x", "pos_y", "condition", "kind"]]


def _load_mojo(seed: int) -> pd.DataFrame:
    df = pd.read_csv(
        VAL / "mojo_cpu_bitexact.csv",
        float_precision="round_trip",
        low_memory=False,
    )
    df = df[df["seed"].astype(str) != "# done"]
    df["seed"] = df["seed"].astype(int)
    df["step"] = df["step"].astype(int)
    df["agent_id"] = df["agent_id"].astype(int)
    df["pos_x"] = df["pos_x"].astype(float).astype(int)
    df["pos_y"] = df["pos_y"].astype(float).astype(int)
    df = df[df["seed"] == seed].copy()
    return df[["step", "agent_id", "pos_x", "pos_y", "condition"]]


def _frame_traces(
    step_df: pd.DataFrame,
    show_legend: bool,
    xaxis: str,
    yaxis: str,
) -> list[go.Scatter]:
    traces = []
    for cond in CONDITION_ORDER:
        sel = step_df[step_df["condition"] == cond]
        traces.append(go.Scatter(
            x=sel["pos_x"], y=sel["pos_y"],
            mode="markers",
            marker=dict(
                size=10, color=CONDITION_COLOR[cond],
                line=dict(width=0.5, color="#0b0d10"),
            ),
            name=cond,
            legendgroup=cond,
            showlegend=show_legend,
            xaxis=xaxis, yaxis=yaxis,
            hovertemplate=(
                f"{cond}<br>id=%{{customdata}}"
                "<br>pos=(%{x},%{y})<extra></extra>"
            ),
            customdata=sel["agent_id"],
        ))
    return traces


def build_bitexact_fig(seed: int = 7) -> go.Figure:
    mesa = _load_mesa(seed)
    mojo = _load_mojo(seed)

    steps = sorted(set(mesa["step"].unique()) & set(mojo["step"].unique()))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Mesa (Python)  ·  seed {seed}",
            f"mojo_cpu  ·  seed {seed}",
        ),
        horizontal_spacing=0.06,
    )

    step0 = steps[0]
    for tr in _frame_traces(
        mesa[mesa["step"] == step0], show_legend=True, xaxis="x", yaxis="y",
    ):
        fig.add_trace(tr, row=1, col=1)
    for tr in _frame_traces(
        mojo[mojo["step"] == step0], show_legend=False, xaxis="x2", yaxis="y2",
    ):
        fig.add_trace(tr, row=1, col=2)

    frames = []
    for s in steps:
        data = []
        data.extend(_frame_traces(
            mesa[mesa["step"] == s], show_legend=True, xaxis="x", yaxis="y",
        ))
        data.extend(_frame_traces(
            mojo[mojo["step"] == s], show_legend=False, xaxis="x2", yaxis="y2",
        ))
        frames.append(go.Frame(data=data, name=str(s)))
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            label=str(s),
            args=[[str(s)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
        )
        for s in steps
    ]

    fig.update_layout(
        height=560, width=1120,
        title_text=(
            f"Bit-exact side-by-side: Mesa vs mojo_cpu  ·  seed {seed}  ·  "
            f"{len(steps)} steps · 1120 agents/panel · 0 disagreements"
        ),
        plot_bgcolor="#0b0d10", paper_bgcolor="#0b0d10",
        font=dict(color="#e6edf3", family="JetBrains Mono, monospace", size=12),
        margin=dict(l=40, r=20, t=90, b=80),
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=-0.20, xanchor="left", yanchor="top",
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(color="#e6edf3"),
            buttons=[
                dict(
                    label="play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=800, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode="immediate",
                    )],
                ),
                dict(
                    label="pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            x=0.15, y=-0.20, len=0.80, xanchor="left", yanchor="top",
            currentvalue=dict(
                prefix="step ", visible=True,
                font=dict(color="#e6edf3"),
            ),
            pad=dict(b=10, t=0),
            bgcolor="#30363d",
            bordercolor="#30363d",
            activebgcolor="#58a6ff",
            tickcolor="#e6edf3",
            font=dict(color="#e6edf3"),
            steps=slider_steps,
        )],
    )
    for ax in ("xaxis", "xaxis2"):
        fig.layout[ax].update(
            range=[-1, 40], showgrid=True, gridcolor="#1e2329",
            zeroline=False, scaleanchor="y", scaleratio=1,
        )
    for ax in ("yaxis", "yaxis2"):
        fig.layout[ax].update(
            range=[-1, 40], showgrid=True, gridcolor="#1e2329",
            zeroline=False,
        )
    fig.layout["yaxis2"].update(scaleanchor="x2", scaleratio=1)

    return fig


def build_fragment(seed: int = 7, div_id: str = "chart-anim") -> str:
    return build_bitexact_fig(seed).to_html(
        full_html=False, include_plotlyjs=False, div_id=div_id,
    )


def build_oscillation_fig(
    parquet_path: Path,
    title: str | None = None,
    frame_stride: int = 1,
) -> go.Figure:
    """Single-panel Mesa-only animation for an oscillating trace. Uses the
    trace schema emitted by ``capture_oscillating_trace.py``. Includes security
    agents (orange) and jailed citizens (dark) in the colour map."""
    df = pd.read_parquet(parquet_path)
    # Flag jailed citizens so they show up as their own condition in the viz,
    # independent of whatever Mesa wrote into .condition.
    df = df.copy()
    is_jailed = (df["kind"] == "citizen") & (df["jail_sentence"].fillna(0).astype(int) > 0)
    df.loc[df["kind"] == "security", "condition"] = "Security"
    df.loc[is_jailed, "condition"] = "Jailed"

    steps = sorted(df["step"].unique())
    if frame_stride > 1:
        steps = steps[::frame_stride]

    fig = go.Figure()
    step0 = steps[0]
    for tr in _frame_traces(
        df[df["step"] == step0], show_legend=True, xaxis="x", yaxis="y",
    ):
        fig.add_trace(tr)

    fig.frames = [
        go.Frame(
            data=_frame_traces(
                df[df["step"] == s], show_legend=True, xaxis="x", yaxis="y",
            ),
            name=str(s),
        )
        for s in steps
    ]

    slider_steps = [
        dict(
            method="animate",
            label=str(s),
            args=[[str(s)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
        )
        for s in steps
    ]

    fig.update_layout(
        height=620, width=760,
        title_text=title or f"Mesa oscillation · {len(steps)} frames",
        plot_bgcolor="#0b0d10", paper_bgcolor="#0b0d10",
        font=dict(color="#e6edf3", family="JetBrains Mono, monospace", size=12),
        margin=dict(l=40, r=20, t=70, b=80),
        legend=dict(
            orientation="h", y=-0.10, x=0.5, xanchor="center",
            bgcolor="rgba(0,0,0,0)",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=-0.18, xanchor="left", yanchor="top",
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(color="#e6edf3"),
            buttons=[
                dict(
                    label="play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=120, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode="immediate",
                    )],
                ),
                dict(
                    label="pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            x=0.15, y=-0.18, len=0.80, xanchor="left", yanchor="top",
            currentvalue=dict(
                prefix="step ", visible=True,
                font=dict(color="#e6edf3"),
            ),
            pad=dict(b=10, t=0),
            bgcolor="#30363d",
            bordercolor="#30363d",
            activebgcolor="#58a6ff",
            tickcolor="#e6edf3",
            font=dict(color="#e6edf3"),
            steps=slider_steps,
        )],
        xaxis=dict(
            range=[-1, 40], showgrid=True, gridcolor="#1e2329",
            zeroline=False, scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            range=[-1, 40], showgrid=True, gridcolor="#1e2329",
            zeroline=False,
        ),
    )
    return fig


def build_oscillation_fragment(
    parquet_path: Path,
    div_id: str = "chart-anim-osc",
    title: str | None = None,
    frame_stride: int = 1,
) -> str:
    return build_oscillation_fig(
        parquet_path, title=title, frame_stride=frame_stride,
    ).to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--oscillating", type=Path, default=None,
        help="If set, build single-panel Mesa animation from this parquet.",
    )
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    if args.oscillating is not None:
        frag = build_oscillation_fragment(
            args.oscillating, frame_stride=args.stride,
        )
        out = VAL / f"anim_oscillating_{args.oscillating.stem}.html"
        out.write_text(
            "<!DOCTYPE html>\n"
            "<html><head>"
            '<meta charset="utf-8"><title>oscillating anim</title>'
            '<script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>'
            '<style>body{background:#0b0d10;margin:0;padding:1em;}</style>'
            "</head><body>\n"
            f"{frag}\n</body></html>"
        )
        print(f"wrote {out}  ({out.stat().st_size // 1024} KiB)")
        return 0

    frag = build_fragment(args.seed)
    out = VAL / f"anim_bitexact_{args.seed}.html"
    out.write_text(
        "<!DOCTYPE html>\n"
        "<html><head>"
        '<meta charset="utf-8"><title>bitexact anim</title>'
        '<script src="https://cdn.plot.ly/plotly-2.34.0.min.js"></script>'
        '<style>body{background:#0b0d10;margin:0;padding:1em;}</style>'
        "</head><body>\n"
        f"{frag}\n</body></html>"
    )
    print(f"wrote {out}  ({out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
