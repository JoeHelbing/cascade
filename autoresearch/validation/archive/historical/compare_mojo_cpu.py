"""
Aggregate-level comparison of mojo_cpu vs Mesa on the 12 picked seeds.

Reads:
    autoresearch/validation/mojo_cpu_model_trace.csv     (mojo_cpu step counts)
    autoresearch/validation/python_model_trace.parquet   (Mesa step counts)

Bit-exact per-agent comparison is *not* expected: mojo_cpu uses LCG/Float32
and uniform init, while Mesa uses Mersenne-Twister/Float64 and Gaussian init.
This script instead reports aggregate behaviour: peak active, revolution step,
final counts, and the per-step active-count trajectory.

Exit 0 if every picked seed reaches revolution in both models. We do NOT
assert on bit-exactness of trajectories -- the two models structurally differ.

Usage:
    uv run autoresearch/validation/compare_mojo_cpu.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_mojo_csv(path: Path) -> pd.DataFrame:
    # Filter out the trailing "# done..." marker line
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(rows)))
    return df


def summarise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-seed summary: peak active, revolution step, revolution flag,
    final counts."""
    g = df.groupby("seed")
    rec = []
    for seed, grp in g:
        peak_active = int(grp["active"].max())
        peak_step = int(grp.loc[grp["active"].idxmax(), "step"])
        rev_rows = grp[grp["revolution"] == 1]
        rev_step = int(rev_rows["step"].min()) if len(rev_rows) else -1
        last = grp.iloc[-1]
        rec.append(dict(
            seed=seed,
            peak_active=peak_active,
            peak_step=peak_step,
            revolution=bool(len(rev_rows)),
            revolution_step=rev_step,
            final_active=int(last["active"]),
            final_support=int(last["support"]),
            final_oppose=int(last["oppose"]),
            final_jail=int(last["jail"]),
            label=label,
        ))
    return pd.DataFrame(rec)


def mesa_to_step_trace(model_df: pd.DataFrame) -> pd.DataFrame:
    """Rename Mesa columns to match mojo_cpu CSV schema."""
    out = model_df.rename(columns={
        "Step": "step",
        "Active Count": "active",
        "Support Count": "support",
        "Oppose Count": "oppose",
        "Jail Count": "jail",
        "Revolution": "revolution",
    }).copy()
    out["revolution"] = out["revolution"].astype(int)
    keep = ["seed", "step", "active", "support", "oppose", "jail", "revolution"]
    return out[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mojo", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/mojo_cpu_model_trace.csv")
    ap.add_argument("--mesa", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/python_model_trace.parquet")
    args = ap.parse_args()

    mojo = load_mojo_csv(args.mojo)
    mesa_full = pd.read_parquet(args.mesa)
    mesa = mesa_to_step_trace(mesa_full)

    mojo_summary = summarise(mojo, "mojo_cpu")
    mesa_summary = summarise(mesa, "mesa")

    merged = mojo_summary.merge(
        mesa_summary, on="seed", suffixes=("_mojo", "_mesa")
    )

    print("Per-seed summary (mojo_cpu vs Mesa)")
    print("=" * 80)
    cols = [
        "seed",
        "peak_active_mojo", "peak_active_mesa",
        "peak_step_mojo", "peak_step_mesa",
        "revolution_step_mojo", "revolution_step_mesa",
    ]
    print(merged[cols].to_string(index=False))
    print()

    all_revolt_mojo = merged["revolution_mojo"].all()
    all_revolt_mesa = merged["revolution_mesa"].all()
    print(f"mojo_cpu revolution on all seeds: {all_revolt_mojo}")
    print(f"mesa     revolution on all seeds: {all_revolt_mesa}")

    if not (all_revolt_mojo and all_revolt_mesa):
        print("\nFAIL: expected both models to reach revolution on every seed")
        return 1

    # Additional aggregate: mean revolution step
    print()
    print(f"mean revolution step -- mojo_cpu: {merged['revolution_step_mojo'].mean():.1f}")
    print(f"mean revolution step -- mesa    : {merged['revolution_step_mesa'].mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
