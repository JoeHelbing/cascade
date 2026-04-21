"""
Diff two per-agent step traces -- reference (original_python) vs candidate
(mojo_cpu or mojo_gpu) -- and report any (seed, step, agent_id) tuples where
state diverges.

Columns compared (float tolerances in parentheses):
    condition              exact
    pos_x, pos_y           exact
    private_preference     1e-6
    epsilon                1e-6
    oppose_threshold       1e-6
    active_threshold       1e-6
    jail_sentence          exact
    activation             1e-6
    perception             1e-6
    arrest_prob            1e-6
    active_level           1e-6
    oppose_level           1e-6

Use `--tol` to override the float tolerance globally. Use `--max-rows` to cap
the number of divergent rows printed.

Exit code 0 if traces agree within tolerance on every row; 1 otherwise.

Usage:
    uv run autoresearch/validation/compare_traces.py \
        --ref autoresearch/validation/python_trace.parquet \
        --cand autoresearch/validation/mojo_cpu_trace.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

FLOAT_COLS = [
    "private_preference",
    "epsilon",
    "oppose_threshold",
    "active_threshold",
    "activation",
    "perception",
    "arrest_prob",
    "active_level",
    "oppose_level",
]
EXACT_COLS = ["condition", "pos_x", "pos_y", "jail_sentence"]
JOIN_KEYS = ["seed", "Step", "AgentID"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normalise column names: mesa uses "Step", "AgentID" -- some callers may lowercase.
    rename = {c: c.lower() for c in df.columns if c.lower() in ("step", "agentid")}
    df = df.rename(columns=rename)
    df = df.rename(columns={"step": "Step", "agentid": "AgentID"})
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-rows", type=int, default=30)
    args = ap.parse_args()

    ref = load(args.ref)
    cand = load(args.cand)

    missing_keys = [k for k in JOIN_KEYS if k not in ref.columns or k not in cand.columns]
    if missing_keys:
        print(f"ERROR: missing join keys {missing_keys} in one of the traces", file=sys.stderr)
        return 2

    merged = ref.merge(cand, on=JOIN_KEYS, suffixes=("_ref", "_cand"), how="outer", indicator=True)
    only_ref = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()

    print(f"rows only in ref : {len(only_ref)}")
    print(f"rows only in cand: {len(only_cand)}")
    print(f"rows in both     : {len(both)}")

    divergent: list[tuple[str, int, pd.DataFrame]] = []
    for col in EXACT_COLS:
        ref_col = f"{col}_ref"
        cand_col = f"{col}_cand"
        if ref_col not in both.columns or cand_col not in both.columns:
            continue
        bad = both[both[ref_col] != both[cand_col]]
        if len(bad):
            divergent.append((col, len(bad), bad[[*JOIN_KEYS, ref_col, cand_col]].head(args.max_rows)))

    for col in FLOAT_COLS:
        ref_col = f"{col}_ref"
        cand_col = f"{col}_cand"
        if ref_col not in both.columns or cand_col not in both.columns:
            continue
        diff = (both[ref_col].astype(float) - both[cand_col].astype(float)).abs()
        bad = both[diff > args.tol]
        if len(bad):
            divergent.append((col, len(bad), bad[[*JOIN_KEYS, ref_col, cand_col]].head(args.max_rows)))

    if not divergent and len(only_ref) == 0 and len(only_cand) == 0:
        print(f"OK: traces agree within tol={args.tol}")
        return 0

    print("\nDIVERGENCE:")
    for col, n, sample in divergent:
        print(f"\n  column={col}  divergent_rows={n}")
        print(sample.to_string(index=False))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
