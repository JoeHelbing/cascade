"""
Bit-exact diff of mojo_cpu per-agent trace against Mesa.

Reads:
    autoresearch/validation/mojo_cpu_bitexact.csv    (mojo output)
    autoresearch/validation/python_trace.parquet     (Mesa reference)

Joins on (seed, step, agent_id) and compares Float64 fields by raw IEEE 754
bit pattern. Exit 0 iff every joinable row agrees on every tracked column.

Usage:
    pixi run python autoresearch/validation/compare_bitexact.py
"""
from __future__ import annotations

import argparse
import struct
from io import StringIO
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Float64 columns compared by exact bit pattern.
FLOAT_COLS = ["opinion", "activation", "active_level", "oppose_level"]
# Integer / categorical columns compared by value equality.
INT_COLS = ["pos_x", "pos_y", "jail_sentence"]
STR_COLS = ["condition"]


def float_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def load_mojo_csv(path: Path) -> pd.DataFrame:
    rows = [ln for ln in path.read_text().splitlines()
            if ln and not ln.startswith("#")]
    # round_trip preserves every bit of the emitted Float64 -- critical
    # because the default parser drops a few low bits on long decimals.
    df = pd.read_csv(StringIO("\n".join(rows)), float_precision="round_trip")
    df = df[df["condition"] != "Security"].copy()
    return df


def load_mesa_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["kind"] == "citizen"].copy()
    df = df.rename(columns={"Step": "step", "AgentID": "agent_id"})
    keep = ["seed", "step", "agent_id", "condition"] + INT_COLS + FLOAT_COLS
    return df[keep]


def diff_frames(mojo: pd.DataFrame, mesa: pd.DataFrame) -> dict:
    key = ["seed", "step", "agent_id"]
    merged = mesa.merge(
        mojo, on=key, how="inner", suffixes=("_mesa", "_mojo"),
    )

    results = {}
    for col in INT_COLS + STR_COLS:
        mismatch = (merged[f"{col}_mesa"] != merged[f"{col}_mojo"]).sum()
        results[col] = int(mismatch)

    for col in FLOAT_COLS:
        mesa_bits = merged[f"{col}_mesa"].apply(float_bits)
        mojo_bits = merged[f"{col}_mojo"].apply(float_bits)
        mismatch = (mesa_bits != mojo_bits).sum()
        results[col] = int(mismatch)
        if mismatch:
            sample = merged[mesa_bits != mojo_bits].head(3)
            results[f"{col}_sample"] = sample[
                key + [f"{col}_mesa", f"{col}_mojo"]
            ]

    results["_rows_joined"] = int(len(merged))
    results["_rows_mesa_only"] = int(len(mesa) - len(merged))
    results["_rows_mojo_only"] = int(len(mojo) - len(merged))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mojo",
        type=Path,
        default=REPO_ROOT / "autoresearch/validation/mojo_cpu_bitexact.csv",
    )
    ap.add_argument(
        "--mesa",
        type=Path,
        default=REPO_ROOT / "autoresearch/validation/python_trace.parquet",
    )
    args = ap.parse_args()

    mojo = load_mojo_csv(args.mojo)
    mesa = load_mesa_parquet(args.mesa)

    print(f"mojo rows: {len(mojo)}")
    print(f"mesa rows: {len(mesa)}")

    results = diff_frames(mojo, mesa)
    print()
    print(f"rows joined on (seed, step, agent_id): {results['_rows_joined']}")
    print(f"mesa-only rows: {results['_rows_mesa_only']}")
    print(f"mojo-only rows: {results['_rows_mojo_only']}")
    print()
    print("Per-column mismatches:")
    print("-" * 60)
    fail = False
    for col in INT_COLS + STR_COLS + FLOAT_COLS:
        n = results[col]
        tag = "OK   " if n == 0 else "FAIL "
        print(f"  {tag} {col:<18} mismatches: {n}")
        if n:
            fail = True
            sample_key = f"{col}_sample"
            if sample_key in results:
                print("    sample diff rows:")
                print(results[sample_key].to_string(index=False))

    print()
    if (
        results["_rows_mesa_only"] == 0
        and results["_rows_mojo_only"] == 0
        and not fail
    ):
        print("PASS: mojo_cpu is bit-for-bit identical to Mesa on every "
              "tracked column.")
        return 0

    print("FAIL: mojo_cpu and Mesa diverge; see details above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
