#!/usr/bin/env python3
"""Single-file Cascade validation chain.

This is the only live validation script in `autoresearch/validation/`. It owns
the whole current chain:

    original_python/ Mesa reference
        -> mojo_cpu.mojo --rng python
        -> mojo_cpu.mojo --rng gpu (bridge mode; exact GPU comparison pending)
        -> mojo_gpu.mojo

Historical helper scripts were moved under `archive/historical/` so a reader can
start here without chasing multiple validation entry points.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]
import pyarrow as pa  # pyright: ignore[reportMissingImports]
import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPO_ROOT / "autoresearch" / "validation"
DEFAULT_SEEDS = VALIDATION_DIR / "picked_seeds.json"
PYTHON_TRACE = VALIDATION_DIR / "python_trace.parquet"
PYTHON_MODEL_TRACE = VALIDATION_DIR / "python_model_trace.parquet"
MOJO_CPU_TRACE = VALIDATION_DIR / "mojo_cpu_bitexact.csv"
MOJO_GPU_OUTPUT = VALIDATION_DIR / "mojo_gpu_output.txt"

sys.path.insert(0, str(REPO_ROOT / "original_python"))
from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]

FLOAT_COLS = ["opinion", "activation", "active_level", "oppose_level"]
INT_COLS = ["pos_x", "pos_y", "jail_sentence"]
STR_COLS = ["condition"]


def run(cmd: list[str], *, stdout_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command from the repo root, optionally writing stdout to a file."""
    print("$", " ".join(cmd), flush=True)
    if stdout_path is None:
        return subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(f"wrote stdout to {stdout_path.relative_to(REPO_ROOT)}")
    return result


def run_python_seed(seed: int, params: dict, steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one original Mesa seed and return agent/model trace frames."""
    model = ResistanceCascade(seed=seed, **params)
    for _ in range(steps):
        if not model.running:
            break
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    model_df = model.datacollector.get_model_vars_dataframe().reset_index().rename(columns={"index": "Step"})
    agent_df["seed"] = seed
    model_df["seed"] = seed

    if "pos" in agent_df.columns:
        agent_df["pos_x"] = agent_df["pos"].apply(lambda p: p[0] if p is not None else -1)
        agent_df["pos_y"] = agent_df["pos"].apply(lambda p: p[1] if p is not None else -1)
        agent_df = agent_df.drop(columns=["pos"])

    agent_df["kind"] = agent_df.apply(
        lambda row: "citizen" if row.get("active_threshold") is not None else "security",
        axis=1,
    )
    return agent_df, model_df


def generate_python_trace(seeds_path: Path, out: Path, out_model: Path) -> None:
    """Generate the Mesa reference traces formerly produced by run_python_trace.py."""
    cfg = json.loads(seeds_path.read_text())
    params = cfg["params"]
    steps = cfg["steps"]
    seeds = [p["seed"] for p in cfg["picked"]]

    agent_dfs: list[pd.DataFrame] = []
    model_dfs: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"running seed={seed}")
        agents, model = run_python_seed(seed, params, steps)
        agent_dfs.append(agents)
        model_dfs.append(model)

    agent_all = pd.concat(agent_dfs, ignore_index=True)
    model_all = pd.concat(model_dfs, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(agent_all), out)
    pq.write_table(pa.Table.from_pandas(model_all), out_model)
    print(f"wrote {len(agent_all)} agent-step rows to {out}")
    print(f"wrote {len(model_all)} model-step rows to {out_model}")


def float_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def load_mojo_csv(path: Path) -> pd.DataFrame:
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(rows)), float_precision="round_trip")
    return df[df["condition"] != "Security"].copy()


def load_mesa_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["kind"] == "citizen"].copy()
    df = df.rename(columns={"Step": "step", "AgentID": "agent_id"})
    keep = ["seed", "step", "agent_id", "condition"] + INT_COLS + FLOAT_COLS
    return df[keep]


def compare_bitexact(mojo_path: Path, mesa_path: Path) -> None:
    """Bit-exact Mesa-vs-Mojo CPU comparison formerly in compare_bitexact.py."""
    mojo = load_mojo_csv(mojo_path)
    mesa = load_mesa_parquet(mesa_path)
    print(f"mojo rows: {len(mojo)}")
    print(f"mesa rows: {len(mesa)}")

    key = ["seed", "step", "agent_id"]
    merged = mesa.merge(mojo, on=key, how="inner", suffixes=("_mesa", "_mojo"))
    mesa_only = len(mesa) - len(merged)
    mojo_only = len(mojo) - len(merged)

    print()
    print(f"rows joined on (seed, step, agent_id): {len(merged)}")
    print(f"mesa-only rows: {mesa_only}")
    print(f"mojo-only rows: {mojo_only}")
    print()
    print("Per-column mismatches:")
    print("-" * 60)

    fail = False
    for col in INT_COLS + STR_COLS:
        mismatch = int((merged[f"{col}_mesa"] != merged[f"{col}_mojo"]).sum())
        print(f"  {'OK   ' if mismatch == 0 else 'FAIL '} {col:<18} mismatches: {mismatch}")
        fail = fail or mismatch != 0

    for col in FLOAT_COLS:
        mesa_bits = merged[f"{col}_mesa"].apply(float_bits)
        mojo_bits = merged[f"{col}_mojo"].apply(float_bits)
        mismatches = mesa_bits != mojo_bits
        mismatch = int(mismatches.sum())
        print(f"  {'OK   ' if mismatch == 0 else 'FAIL '} {col:<18} mismatches: {mismatch}")
        if mismatch:
            print("    sample diff rows:")
            print(merged[mismatches].head(3)[key + [f"{col}_mesa", f"{col}_mojo"]].to_string(index=False))
        fail = fail or mismatch != 0

    print()
    if mesa_only == 0 and mojo_only == 0 and not fail:
        print("PASS: mojo_cpu is bit-for-bit identical to Mesa on every tracked column.")
        return
    raise SystemExit("FAIL: mojo_cpu and Mesa diverge; see details above.")


def validate_cpu(args: argparse.Namespace) -> None:
    """Validate mojo_cpu.mojo --rng python against original Python/Mesa."""
    print("\n=== CPU validation: original_python -> mojo_cpu --rng python ===", flush=True)
    if not args.skip_python_trace:
        generate_python_trace(args.seeds, PYTHON_TRACE, PYTHON_MODEL_TRACE)
    elif not PYTHON_TRACE.exists():
        raise SystemExit(
            "--skip-python-trace was set, but "
            f"{PYTHON_TRACE.relative_to(REPO_ROOT)} does not exist."
        )

    run(["pixi", "run", "build-cpu"])
    run([str(REPO_ROOT / "build" / "mojo_cpu"), "--rng", "python"], stdout_path=MOJO_CPU_TRACE)
    compare_bitexact(MOJO_CPU_TRACE, PYTHON_TRACE)
    print("CPU validation PASS: mojo_cpu --rng python matches Mesa on tracked per-agent columns.")


def validate_gpu() -> None:
    """Run the current GPU aggregate validation/smoke gate."""
    print("\n=== GPU validation: mojo_cpu --rng gpu boundary -> mojo_gpu aggregate gate ===", flush=True)
    run(["pixi", "run", "build-gpu"])
    result = run([str(REPO_ROOT / "build" / "mojo_gpu")], stdout_path=MOJO_GPU_OUTPUT)
    sim_lines = [line for line in result.stdout.splitlines() if line.startswith("Sim ")]
    if len(sim_lines) != 45:
        raise SystemExit(
            f"GPU validation FAIL: expected 45 'Sim ...' lines, got {len(sim_lines)}. "
            f"See {MOJO_GPU_OUTPUT.relative_to(REPO_ROOT)}."
        )
    print("GPU validation PASS: produced the expected 45 aggregate simulation lines.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cpu", "gpu", "all"], default="all")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--skip-python-trace",
        action="store_true",
        help="Reuse an existing python_trace.parquet instead of regenerating Mesa traces.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"cpu", "all"}:
        validate_cpu(args)
    if args.stage in {"gpu", "all"}:
        validate_gpu()
    print("\nValidation pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
