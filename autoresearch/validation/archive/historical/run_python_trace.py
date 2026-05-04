"""
Run the original Mesa model for a set of picked seeds and dump full
per-agent per-step state to a parquet file.

Output schema (one row per (seed, step, agent_id)):
    seed, step, agent_id, kind ('citizen'|'security'),
    pos_x, pos_y, condition, private_preference, epsilon, epsilon_probability,
    oppose_threshold, active_threshold, jail_sentence, perception, arrest_prob,
    active_level, oppose_level, flip, ever_flipped, activation

Also dumps the model-level data (active count, support count, etc.) into
a sibling parquet file for aggregate comparison.

Usage:
    uv run autoresearch/validation/run_python_trace.py \
        --seeds autoresearch/validation/picked_seeds.json \
        --out autoresearch/validation/python_trace.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]


def run_seed(seed: int, params: dict, steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = ResistanceCascade(seed=seed, **params)
    # mesa collects step-0 in __init__ already
    for _ in range(steps):
        if not model.running:
            break
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    model_df = model.datacollector.get_model_vars_dataframe().reset_index().rename(columns={"index": "Step"})

    agent_df["seed"] = seed
    model_df["seed"] = seed

    # normalize: flatten tuple pos to two columns
    if "pos" in agent_df.columns:
        agent_df["pos_x"] = agent_df["pos"].apply(lambda p: p[0] if p is not None else -1)
        agent_df["pos_y"] = agent_df["pos"].apply(lambda p: p[1] if p is not None else -1)
        agent_df = agent_df.drop(columns=["pos"])

    # mesa's AgentID includes both citizen and security agents. Tag kind.
    agent_df["kind"] = agent_df.apply(
        lambda row: "citizen" if row.get("active_threshold") is not None else "security",
        axis=1,
    )
    return agent_df, model_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=REPO_ROOT / "autoresearch/validation/picked_seeds.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "autoresearch/validation/python_trace.parquet")
    ap.add_argument("--out-model", type=Path, default=REPO_ROOT / "autoresearch/validation/python_model_trace.parquet")
    args = ap.parse_args()

    cfg = json.loads(args.seeds.read_text())
    params = cfg["params"]
    steps = cfg["steps"]
    seeds = [p["seed"] for p in cfg["picked"]]

    agent_dfs: list[pd.DataFrame] = []
    model_dfs: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"running seed={seed}")
        a, m = run_seed(seed, params, steps)
        agent_dfs.append(a)
        model_dfs.append(m)

    agent_all = pd.concat(agent_dfs, ignore_index=True)
    model_all = pd.concat(model_dfs, ignore_index=True)

    pq.write_table(pa.Table.from_pandas(agent_all), args.out)
    pq.write_table(pa.Table.from_pandas(model_all), args.out_model)

    print(f"wrote {len(agent_all)} agent-step rows to {args.out}")
    print(f"wrote {len(model_all)} model-step rows to {args.out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
