"""
Run Mesa at the picked oscillating params (sec_density > 0) and emit a
per-agent per-step parquet suitable for animation. Mesa-only -- mojo_cpu does
not yet cover sec_density > 0 so there is no side-by-side panel.

Inputs:
    --seeds   path to picked_oscillating_mesa.json from probe_oscillation.py
    --seed    single integer seed to trace (falls back to winners[0])
    --steps   number of steps to run (default 500, matches probe)

Output:
    autoresearch/validation/oscillating_trace_seed<seed>.parquet
    columns: step, agent_id, pos_x, pos_y, condition, kind, jail_sentence
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL = REPO_ROOT / "autoresearch/validation"
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]
from resistance_cascade.agent import Citizen  # noqa: E402  # pyright: ignore[reportMissingImports]


def run_trace(seed: int, params: dict, steps: int) -> pd.DataFrame:
    model = ResistanceCascade(seed=seed, **params)
    rows: list[dict] = []

    def snap(step: int):
        for a in model.schedule.agents:
            if a.pos is None:  # jailed citizens are removed from the grid
                continue
            rows.append(dict(
                step=step,
                agent_id=a.unique_id,
                pos_x=a.pos[0],
                pos_y=a.pos[1],
                condition=a.condition,
                kind="citizen" if isinstance(a, Citizen) else "security",
                jail_sentence=getattr(a, "jail_sentence", 0),
            ))

    snap(0)
    for step in range(1, steps + 1):
        if not model.running:
            break
        model.step()
        snap(step)

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path,
                    default=VAL / "picked_oscillating_mesa.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    picked = json.loads(args.seeds.read_text())
    base = picked["base"]
    winners = picked["winners"]
    if not winners:
        print("No winners in picked JSON.", file=sys.stderr)
        return 1

    target = winners[0] if args.seed is None else next(
        (w for w in winners if w["seed"] == args.seed), None
    )
    if target is None:
        print(f"seed {args.seed} not in winners", file=sys.stderr)
        return 1

    params = {**base, **target["config"]}
    out = args.out or VAL / f"oscillating_trace_seed{target['seed']}.parquet"

    print(
        f"seed={target['seed']}  sec_density={params['security_density']}  "
        f"threshold={params['threshold']}  steps={args.steps}",
        flush=True,
    )
    df = run_trace(target["seed"], params, args.steps)
    df.to_parquet(out)
    print(f"wrote {out}  rows={len(df)}  steps={df['step'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
