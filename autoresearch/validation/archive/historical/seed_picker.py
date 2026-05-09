"""
Pick seeds that produce non-trivial activation dynamics.

Runs the original Mesa model across a range of seeds at a parameter setting
that is known to generate movement (epsilon high, threshold moderate, mean
private preference shifted toward opposition). Records peak active count per
seed and writes the top-N seeds to `picked_seeds.json`.

Why: correctness validation against trajectories where the model never
activates is vacuous -- all conditions stay Support, so any bug in the
activation decision would go undetected.

Usage:
    pixi run python autoresearch/validation/seed_picker.py --n-seeds 200 --n-keep 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]


DEFAULT_PARAMS = dict(
    width=40,
    height=40,
    citizen_vision=7,
    citizen_density=0.7,
    security_density=0.0,
    security_vision=7,
    max_jail_term=100,
    movement=True,
    multiple_agents_per_cell=True,
    private_preference_distribution_mean=0.0,
    standard_deviation=1.0,
    epsilon=0.5,
    threshold=3.5,
    max_iters=500,
)


def peak_active_for_seed(seed: int, params: dict, steps: int) -> tuple[int, int, bool]:
    model = ResistanceCascade(seed=seed, **params)
    peak = 0
    peak_step = 0
    for i in range(steps):
        if not model.running:
            break
        model.step()
        if model.active_count > peak:
            peak = model.active_count
            peak_step = i
    return peak, peak_step, bool(model.revolution)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=200, help="number of candidate seeds to scan")
    ap.add_argument("--n-keep", type=int, default=12, help="number of seeds to keep (top active)")
    ap.add_argument("--steps", type=int, default=500, help="simulation steps per seed")
    ap.add_argument("--min-peak", type=int, default=5, help="minimum peak active count to keep")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "autoresearch/validation/picked_seeds.json")
    args = ap.parse_args()

    results: list[dict] = []
    for seed in range(args.n_seeds):
        peak, peak_step, revolution = peak_active_for_seed(seed, DEFAULT_PARAMS, args.steps)
        results.append(dict(seed=seed, peak_active=peak, peak_step=peak_step, revolution=revolution))
        print(f"seed={seed:4d}  peak_active={peak:4d}  peak_step={peak_step:4d}  revolution={revolution}")

    kept = [r for r in results if r["peak_active"] >= args.min_peak]
    kept.sort(key=lambda r: r["peak_active"], reverse=True)
    kept = kept[: args.n_keep]

    args.out.write_text(
        json.dumps(
            dict(
                params=DEFAULT_PARAMS,
                steps=args.steps,
                min_peak=args.min_peak,
                picked=kept,
            ),
            indent=2,
        )
    )
    print(f"\nwrote {len(kept)} seeds to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
