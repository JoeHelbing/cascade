"""
Sweep for oscillating seeds with sec_density > 0 (true oscillation regime).

Per the cascade phase-transition analysis, oscillation peaks at ~2-4%
security density. This is separate from the sweep_oscillating.py that stays
at sec_density=0 for bit-exact mojo_cpu comparison -- this sweep explores
Mesa's dynamics in the regime where oscillation actually exists.

Use this to: (a) confirm oscillating regime exists, (b) harvest per-agent
traces for rendering, (c) determine whether the mojo_cpu port needs the
arrest/jail code paths to match the user's "200+ step oscillation" goal.

Usage:
    uv run autoresearch/validation/sweep_oscillating_with_sec.py \
        --n-seeds 200 --n-keep 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "original_python"))
sys.path.insert(0, str(HERE))

from sweep_oscillating import run_trace, score, BASE_PARAMS  # noqa: E402  # pyright: ignore[reportMissingImports]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--n-keep", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument(
        "--sec-densities", type=str, default="0.02,0.03,0.04",
        help="Per phase-transition analysis, oscillation lives here",
    )
    ap.add_argument("--thresholds", type=str, default="1.5,2.5,3.5")
    ap.add_argument("--pp-means", type=str, default="0.0")
    ap.add_argument("--visions", type=str, default="7")
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "autoresearch/validation/picked_seeds_oscillating_withsec.json",
    )
    args = ap.parse_args()

    sec_ds = [float(s) for s in args.sec_densities.split(",")]
    thrs = [float(s) for s in args.thresholds.split(",")]
    pp_ms = [float(s) for s in args.pp_means.split(",")]
    vis = [int(s) for s in args.visions.split(",")]

    traces: list = []
    for sd in sec_ds:
        for th in thrs:
            for pp in pp_ms:
                for v in vis:
                    p = dict(BASE_PARAMS)
                    p["security_density"] = sd
                    p["threshold"] = th
                    p["private_preference_distribution_mean"] = pp
                    p["citizen_vision"] = v
                    p["security_vision"] = v
                    for seed in range(args.n_seeds):
                        t = run_trace(seed, p, args.steps)
                        traces.append(t)
                    best = max(
                        (t for t in traces
                         if t.threshold == th
                         and t.pp_mean == pp),
                        key=score,
                    )
                    print(
                        f"sd={sd:.3f} th={th:.2f} pp={pp:+.2f} v={v}  "
                        f"best seed={best.seed:4d} score={score(best):.1f} "
                        f"peak={best.peak_active} peak_step={best.peak_step} "
                        f"n_peaks={best.n_peaks} rev_step={best.revolution_step} "
                        f"time_above_10={best.time_above_10}",
                        flush=True,
                    )

    ranked = sorted(traces, key=score, reverse=True)
    top = ranked[: args.n_keep]

    payload = {
        "base_params": BASE_PARAMS,
        "sec_densities": sec_ds,
        "thresholds": thrs,
        "pp_means": pp_ms,
        "visions": vis,
        "n_seeds_scanned": args.n_seeds,
        "picked": [
            dict(
                seed=t.seed,
                pp_mean=t.pp_mean,
                threshold=t.threshold,
                peak_active=t.peak_active,
                peak_step=t.peak_step,
                n_peaks=t.n_peaks,
                active_var=t.active_var,
                time_above_10=t.time_above_10,
                revolution_step=t.revolution_step,
                final_active=t.final_active,
                score=score(t),
                history=t.history,
            )
            for t in top
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(top)} seeds to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
