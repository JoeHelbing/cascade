"""
Wider seed sweep to find trajectories with interesting oscillation dynamics.

The original picked_seeds set all cascade to revolution in ~5 steps, which is
visually a non-event. This script scans N seeds across a small grid of
threshold / private-preference-mean values, scoring each trajectory by metrics
that favour slow / multi-peak / non-monotonic dynamics:

    peak_step      step at which active_count was highest
    n_peaks        local maxima in active_count (dampening / resurgence)
    active_var     variance of active_count over steps 0..peak_step*3
    time_above_10  number of steps with active_count >= 10

Writes a JSON with the top-N interesting seeds + the parameter triple at which
they were run. Downstream capture / mojo_cpu runs pick this up.

Usage:
    pixi run python autoresearch/validation/sweep_oscillating.py --n-seeds 200 --n-keep 6
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]


# Keep the picked-seed constraint: sec_density=0 (arrests / jailing removed),
# multiple_agents_per_cell=True (no set-iteration-order dependency). These are
# the params mojo_cpu is bit-exact against; the sweep only varies threshold and
# private-preference-mean to shift the equilibrium.
BASE_PARAMS = dict(
    width=40,
    height=40,
    citizen_vision=7,
    citizen_density=0.7,
    security_density=0.0,
    security_vision=7,
    max_jail_term=100,
    movement=True,
    multiple_agents_per_cell=True,
    standard_deviation=1.0,
    epsilon=0.5,
    max_iters=500,
)


@dataclass
class Trace:
    seed: int
    pp_mean: float
    threshold: float
    peak_active: int
    peak_step: int
    n_peaks: int
    active_var: float
    time_above_10: int
    revolution_step: int  # -1 if never
    final_active: int
    history: list[int]


def _count_local_peaks(hist: list[int], min_prom: int = 10) -> int:
    """Count local maxima with prominence >= min_prom (distance from the
    surrounding troughs). Crude but enough to rank trajectories."""
    peaks = 0
    n = len(hist)
    i = 1
    while i < n - 1:
        if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]:
            # Look back to find prior trough
            j = i - 1
            while j > 0 and hist[j - 1] <= hist[j]:
                j -= 1
            prom_left = hist[i] - hist[j]
            # Look forward to find next trough
            k = i + 1
            while k < n - 1 and hist[k + 1] <= hist[k]:
                k += 1
            prom_right = hist[i] - hist[k]
            if min(prom_left, prom_right) >= min_prom:
                peaks += 1
        i += 1
    return peaks


def run_trace(seed: int, params: dict, steps: int) -> Trace:
    model = ResistanceCascade(seed=seed, **params)
    hist: list[int] = [int(model.active_count)]
    rev_step = -1
    for step in range(steps):
        if not model.running:
            break
        model.step()
        hist.append(int(model.active_count))
        if rev_step == -1 and getattr(model, "revolution", False):
            rev_step = step + 1

    peak = max(hist)
    peak_step = hist.index(peak)
    n_peaks = _count_local_peaks(hist)
    if len(hist) > 1:
        mean = sum(hist) / len(hist)
        var = sum((h - mean) ** 2 for h in hist) / len(hist)
    else:
        var = 0.0
    time_above = sum(1 for h in hist if h >= 10)
    return Trace(
        seed=seed,
        pp_mean=params["private_preference_distribution_mean"],
        threshold=params["threshold"],
        peak_active=peak,
        peak_step=peak_step,
        n_peaks=n_peaks,
        active_var=var,
        time_above_10=time_above,
        revolution_step=rev_step,
        final_active=hist[-1] if hist else 0,
        history=hist,
    )


def score(tr: Trace) -> float:
    """Higher score = more interesting. Rewards: (a) multiple peaks,
    (b) time spent above noise floor, (c) peak arriving later than step 5."""
    multi_peak = 50.0 * max(tr.n_peaks - 1, 0)
    linger = float(tr.time_above_10)
    slow_peak = max(tr.peak_step - 5, 0) * 2.0
    # Penalise both extremes: trajectories that instantly cascade to full
    # population AND trajectories that never leave noise.
    instant_rev = 30.0 if (tr.revolution_step != -1 and tr.revolution_step <= 6) else 0.0
    dead = 40.0 if tr.peak_active < 5 else 0.0
    return multi_peak + linger + slow_peak - instant_rev - dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--n-keep", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument(
        "--pp-means",
        type=str,
        default="-0.5,-0.3,0.0",
        help="comma-separated pp_mean values to try",
    )
    ap.add_argument(
        "--thresholds",
        type=str,
        default="3.5,4.0,4.5",
        help="comma-separated threshold values to try",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "autoresearch/validation/picked_seeds_oscillating.json",
    )
    args = ap.parse_args()

    pp_means = [float(s) for s in args.pp_means.split(",")]
    thresholds = [float(s) for s in args.thresholds.split(",")]

    traces: list[Trace] = []
    for pp in pp_means:
        for th in thresholds:
            params = dict(BASE_PARAMS)
            params["private_preference_distribution_mean"] = pp
            params["threshold"] = th
            for seed in range(args.n_seeds):
                tr = run_trace(seed, params, args.steps)
                traces.append(tr)
            best_here = max(
                (t for t in traces if t.pp_mean == pp and t.threshold == th),
                key=score,
            )
            print(
                f"pp={pp:+.2f} th={th:.2f}  best seed={best_here.seed:4d} "
                f"score={score(best_here):.1f} peak={best_here.peak_active} "
                f"peak_step={best_here.peak_step} peaks={best_here.n_peaks} "
                f"rev_step={best_here.revolution_step}",
                flush=True,
            )

    ranked = sorted(traces, key=score, reverse=True)
    top = ranked[: args.n_keep]

    payload = {
        "base_params": BASE_PARAMS,
        "pp_means": pp_means,
        "thresholds": thresholds,
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
