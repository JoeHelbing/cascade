"""Fast-fail probe: find 2-3 seeds at sec_density~0.025-0.035 that show
long-lived oscillation (many steps >= some threshold, multiple local peaks).
Writes the winners as a JSON payload ready for the trace/render pipeline.

Runs Mesa directly; this is not bit-exact with mojo_cpu yet (mojo_cpu only
handles sec_density=0), so the oscillation section of the showboat is
Mesa-only until the mojo_cpu extension ships.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]


BASE = dict(
    width=40, height=40,
    citizen_vision=7, security_vision=7,
    citizen_density=0.7,
    max_jail_term=100,
    movement=True, multiple_agents_per_cell=True,
    standard_deviation=1.0, epsilon=0.5,
    private_preference_distribution_mean=0.0,
    max_iters=500,
)


def run(seed: int, params: dict, steps: int = 500) -> list[int]:
    m = ResistanceCascade(seed=seed, **params)
    hist: list[int] = [int(m.active_count)]
    for _ in range(steps):
        if not m.running:
            break
        m.step()
        hist.append(int(m.active_count))
    return hist


def score(hist: list[int]) -> tuple[int, int, int, float]:
    """Returns (length, time_above_30, n_meaningful_peaks, mean_active)."""
    length = len(hist)
    above = sum(1 for h in hist if h >= 30)
    peaks = 0
    for i in range(5, length - 5):
        left = hist[i - 5:i]
        right = hist[i + 1:i + 6]
        if hist[i] > max(left) and hist[i] >= max(right) and hist[i] > 50:
            peaks += 1
    mean = sum(hist) / max(length, 1)
    return length, above, peaks, mean


def main():
    t0 = time.time()
    configs = [
        dict(security_density=0.025, threshold=2.5),
        dict(security_density=0.03,  threshold=3.0),
        dict(security_density=0.04,  threshold=3.5),
    ]
    winners = []
    for cfg in configs:
        label = f"sd={cfg['security_density']:.3f}_th={cfg['threshold']:.1f}"
        print(f"\n=== {label} ===", flush=True)
        for seed in range(30):
            params = {**BASE, **cfg}
            hist = run(seed, params, 500)
            length, above, peaks, mean = score(hist)
            print(f"  seed={seed:3d}  len={length:3d} "
                  f"above30={above:3d}  peaks={peaks}  mean={mean:.1f}",
                  flush=True)
            # Winner criteria: long, many steps above, multiple peaks
            if length >= 200 and above >= 80 and peaks >= 2:
                winners.append(dict(
                    seed=seed, config=cfg, length=length,
                    above30=above, n_peaks=peaks, mean_active=mean,
                    history=hist,
                ))
    winners.sort(key=lambda w: (w["n_peaks"], w["above30"]), reverse=True)
    out = REPO_ROOT / "autoresearch/validation/picked_oscillating_mesa.json"
    out.write_text(json.dumps({
        "base": BASE, "winners": winners[:6],
    }, indent=2))
    print(f"\n{len(winners)} winners; elapsed {time.time() - t0:.0f}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
