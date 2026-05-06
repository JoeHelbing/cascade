#!/usr/bin/env python3
"""Analyze `mojo_gpu --seed-search` output for full-length oscillating cases.

The GPU emits one `STEP,...` CSV row per simulation step. This script ranks
(seed, epsilon, security_density) cases that reached the full 500-step horizon
without revolution and whose active-count trajectory keeps moving late in the
run. The resulting JSON is a compact candidate seed set for future validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "autoresearch" / "validation" / "mojo_gpu_seed_search_output.txt"
DEFAULT_OUTPUT = REPO_ROOT / "autoresearch" / "validation" / "gpu_oscillating_seed_candidates.json"
EXPECTED_STEPS = 500


@dataclass(frozen=True)
class Candidate:
    seed: int
    epsilon: float
    security_density: float
    score: float
    active_range: int
    late_active_range: int
    late_direction_changes: int
    final_active: int
    final_support: int
    final_oppose: int
    final_jail: int
    final_revolution: bool


def parse_step_rows(path: Path) -> dict[tuple[int, float, float], list[dict[str, int | float | bool]]]:
    cases: dict[tuple[int, float, float], list[dict[str, int | float | bool]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith("STEP,") or line.startswith("STEP,sim,"):
            continue
        row = next(csv.DictReader([line], fieldnames=[
            "tag", "sim", "seed", "epsilon", "security_density", "step",
            "active", "support", "oppose", "jail", "revolution",
        ]))
        key = (int(row["seed"]), float(row["epsilon"]), float(row["security_density"]))
        cases[key].append({
            "step": int(row["step"]),
            "active": int(row["active"]),
            "support": int(row["support"]),
            "oppose": int(row["oppose"]),
            "jail": int(row["jail"]),
            "revolution": bool(int(row["revolution"])),
        })
    return cases


def direction_changes(values: list[int], min_delta: int = 3) -> int:
    last_sign = 0
    changes = 0
    for prev, cur in zip(values, values[1:], strict=False):
        delta = cur - prev
        if abs(delta) < min_delta:
            continue
        sign = 1 if delta > 0 else -1
        if last_sign and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


def score_case(rows: list[dict[str, int | float | bool]]) -> Candidate | None:
    if len(rows) != EXPECTED_STEPS:
        return None
    rows = sorted(rows, key=lambda r: int(r["step"]))
    if int(rows[-1]["step"]) != EXPECTED_STEPS - 1:
        return None
    if any(bool(row["revolution"]) for row in rows):
        return None
    active = [int(row["active"]) for row in rows]
    late = active[EXPECTED_STEPS // 2 :]
    active_range = max(active) - min(active)
    late_range = max(late) - min(late)
    changes = direction_changes(late)
    # Prefer non-trivial swings that persist late, penalize monotonic ramps.
    score = late_range * 1.5 + changes * 12.0 + pstdev(late) + mean(abs(b - a) for a, b in zip(late, late[1:]))
    first = rows[0]
    final = rows[-1]
    return Candidate(
        seed=int(first.get("seed", 0)),  # patched by caller
        epsilon=0.0,
        security_density=0.0,
        score=round(score, 3),
        active_range=active_range,
        late_active_range=late_range,
        late_direction_changes=changes,
        final_active=int(final["active"]),
        final_support=int(final["support"]),
        final_oppose=int(final["oppose"]),
        final_jail=int(final["jail"]),
        final_revolution=bool(final["revolution"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=24)
    args = parser.parse_args()

    cases = parse_step_rows(args.input)
    candidates: list[Candidate] = []
    full_length = 0
    full_length_security = 0
    for (seed, epsilon, security_density), rows in cases.items():
        scored = score_case(rows)
        if scored is None:
            continue
        full_length += 1
        if security_density > 0:
            full_length_security += 1
        candidates.append(
            Candidate(
                seed=seed,
                epsilon=epsilon,
                security_density=security_density,
                score=scored.score,
                active_range=scored.active_range,
                late_active_range=scored.late_active_range,
                late_direction_changes=scored.late_direction_changes,
                final_active=scored.final_active,
                final_support=scored.final_support,
                final_oppose=scored.final_oppose,
                final_jail=scored.final_jail,
                final_revolution=scored.final_revolution,
            )
        )

    candidates.sort(key=lambda c: (c.score, c.security_density > 0, c.late_active_range), reverse=True)
    top = candidates[: args.top]
    payload = {
        "source": str(args.input.relative_to(REPO_ROOT) if args.input.is_relative_to(REPO_ROOT) else args.input),
        "expected_steps": EXPECTED_STEPS,
        "cases_seen": len(cases),
        "full_length_non_revolution_cases": full_length,
        "full_length_non_revolution_security_cases": full_length_security,
        "selection_rule": "full 500 STEP rows, no revolution flag, ranked by late active-count range/direction changes/stddev",
        "top_candidates": [c.__dict__ for c in top],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
