"""
Capture every RNG draw Mesa consumes during a ResistanceCascade run.

Produces a per-seed injection bundle: init snapshot + ordered decision log +
per-step boundary markers. A deterministic replay can consume the bundle and
reproduce Mesa's trajectory without re-running Mersenne Twister.

Why
---
`mojo_cpu.mojo` uses LCG/Float32/uniform-init while Mesa uses Mersenne-Twister/
Float64/Gaussian-init. A port cannot match bit-exact without either reimple-
menting Mesa's RNG+init in mojo (big) or feeding Mesa's stochastic outcomes
directly into the port (this approach: "decision injection"). Once captured
here, the mojo_cpu replay only has to reproduce Mesa's deterministic math.

Decisions captured
------------------
- init: `randrange(w)` + `randrange(h)` per agent; `gauss` per citizen for
  private_preference, epsilon, and the threshold pair; `gauss` for security
  private_preference.
- per step, per citizen (alive): `uniform(0,1)` for random_activation.
- per step, per citizen (un-jailing): `choice(empties)` for re-placement.
- per step, per citizen/security: `choice(next_moves)` in random_move.
- per step, per security arresting: `choice(actives/opposed)` + `randint`.

Usage:
    uv run autoresearch/validation/capture_mesa.py \
        --seeds autoresearch/validation/picked_seeds.json \
        --out   autoresearch/validation/captures/
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]
from resistance_cascade.agent import Citizen, Security  # noqa: E402  # pyright: ignore[reportMissingImports]


class RecordingRandom(random.Random):
    """Drop-in `random.Random` that logs every top-level call.

    Inner dispatch: some high-level methods (uniform, gauss, choice, randrange)
    internally call `self.random()` or `self.getrandbits()`. We log only the
    *outermost* call so the log has exactly one entry per caller-visible draw.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self.log: list[tuple[str, tuple, Any]] = []
        self._depth: int = 0

    def getrandbits(self, k: int) -> int:
        # Defined so Python's __init_subclass__ picks `_randbelow_with_getrandbits`
        # rather than the `_randbelow_without_getrandbits` fallback (the fallback
        # would consume extra `self.random()` calls and change trajectories).
        return super().getrandbits(k)

    def _log(self, method: str, args: tuple, value: Any) -> None:
        if self._depth == 1:  # only log when we are the outermost call
            self.log.append((method, args, value))

    def random(self) -> float:
        self._depth += 1
        try:
            v = super().random()
            self._log("random", (), v)
            return v
        finally:
            self._depth -= 1

    def uniform(self, a: float, b: float) -> float:
        self._depth += 1
        try:
            v = super().uniform(a, b)
            self._log("uniform", (a, b), v)
            return v
        finally:
            self._depth -= 1

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        self._depth += 1
        try:
            v = super().gauss(mu, sigma)
            self._log("gauss", (mu, sigma), v)
            return v
        finally:
            self._depth -= 1

    def randrange(self, *args, **kwargs):
        self._depth += 1
        try:
            v = super().randrange(*args, **kwargs)
            self._log("randrange", args, v)
            return v
        finally:
            self._depth -= 1

    def randint(self, a: int, b: int) -> int:
        self._depth += 1
        try:
            v = super().randint(a, b)
            self._log("randint", (a, b), v)
            return v
        finally:
            self._depth -= 1

    def choice(self, seq):  # type: ignore[override]
        self._depth += 1
        try:
            seq_list = list(seq)  # type: ignore[arg-type]
            value = super().choice(seq_list)
            self._log("choice", (tuple(seq_list),), value)
            return value
        finally:
            self._depth -= 1


class RecordingCascade(ResistanceCascade):
    """Install the recording RNG before any agent construction draws."""

    def reset_randomizer(self, seed=None):  # type: ignore[override]
        self.random = RecordingRandom(seed)
        self._seed = seed


def _snapshot_agents(model) -> list[dict]:
    """Per-agent state in schedule order (matches insertion order)."""
    snap: list[dict] = []
    for agent in model.schedule.agents:
        row: dict[str, Any] = {
            "unique_id": agent.unique_id,
            "kind": "citizen" if isinstance(agent, Citizen) else "security",
            "pos_x": agent.pos[0],
            "pos_y": agent.pos[1],
            "condition": agent.condition,
        }
        if isinstance(agent, Citizen):
            row.update(
                private_preference=agent.private_preference,
                epsilon=agent.epsilon,
                epsilon_probability=agent.epsilon_probability,
                oppose_threshold=agent.oppose_threshold,
                active_threshold=agent.active_threshold,
                jail_sentence=agent.jail_sentence,
            )
        elif isinstance(agent, Security):
            row.update(private_preference=agent.private_preference)
        snap.append(row)
    return snap


def run_and_capture(seed: int, params: dict, steps: int) -> dict:
    model = RecordingCascade(seed=seed, **params)
    rng: RecordingRandom = model.random  # type: ignore[assignment]

    init_state = _snapshot_agents(model)

    # step_markers[k] = len(rng.log) at the START of step k.
    step_markers: list[int] = [len(rng.log)]
    for _ in range(steps):
        if not model.running:
            break
        model.step()
        step_markers.append(len(rng.log))

    final_state = _snapshot_agents(model)

    return {
        "seed": seed,
        "params": params,
        "steps_run": len(step_markers) - 1,
        "max_steps": steps,
        "init_state": init_state,
        "final_state": final_state,
        "rng_log": rng.log,
        "step_markers": step_markers,
        "revolution": bool(model.revolution),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/picked_seeds.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/captures")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    picked = json.loads(args.seeds.read_text())
    params = picked["params"]
    steps = args.steps or picked["steps"]
    seeds = [r["seed"] for r in picked["picked"]]
    if args.limit:
        seeds = seeds[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"Capturing seed={seed}...", flush=True)
        bundle = run_and_capture(seed, params, steps)
        out_path = args.out / f"capture_seed{seed:04d}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {out_path.name}  draws={len(bundle['rng_log'])}  "
              f"steps={bundle['steps_run']}  rev={bundle['revolution']}")

    print(f"\nDone. {len(seeds)} captures at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
