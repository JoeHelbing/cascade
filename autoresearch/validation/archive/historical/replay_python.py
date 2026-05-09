"""
Replay a captured Mesa run using injected RNG draws.

Reads a pickle from `capture_mesa.py`, feeds Mesa a drop-in `random.Random`
that returns the captured values in order, and emits a trace identical to
`run_python_trace.py` output. If `compare_traces.py` shows zero rows of
difference between the original trace and this replay's trace, the
decision-injection protocol is bit-exact.

This validates that the *protocol* works. Once verified in pure Python, the
same protocol can be ported into mojo_cpu (read the same log, serve values,
run Mesa's deterministic math) and yield a bit-exact mojo trace.

Usage:
    pixi run python autoresearch/validation/replay_python.py
    pixi run python autoresearch/validation/compare_traces.py \
        --ref autoresearch/validation/python_trace.parquet \
        --cand autoresearch/validation/replay_trace.parquet
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "original_python"))

from resistance_cascade.model import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]


class ReplayRandom(random.Random):
    """Serves pre-captured RNG values in sequence. Strict mode fails fast
    when the caller asks for a draw type that doesn't match the log."""

    def __init__(self, log: list[tuple[str, tuple, Any]], strict: bool = True):
        super().__init__()
        self.log = log
        self.idx = 0
        self.strict = strict

    def getrandbits(self, k: int) -> int:
        # Match RecordingRandom so Mesa's randrange path is identical across
        # capture and replay (_randbelow_with_getrandbits rather than the
        # _without variant that consumes self.random()).
        return super().getrandbits(k)

    def _pop(self, expected_method: str) -> Any:
        if self.idx >= len(self.log):
            raise RuntimeError(
                f"log exhausted at draw {self.idx} (expected {expected_method})")
        method, _args, value = self.log[self.idx]
        if self.strict and method != expected_method:
            raise RuntimeError(
                f"draw {self.idx}: replay wants {expected_method} but log "
                f"has {method}; semantics diverged"
            )
        self.idx += 1
        return value

    def random(self) -> float:
        return self._pop("random")

    def uniform(self, a: float, b: float) -> float:
        return self._pop("uniform")

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._pop("gauss")

    def randrange(self, *args, **kwargs):
        return self._pop("randrange")

    def randint(self, a: int, b: int) -> int:
        return self._pop("randint")

    def choice(self, seq):  # type: ignore[override]
        return self._pop("choice")


class ReplayCascade(ResistanceCascade):
    """Installs a ReplayRandom in place of Mesa's default RNG."""

    def __init__(self, rng: ReplayRandom, **kwargs):
        self._replay_rng = rng
        super().__init__(**kwargs)

    def reset_randomizer(self, seed=None):  # type: ignore[override]
        # Ignore the seed arg -- replay RNG is pre-loaded with captured values.
        self.random = self._replay_rng
        self._seed = seed


def replay_one(capture_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (agent_df, model_df) matching run_python_trace.py schema."""
    with capture_path.open("rb") as f:
        bundle = pickle.load(f)

    seed = bundle["seed"]
    params = bundle["params"]
    steps = bundle["max_steps"]
    log = bundle["rng_log"]

    rng = ReplayRandom(log, strict=True)
    model = ReplayCascade(rng=rng, seed=seed, **params)

    for _ in range(steps):
        if not model.running:
            break
        model.step()

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    model_df = model.datacollector.get_model_vars_dataframe().reset_index()

    # Normalize pos tuple -> pos_x/pos_y to match run_python_trace schema
    if "pos" in agent_df.columns:
        agent_df["pos_x"] = agent_df["pos"].apply(
            lambda t: t[0] if isinstance(t, tuple) else None
        )
        agent_df["pos_y"] = agent_df["pos"].apply(
            lambda t: t[1] if isinstance(t, tuple) else None
        )
        agent_df = agent_df.drop(columns=["pos"])
    agent_df["seed"] = seed
    agent_df["kind"] = agent_df["active_threshold"].apply(
        lambda v: "citizen" if pd.notna(v) else "security"
    )
    model_df["seed"] = seed
    return agent_df, model_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/captures")
    ap.add_argument("--out-agent", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/replay_trace.parquet")
    ap.add_argument("--out-model", type=Path,
                    default=REPO_ROOT / "autoresearch/validation/replay_model_trace.parquet")
    args = ap.parse_args()

    capture_files = sorted(args.captures.glob("capture_seed*.pkl"))
    if not capture_files:
        print(f"no captures at {args.captures}")
        return 1

    all_agent: list[pd.DataFrame] = []
    all_model: list[pd.DataFrame] = []
    for p in capture_files:
        print(f"replay {p.name}...", flush=True)
        a, m = replay_one(p)
        all_agent.append(a)
        all_model.append(m)

    agent_df = pd.concat(all_agent, ignore_index=True)
    model_df = pd.concat(all_model, ignore_index=True)

    agent_df.to_parquet(args.out_agent, index=False)
    model_df.to_parquet(args.out_model, index=False)
    print(f"\nwrote {args.out_agent}  rows={len(agent_df)}")
    print(f"wrote {args.out_model}  rows={len(model_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
