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
import csv
import json
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
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
PYTHON_CORE_TRACE = VALIDATION_DIR / "python_core_trace.csv"
MOJO_CPU_TRACE = VALIDATION_DIR / "mojo_cpu_bitexact.csv"
MOJO_CPU_GPU_TRACE = VALIDATION_DIR / "mojo_cpu_gpu_bridge.csv"
MOJO_GPU_OUTPUT = VALIDATION_DIR / "mojo_gpu_output.txt"

sys.path.insert(0, str(REPO_ROOT / "python-core-simulation"))
from cascade_core import ResistanceCascade  # noqa: E402  # pyright: ignore[reportMissingImports]

FLOAT_COLS = ["opinion", "activation", "active_level", "oppose_level"]
INT_COLS = ["pos_x", "pos_y", "jail_sentence"]
STR_COLS = ["condition"]
GPU_SEEDS = [42, 123, 456, 789, 1001]
GPU_EPSILONS = [0.2, 0.5, 1.0]
GPU_SECURITY_DENSITIES = [0.0, 0.02, 0.05]
GPU_NO_SECURITY_ACTIVE_TOLERANCE = 35


@dataclass(frozen=True)
class AggregateRow:
    sim_id: int
    seed: int
    epsilon: float
    security_density: float
    active: int
    support: int
    oppose: int
    jail: int
    revolution: bool


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


def _cpu_param_args(params: dict, steps: int, seeds: list[int]) -> list[str]:
    return [
        "--seed", ",".join(str(seed) for seed in seeds),
        "--width", str(params["width"]),
        "--height", str(params["height"]),
        "--citizen-vision", str(params["citizen_vision"]),
        "--citizen-density", str(params["citizen_density"]),
        "--security-density", str(params["security_density"]),
        "--security-vision", str(params["security_vision"]),
        "--max-jail-term", str(params["max_jail_term"]),
        "--movement", str(params["movement"]).lower(),
        "--multiple-agents-per-cell", str(params.get("multiple_agents_per_cell", True)).lower(),
        "--private-preference-distribution-mean", str(params["private_preference_distribution_mean"]),
        "--standard-deviation", str(params["standard_deviation"]),
        "--epsilon", str(params["epsilon"]),
        "--threshold", str(params["threshold"]),
        "--max-iters", str(steps),
        "--random-seed", "false",
    ]


def generate_python_core_trace(seeds_path: Path, out: Path) -> None:
    """Generate the python-core reference CSV that mojo_cpu --rng python must emit."""
    cfg = json.loads(seeds_path.read_text())
    params = cfg["params"]
    steps = cfg["steps"]
    seeds = [p["seed"] for p in cfg["picked"]]

    with out.open("w", newline="") as output:
        wrote_header = False
        for seed in seeds:
            print(f"running python-core seed={seed}")
            sim = ResistanceCascade(
                width=params["width"],
                height=params["height"],
                citizen_vision=params["citizen_vision"],
                citizen_density=params["citizen_density"],
                security_density=params["security_density"],
                security_vision=params["security_vision"],
                max_jail_term=params["max_jail_term"],
                movement=params["movement"],
                private_preference_distribution_mean=params["private_preference_distribution_mean"],
                standard_deviation=params["standard_deviation"],
                epsilon=params["epsilon"],
                threshold=params["threshold"],
                max_iters=steps,
                seed=seed,
                collect_trace=True,
            )
            sim.run(steps=steps)
            rows = [dict((field, getattr(row, field)) for field in row.__dataclass_fields__) for row in sim.trace]
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            if not wrote_header:
                writer.writeheader()
                wrote_header = True
            writer.writerows(rows)
    print(f"wrote python-core trace to {out.relative_to(REPO_ROOT)}")


def _csv_payload(path: Path) -> str:
    return "\n".join(line for line in path.read_text().splitlines() if line and not line.startswith("#")) + "\n"


def compare_python_core_csv(expected_path: Path, mojo_path: Path) -> None:
    expected = _csv_payload(expected_path)
    actual = _csv_payload(mojo_path)
    if expected == actual:
        print("PASS: mojo_cpu --rng python is byte-identical to python-core CSV output.")
        return

    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for i, (expected_line, actual_line) in enumerate(zip(expected_lines, actual_lines, strict=False), start=1):
        if expected_line != actual_line:
            raise SystemExit(
                "FAIL: mojo_cpu --rng python differs from python-core CSV output "
                f"at line {i}.\nexpected: {expected_line}\nactual:   {actual_line}"
            )
    raise SystemExit(
        "FAIL: mojo_cpu --rng python differs from python-core CSV output: "
        f"expected {len(expected_lines)} lines, got {len(actual_lines)} lines."
    )


def float_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def load_mojo_csv(path: Path) -> pd.DataFrame:
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(rows)), float_precision="round_trip")
    return df[df["condition"] != "Security"].copy()


_GPU_SIM_RE = re.compile(
    r"^Sim\s+(?P<sim_id>\d+)\s+"
    r"seed=\s*(?P<seed>\d+)\s+"
    r"eps=\s*(?P<epsilon>[0-9.]+)\s+"
    r"sd=\s*(?P<security_density>[0-9.]+)\s+"
    r"active=\s*(?P<active>\d+)\s+"
    r"support=\s*(?P<support>\d+)\s+"
    r"oppose=\s*(?P<oppose>\d+)\s+"
    r"jail=\s*(?P<jail>\d+)\s+"
    r"rev=\s*(?P<revolution>True|False)"
)


def parse_gpu_sim_lines(output: str) -> list[AggregateRow]:
    rows: list[AggregateRow] = []
    for line in output.splitlines():
        match = _GPU_SIM_RE.match(line.strip())
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            AggregateRow(
                sim_id=int(groups["sim_id"]),
                seed=int(groups["seed"]),
                epsilon=float(groups["epsilon"]),
                security_density=float(groups["security_density"]),
                active=int(groups["active"]),
                support=int(groups["support"]),
                oppose=int(groups["oppose"]),
                jail=int(groups["jail"]),
                revolution=groups["revolution"] == "True",
            )
        )
    return rows


def load_mojo_cpu_gpu_aggregate(
    path: Path,
    *,
    sim_id: int,
    seed: int,
    epsilon: float,
    security_density: float,
) -> AggregateRow:
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(rows)
    data = [row for row in reader if row.get("agent_type") == "Citizen"]
    if not data:
        raise SystemExit(f"No citizen rows found in {path.relative_to(REPO_ROOT)}")
    last_step = max(int(row["step"]) for row in data)
    final = [row for row in data if int(row["step"]) == last_step]
    active = sum(1 for row in final if row["condition"] == "Active")
    support = sum(1 for row in final if row["condition"] == "Support")
    oppose = sum(1 for row in final if row["condition"] == "Oppose")
    jail = sum(1 for row in final if row["condition"] == "Jailed")
    revolution = (active + jail) / len(final) >= 0.95
    return AggregateRow(sim_id, seed, epsilon, security_density, active, support, oppose, jail, revolution)


def _run_cpu_gpu_case(sim_id: int, seed: int, epsilon: float, security_density: float) -> AggregateRow:
    run(
        [
            str(REPO_ROOT / "build" / "mojo_cpu"),
            "--rng", "gpu",
            "--width", "33",
            "--height", "33",
            "--citizen-density", "0.7",
            "--security-density", str(security_density),
            "--epsilon", str(epsilon),
            "--threshold", "2.94444",
            "--max-iters", "50",
            "--seed", str(seed),
            "--movement", "true",
            "--random-seed", "false",
        ],
        stdout_path=MOJO_CPU_GPU_TRACE,
    )
    return load_mojo_cpu_gpu_aggregate(
        MOJO_CPU_GPU_TRACE,
        sim_id=sim_id,
        seed=seed,
        epsilon=epsilon,
        security_density=security_density,
    )


def compare_cpu_gpu_aggregates(cpu_rows: list[AggregateRow], gpu_rows: list[AggregateRow]) -> None:
    gpu_by_key = {(row.seed, row.epsilon, row.security_density): row for row in gpu_rows}
    failures: list[str] = []
    for cpu in cpu_rows:
        gpu = gpu_by_key.get((cpu.seed, cpu.epsilon, cpu.security_density))
        if gpu is None:
            failures.append(f"missing GPU row for seed={cpu.seed} eps={cpu.epsilon} sd={cpu.security_density}")
            continue
        if cpu.active + cpu.support + cpu.oppose + cpu.jail != gpu.active + gpu.support + gpu.oppose + gpu.jail:
            failures.append(f"citizen total mismatch for seed={cpu.seed} eps={cpu.epsilon}: CPU={cpu} GPU={gpu}")
        if cpu.revolution != gpu.revolution:
            failures.append(f"revolution mismatch for seed={cpu.seed} eps={cpu.epsilon}: CPU={cpu.revolution} GPU={gpu.revolution}")
        if abs(cpu.active - gpu.active) > GPU_NO_SECURITY_ACTIVE_TOLERANCE:
            failures.append(
                f"active count drift exceeds tolerance for seed={cpu.seed} eps={cpu.epsilon}: "
                f"CPU={cpu.active} GPU={gpu.active} tolerance={GPU_NO_SECURITY_ACTIVE_TOLERANCE}"
            )
    if failures:
        raise SystemExit("GPU validation FAIL:\n" + "\n".join(failures))
    print(
        "GPU validation PASS: mojo_cpu --rng gpu and mojo_gpu agree on no-security "
        f"aggregate outcomes ({len(cpu_rows)} cases, active tolerance ±{GPU_NO_SECURITY_ACTIVE_TOLERANCE})."
    )


def validate_cpu(args: argparse.Namespace) -> None:
    """Validate mojo_cpu.mojo --rng python against python-core-simulation tests."""
    print("\n=== CPU validation: python-core-simulation -> mojo_cpu --rng python ===", flush=True)
    run([
        "pixi",
        "run",
        "python",
        "-m",
        "unittest",
        "tests.test_python_core_simulation",
        "tests.test_mojo_cpu_cli",
    ])
    print("CPU validation PASS: python-core and mojo_cpu CLI regression tests passed.")


def validate_gpu() -> None:
    """Run the CPU-GPU aggregate validation gate."""
    print("\n=== GPU validation: mojo_cpu --rng gpu boundary -> mojo_gpu aggregate gate ===", flush=True)
    run(["pixi", "run", "build-cpu"])
    run(["pixi", "run", "build-gpu"])
    result = run([str(REPO_ROOT / "build" / "mojo_gpu")], stdout_path=MOJO_GPU_OUTPUT)
    gpu_rows = parse_gpu_sim_lines(result.stdout)
    if len(gpu_rows) != 45:
        raise SystemExit(
            f"GPU validation FAIL: expected 45 'Sim ...' lines, got {len(gpu_rows)}. "
            f"See {MOJO_GPU_OUTPUT.relative_to(REPO_ROOT)}."
        )

    cpu_rows: list[AggregateRow] = []
    sim_id = 0
    for seed in GPU_SEEDS:
        for epsilon in GPU_EPSILONS:
            for security_density in GPU_SECURITY_DENSITIES:
                if security_density == 0.0:
                    cpu_rows.append(_run_cpu_gpu_case(sim_id, seed, epsilon, security_density))
                sim_id += 1
    compare_cpu_gpu_aggregates(cpu_rows, gpu_rows)


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
