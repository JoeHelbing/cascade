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
import hashlib
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
PYTHON_CORE_TRACE = VALIDATION_DIR / "python_core_trace.parquet"
MOJO_CPU_TRACE = VALIDATION_DIR / "mojo_cpu_python_rng_trace.parquet"
MOJO_CPU_GPU_TRACE = VALIDATION_DIR / "mojo_cpu_gpu_rng_trace.parquet"
PYTHON_CORE_STATE_TRACE = VALIDATION_DIR / "python_core_state_trace.parquet"
MOJO_CPU_PYTHON_STATE_TRACE = VALIDATION_DIR / "mojo_cpu_python_rng_state_trace.parquet"
MOJO_CPU_MOJO_STATE_TRACE = VALIDATION_DIR / "mojo_cpu_mojo_rng_state_trace.parquet"
MOJO_GPU_STATE_TRACE = VALIDATION_DIR / "mojo_gpu_state_trace.parquet"
MOJO_CPU_GPU_AGGREGATE = VALIDATION_DIR / "mojo_cpu_gpu_rng_aggregate.parquet"
MOJO_GPU_AGGREGATE = VALIDATION_DIR / "mojo_gpu_aggregate.parquet"
MOJO_GPU_OUTPUT = VALIDATION_DIR / "mojo_gpu_output.txt"
SHA_MANIFEST = VALIDATION_DIR / "validation_sha256.json"

sys.path.insert(0, str(REPO_ROOT / "python-core-simulation"))
from cascade_core import ResistanceCascade, TRACE_FIELDS  # noqa: E402  # pyright: ignore[reportMissingImports]

FLOAT_COLS = ["opinion", "activation", "active_level", "oppose_level"]
INT_COLS = ["pos_x", "pos_y", "jail_sentence"]
STR_COLS = ["condition"]
TRACE_FLOAT_FIELDS = [
    "opinion",
    "activation",
    "private_preference",
    "epsilon",
    "oppose_threshold",
    "active_threshold",
    "perception",
    "arrest_prob",
    "active_level",
    "oppose_level",
]
TRACE_INT_FIELDS = [
    "step",
    "agent_id",
    "x",
    "y",
    "jail_sentence",
    "active_in_vision",
    "oppose_in_vision",
    "support_in_vision",
    "security_in_vision",
]
TRACE_BOOL_FIELDS = ["flip", "ever_flipped"]
TRACE_STRING_FIELDS = ["agent_type", "condition"]
GPU_SEEDS = [42, 123, 456, 789, 1001]
GPU_EPSILONS = [0.2, 0.5, 1.0]
GPU_SECURITY_DENSITIES = [0.0, 0.02, 0.05]
GPU_NO_SECURITY_ACTIVE_TOLERANCE = 35
GPU_TRACE_VALIDATION_SEED = 16
GPU_TRACE_VALIDATION_EPSILON = 0.5
GPU_TRACE_VALIDATION_SECURITY_DENSITY = 0.0
GPU_TRACE_VALIDATION_THRESHOLD = 2.5
GPU_TRACE_VALIDATION_STEPS = 500


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


@dataclass(frozen=True)
class TraceChunk:
    sim_id: int
    seed: int
    epsilon: float
    security_density: float
    rows: list[dict]


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


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _none_if_empty(value: object) -> object | None:
    return None if value is None or value == "" else value


def _as_int(value: object) -> int | None:
    value = _none_if_empty(value)
    return None if value is None else int(value)


def _as_float(value: object) -> float | None:
    value = _none_if_empty(value)
    return None if value is None else float(value)


def _as_bool(value: object) -> bool | None:
    value = _none_if_empty(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"Expected boolean-ish value, got {value!r}")


def _trace_schema() -> pa.Schema:
    fields = [
        pa.field("sim_id", pa.int64()),
        pa.field("seed", pa.int64()),
        pa.field("epsilon_config", pa.float64()),
        pa.field("security_density_config", pa.float64()),
    ]
    for field in TRACE_FIELDS:
        if field in TRACE_INT_FIELDS:
            fields.append(pa.field(field, pa.int64()))
        elif field in TRACE_FLOAT_FIELDS:
            fields.append(pa.field(field, pa.float64()))
        elif field in TRACE_BOOL_FIELDS:
            fields.append(pa.field(field, pa.bool_()))
        else:
            fields.append(pa.field(field, pa.string()))
    for field in TRACE_FLOAT_FIELDS:
        fields.append(pa.field(f"{field}_bits", pa.uint64()))
    return pa.schema(fields)


def _normalise_trace_row(
    row: dict,
    *,
    sim_id: int,
    seed: int,
    epsilon: float,
    security_density: float,
) -> dict:
    out: dict[str, object | None] = {
        "sim_id": sim_id,
        "seed": seed,
        "epsilon_config": epsilon,
        "security_density_config": security_density,
    }
    for field in TRACE_FIELDS:
        value = row.get(field)
        if field in TRACE_INT_FIELDS:
            out[field] = _as_int(value)
        elif field in TRACE_FLOAT_FIELDS:
            out[field] = _as_float(value)
        elif field in TRACE_BOOL_FIELDS:
            out[field] = _as_bool(value)
        else:
            value = _none_if_empty(value)
            out[field] = None if value is None else str(value)
    for field in TRACE_FLOAT_FIELDS:
        value = out[field]
        out[f"{field}_bits"] = None if value is None else float_bits(float(value))
    return out


def write_trace_parquet(path: Path, chunks: list[TraceChunk] | list[tuple[int, int, float, float, list[dict]]]) -> str:
    """Overwrite a trace artifact and write all simulation chunks into one Parquet file."""
    path.unlink(missing_ok=True)
    normalised: list[dict] = []
    for chunk in chunks:
        if isinstance(chunk, TraceChunk):
            sim_id, seed, epsilon, security_density, rows = (
                chunk.sim_id,
                chunk.seed,
                chunk.epsilon,
                chunk.security_density,
                chunk.rows,
            )
        else:
            sim_id, seed, epsilon, security_density, rows = chunk
        normalised.extend(
            _normalise_trace_row(
                row,
                sim_id=sim_id,
                seed=seed,
                epsilon=epsilon,
                security_density=security_density,
            )
            for row in rows
        )
    table = pa.Table.from_pylist(normalised, schema=_trace_schema())
    pq.write_table(table, path, compression="NONE", use_dictionary=False, write_statistics=False)
    digest = sha256_file(path)
    print(f"wrote {display_path(path)} sha256={digest}")
    return digest


def split_mojo_trace_output_by_seed(
    output: str,
    seeds: list[int],
    *,
    epsilon: float = 0.0,
    security_density: float = 0.0,
) -> list[TraceChunk]:
    """Split concatenated mojo_cpu CSV stdout into seed chunks by step-0 reset."""
    data_lines = [line for line in output.splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(data_lines)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    previous_step: int | None = None
    for row in reader:
        step = int(row["step"])
        if current and step == 0 and previous_step != 0:
            chunks.append(current)
            current = []
        current.append(row)
        previous_step = step
    if current:
        chunks.append(current)
    if len(chunks) != len(seeds):
        raise SystemExit(f"Expected {len(seeds)} mojo_cpu trace chunks, found {len(chunks)}")
    return [
        TraceChunk(sim_id=sim_id, seed=seed, epsilon=epsilon, security_density=security_density, rows=rows)
        for sim_id, (seed, rows) in enumerate(zip(seeds, chunks, strict=True))
    ]


def _state_trace_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sim_id", pa.int64()),
            pa.field("seed", pa.int64()),
            pa.field("epsilon_config", pa.float64()),
            pa.field("security_density_config", pa.float64()),
            pa.field("step", pa.int64()),
            pa.field("agent_id", pa.int64()),
            pa.field("agent_type", pa.string()),
            pa.field("x", pa.int64()),
            pa.field("y", pa.int64()),
            pa.field("condition", pa.string()),
        ]
    )


def pad_state_trace_chunk(chunk: TraceChunk, final_step: int) -> TraceChunk:
    rows = list(chunk.rows)
    if not rows:
        return chunk
    max_step = max(int(row["step"]) for row in rows)
    if max_step >= final_step:
        return chunk
    final_rows = [row for row in rows if int(row["step"]) == max_step]
    for step in range(max_step + 1, final_step + 1):
        for row in final_rows:
            padded = dict(row)
            padded["step"] = str(step)
            rows.append(padded)
    return TraceChunk(
        sim_id=chunk.sim_id,
        seed=chunk.seed,
        epsilon=chunk.epsilon,
        security_density=chunk.security_density,
        rows=rows,
    )


def write_state_trace_parquet(path: Path, chunks: list[TraceChunk]) -> str:
    path.unlink(missing_ok=True)
    rows: list[dict] = []
    for chunk in chunks:
        for row in chunk.rows:
            rows.append(
                {
                    "sim_id": chunk.sim_id,
                    "seed": chunk.seed,
                    "epsilon_config": chunk.epsilon,
                    "security_density_config": chunk.security_density,
                    "step": _as_int(row.get("step")),
                    "agent_id": _as_int(row.get("agent_id")),
                    "agent_type": str(row.get("agent_type", "Citizen")),
                    "x": _as_int(row.get("x")),
                    "y": _as_int(row.get("y")),
                    "condition": str(row.get("condition")),
                }
            )
    table = pa.Table.from_pylist(rows, schema=_state_trace_schema())
    pq.write_table(table, path, compression="NONE", use_dictionary=False, write_statistics=False)
    digest = sha256_file(path)
    print(f"wrote {display_path(path)} sha256={digest}")
    return digest


def write_aggregate_parquet(path: Path, rows: list[AggregateRow]) -> str:
    path.unlink(missing_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "sim_id": row.sim_id,
                "seed": row.seed,
                "epsilon": row.epsilon,
                "security_density": row.security_density,
                "active": row.active,
                "support": row.support,
                "oppose": row.oppose,
                "jail": row.jail,
                "revolution": row.revolution,
            }
            for row in rows
        ],
        schema=pa.schema(
            [
                pa.field("sim_id", pa.int64()),
                pa.field("seed", pa.int64()),
                pa.field("epsilon", pa.float64()),
                pa.field("security_density", pa.float64()),
                pa.field("active", pa.int64()),
                pa.field("support", pa.int64()),
                pa.field("oppose", pa.int64()),
                pa.field("jail", pa.int64()),
                pa.field("revolution", pa.bool_()),
            ]
        ),
    )
    pq.write_table(table, path, compression="NONE", use_dictionary=False, write_statistics=False)
    digest = sha256_file(path)
    print(f"wrote {display_path(path)} sha256={digest}")
    return digest


def write_sha_manifest(entries: dict[str, str]) -> None:
    SHA_MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    print(f"wrote {SHA_MANIFEST.relative_to(REPO_ROOT)}")


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


def generate_python_core_trace(seeds_path: Path, out: Path) -> str:
    """Generate the python-core reference Parquet that mojo_cpu --rng python must emit."""
    cfg = json.loads(seeds_path.read_text())
    params = cfg["params"]
    steps = cfg["steps"]
    seeds = [p["seed"] for p in cfg["picked"]]

    chunks: list[TraceChunk] = []
    for sim_id, seed in enumerate(seeds):
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
        rows = [dict((field, getattr(row, field)) for field in TRACE_FIELDS) for row in sim.trace]
        chunks.append(
            TraceChunk(
                sim_id=sim_id,
                seed=seed,
                epsilon=float(params["epsilon"]),
                security_density=float(params["security_density"]),
                rows=rows,
            )
        )
    return write_trace_parquet(out, chunks)


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


def parse_gpu_trace_lines(output: str) -> list[TraceChunk]:
    rows_by_key: dict[tuple[int, int, float, float], list[dict]] = {}
    for line in output.splitlines():
        if not line.startswith("TRACE,"):
            continue
        _, sim_id_s, seed_s, epsilon_s, security_density_s, step_s, agent_id_s, x_s, y_s, condition = line.split(",", 9)
        key = (int(sim_id_s), int(seed_s), float(epsilon_s), float(security_density_s))
        rows_by_key.setdefault(key, []).append(
            {
                "step": step_s,
                "agent_id": agent_id_s,
                "agent_type": "Citizen",
                "x": x_s,
                "y": y_s,
                "condition": condition,
            }
        )
    return [
        TraceChunk(sim_id=sim_id, seed=seed, epsilon=epsilon, security_density=security_density, rows=rows)
        for (sim_id, seed, epsilon, security_density), rows in sorted(rows_by_key.items())
    ]


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


def _aggregate_from_trace_rows(
    data: list[dict],
    *,
    sim_id: int,
    seed: int,
    epsilon: float,
    security_density: float,
) -> AggregateRow:
    citizens = [row for row in data if row.get("agent_type") == "Citizen"]
    if not citizens:
        raise SystemExit("No citizen rows found in mojo_cpu trace output")
    last_step = max(int(row["step"]) for row in citizens)
    final = [row for row in citizens if int(row["step"]) == last_step]
    active = sum(1 for row in final if row["condition"] == "Active")
    support = sum(1 for row in final if row["condition"] == "Support")
    oppose = sum(1 for row in final if row["condition"] == "Oppose")
    jail = sum(1 for row in final if row["condition"] == "Jailed")
    revolution = (active + jail) / len(final) >= 0.95
    return AggregateRow(sim_id, seed, epsilon, security_density, active, support, oppose, jail, revolution)


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
    return _aggregate_from_trace_rows(
        list(reader),
        sim_id=sim_id,
        seed=seed,
        epsilon=epsilon,
        security_density=security_density,
    )


def _run_cpu_gpu_case(sim_id: int, seed: int, epsilon: float, security_density: float) -> tuple[TraceChunk, AggregateRow]:
    result = run_capture(
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
        ]
    )
    data_lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    rows = list(csv.DictReader(data_lines))
    chunk = TraceChunk(sim_id=sim_id, seed=seed, epsilon=epsilon, security_density=security_density, rows=rows)
    aggregate = _aggregate_from_trace_rows(
        rows,
        sim_id=sim_id,
        seed=seed,
        epsilon=epsilon,
        security_density=security_density,
    )
    return chunk, aggregate


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


def validate_cpu(args: argparse.Namespace) -> dict[str, str]:
    """Validate mojo_cpu.mojo --rng python against python-core-simulation Parquet."""
    print("\n=== CPU validation: python-core-simulation -> mojo_cpu --rng python ===", flush=True)
    run(["pixi", "run", "python", "tests/test_python_core_simulation.py"])
    run([
        "pixi",
        "run",
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_mojo_cpu_cli.py",
    ])
    run(["pixi", "run", "build-cpu"])

    cfg = json.loads(args.seeds.read_text())
    params = cfg["params"]
    steps = cfg["steps"]
    seeds = [p["seed"] for p in cfg["picked"]]

    python_digest = generate_python_core_trace(args.seeds, PYTHON_CORE_TRACE)
    mojo_chunks: list[TraceChunk] = []
    for sim_id, seed in enumerate(seeds):
        result = run_capture(
            [str(REPO_ROOT / "build" / "mojo_cpu"), "--rng", "python", *_cpu_param_args(params, steps, [seed])]
        )
        chunk = split_mojo_trace_output_by_seed(
            result.stdout,
            [seed],
            epsilon=float(params["epsilon"]),
            security_density=float(params["security_density"]),
        )[0]
        mojo_chunks.append(
            TraceChunk(
                sim_id=sim_id,
                seed=chunk.seed,
                epsilon=chunk.epsilon,
                security_density=chunk.security_density,
                rows=chunk.rows,
            )
        )
    mojo_digest = write_trace_parquet(MOJO_CPU_TRACE, mojo_chunks)
    if python_digest != mojo_digest:
        raise SystemExit(
            "CPU validation FAIL: python-core and mojo_cpu --rng python Parquet SHA256 differ\n"
            f"python-core: {python_digest}\nmojo_cpu:    {mojo_digest}"
        )
    print(f"CPU validation PASS: Parquet SHA256 match ({python_digest}).")
    return {
        "python_core_trace.parquet": python_digest,
        "mojo_cpu_python_rng_trace.parquet": mojo_digest,
    }


def _gpu_trace_params() -> dict:
    return {
        "width": 33,
        "height": 33,
        "citizen_vision": 7,
        "citizen_density": 0.7,
        "security_density": GPU_TRACE_VALIDATION_SECURITY_DENSITY,
        "security_vision": 7,
        "max_jail_term": 100,
        "movement": True,
        "multiple_agents_per_cell": True,
        "private_preference_distribution_mean": 0.0,
        "standard_deviation": 1.0,
        "epsilon": GPU_TRACE_VALIDATION_EPSILON,
        "threshold": GPU_TRACE_VALIDATION_THRESHOLD,
        "max_iters": GPU_TRACE_VALIDATION_STEPS,
    }


def _run_python_state_trace_case() -> TraceChunk:
    params = _gpu_trace_params()
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
        max_iters=params["max_iters"],
        seed=GPU_TRACE_VALIDATION_SEED,
        collect_trace=True,
    )
    sim.run(steps=params["max_iters"])
    rows = [dict((field, getattr(row, field)) for field in TRACE_FIELDS) for row in sim.trace]
    return pad_state_trace_chunk(
        TraceChunk(
            sim_id=0,
            seed=GPU_TRACE_VALIDATION_SEED,
            epsilon=GPU_TRACE_VALIDATION_EPSILON,
            security_density=GPU_TRACE_VALIDATION_SECURITY_DENSITY,
            rows=rows,
        ),
        GPU_TRACE_VALIDATION_STEPS,
    )


def _run_mojo_cpu_state_trace_case(rng_mode: str) -> TraceChunk:
    params = _gpu_trace_params()
    result = run_capture(
        [
            str(REPO_ROOT / "build" / "mojo_cpu"),
            "--rng", rng_mode,
            *_cpu_param_args(params, params["max_iters"], [GPU_TRACE_VALIDATION_SEED]),
        ]
    )
    chunk = split_mojo_trace_output_by_seed(
        result.stdout,
        [GPU_TRACE_VALIDATION_SEED],
        epsilon=GPU_TRACE_VALIDATION_EPSILON,
        security_density=GPU_TRACE_VALIDATION_SECURITY_DENSITY,
    )[0]
    return pad_state_trace_chunk(
        TraceChunk(
            sim_id=0,
            seed=chunk.seed,
            epsilon=chunk.epsilon,
            security_density=chunk.security_density,
            rows=chunk.rows,
        ),
        GPU_TRACE_VALIDATION_STEPS,
    )


def _run_gpu_state_trace_case() -> TraceChunk:
    result = run([str(REPO_ROOT / "build" / "mojo_gpu"), "--trace-validation"], stdout_path=MOJO_GPU_OUTPUT)
    chunks = parse_gpu_trace_lines(result.stdout)
    if len(chunks) != 1:
        raise SystemExit(f"Expected one GPU trace chunk, found {len(chunks)}")
    return chunks[0]


def validate_gpu() -> dict[str, str]:
    """Run the CPU-GPU aggregate validation gate and write Parquet artifacts."""
    print("\n=== GPU validation: mojo_cpu --rng mojo boundary -> mojo_gpu trace gate ===", flush=True)
    run(["pixi", "run", "build-cpu"])
    run(["pixi", "run", "build-gpu"])

    python_state_digest = write_state_trace_parquet(PYTHON_CORE_STATE_TRACE, [_run_python_state_trace_case()])
    mojo_cpu_python_state_digest = write_state_trace_parquet(
        MOJO_CPU_PYTHON_STATE_TRACE, [_run_mojo_cpu_state_trace_case("python")]
    )
    mojo_cpu_mojo_state_digest = write_state_trace_parquet(
        MOJO_CPU_MOJO_STATE_TRACE, [_run_mojo_cpu_state_trace_case("mojo")]
    )
    gpu_state_digest = write_state_trace_parquet(MOJO_GPU_STATE_TRACE, [_run_gpu_state_trace_case()])
    if mojo_cpu_mojo_state_digest != gpu_state_digest:
        raise SystemExit(
            "GPU trace validation FAIL: mojo_cpu --rng mojo and mojo_gpu state trace SHA256 differ\n"
            f"mojo_cpu --rng mojo: {mojo_cpu_mojo_state_digest}\n"
            f"mojo_gpu:           {gpu_state_digest}"
        )

    result = run([str(REPO_ROOT / "build" / "mojo_gpu")], stdout_path=MOJO_GPU_OUTPUT)
    gpu_rows = parse_gpu_sim_lines(result.stdout)
    if len(gpu_rows) != 45:
        raise SystemExit(
            f"GPU validation FAIL: expected 45 'Sim ...' lines, got {len(gpu_rows)}. "
            f"See {MOJO_GPU_OUTPUT.relative_to(REPO_ROOT)}."
        )

    cpu_trace_chunks: list[TraceChunk] = []
    cpu_rows: list[AggregateRow] = []
    sim_id = 0
    for seed in GPU_SEEDS:
        for epsilon in GPU_EPSILONS:
            for security_density in GPU_SECURITY_DENSITIES:
                if security_density == 0.0:
                    trace_chunk, aggregate = _run_cpu_gpu_case(sim_id, seed, epsilon, security_density)
                    cpu_trace_chunks.append(trace_chunk)
                    cpu_rows.append(aggregate)
                sim_id += 1
    trace_digest = write_trace_parquet(MOJO_CPU_GPU_TRACE, cpu_trace_chunks)
    cpu_aggregate_digest = write_aggregate_parquet(MOJO_CPU_GPU_AGGREGATE, cpu_rows)
    gpu_aggregate_digest = write_aggregate_parquet(MOJO_GPU_AGGREGATE, gpu_rows)
    compare_cpu_gpu_aggregates(cpu_rows, gpu_rows)
    return {
        "python_core_state_trace.parquet": python_state_digest,
        "mojo_cpu_python_rng_state_trace.parquet": mojo_cpu_python_state_digest,
        "mojo_cpu_mojo_rng_state_trace.parquet": mojo_cpu_mojo_state_digest,
        "mojo_gpu_state_trace.parquet": gpu_state_digest,
        "mojo_cpu_gpu_rng_trace.parquet": trace_digest,
        "mojo_cpu_gpu_rng_aggregate.parquet": cpu_aggregate_digest,
        "mojo_gpu_aggregate.parquet": gpu_aggregate_digest,
    }


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
    sha_entries: dict[str, str] = {}
    if args.stage in {"cpu", "all"}:
        sha_entries.update(validate_cpu(args))
    if args.stage in {"gpu", "all"}:
        sha_entries.update(validate_gpu())
    write_sha_manifest(sha_entries)
    print("\nValidation pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
