#!/usr/bin/env python3
"""Run the canonical Cascade validation pipeline.

The validation chain is intentionally split at implementation boundaries:

    original_python/ -> mojo_cpu.mojo -> mojo_gpu.mojo

CPU validation is bit-exact against the Mesa reference traces. GPU validation is
currently an aggregate smoke/fingerprint gate around the self-contained GPU
kernel output; it is not Mesa bit-exact because the GPU kernel uses GPU-safe RNG
and Float32 math.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPO_ROOT / "autoresearch" / "validation"
PYTHON_TRACE = VALIDATION_DIR / "python_trace.parquet"
MOJO_CPU_TRACE = VALIDATION_DIR / "mojo_cpu_bitexact.csv"
MOJO_GPU_OUTPUT = VALIDATION_DIR / "mojo_gpu_output.txt"


def run(cmd: list[str], *, stdout_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command from the repo root, optionally teeing stdout to a file."""
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


def validate_cpu(skip_python_trace: bool) -> None:
    """Validate mojo_cpu.mojo against the original Mesa Python trace."""
    print("\n=== CPU validation: original_python -> mojo_cpu ===", flush=True)
    if not skip_python_trace:
        run(["uv", "run", "autoresearch/validation/run_python_trace.py"])
    elif not PYTHON_TRACE.exists():
        raise SystemExit(
            "--skip-python-trace was set, but "
            f"{PYTHON_TRACE.relative_to(REPO_ROOT)} does not exist."
        )

    run(["pixi", "run", "build-cpu"])
    run([str(REPO_ROOT / "build" / "mojo_cpu")], stdout_path=MOJO_CPU_TRACE)
    run(
        [
            "uv",
            "run",
            "autoresearch/validation/compare_bitexact.py",
            "--mesa",
            str(PYTHON_TRACE.relative_to(REPO_ROOT)),
            "--mojo",
            str(MOJO_CPU_TRACE.relative_to(REPO_ROOT)),
        ]
    )
    print("CPU validation PASS: mojo_cpu matches Mesa on tracked per-agent columns.")


def validate_gpu() -> None:
    """Run the current GPU aggregate validation/smoke gate."""
    print("\n=== GPU validation: mojo_cpu boundary -> mojo_gpu aggregate gate ===", flush=True)
    run(["pixi", "run", "build-gpu"])
    result = run([str(REPO_ROOT / "build" / "mojo_gpu")], stdout_path=MOJO_GPU_OUTPUT)
    sim_lines = [line for line in result.stdout.splitlines() if line.startswith("Sim ")]
    if len(sim_lines) != 45:
        raise SystemExit(
            f"GPU validation FAIL: expected 45 'Sim ...' lines, got {len(sim_lines)}. "
            f"See {MOJO_GPU_OUTPUT.relative_to(REPO_ROOT)}."
        )
    print("GPU validation PASS: produced the expected 45 aggregate simulation lines.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["cpu", "gpu", "all"],
        default="all",
        help="Which validation boundary to run.",
    )
    parser.add_argument(
        "--skip-python-trace",
        action="store_true",
        help="Reuse an existing python_trace.parquet instead of regenerating Mesa traces.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"cpu", "all"}:
        validate_cpu(skip_python_trace=args.skip_python_trace)
    if args.stage in {"gpu", "all"}:
        validate_gpu()
    print("\nValidation pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
