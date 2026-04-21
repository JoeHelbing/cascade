"""
7D Coarse Parameter Sweep Orchestrator.

Generates the 7D parameter grid, writes params.bin for the Mojo GPU runner,
reads results, computes per-sim summaries, and inserts into SQLite.

This replaces manifold_search_gpu.mojo's Python interop with a clean
Mojo-binary + Python-orchestrator architecture.

Usage:
    cd mojo_cascade
    pixi run python run_7d_sweep.py [--steps 100] [--points 7] [--seeds 10]
"""

import argparse
import itertools
import struct
import subprocess
import time
from pathlib import Path

import apsw
import numpy as np

GRID_SIZE = 33 * 33  # 1089
MAX_AGENTS = 1024
N_STEP_FIELDS = 5
RUNNER_BIN = "./cascade_gpu_runner"


def build_param_grid(n_points: int, n_seeds: int, num_steps: int):
    """Build 7D parameter grid. Returns (params_array, param_labels)."""

    def linspace(lo, hi, n):
        if n == 1:
            return [0.5 * (lo + hi)]
        return [lo + i * (hi - lo) / (n - 1) for i in range(n)]

    pp_means = linspace(-1.0, 1.0, n_points)
    sec_densities = linspace(0.0, 0.10, n_points)
    epsilons = linspace(0.01, 2.0, n_points)
    thresholds = linspace(1.0, 6.0, n_points)
    citizen_densities = linspace(0.3, 0.9, n_points)
    max_jails = [int(10 + i * 190 / (n_points - 1)) for i in range(n_points)] if n_points > 1 else [100]
    visions = [int(1 + i * 9 / (n_points - 1)) for i in range(n_points)] if n_points > 1 else [5]

    seeds = [42 + i * 7919 for i in range(n_seeds)]

    print(f"Parameter grid: {n_points}^7 = {n_points**7:,} configs x {n_seeds} seeds")
    print(f"  pp_mean:          {pp_means[0]:.2f} to {pp_means[-1]:.2f}")
    print(f"  sec_density:      {sec_densities[0]:.4f} to {sec_densities[-1]:.4f}")
    print(f"  epsilon:          {epsilons[0]:.2f} to {epsilons[-1]:.2f}")
    print(f"  threshold:        {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print(f"  citizen_density:  {citizen_densities[0]:.2f} to {citizen_densities[-1]:.2f}")
    print(f"  max_jail:         {max_jails[0]} to {max_jails[-1]}")
    print(f"  vision:           {visions[0]} to {visions[-1]}")

    # Build flat param array: 9 float32s per sim
    # Order: seed varies fastest, then vis, mj, cd, th, eps, sd, pp
    configs = list(itertools.product(
        pp_means, sec_densities, epsilons, thresholds,
        citizen_densities, max_jails, visions, seeds,
    ))

    n_sims = len(configs)
    params = np.zeros((n_sims, 9), dtype=np.float32)

    for i, (pp, sd, eps, th, cd, mj, vis, seed) in enumerate(configs):
        params[i, 0] = seed
        params[i, 1] = cd
        params[i, 2] = sd
        params[i, 3] = pp
        params[i, 4] = eps
        params[i, 5] = th
        params[i, 6] = mj
        params[i, 7] = num_steps
        params[i, 8] = vis

    # Also build a label array for SQL insert
    labels = []
    for pp, sd, eps, th, cd, mj, vis, seed in configs:
        labels.append((seed, cd, sd, pp, eps, th, mj, vis))

    return params, labels


def write_params_bin(path: Path, params: np.ndarray, num_steps: int):
    """Write params.bin: [n_sims:i32, num_steps:i32, 9*n_sims float32s]"""
    n_sims = len(params)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", n_sims, num_steps))
        f.write(params.tobytes())
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {n_sims:,} sims)")


def compute_summary_for_chunk(metrics_chunk, step_chunk, num_steps):
    """Compute per-sim summaries for a chunk. Vectorized where possible."""
    n = len(metrics_chunk)
    active = step_chunk[:, :, 0].astype(np.int64)  # (n, num_steps)
    rev_col = step_chunk[:, :, 4]  # (n, num_steps)

    max_active = active.max(axis=1)
    sum_active = active.sum(axis=1)

    # Revolution step per sim
    rev_steps = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        revs = np.where(rev_col[i] == 1)[0]
        if len(revs) > 0:
            rev_steps[i] = revs[0]

    # Cascade peak counting (vectorized diff approach)
    peak_counts = np.zeros(n, dtype=np.int64)
    if num_steps > 1:
        diffs = np.diff(active, axis=1)  # (n, num_steps-1)
        for i in range(n):
            rising = False
            for s in range(num_steps - 1):
                if diffs[i, s] > 0:
                    rising = True
                elif diffs[i, s] < 0:
                    if rising:
                        peak_counts[i] += 1
                    rising = False

    periodic = (peak_counts >= 2).astype(np.int64)
    return max_active, sum_active, rev_steps, peak_counts, periodic


def stream_to_db(
    db_path: Path,
    labels: list,
    metrics_path: Path,
    steps_path: Path,
    num_steps: int,
    sim_offset: int,
    n_sims: int,
):
    """Stream binary files to SQLite in chunks to avoid OOM."""
    conn = apsw.Connection(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")

    conn.execute("""CREATE TABLE IF NOT EXISTS simulations (
        sim_id INTEGER PRIMARY KEY,
        seed INTEGER,
        citizen_density REAL,
        sec_density REAL,
        pp_mean REAL,
        epsilon REAL,
        threshold REAL,
        max_jail INTEGER,
        vision INTEGER,
        n_citizens INTEGER,
        n_security INTEGER,
        max_active INTEGER,
        revolution_step INTEGER,
        n_cascades INTEGER,
        cascade_periodic INTEGER,
        sum_active INTEGER,
        num_steps INTEGER
    )""")

    conn.execute("""CREATE TABLE IF NOT EXISTS model_steps (
        sim_id INTEGER,
        step INTEGER,
        active_count INTEGER,
        support_count INTEGER,
        oppose_count INTEGER,
        jail_count INTEGER,
        revolution INTEGER,
        PRIMARY KEY (sim_id, step)
    ) WITHOUT ROWID""")

    # Read metrics.bin fully (small: ~197 MB for 8.2M sims)
    metrics = np.fromfile(str(metrics_path), dtype=np.int32).reshape(n_sims, 6)

    # Process step_metrics.bin in chunks
    CHUNK = 50_000  # sims per chunk
    step_bytes_per_sim = num_steps * N_STEP_FIELDS * 4  # 4 bytes per int32
    n_chunks = (n_sims + CHUNK - 1) // CHUNK

    t0 = time.time()
    steps_fp = open(steps_path, "rb")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK
        end = min(start + CHUNK, n_sims)
        chunk_n = end - start

        # Read chunk of step_metrics
        raw = np.frombuffer(steps_fp.read(chunk_n * step_bytes_per_sim), dtype=np.int32)
        step_chunk = raw.reshape(chunk_n, num_steps, N_STEP_FIELDS)

        metrics_chunk = metrics[start:end]

        # Compute summaries
        max_act, sum_act, rev_step, n_casc, periodic = compute_summary_for_chunk(
            metrics_chunk, step_chunk, num_steps
        )

        # Insert simulation rows
        sim_rows = []
        for i in range(chunk_n):
            idx = start + i
            seed, cd, sd, pp, eps, th, mj, vis = labels[idx]
            n_cit = int(metrics_chunk[i, 5])
            n_sec = int(round(GRID_SIZE * sd))
            sim_id = sim_offset + idx

            sim_rows.append((
                sim_id, int(seed), float(cd), float(sd), float(pp), float(eps),
                float(th), int(mj), int(vis), n_cit, n_sec,
                int(max_act[i]), int(rev_step[i]), int(n_casc[i]),
                int(periodic[i]), int(sum_act[i]), num_steps,
            ))

        conn.execute("BEGIN")
        conn.executemany(
            "INSERT OR REPLACE INTO simulations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            sim_rows,
        )
        conn.execute("COMMIT")

        # Insert step data rows
        STEP_BATCH = 500_000
        step_rows = []
        for i in range(chunk_n):
            sim_id = sim_offset + start + i
            for s in range(num_steps):
                step_rows.append((
                    sim_id, s,
                    int(step_chunk[i, s, 0]),
                    int(step_chunk[i, s, 1]),
                    int(step_chunk[i, s, 2]),
                    int(step_chunk[i, s, 3]),
                    int(step_chunk[i, s, 4]),
                ))
                if len(step_rows) >= STEP_BATCH:
                    conn.execute("BEGIN")
                    conn.executemany(
                        "INSERT OR REPLACE INTO model_steps VALUES (?,?,?,?,?,?,?)",
                        step_rows,
                    )
                    conn.execute("COMMIT")
                    step_rows = []

        if step_rows:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT OR REPLACE INTO model_steps VALUES (?,?,?,?,?,?,?)",
                step_rows,
            )
            conn.execute("COMMIT")

        elapsed = time.time() - t0
        pct = end / n_sims * 100
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n_sims - end) / rate if rate > 0 else 0
        print(f"  Chunk {chunk_idx + 1}/{n_chunks} | {end:,}/{n_sims:,} sims ({pct:.1f}%) | "
              f"{elapsed:.0f}s elapsed | ETA: {eta:.0f}s")

    steps_fp.close()

    elapsed = time.time() - t0
    print(f"DB insert done in {elapsed:.1f}s")

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sims_params ON simulations(pp_mean, sec_density, epsilon, threshold, citizen_density, max_jail, vision)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_sim ON model_steps(sim_id)")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="7D Coarse Parameter Sweep")
    parser.add_argument("--steps", type=int, default=100, help="Steps per simulation")
    parser.add_argument("--points", type=int, default=7, help="Grid points per parameter")
    parser.add_argument("--seeds", type=int, default=10, help="Seeds per config")
    parser.add_argument("--outdir", default="manifold_results", help="Output directory")
    parser.add_argument("--sim-offset", type=int, default=0, help="Starting sim_id")
    parser.add_argument("--runner", default=RUNNER_BIN, help="Path to cascade_gpu_runner binary")
    parser.add_argument("--load-only", action="store_true",
                        help="Skip GPU run, just load existing binary files into DB")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params_path = outdir / "params.bin"
    metrics_path = outdir / "metrics.bin"
    steps_path = outdir / "step_metrics.bin"
    db_path = outdir / "manifold.db"

    # Build parameter grid (needed for labels even in load-only mode)
    print(f"\n=== 7D Sweep: {args.points}^7 x {args.seeds} seeds x {args.steps} steps ===\n")
    params, labels = build_param_grid(args.points, args.seeds, args.steps)
    n_sims = len(params)
    print(f"Total simulations: {n_sims:,}")

    if not args.load_only:
        print(f"Estimated time at ~5,500 sims/sec: {n_sims / 5500 / 60:.1f} min")
        print()

        # Write params
        write_params_bin(params_path, params, args.steps)

        # Run GPU runner
        print(f"Launching {args.runner}...")
        t0 = time.time()
        subprocess.run(
            [args.runner, str(params_path), str(metrics_path), str(steps_path)],
            check=True,
        )
        gpu_time = time.time() - t0
        print(f"GPU runner completed in {gpu_time:.1f}s ({n_sims / gpu_time:.0f} sims/sec)")
        print()
    else:
        print("Load-only mode: skipping GPU run, loading existing binary files")
        t0 = time.time()

    # Stream results to DB in chunks (avoids OOM for large step counts)
    print("Streaming results to database...")
    stream_to_db(db_path, labels, metrics_path, steps_path, args.steps,
                 args.sim_offset, n_sims)

    # Cleanup binary intermediates
    params_path.unlink(missing_ok=True)
    metrics_path.unlink()
    steps_path.unlink()
    print(f"\nCleaned up binary files")
    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.2f} GB)")

    print(f"\n=== Done: {n_sims:,} sims in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
