"""
Phase 1D: High-Resolution Targeted Parameter Sweep.

Unlike the 7D coarse sweep (uniform grid), Phase 1D uses:
- Different resolution per parameter (25, 25, 10, 15, 10 points)
- Fixed citizen_density=0.7 and max_jail=100
- 500 steps, 20 seeds
- 937,500 configs x 20 seeds = 18,750,000 sims

Uses the same cascade_gpu_runner binary and DB format.

Usage:
    cd mojo_cascade
    pixi run python run_1d_sweep.py [--outdir manifold_results] [--load-only]
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
NUM_STEPS = 500
N_SEEDS = 20

# Phase 1D parameter grid: variable resolution per parameter
PARAM_GRID = {
    "pp_mean":     {"lo": -1.0,  "hi": 1.0,  "n": 15},
    "sec_density": {"lo": 0.0,   "hi": 0.10, "n": 25},
    "epsilon":     {"lo": 0.01,  "hi": 2.0,  "n": 10},
    "threshold":   {"lo": 1.0,   "hi": 6.0,  "n": 25},
    "vision":      {"lo": 1,     "hi": 10,   "n": 10, "int": True},
}

# Fixed parameters
FIXED = {
    "citizen_density": 0.7,
    "max_jail": 100,
}


def linspace(lo, hi, n):
    if n == 1:
        return [0.5 * (lo + hi)]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def int_linspace(lo, hi, n):
    if n == 1:
        return [int(0.5 * (lo + hi))]
    return sorted(set(int(lo + i * (hi - lo) / (n - 1)) for i in range(n)))


def build_param_grid():
    """Build Phase 1D parameter grid with variable resolution."""
    pp_means = linspace(-1.0, 1.0, 15)
    sec_densities = linspace(0.0, 0.10, 25)
    epsilons = linspace(0.01, 2.0, 10)
    thresholds = linspace(1.0, 6.0, 25)
    visions = int_linspace(1, 10, 10)
    seeds = [42 + i * 7919 for i in range(N_SEEDS)]

    n_configs = len(pp_means) * len(sec_densities) * len(epsilons) * len(thresholds) * len(visions)
    n_sims = n_configs * N_SEEDS

    print(f"Phase 1D Parameter Grid:")
    print(f"  pp_mean:       {len(pp_means)} pts, {pp_means[0]:.2f} to {pp_means[-1]:.2f}")
    print(f"  sec_density:   {len(sec_densities)} pts, {sec_densities[0]:.4f} to {sec_densities[-1]:.4f}")
    print(f"  epsilon:       {len(epsilons)} pts, {epsilons[0]:.2f} to {epsilons[-1]:.2f}")
    print(f"  threshold:     {len(thresholds)} pts, {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    print(f"  vision:        {len(visions)} pts, {visions[0]} to {visions[-1]}")
    print(f"  citizen_density: FIXED at {FIXED['citizen_density']}")
    print(f"  max_jail:      FIXED at {FIXED['max_jail']}")
    print(f"  seeds:         {N_SEEDS}")
    print(f"  step count:    {NUM_STEPS}")
    print(f"  configs:       {n_configs:,}")
    print(f"  total sims:    {n_sims:,}")

    # Build flat param array: 9 float32s per sim
    configs = list(itertools.product(
        pp_means, sec_densities, epsilons, thresholds, visions, seeds,
    ))

    params = np.zeros((len(configs), 9), dtype=np.float32)
    labels = []

    cd = FIXED["citizen_density"]
    mj = FIXED["max_jail"]

    for i, (pp, sd, eps, th, vis, seed) in enumerate(configs):
        params[i, 0] = seed
        params[i, 1] = cd
        params[i, 2] = sd
        params[i, 3] = pp
        params[i, 4] = eps
        params[i, 5] = th
        params[i, 6] = mj
        params[i, 7] = NUM_STEPS
        params[i, 8] = vis
        labels.append((seed, cd, sd, pp, eps, th, mj, vis))

    return params, labels


def write_params_bin(path, params):
    n_sims = len(params)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", n_sims, NUM_STEPS))
        f.write(params.tobytes())
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {n_sims:,} sims)")


def compute_summary_for_chunk(metrics_chunk, step_chunk, num_steps):
    n = len(metrics_chunk)
    active = step_chunk[:, :, 0].astype(np.int64)
    rev_col = step_chunk[:, :, 4]

    max_active = active.max(axis=1)
    sum_active = active.sum(axis=1)

    rev_steps = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        revs = np.where(rev_col[i] == 1)[0]
        if len(revs) > 0:
            rev_steps[i] = revs[0]

    peak_counts = np.zeros(n, dtype=np.int64)
    if num_steps > 1:
        diffs = np.diff(active, axis=1)
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


def stream_to_db(db_path, labels, metrics_path, steps_path, sim_offset, n_sims):
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

    # Tag Phase 1D sims with a sweep_id
    conn.execute("""CREATE TABLE IF NOT EXISTS sweep_metadata (
        sweep_id TEXT PRIMARY KEY,
        description TEXT,
        start_sim_id INTEGER,
        end_sim_id INTEGER,
        n_sims INTEGER,
        num_steps INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    metrics = np.fromfile(str(metrics_path), dtype=np.int32).reshape(n_sims, 6)

    CHUNK = 50_000
    step_bytes_per_sim = NUM_STEPS * N_STEP_FIELDS * 4
    n_chunks = (n_sims + CHUNK - 1) // CHUNK

    t0 = time.time()
    steps_fp = open(steps_path, "rb")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK
        end = min(start + CHUNK, n_sims)
        chunk_n = end - start

        raw = np.frombuffer(steps_fp.read(chunk_n * step_bytes_per_sim), dtype=np.int32)
        step_chunk = raw.reshape(chunk_n, NUM_STEPS, N_STEP_FIELDS)
        metrics_chunk = metrics[start:end]

        max_act, sum_act, rev_step, n_casc, periodic_arr = compute_summary_for_chunk(
            metrics_chunk, step_chunk, NUM_STEPS
        )

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
                int(periodic_arr[i]), int(sum_act[i]), NUM_STEPS,
            ))

        conn.execute("BEGIN")
        conn.executemany(
            "INSERT OR REPLACE INTO simulations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            sim_rows,
        )
        conn.execute("COMMIT")

        STEP_BATCH = 500_000
        step_rows = []
        for i in range(chunk_n):
            sim_id = sim_offset + start + i
            for s in range(NUM_STEPS):
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

    # Record sweep metadata
    conn.execute(
        "INSERT OR REPLACE INTO sweep_metadata VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
        ("phase_1d", "High-resolution targeted sweep (5 params, fixed cd=0.7/mj=100)",
         sim_offset, sim_offset + n_sims - 1, n_sims, NUM_STEPS),
    )

    elapsed = time.time() - t0
    print(f"DB insert done in {elapsed:.1f}s")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_sims_params ON simulations(pp_mean, sec_density, epsilon, threshold, citizen_density, max_jail, vision)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_sim ON model_steps(sim_id)")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1D High-Resolution Sweep")
    parser.add_argument("--outdir", default="manifold_results", help="Output directory")
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

    # Get sim_id offset from existing DB
    if db_path.exists():
        conn = apsw.Connection(str(db_path), apsw.SQLITE_OPEN_READONLY)
        max_id = list(conn.execute("SELECT COALESCE(MAX(sim_id), -1) FROM simulations"))[0][0]
        conn.close()
        sim_offset = max_id + 1
    else:
        sim_offset = 0

    print(f"\n=== Phase 1D: High-Resolution Targeted Sweep ===")
    print(f"Starting sim_id: {sim_offset:,}\n")

    params, labels = build_param_grid()
    n_sims = len(params)

    if not args.load_only:
        est_hours = n_sims / 530 / 3600
        print(f"\nEstimated GPU time at ~530 sims/sec: {est_hours:.1f} hrs")
        print()

        write_params_bin(params_path, params)

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
        print("Load-only mode: skipping GPU run")

    print("Streaming results to database...")
    stream_to_db(db_path, labels, metrics_path, steps_path, sim_offset, n_sims)

    # Cleanup binary intermediates
    params_path.unlink(missing_ok=True)
    metrics_path.unlink()
    steps_path.unlink()
    print(f"\nCleaned up binary files")
    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
