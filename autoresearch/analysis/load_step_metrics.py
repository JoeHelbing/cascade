"""
Load step_metrics.bin into the manifold.db SQLite database.

The binary file has a 4-byte header (num_steps as int32), followed by raw int32
arrays. Layout: [n_sims, num_steps, N_STEP_FIELDS] where fields are:
  0: active_count, 1: support_count, 2: oppose_count, 3: jail_count, 4: revolution

Sims are ordered by flat index: seed varies fastest, then vis, mj, cd, th,
eps, sd, pp.

After successful load, the binary file is deleted.

Usage:
    cd mojo_cascade
    pixi run python load_step_metrics.py [--outdir manifold_results] [--sim-offset 0]
"""

import argparse
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np

N_STEP_FIELDS = 5
BATCH_SIZE = 100_000  # rows per INSERT batch


def main():
    parser = argparse.ArgumentParser(description="Load step_metrics.bin into SQLite")
    parser.add_argument("--outdir", default="manifold_results", help="Output directory")
    parser.add_argument("--sim-offset", type=int, default=0, help="Sim ID offset")
    args = parser.parse_args()

    bin_path = Path(args.outdir) / "step_metrics.bin"
    db_path = Path(args.outdir) / "manifold.db"

    print(f"Loading {bin_path}...")

    with open(bin_path, "rb") as f:
        # Read header: num_steps as int32
        header = f.read(4)
        (num_steps,) = struct.unpack("i", header)
        print(f"  Header: num_steps = {num_steps}")

        # Read remaining data
        raw = np.frombuffer(f.read(), dtype=np.int32)

    n_values = len(raw)
    n_sims = n_values // (num_steps * N_STEP_FIELDS)
    print(f"  {n_values:,} int32 values = {n_sims:,} sims x {num_steps} steps x {N_STEP_FIELDS} fields")
    print(f"  File size: {bin_path.stat().st_size / 1e9:.2f} GB")

    # Reshape: [n_sims, num_steps, N_STEP_FIELDS]
    data = raw.reshape(n_sims, num_steps, N_STEP_FIELDS)

    # Build flat rows: (sim_id, step, active, support, oppose, jail, revolution)
    print("Building row arrays...")
    t0 = time.time()

    sim_ids = np.repeat(np.arange(n_sims, dtype=np.int64) + args.sim_offset, num_steps)
    steps = np.tile(np.arange(num_steps, dtype=np.int32), n_sims)
    flat = data.reshape(-1, N_STEP_FIELDS)

    # Stack into (n_rows, 7) array
    rows = np.column_stack([sim_ids, steps, flat])
    n_rows = len(rows)
    print(f"  {n_rows:,} rows in {time.time() - t0:.1f}s")

    # Insert into SQLite
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")

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

    print(f"Inserting {n_rows:,} rows into model_steps...")
    t0 = time.time()

    # Convert to list of tuples in batches for executemany
    for start in range(0, n_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_rows)
        batch = [tuple(int(x) for x in row) for row in rows[start:end]]
        conn.executemany(
            "INSERT OR REPLACE INTO model_steps VALUES (?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

        if (start // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - t0
            done = end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_rows - done) / rate if rate > 0 else 0
            print(f"  {done:,} / {n_rows:,} ({done/n_rows*100:.1f}%) | {rate:.0f} rows/sec | ETA: {eta:.0f}s")

    print("Creating index...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_sim ON model_steps(sim_id)")
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n_rows/elapsed:.0f} rows/sec)")
    print(f"Database: {db_path} ({db_path.stat().st_size / 1e9:.2f} GB)")

    # Delete binary file after successful load
    bin_path.unlink()
    print(f"Deleted {bin_path}")


if __name__ == "__main__":
    main()
