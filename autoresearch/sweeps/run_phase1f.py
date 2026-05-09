"""
Phase 1F: Run all agent-level simulations in batches.

Reads autoresearch/configs/agent_sim_params.json, converts to the config format
expected by run_agent_sims.py, and runs in batches of 16 sims to fit in GPU
memory.

Usage:
    pixi run python autoresearch/sweeps/run_phase1f.py
"""

import json
import struct
import subprocess
import sqlite3
import sys
import tempfile
import numpy as np
from pathlib import Path

MAX_AGENTS = 1024
N_AGENT_FIELDS = 6
GRID_SIZE = 33 * 33
BATCH_SIZE = 4  # Very small batches for agent-level output (GPU memory constrained)


def write_params_bin(path, simulations, num_steps):
    """Write params.bin in cascade_gpu_runner format."""
    n_sims = len(simulations)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n_sims, num_steps))
        for sim in simulations:
            params = [
                float(sim["seed"]),
                float(sim.get("citizen_density", 0.7)),
                float(sim["sec_density"]),
                float(sim["pp_mean"]),
                float(sim["epsilon"]),
                float(sim["threshold"]),
                float(sim.get("max_jail", 100)),
                float(num_steps),
                float(sim.get("vision", 7)),
            ]
            f.write(struct.pack("<9f", *params))


def load_agent_batch_to_db(conn, agent_data_path, batch_sims, num_steps, batch_offset):
    """Load one batch of agent data from binary into SQLite."""
    n_sims = len(batch_sims)
    expected_size = n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS * 4
    data = np.fromfile(agent_data_path, dtype=np.float32)

    if data.size != n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS:
        print(f"WARNING: Expected {n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS} floats, got {data.size}")
        return 0

    data = data.reshape(n_sims, num_steps, MAX_AGENTS, N_AGENT_FIELDS)
    cur = conn.cursor()
    rows_inserted = 0

    for sim_idx, sim in enumerate(batch_sims):
        global_sim_id = batch_offset + sim_idx
        n_citizens = round(GRID_SIZE * sim.get("citizen_density", 0.7))
        n_security = round(GRID_SIZE * sim.get("sec_density", 0.0))
        n_agents = n_citizens + n_security

        for step in range(num_steps):
            # Load citizens (indices 0..n_citizens-1)
            for cid in range(n_citizens):
                row = data[sim_idx, step, cid]
                pos_x = int(row[0])
                pos_y = int(row[1])
                activation = float(row[2])
                condition = int(row[3])
                jail_sent = int(row[4])
                arrest_prob = float(row[5])

                cur.execute(
                    """INSERT INTO agent_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        global_sim_id, step, cid,
                        pos_x, pos_y, activation, condition, jail_sent, arrest_prob,
                        sim["seed"], sim["pp_mean"], sim["sec_density"],
                        sim["epsilon"], sim["threshold"], sim.get("vision", 7),
                    ),
                )
                rows_inserted += 1

            # Load security agents (indices n_citizens..n_agents-1, condition=4)
            for sid in range(n_citizens, n_agents):
                row = data[sim_idx, step, sid]
                pos_x = int(row[0])
                pos_y = int(row[1])
                # Security: condition=4, no activation/jail/arrest
                cur.execute(
                    """INSERT INTO agent_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        global_sim_id, step, sid,
                        pos_x, pos_y, 0.0, 4, 0, 0.0,
                        sim["seed"], sim["pp_mean"], sim["sec_density"],
                        sim["epsilon"], sim["threshold"], sim.get("vision", 7),
                    ),
                )
                rows_inserted += 1

        if (sim_idx + 1) % 4 == 0:
            conn.commit()

    conn.commit()
    return rows_inserted


def main():
    autoresearch_dir = Path(__file__).resolve().parents[1]
    params_file = autoresearch_dir / "configs" / "agent_sim_params.json"
    if not params_file.exists():
        print(f"ERROR: {params_file} not found")
        print("Generate it with: pixi run python autoresearch/sweeps/generate_agent_params.py")
        sys.exit(1)

    with open(params_file) as f:
        all_sims = json.load(f)

    runner = Path(__file__).parent / "cascade_gpu_runner"
    if not runner.exists():
        print("ERROR: cascade_gpu_runner not found. Build with:")
        print("  pixi run mojo build cascade_gpu_runner.mojo -o cascade_gpu_runner")
        sys.exit(1)

    db_path = Path("manifold_results/agent_data.db")
    db_path.parent.mkdir(exist_ok=True)

    # Create database
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS agent_steps")
    cur.execute("""
        CREATE TABLE agent_steps (
            sim_id INTEGER,
            step INTEGER,
            agent_id INTEGER,
            pos_x INTEGER,
            pos_y INTEGER,
            activation_val REAL,
            condition INTEGER,
            jail_sentence INTEGER,
            arrest_prob REAL,
            seed INTEGER,
            pp_mean REAL,
            sec_density REAL,
            epsilon REAL,
            threshold REAL,
            vision INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sim_params (
            sim_id INTEGER PRIMARY KEY,
            seed INTEGER,
            pp_mean REAL,
            sec_density REAL,
            epsilon REAL,
            threshold REAL,
            vision INTEGER,
            citizen_density REAL,
            max_jail INTEGER,
            num_steps INTEGER,
            sim_group TEXT,
            note TEXT
        )
    """)
    conn.commit()

    n_total = len(all_sims)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    num_steps = 500
    total_rows = 0

    print(f"=== Phase 1F Agent-Level Simulations ===")
    print(f"Total simulations: {n_total}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches: {n_batches}")
    print(f"Steps per sim: {num_steps}")
    print(f"Output: {db_path}")
    print()

    # Insert sim_params
    for i, sim in enumerate(all_sims):
        cur.execute(
            "INSERT INTO sim_params VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, sim["seed"], sim["pp_mean"], sim["sec_density"],
             sim["epsilon"], sim["threshold"], sim.get("vision", 7),
             sim.get("citizen_density", 0.7), sim.get("max_jail", 100),
             num_steps, sim.get("group", ""), sim.get("note", "")),
        )
    conn.commit()

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_total)
        batch_sims = all_sims[start:end]

        print(f"Batch {batch_idx + 1}/{n_batches}: sims {start}-{end - 1} ({len(batch_sims)} sims)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            params_bin = tmpdir / "params.bin"
            metrics_bin = tmpdir / "metrics.bin"
            step_metrics_bin = tmpdir / "step_metrics.bin"
            agent_data_bin = tmpdir / "agent_data.bin"

            write_params_bin(params_bin, batch_sims, num_steps)

            cmd = [
                str(runner),
                str(params_bin),
                str(metrics_bin),
                str(step_metrics_bin),
                str(agent_data_bin),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"  ERROR: GPU runner failed")
                print(f"  STDOUT: {result.stdout[-500:]}")
                print(f"  STDERR: {result.stderr[-500:]}")
                continue

            # Parse throughput from output
            for line in result.stdout.split("\n"):
                if "sims/sec" in line.lower() or "throughput" in line.lower():
                    print(f"  {line.strip()}")

            # Check agent_data.bin exists and has data
            if not agent_data_bin.exists():
                print(f"  WARNING: agent_data.bin not created")
                continue

            ad_size = agent_data_bin.stat().st_size
            expected = len(batch_sims) * num_steps * MAX_AGENTS * N_AGENT_FIELDS * 4
            print(f"  agent_data.bin: {ad_size / 1e6:.1f} MB (expected {expected / 1e6:.1f} MB)")

            if ad_size != expected:
                print(f"  WARNING: Size mismatch, skipping batch")
                continue

            rows = load_agent_batch_to_db(conn, agent_data_bin, batch_sims, num_steps, start)
            total_rows += rows
            print(f"  Loaded {rows:,} rows into DB")

    # Create indices after all data loaded
    print("\nCreating indices...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_step ON agent_steps(sim_id, step)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_agent ON agent_steps(sim_id, agent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_condition ON agent_steps(condition)")
    conn.commit()
    conn.close()

    db_size_mb = db_path.stat().st_size / (1024**2)
    print(f"\n=== Done ===")
    print(f"Database: {db_path}")
    print(f"Total rows: {total_rows:,}")
    print(f"Database size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
