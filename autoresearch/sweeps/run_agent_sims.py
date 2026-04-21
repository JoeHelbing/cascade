"""
Run small-batch GPU simulations with per-agent per-step output for Phase 1F analysis.

Writes params.bin for a small batch of simulations, invokes cascade_gpu_runner
with agent_data output enabled, and loads results into SQLite for analysis.

Usage:
    pixi run python run_agent_sims.py config.json

Config JSON format:
{
    "output_db": "agent_data.db",
    "num_steps": 500,
    "max_jail": 30,
    "vision": 3,
    "simulations": [
        {
            "seed": 42,
            "citizen_density": 0.7,
            "security_density": 0.04,
            "pp_mean": 2.5,
            "epsilon": 0.1,
            "threshold": 2.5
        },
        ...
    ]
}

Binary layout for agent_data.bin:
    float32[n_sims * num_steps * MAX_AGENTS * 6]
    Fields per agent per step: pos_x, pos_y, activation, cond, jail_sent, arrest_prob
    MAX_AGENTS = 1024 (fixed in kernel)
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
GRID_SIZE = 33 * 33  # GRID_W * GRID_H


def write_params_bin(path: Path, simulations: list, num_steps: int, max_jail: int, vision: int):
    """Write params.bin in the format expected by cascade_gpu_runner."""
    n_sims = len(simulations)

    with open(path, "wb") as f:
        # Header: n_sims (int32), num_steps (int32)
        f.write(struct.pack("<ii", n_sims, num_steps))

        # 9 float32s per simulation
        for sim in simulations:
            params = [
                float(sim["seed"]),
                float(sim["citizen_density"]),
                float(sim["security_density"]),
                float(sim["pp_mean"]),
                float(sim["epsilon"]),
                float(sim["threshold"]),
                float(max_jail),
                float(num_steps),
                float(vision),
            ]
            f.write(struct.pack("<9f", *params))


def compute_n_citizens(citizen_density: float) -> int:
    """Compute number of citizens from density, matching kernel logic."""
    return round(GRID_SIZE * citizen_density)


def load_agent_data_to_sqlite(
    db_path: Path,
    agent_data_path: Path,
    simulations: list,
    num_steps: int,
):
    """Load agent_data.bin into SQLite database."""
    n_sims = len(simulations)

    # Read raw binary
    raw = np.fromfile(agent_data_path, dtype=np.float32)
    expected_size = n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS
    if raw.size != expected_size:
        print(f"WARNING: expected {expected_size} floats, got {raw.size}")
        print(f"  n_sims={n_sims}, num_steps={num_steps}, MAX_AGENTS={MAX_AGENTS}, N_AGENT_FIELDS={N_AGENT_FIELDS}")

    # Reshape: [n_sims, num_steps, MAX_AGENTS, N_AGENT_FIELDS]
    data = raw.reshape(n_sims, num_steps, MAX_AGENTS, N_AGENT_FIELDS)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tables
    cur.execute("DROP TABLE IF EXISTS simulations")
    cur.execute("DROP TABLE IF EXISTS agent_steps")

    cur.execute("""
        CREATE TABLE simulations (
            sim_id INTEGER PRIMARY KEY,
            seed INTEGER,
            citizen_density REAL,
            security_density REAL,
            pp_mean REAL,
            epsilon REAL,
            threshold REAL,
            n_citizens INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE agent_steps (
            sim_id INTEGER,
            step INTEGER,
            agent_id INTEGER,
            pos_x INTEGER,
            pos_y INTEGER,
            activation REAL,
            cond INTEGER,
            jail_sent INTEGER,
            arrest_prob REAL,
            PRIMARY KEY (sim_id, step, agent_id)
        )
    """)

    # Insert simulation metadata
    for i, sim in enumerate(simulations):
        n_cit = compute_n_citizens(sim["citizen_density"])
        cur.execute(
            "INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, sim["seed"], sim["citizen_density"], sim["security_density"],
             sim["pp_mean"], sim["epsilon"], sim["threshold"], n_cit),
        )

    # Insert agent data (only for actual citizens, skip padding)
    print(f"Loading agent data into {db_path}...")
    batch_rows = []
    batch_size = 50000

    for sim_idx in range(n_sims):
        n_citizens = compute_n_citizens(simulations[sim_idx]["citizen_density"])
        sim_data = data[sim_idx]  # [num_steps, MAX_AGENTS, N_AGENT_FIELDS]

        for step in range(num_steps):
            step_data = sim_data[step]  # [MAX_AGENTS, N_AGENT_FIELDS]
            for agent_id in range(n_citizens):
                row = step_data[agent_id]
                batch_rows.append((
                    sim_idx,
                    step,
                    agent_id,
                    int(row[0]),   # pos_x
                    int(row[1]),   # pos_y
                    float(row[2]), # activation
                    int(row[3]),   # cond
                    int(row[4]),   # jail_sent
                    float(row[5]), # arrest_prob
                ))

                if len(batch_rows) >= batch_size:
                    cur.executemany(
                        "INSERT INTO agent_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch_rows,
                    )
                    batch_rows = []

    if batch_rows:
        cur.executemany(
            "INSERT INTO agent_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch_rows,
        )

    # Create indices for common queries
    cur.execute("CREATE INDEX idx_agent_steps_sim_step ON agent_steps(sim_id, step)")
    cur.execute("CREATE INDEX idx_agent_steps_sim_agent ON agent_steps(sim_id, agent_id)")

    conn.commit()
    conn.close()
    print(f"Done. {db_path} created.")


def main():
    if len(sys.argv) < 2:
        print("Usage: pixi run python run_agent_sims.py <config.json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = json.load(f)

    simulations = config["simulations"]
    num_steps = config.get("num_steps", 500)
    max_jail = config.get("max_jail", 30)
    vision = config.get("vision", 3)
    output_db = Path(config.get("output_db", "agent_data.db"))

    n_sims = len(simulations)
    print(f"=== Agent Data Runner ===")
    print(f"Simulations: {n_sims}")
    print(f"Steps: {num_steps}")

    # Size estimate
    n_citizens_max = max(compute_n_citizens(s["citizen_density"]) for s in simulations)
    buffer_gb = n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS * 4 / (1024**3)
    print(f"GPU buffer size: {buffer_gb:.2f} GB")
    if buffer_gb > 8:
        print(f"WARNING: Buffer size {buffer_gb:.1f} GB may exceed GPU memory.")
        print("Consider reducing batch size or num_steps.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        params_bin = tmpdir / "params.bin"
        metrics_bin = tmpdir / "metrics.bin"
        step_metrics_bin = tmpdir / "step_metrics.bin"
        agent_data_bin = tmpdir / "agent_data.bin"

        # Write params
        write_params_bin(params_bin, simulations, num_steps, max_jail, vision)
        print(f"Wrote {params_bin} ({params_bin.stat().st_size} bytes)")

        # Run GPU kernel
        runner = Path(__file__).parent / "cascade_gpu_runner"
        if not runner.exists():
            print(f"ERROR: {runner} not found. Build with:")
            print("  pixi run mojo build cascade_gpu_runner.mojo -o cascade_gpu_runner")
            sys.exit(1)

        cmd = [str(runner), str(params_bin), str(metrics_bin), str(step_metrics_bin), str(agent_data_bin)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            sys.exit(1)

        # Verify output
        ad_size = agent_data_bin.stat().st_size
        expected_bytes = n_sims * num_steps * MAX_AGENTS * N_AGENT_FIELDS * 4
        print(f"agent_data.bin: {ad_size} bytes (expected {expected_bytes})")

        # Load into SQLite
        load_agent_data_to_sqlite(output_db, agent_data_bin, simulations, num_steps)

    # Summary stats
    conn = sqlite3.connect(output_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM agent_steps")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT sim_id) FROM agent_steps")
    n_sims_loaded = cur.fetchone()[0]
    conn.close()

    print(f"\n=== Summary ===")
    print(f"Database: {output_db}")
    print(f"Simulations loaded: {n_sims_loaded}")
    print(f"Total agent-step rows: {total_rows:,}")
    db_size_mb = output_db.stat().st_size / (1024**2)
    print(f"Database size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
