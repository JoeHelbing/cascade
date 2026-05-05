import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from autoresearch.validation import run_pipeline


class ValidationPipelineTests(unittest.TestCase):
    def test_trace_parquet_writer_overwrites_existing_artifact_and_adds_seed_metadata(self):
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.parquet"
            path.write_text("stale artifact")
            rows = [
                {
                    "step": "0",
                    "agent_id": "0",
                    "agent_type": "Citizen",
                    "x": "1",
                    "y": "2",
                    "condition": "Support",
                    "opinion": "0.5",
                    "activation": "0.6224593312018546",
                    "private_preference": "-0.1",
                    "epsilon": "0.2",
                    "oppose_threshold": "1.0",
                    "active_threshold": "2.0",
                    "jail_sentence": "0",
                    "active_in_vision": "1",
                    "oppose_in_vision": "0",
                    "support_in_vision": "1",
                    "security_in_vision": "0",
                    "perception": "1.0",
                    "arrest_prob": "0.0",
                    "active_level": "0.1",
                    "oppose_level": "0.2",
                    "flip": "False",
                    "ever_flipped": "False",
                }
            ]

            # Act
            digest = run_pipeline.write_trace_parquet(
                path,
                [(7, 42, 0.5, 0.0, rows)],
            )

            # Assert
            self.assertEqual(len(digest), 64)
            table = pq.read_table(path)
            self.assertEqual(table.column_names[:4], ["sim_id", "seed", "epsilon_config", "security_density_config"])
            self.assertEqual(table.column("seed").to_pylist(), [42])
            self.assertEqual(table.column("opinion_bits").to_pylist(), [run_pipeline.float_bits(0.5)])

    def test_split_mojo_trace_output_uses_step_zero_boundaries_for_seed_metadata(self):
        # Arrange
        output = "\n".join(
            [
                "step,agent_id,agent_type,x,y,condition",
                "0,0,Citizen,1,1,Support",
                "1,0,Citizen,1,2,Active",
                "0,0,Citizen,3,3,Support",
                "# done 2 sims",
            ]
        )

        # Act
        chunks = run_pipeline.split_mojo_trace_output_by_seed(output, [10, 20])

        # Assert
        self.assertEqual([(chunk.seed, len(chunk.rows)) for chunk in chunks], [(10, 2), (20, 1)])
        self.assertEqual(chunks[1].rows[0]["x"], "3")

    def test_parse_gpu_trace_lines_extracts_per_agent_state_rows(self):
        # Arrange
        output = "\n".join(
            [
                "TRACE,0,16,0.5,0.0,0,0,1,2,Support",
                "TRACE,0,16,0.5,0.0,1,0,2,2,Active",
            ]
        )

        # Act
        chunks = run_pipeline.parse_gpu_trace_lines(output)

        # Assert
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].seed, 16)
        self.assertEqual(chunks[0].rows[1]["condition"], "Active")

    def test_state_trace_parquet_sha_matches_for_identical_cpu_gpu_rows(self):
        # Arrange
        rows = [
            {"step": "0", "agent_id": "0", "agent_type": "Citizen", "x": "1", "y": "2", "condition": "Support"},
            {"step": "1", "agent_id": "0", "agent_type": "Citizen", "x": "2", "y": "2", "condition": "Active"},
        ]
        chunk = run_pipeline.TraceChunk(sim_id=0, seed=16, epsilon=0.5, security_density=0.0, rows=rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_path = Path(tmpdir) / "cpu.parquet"
            gpu_path = Path(tmpdir) / "gpu.parquet"

            # Act
            cpu_sha = run_pipeline.write_state_trace_parquet(cpu_path, [chunk])
            gpu_sha = run_pipeline.write_state_trace_parquet(gpu_path, [chunk])

            # Assert
            self.assertEqual(cpu_sha, gpu_sha)

    def test_parse_gpu_sim_lines_extracts_aggregate_metrics(self):
        # Arrange
        output = """
noise
Sim 0 seed= 42 eps= 0.2 sd= 0.0 active= 762 support= 0 oppose= 0 jail= 0 rev= True
Sim 1 seed= 42 eps= 0.2 sd= 0.02 active= 0 support= 762 oppose= 0 jail= 0 rev= False
"""

        # Act
        rows = run_pipeline.parse_gpu_sim_lines(output)

        # Assert
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].seed, 42)
        self.assertEqual(rows[0].epsilon, 0.2)
        self.assertEqual(rows[0].security_density, 0.0)
        self.assertEqual(rows[0].active, 762)
        self.assertTrue(rows[0].revolution)
        self.assertFalse(rows[1].revolution)

    def test_load_mojo_cpu_gpu_aggregate_uses_last_step_and_citizens_only(self):
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpu.csv"
            rows = [
                ["step", "agent_id", "agent_type", "x", "y", "condition"],
                ["0", "0", "Citizen", "1", "1", "Support"],
                ["0", "1", "Security", "2", "2", "Security"],
                ["1", "0", "Citizen", "1", "1", "Active"],
                ["1", "1", "Security", "2", "2", "Security"],
                ["# done", "ignored", "ignored", "ignored", "ignored", "ignored"],
            ]
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)

            # Act
            aggregate = run_pipeline.load_mojo_cpu_gpu_aggregate(
                path,
                sim_id=7,
                seed=42,
                epsilon=0.5,
                security_density=0.0,
            )

        # Assert
        self.assertEqual(aggregate.sim_id, 7)
        self.assertEqual(aggregate.active, 1)
        self.assertEqual(aggregate.support, 0)
        self.assertEqual(aggregate.oppose, 0)
        self.assertEqual(aggregate.jail, 0)
        self.assertTrue(aggregate.revolution)


if __name__ == "__main__":
    unittest.main()
