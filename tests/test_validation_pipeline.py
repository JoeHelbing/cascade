import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from autoresearch.validation import run_pipeline


class ValidationPipelineTests(unittest.TestCase):
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
