import csv
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MojoCpuCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["pixi", "run", "build-cpu"], cwd=REPO_ROOT, check=True)

    def test_accepts_original_model_parameters_from_cli(self):
        # Arrange
        cmd = [
            str(REPO_ROOT / "build" / "mojo_cpu"),
            "--width", "10",
            "--height", "10",
            "--citizen-vision", "1",
            "--citizen-density", "0.1",
            "--security-density", "0.0",
            "--security-vision", "1",
            "--max-jail-term", "7",
            "--movement", "false",
            "--multiple-agents-per-cell", "true",
            "--private-preference-distribution-mean", "0.2",
            "--standard-deviation", "0.3",
            "--epsilon", "0.4",
            "--max-iters", "1",
            "--threshold", "1.5",
            "--seed", "42",
            "--random-seed", "false",
        ]

        # Act
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)

        # Assert
        data_lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
        rows = list(csv.DictReader(data_lines))
        self.assertEqual({row["seed"] for row in rows}, {"42"})
        self.assertEqual({row["step"] for row in rows}, {"0", "1"})
        self.assertEqual(len(rows), 20)  # 10 citizens x (initial row + one iteration)


if __name__ == "__main__":
    unittest.main()
