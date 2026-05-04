import csv
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MojoCpuCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["pixi", "run", "build-cpu"], cwd=REPO_ROOT, check=True)

    def run_small_cpu(self, rng_mode: str) -> list[dict[str, str]]:
        cmd = [
            str(REPO_ROOT / "build" / "mojo_cpu"),
            "--rng", rng_mode,
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

        result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
        data_lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
        return list(csv.DictReader(data_lines))

    def test_accepts_original_model_parameters_from_cli(self):
        # Act
        rows = self.run_small_cpu("python")

        # Assert
        self.assertEqual({row["seed"] for row in rows}, {"42"})
        self.assertEqual({row["step"] for row in rows}, {"0", "1"})
        self.assertEqual(len(rows), 20)  # 10 citizens x (initial row + one iteration)

    def test_accepts_gpu_rng_switch_with_same_schema(self):
        # Act
        python_rows = self.run_small_cpu("python")
        gpu_rows = self.run_small_cpu("gpu")

        # Assert
        self.assertEqual(list(python_rows[0].keys()), list(gpu_rows[0].keys()))
        self.assertEqual({row["seed"] for row in gpu_rows}, {"42"})
        self.assertEqual({row["step"] for row in gpu_rows}, {"0", "1"})
        self.assertEqual(len(gpu_rows), 20)
        self.assertNotEqual(
            [(row["pos_x"], row["pos_y"], row["activation"]) for row in python_rows],
            [(row["pos_x"], row["pos_y"], row["activation"]) for row in gpu_rows],
        )


if __name__ == "__main__":
    unittest.main()
