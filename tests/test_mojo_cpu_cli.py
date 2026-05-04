import csv
import struct
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python-core-simulation"))

from cascade_core import ResistanceCascade  # noqa: E402


def _float_bits(value: float) -> bytes:
    return struct.pack("!d", value)


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
        self.assertEqual({row["step"] for row in rows}, {"0", "1"})
        self.assertEqual(len(rows), 20)  # 10 citizens x (initial row + one iteration)

    def test_accepts_gpu_rng_switch_with_same_schema(self):
        # Act
        python_rows = self.run_small_cpu("python")
        gpu_rows = self.run_small_cpu("gpu")

        # Assert
        self.assertEqual(list(python_rows[0].keys()), list(gpu_rows[0].keys()))
        self.assertEqual({row["step"] for row in gpu_rows}, {"0", "1"})
        self.assertEqual(len(gpu_rows), 20)
        self.assertNotEqual(
            [(row["x"], row["y"], row["activation"]) for row in python_rows],
            [(row["x"], row["y"], row["activation"]) for row in gpu_rows],
        )

    def test_python_rng_matches_python_core_simulation_bit_for_bit(self):
        # Arrange
        cmd = [
            str(REPO_ROOT / "build" / "mojo_cpu"),
            "--rng", "python",
            "--width", "6",
            "--height", "5",
            "--citizen-vision", "1",
            "--citizen-density", "0.2",
            "--security-density", "0.1",
            "--security-vision", "1",
            "--max-jail-term", "7",
            "--movement", "true",
            "--private-preference-distribution-mean", "0.2",
            "--standard-deviation", "0.3",
            "--epsilon", "0.4",
            "--max-iters", "1",
            "--threshold", "1.5",
            "--seed", "42",
            "--random-seed", "false",
        ]
        expected_sim = ResistanceCascade(
            width=6,
            height=5,
            citizen_vision=1,
            citizen_density=0.2,
            security_density=0.1,
            security_vision=1,
            max_jail_term=7,
            movement=True,
            private_preference_distribution_mean=0.2,
            standard_deviation=0.3,
            epsilon=0.4,
            max_iters=1,
            threshold=1.5,
            seed=42,
        )
        expected_sim.step()

        # Act
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
        actual_rows = list(csv.DictReader(line for line in result.stdout.splitlines() if not line.startswith("#")))

        # Assert
        self.assertEqual(len(actual_rows), len(expected_sim.trace))
        for actual, expected in zip(actual_rows, expected_sim.trace, strict=True):
            self.assertEqual(actual["step"], str(expected.step))
            self.assertEqual(actual["agent_id"], str(expected.agent_id))
            self.assertEqual(actual["agent_type"], expected.agent_type)
            self.assertEqual(actual["x"], "" if expected.x is None else str(expected.x))
            self.assertEqual(actual["y"], "" if expected.y is None else str(expected.y))
            self.assertEqual(actual["condition"], expected.condition)
            for field in (
                "opinion",
                "activation",
                "private_preference",
                "epsilon",
                "oppose_threshold",
                "active_threshold",
                "perception",
                "arrest_prob",
                "active_level",
                "oppose_level",
            ):
                expected_value = getattr(expected, field)
                if expected_value is None:
                    self.assertEqual(actual[field], "")
                else:
                    self.assertEqual(_float_bits(float(actual[field])), _float_bits(expected_value), field)
            for field in (
                "jail_sentence",
                "active_in_vision",
                "oppose_in_vision",
                "support_in_vision",
                "security_in_vision",
            ):
                expected_value = getattr(expected, field)
                self.assertEqual(actual[field], "" if expected_value is None else str(expected_value))
            for field in ("flip", "ever_flipped"):
                expected_value = getattr(expected, field)
                self.assertEqual(actual[field], "" if expected_value is None else str(expected_value))


if __name__ == "__main__":
    unittest.main()
