import csv
import math
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = REPO_ROOT / "core_cpu_mojo.mojo"
CORE_BINARY = REPO_ROOT / "build" / "core_cpu_mojo"


class CoreCpuMojoTests(unittest.TestCase):
    def test_source_is_pure_mojo_without_python_interop(self):
        source = CORE_SOURCE.read_text()
        self.assertNotIn("PythonObject", source)
        self.assertNotIn("std.python", source)
        self.assertNotIn("Python.import_module", source)

    def test_source_uses_std_random_for_native_rng(self):
        source = CORE_SOURCE.read_text()
        self.assertIn("from std.random import", source)
        self.assertIn("randn_float64", source)
        self.assertIn("random_float64", source)
        self.assertNotIn("lcg_next", source)
        self.assertNotIn("struct NativeRng", source)

    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["pixi", "run", "mojo", "build", "core_cpu_mojo.mojo", "-o", str(CORE_BINARY), "-Xlinker", "-lm"],
            cwd=REPO_ROOT,
            check=True,
        )

    def test_core_cpu_mojo_builds_and_emits_trace_schema(self):
        result = subprocess.run(
            [
                str(CORE_BINARY),
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
                "--max-iters", "2",
                "--threshold", "1.5",
                "--seed", "42",
                "--random-seed", "false",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        data_lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
        rows = list(csv.DictReader(data_lines))
        self.assertEqual(list(rows[0].keys()), [
            "step", "agent_id", "agent_type", "x", "y", "condition", "opinion",
            "activation", "private_preference", "epsilon", "oppose_threshold",
            "active_threshold", "jail_sentence", "active_in_vision", "oppose_in_vision",
            "support_in_vision", "security_in_vision", "perception", "arrest_prob",
            "active_level", "oppose_level", "flip", "ever_flipped",
        ])
        self.assertEqual({row["step"] for row in rows}, {"0", "1", "2"})
        self.assertEqual(len(rows), 27)  # 6 citizens + 3 security across 3 trace snapshots
        self.assertEqual(sum(1 for row in rows if row["agent_type"] == "Security"), 9)

    def run_core_rows(self, *args: str) -> list[dict[str, str]]:
        result = subprocess.run(
            [str(CORE_BINARY), *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return list(csv.DictReader(line for line in result.stdout.splitlines() if not line.startswith("#")))

    def test_initial_agent_placement_is_unique_and_spread_across_grid(self):
        # Arrange / Act
        rows = self.run_core_rows(
            "--width", "8",
            "--height", "8",
            "--citizen-density", "0.5",
            "--security-density", "0.0",
            "--max-iters", "0",
            "--seed", "1",
            "--movement", "false",
        )
        positions = [(int(row["x"]), int(row["y"])) for row in rows if row["step"] == "0"]

        # Assert
        self.assertEqual(len(positions), 32)
        self.assertEqual(len(set(positions)), len(positions))
        self.assertGreater(len({x for x, _ in positions}), 4)
        self.assertGreater(len({y for _, y in positions}), 4)

    def test_agent_placement_remains_unique_after_movement_steps(self):
        # Arrange / Act
        rows = self.run_core_rows(
            "--width", "8",
            "--height", "8",
            "--citizen-density", "0.5",
            "--security-density", "0.0",
            "--max-iters", "4",
            "--seed", "5",
            "--movement", "true",
        )

        # Assert
        for step in {row["step"] for row in rows}:
            positions = [(int(row["x"]), int(row["y"])) for row in rows if row["step"] == step and row["x"]]
            self.assertEqual(len(set(positions)), len(positions), step)

    def test_decision_math_matches_original_python_formulas_from_trace_fields(self):
        # Arrange / Act
        rows = self.run_core_rows(
            "--width", "8",
            "--height", "8",
            "--citizen-vision", "2",
            "--citizen-density", "0.2",
            "--security-density", "0.05",
            "--max-iters", "0",
            "--seed", "9",
            "--movement", "false",
        )
        citizen = next(row for row in rows if row["agent_type"] == "Citizen")

        # Assert
        active = int(citizen["active_in_vision"])
        oppose = int(citizen["oppose_in_vision"])
        support = int(citizen["support_in_vision"])
        security = int(citizen["security_in_vision"])
        epsilon = float(citizen["epsilon"])
        epsilon_probability = 1.0 / (1.0 + math.exp(-epsilon))
        private_preference = float(citizen["private_preference"])
        perception = (active + oppose * epsilon_probability) ** ((epsilon**2 + 1.0) ** -1)
        arrest_prob = 1.0 - math.exp(-2.3 * (security / active) * (2.0 * epsilon_probability))
        opinion = -private_preference + perception * ((active + oppose) / support)
        activation = 1.0 / (1.0 + math.exp(-opinion))
        active_level = activation if False else 1.0 / (1.0 + math.exp(-(opinion - float(citizen["active_threshold"])))) - arrest_prob
        oppose_level = 1.0 / (1.0 + math.exp(-(opinion - float(citizen["oppose_threshold"])))) - arrest_prob

        self.assertAlmostEqual(float(citizen["perception"]), perception, places=12)
        self.assertAlmostEqual(float(citizen["arrest_prob"]), arrest_prob, places=12)
        self.assertAlmostEqual(float(citizen["opinion"]), opinion, places=12)
        self.assertAlmostEqual(float(citizen["activation"]), activation, places=12)
        self.assertAlmostEqual(float(citizen["active_level"]), active_level, places=12)
        self.assertAlmostEqual(float(citizen["oppose_level"]), oppose_level, places=12)


if __name__ == "__main__":
    unittest.main()
