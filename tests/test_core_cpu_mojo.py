import csv
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

    def test_core_cpu_mojo_builds_and_emits_trace_schema(self):
        subprocess.run(
            ["pixi", "run", "mojo", "build", "core_cpu_mojo.mojo", "-o", str(CORE_BINARY)],
            cwd=REPO_ROOT,
            check=True,
        )
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


if __name__ == "__main__":
    unittest.main()
