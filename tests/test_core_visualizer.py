import tomllib
import unittest
from pathlib import Path

from visualizer.server import STATIC_ROOT, build_core_command, parse_trace_csv, validate_params


SAMPLE_TRACE = """step,agent_id,agent_type,x,y,condition,opinion,activation,private_preference,epsilon,oppose_threshold,active_threshold,jail_sentence,active_in_vision,oppose_in_vision,support_in_vision,security_in_vision,perception,arrest_prob,active_level,oppose_level,flip,ever_flipped
0,0,Citizen,1,2,Support,0.1,0.5,0.2,0.3,1.0,2.0,0,1,0,3,1,1.0,0.1,0.2,0.3,False,False
0,1,Security,3,4,Security,,,0.0,,,,,,,,,,,,,
1,0,Citizen,2,2,Active,0.1,0.5,0.2,0.3,1.0,2.0,0,1,0,3,1,1.0,0.1,0.2,0.3,True,True
1,1,Security,4,4,Security,,,0.0,,,,,,,,,,,,,
# done,1 sims, 0.01 s, accel=no
"""


class CoreVisualizerTests(unittest.TestCase):
    def test_validate_params_clamps_to_supported_ranges(self):
        params = validate_params({"width": 5, "height": 200, "max_iters": 999, "citizen_density": 2})

        self.assertEqual(params["width"], 8)
        self.assertEqual(params["height"], 80)
        self.assertEqual(params["max_iters"], 500)
        self.assertEqual(params["citizen_density"], 1.0)

    def test_build_core_command_uses_core_binary_and_cli_flags(self):
        params = validate_params({"seed": 42, "width": 12, "height": 10, "max_iters": 25})
        command = build_core_command(Path("/repo"), params)

        self.assertEqual(command[0], "/repo/build/core_cpu_mojo")
        self.assertIn("--seed", command)
        self.assertIn("42", command)
        self.assertIn("--max-iters", command)
        self.assertIn("25", command)

    def test_parse_trace_groups_agents_by_step_and_summarizes_counts(self):
        trace = parse_trace_csv(SAMPLE_TRACE)

        self.assertEqual(trace["step_count"], 2)
        self.assertEqual(trace["agent_count"], 2)
        self.assertEqual(trace["steps"][0]["counts"], {"Support": 1, "Oppose": 0, "Active": 0, "Jailed": 0, "Security": 1})
        self.assertEqual(trace["steps"][1]["counts"]["Active"], 1)
        self.assertEqual(trace["steps"][1]["agents"][0]["condition"], "Active")

    def test_static_ui_assets_define_controls_canvas_and_api_call(self):
        index = (STATIC_ROOT / "index.html").read_text()
        app = (STATIC_ROOT / "app.js").read_text()

        self.assertIn("id=\"simulation-canvas\"", index)
        self.assertIn("name=\"max_iters\"", index)
        self.assertIn("name=\"epsilon\"", index)
        self.assertIn("fetch('/api/run'", app)
        self.assertIn("drawFrame", app)

    def test_pixi_task_serves_visualizer_after_building_core_binary(self):
        pixi = tomllib.loads((Path(__file__).resolve().parents[1] / "pixi.toml").read_text())
        task = pixi["tasks"]["visualize-core-cpu"]

        self.assertEqual(task["depends-on"], ["build-core-cpu"])
        self.assertIn("visualizer/server.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
