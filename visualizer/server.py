from __future__ import annotations

import csv
import json
import subprocess
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"

DEFAULT_PARAMS: dict[str, Any] = {
    "seed": 42,
    "width": 40,
    "height": 40,
    "citizen_vision": 7,
    "citizen_density": 0.7,
    "security_density": 0.0,
    "security_vision": 7,
    "max_jail_term": 100,
    "movement": True,
    "private_preference_distribution_mean": 0.0,
    "standard_deviation": 1.0,
    "epsilon": 0.5,
    "threshold": 3.66356,
    "max_iters": 200,
}

PARAM_SPECS = {
    "seed": (0, 999_999, int),
    "width": (8, 80, int),
    "height": (8, 80, int),
    "citizen_vision": (1, 20, int),
    "citizen_density": (0.0, 1.0, float),
    "security_density": (0.0, 0.25, float),
    "security_vision": (1, 20, int),
    "max_jail_term": (0, 500, int),
    "private_preference_distribution_mean": (-5.0, 5.0, float),
    "standard_deviation": (0.01, 5.0, float),
    "epsilon": (0.01, 3.0, float),
    "threshold": (-5.0, 8.0, float),
    "max_iters": (0, 500, int),
}

CLI_FLAGS = {
    "seed": "--seed",
    "width": "--width",
    "height": "--height",
    "citizen_vision": "--citizen-vision",
    "citizen_density": "--citizen-density",
    "security_density": "--security-density",
    "security_vision": "--security-vision",
    "max_jail_term": "--max-jail-term",
    "movement": "--movement",
    "private_preference_distribution_mean": "--private-preference-distribution-mean",
    "standard_deviation": "--standard-deviation",
    "epsilon": "--epsilon",
    "threshold": "--threshold",
    "max_iters": "--max-iters",
}


def _clamp(value: Any, minimum: float, maximum: float, cast: type) -> Any:
    try:
        parsed = cast(value)
    except (TypeError, ValueError):
        parsed = cast(minimum)
    if parsed < minimum:
        return cast(minimum)
    if parsed > maximum:
        return cast(maximum)
    return parsed


def validate_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    params = DEFAULT_PARAMS.copy()
    if raw:
        params.update(raw)
    for name, (minimum, maximum, cast) in PARAM_SPECS.items():
        params[name] = _clamp(params.get(name), minimum, maximum, cast)
    params["movement"] = bool(params.get("movement"))
    return params


def build_core_command(repo_root: Path, params: dict[str, Any]) -> list[str]:
    binary = repo_root / "build" / "core_cpu_mojo"
    command = [str(binary)]
    for name, flag in CLI_FLAGS.items():
        value = params[name]
        if isinstance(value, bool):
            value = "true" if value else "false"
        command.extend([flag, str(value)])
    command.extend(["--random-seed", "false"])
    return command


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def parse_trace_csv(stdout: str) -> dict[str, Any]:
    data_lines = [line for line in stdout.splitlines() if line and not line.startswith("#")]
    rows = list(csv.DictReader(data_lines))
    steps_by_id: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        step = int(row["step"])
        agent = {
            "id": int(row["agent_id"]),
            "type": row["agent_type"],
            "x": _optional_int(row["x"]),
            "y": _optional_int(row["y"]),
            "condition": row["condition"],
            "opinion": _optional_float(row["opinion"]),
            "activation": _optional_float(row["activation"]),
            "activeLevel": _optional_float(row["active_level"]),
            "opposeLevel": _optional_float(row["oppose_level"]),
            "jailSentence": _optional_int(row["jail_sentence"]),
            "flip": row["flip"] == "True",
            "everFlipped": row["ever_flipped"] == "True",
        }
        steps_by_id.setdefault(step, []).append(agent)

    steps: list[dict[str, Any]] = []
    for step_id in sorted(steps_by_id):
        agents = sorted(steps_by_id[step_id], key=lambda agent: agent["id"])
        counts = {"Support": 0, "Oppose": 0, "Active": 0, "Jailed": 0, "Security": 0}
        for agent in agents:
            counts[agent["condition"]] = counts.get(agent["condition"], 0) + 1
        citizen_count = len([agent for agent in agents if agent["type"] == "Citizen"])
        active_or_jailed = counts["Active"] + counts["Jailed"]
        steps.append({
            "step": step_id,
            "agents": agents,
            "counts": counts,
            "revolutionShare": active_or_jailed / citizen_count if citizen_count else 0.0,
        })

    return {
        "step_count": len(steps),
        "agent_count": len(steps[0]["agents"]) if steps else 0,
        "steps": steps,
    }


def run_simulation(raw_params: dict[str, Any] | None, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    params = validate_params(raw_params)
    binary = repo_root / "build" / "core_cpu_mojo"
    if not binary.exists():
        raise FileNotFoundError("build/core_cpu_mojo is missing; run `pixi run build-core-cpu` first")
    result = subprocess.run(
        build_core_command(repo_root, params),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    trace = parse_trace_csv(result.stdout)
    trace["params"] = params
    return trace


class VisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            response = run_simulation(payload)
            self._send_json(200, response)
        except Exception as exc:  # noqa: BLE001 - surfaced to local dev UI
            self._send_json(400, {"error": str(exc)})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the core_cpu_mojo visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"core_cpu_mojo visualizer: {url}")
    print("Run `pixi run build-core-cpu` first if the binary is missing.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
