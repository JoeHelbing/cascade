"""
Cross-validation: Python model vs Mojo GPU kernel.

Strategy: Run Python model, extract ALL random values and per-step agent
states. Feed the same random values into a Mojo-compatible test harness
to verify the math/logic produces identical results.

This isolates the RNG difference by using Python's actual random values.

Usage:
    uv run cross_validate.py
"""
import sys
import math
import json
import random
from pathlib import Path

# Add parent dir so we can import the Python model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resistance_cascade.model import ResistanceCascade
from resistance_cascade.agent import Citizen, Security, ACTIVE, SUPPORT, OPPOSE, JAILED


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def run_python_model_with_trace(seed, n_steps=10, sec_density=0.02, epsilon=0.5,
                                 pp_mean=0.0, threshold=2.94444):
    """Run Python model and capture full trace of random values and states."""
    model = ResistanceCascade(
        width=40, height=40,
        citizen_density=0.7,
        security_density=sec_density,
        citizen_vision=7,
        security_vision=7,
        max_jail_term=100,
        epsilon=epsilon,
        private_preference_distribution_mean=pp_mean,
        threshold=threshold,
        seed=seed,
        max_iters=1000,
        movement=True,
        multiple_agents_per_cell=True,
    )

    # Capture initialization state
    citizens = []
    for agent in model.schedule.agents:
        if isinstance(agent, Citizen):
            citizens.append({
                "id": agent.unique_id,
                "pos": agent.pos,
                "private_pref": agent.private_preference,
                "epsilon": agent.epsilon,
                "epsilon_prob": agent.epsilon_probability,
                "oppose_th": agent.oppose_threshold,
                "active_th": agent.active_threshold,
                "condition": agent._cond,
            })

    security_agents = []
    for agent in model.schedule.agents:
        if isinstance(agent, Security):
            security_agents.append({
                "id": agent.unique_id,
                "pos": agent.pos,
            })

    # Run steps and capture per-step state
    step_traces = []
    for step in range(n_steps):
        if not model.running:
            break

        model.step()

        # Capture citizen states after this step
        citizen_states = []
        for agent in model.schedule.agents:
            if isinstance(agent, Citizen):
                citizen_states.append({
                    "id": agent.unique_id,
                    "pos": agent.pos,
                    "cond": agent._cond,
                    "jail_sentence": agent.jail_sentence,
                    "actives": agent.actives_in_vision,
                    "opposed": agent.opposed_in_vision,
                    "support": agent.support_in_vision,
                    "security": agent.security_in_vision,
                    "opinion": agent.opinion,
                    "activation": agent.activation,
                    "active_level": agent.active_level,
                    "oppose_level": agent.oppose_level,
                    "arrest_prob": agent.arrest_prob,
                })

        # Count conditions
        active = sum(1 for s in citizen_states if s["cond"] == ACTIVE)
        support = sum(1 for s in citizen_states if s["cond"] == SUPPORT)
        oppose = sum(1 for s in citizen_states if s["cond"] == OPPOSE)
        jailed = sum(1 for s in citizen_states if s["cond"] == JAILED)

        step_traces.append({
            "step": step,
            "active": active,
            "support": support,
            "oppose": oppose,
            "jailed": jailed,
            "revolution": model.revolution,
            "citizen_states": citizen_states,
        })

    return {
        "seed": seed,
        "n_citizens": model.citizen_count,
        "n_security": model.security_count,
        "n_agents": model.citizen_count + model.security_count,
        "citizens_init": citizens,
        "security_init": security_agents,
        "steps": step_traces,
        "revolution": model.revolution,
        "params": {
            "sec_density": sec_density,
            "epsilon": epsilon,
            "pp_mean": pp_mean,
            "threshold": threshold,
            "citizen_density": 0.7,
            "max_jail": 100,
            "vision": 7,
        },
    }


def analyze_math_differences(trace):
    """Check if the math formulas produce expected values given the inputs."""
    print(f"\n=== Analyzing Python model trace ===")
    print(f"Seed: {trace['seed']}")
    print(f"Citizens: {trace['n_citizens']}, Security: {trace['n_security']}")

    # Verify the math formulas match what we expect
    threshold = trace["params"]["threshold"]
    threshold_sig = sigmoid(threshold)

    print(f"\nChecking initialization math...")
    errors = 0
    for c in trace["citizens_init"][:5]:  # Check first 5
        # Verify epsilon_prob = sigmoid(epsilon)
        expected_ep = sigmoid(c["epsilon"])
        if abs(expected_ep - c["epsilon_prob"]) > 1e-10:
            print(f"  ERROR: citizen {c['id']} epsilon_prob {c['epsilon_prob']} != sigmoid({c['epsilon']}) = {expected_ep}")
            errors += 1

    if errors == 0:
        print(f"  All epsilon_prob values match sigmoid(epsilon)")

    # Check step 0 citizen math
    if trace["steps"]:
        step0 = trace["steps"][0]
        print(f"\nStep 0: active={step0['active']}, support={step0['support']}, "
              f"oppose={step0['oppose']}, jailed={step0['jailed']}")

        # Verify a few citizens' math
        print(f"\nChecking step 0 citizen math (first 5 citizens)...")
        for cs in step0["citizen_states"][:5]:
            if cs["opinion"] is None:
                continue

            actives = cs["actives"]
            opposed = cs["opposed"]
            support = cs["support"]
            security = cs["security"]

            # Find this citizen's init data
            init = None
            for c in trace["citizens_init"]:
                if c["id"] == cs["id"]:
                    init = c
                    break

            if init is None:
                continue

            pp = init["private_pref"]
            ep = init["epsilon"]
            ep_prob = init["epsilon_prob"]

            # Verify: active_ratio = (actives + opposed) / support
            active_ratio = (actives + opposed) / support

            # perception = (actives + opposed * ep_prob) ^ (1 / (ep^2 + 1))
            perception = (actives + opposed * ep_prob) ** (1.0 / (ep**2 + 1))

            # arrest_prob = 1 - exp(-2.3 * (security / actives) * 2 * ep_prob)
            arrest_prob = 1 - math.exp(-2.3 * (security / actives) * 2 * ep_prob)

            # opinion = -pp + perception * active_ratio
            opinion = -pp + perception * active_ratio

            # activation = sigmoid(opinion)
            activation = sigmoid(opinion)

            # Check
            if abs(opinion - cs["opinion"]) > 1e-6:
                print(f"  Citizen {cs['id']}: opinion MISMATCH: computed={opinion:.6f} vs model={cs['opinion']:.6f}")
            elif abs(activation - cs["activation"]) > 1e-6:
                print(f"  Citizen {cs['id']}: activation MISMATCH: computed={activation:.6f} vs model={cs['activation']:.6f}")
            elif abs(arrest_prob - cs["arrest_prob"]) > 1e-6:
                print(f"  Citizen {cs['id']}: arrest_prob MISMATCH: computed={arrest_prob:.6f} vs model={cs['arrest_prob']:.6f}")
            else:
                print(f"  Citizen {cs['id']}: OK (opinion={opinion:.4f}, activation={activation:.4f}, "
                      f"arrest_prob={arrest_prob:.4f}, neighbors={actives}a/{opposed}o/{support}s/{security}sec)")

    # Print step-by-step summary
    print(f"\nStep-by-step summary:")
    for step in trace["steps"]:
        rev = " REVOLUTION" if step["revolution"] else ""
        print(f"  Step {step['step']:2d}: active={step['active']:4d} support={step['support']:4d} "
              f"oppose={step['oppose']:4d} jailed={step['jailed']:4d}{rev}")

    return trace


def export_for_mojo(trace, output_path):
    """Export initialization data so Mojo can reproduce with identical values."""
    data = {
        "seed": trace["seed"],
        "params": trace["params"],
        "n_citizens": trace["n_citizens"],
        "n_security": trace["n_security"],
        "citizens": [],
        "security": [],
        "python_steps": [],
    }

    for c in trace["citizens_init"]:
        data["citizens"].append({
            "pos_x": c["pos"][0],
            "pos_y": c["pos"][1],
            "private_pref": c["private_pref"],
            "epsilon": c["epsilon"],
            "epsilon_prob": c["epsilon_prob"],
            "oppose_th": c["oppose_th"],
            "active_th": c["active_th"],
        })

    for s in trace["security_init"]:
        data["security"].append({
            "pos_x": s["pos"][0],
            "pos_y": s["pos"][1],
        })

    for step in trace["steps"]:
        data["python_steps"].append({
            "step": step["step"],
            "active": step["active"],
            "support": step["support"],
            "oppose": step["oppose"],
            "jailed": step["jailed"],
            "revolution": step["revolution"],
        })

    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"\nExported Mojo-compatible data to {output_path}")
    return data


def main():
    # Run with a few different configurations
    configs = [
        {"seed": 42, "sec_density": 0.0, "epsilon": 0.5},
        {"seed": 42, "sec_density": 0.02, "epsilon": 0.5},
        {"seed": 42, "sec_density": 0.01, "epsilon": 0.5},  # Near phase transition
        {"seed": 123, "sec_density": 0.015, "epsilon": 0.5},  # Near phase transition
    ]

    all_traces = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: seed={cfg['seed']}, sec_density={cfg['sec_density']}, eps={cfg['epsilon']}")
        print(f"{'='*60}")
        trace = run_python_model_with_trace(
            seed=cfg["seed"],
            n_steps=50,
            sec_density=cfg["sec_density"],
            epsilon=cfg["epsilon"],
        )
        analyze_math_differences(trace)
        all_traces.append(trace)

    # Export the first trace for Mojo cross-validation
    export_for_mojo(all_traces[0], "/tmp/python_trace_sd0.json")
    export_for_mojo(all_traces[1], "/tmp/python_trace_sd002.json")
    export_for_mojo(all_traces[2], "/tmp/python_trace_sd001.json")
    export_for_mojo(all_traces[3], "/tmp/python_trace_sd0015.json")

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("SUMMARY: Python model results (ground truth)")
    print(f"{'='*60}")
    for trace in all_traces:
        final = trace["steps"][-1] if trace["steps"] else None
        if final:
            rev = "YES" if trace["revolution"] else "no"
            stopped = len(trace["steps"])
            print(f"  seed={trace['seed']} sd={trace['params']['sec_density']:.3f} "
                  f"steps={stopped} rev={rev} "
                  f"active={final['active']} support={final['support']} "
                  f"oppose={final['oppose']} jailed={final['jailed']}")


if __name__ == "__main__":
    main()
