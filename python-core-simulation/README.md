# Python Core Simulation

A simplified, non-Mesa implementation of the Resistance Cascade simulation mechanics.

This folder is intended as a readable semantic bridge between `original_python/` and the Mojo ports. It keeps the core ABM mechanics without Mesa classes, visualization, batch-run plumbing, or notebook/reporting artifacts.

## Core mechanics included

- Citizen and Security agent types.
- Citizen states: `Support`, `Oppose`, `Active`, `Jailed`.
- Gaussian initialization for private preference, epsilon, and thresholds.
- Epsilon sigmoid transform.
- Toroidal grid coordinates.
- Moore-neighborhood vision with Mesa-style center-cell exclusion for perception/arrest scans.
- Citizen self-counting for Active and Support in the activation calculation.
- Perception, arrest probability, opinion, activation, active level, and oppose level calculations.
- One uniform random draw per citizen decision, with Active checked before Oppose.
- Simultaneous citizen decision/application semantics.
- Random movement including stay-put as one possible movement outcome.
- Security arrest behavior and jail/release mechanics.
- Revolution metric: active or jailed citizens >= 95%.
- Full per-agent trace collection and CSV export.

## Minimal example

```python
from cascade_core import ResistanceCascade

sim = ResistanceCascade(seed=42, security_density=0.02, max_iters=100)
sim.run()
print(sim.count_conditions())
sim.write_trace_csv("trace.csv")
```

## Notes

This implementation prioritizes clarity and faithful mechanics over speed. It uses Python standard-library `random.Random`, dataclasses, and lists. It does not attempt Mesa bit-exact behavior for every random-list ordering edge case, but it preserves the core model decisions that should be ported to CPU/GPU implementations.
