# Cascade AutoResearch Program

This is an autoresearch program for optimizing the Resistance Cascade ABM simulation.
Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology.

## Setup

1. **Create a branch**: `git checkout -b autoresearch/<tag>` from current main.
2. **Read the in-scope files**:
   - `benchmark.py` — fixed evaluation harness. DO NOT modify.
   - `resistance_cascade/model.py` — model class, data collector, step logic
   - `resistance_cascade/agent.py` — citizen and security agent classes
   - `resistance_cascade/grid.py` — toroidal multi-grid
   - `resistance_cascade/scheduler.py` — two-phase simultaneous activation
   - `resistance_cascade/random_walker.py` — base agent class
3. **Run baseline**: `uv run benchmark.py --compare --steps 200`
4. **Confirm and go**.

## Constraints

**What you CAN modify:**
- Any file in `resistance_cascade/` — model, agents, grid, scheduler

**What you CANNOT modify:**
- `benchmark.py` — this is the fixed evaluation harness
- The mathematical interactions: sigmoid, perception, arrest probability, opinion formation
- The RNG sequence: same seed must produce same fingerprint

**The goal**: Lower `run_time_s` while maintaining `fingerprint` match.

**Correctness is absolute**: If the fingerprint changes, the experiment is a FAILURE regardless
of speedup. Revert immediately.

**Simplicity criterion**: All else being equal, simpler is better. Removing code for equal
performance is a win. Adding complexity for <5% speedup is not worth it.

## Output format

After `uv run benchmark.py --compare --steps 200`:

```
Correctness:  PASS/FAIL
Performance:
  baseline run_time:    Xs
  current run_time:     Ys
  speedup:              Z.ZZx
```

## Logging results

Record each experiment in `results.tsv` (tab-separated):

```
commit	run_time_s	steps_per_sec	speedup	fingerprint	status	description
```

Status: `keep`, `discard`, or `crash`.

## The experiment loop

LOOP FOREVER:

1. Read the current state: `uv run benchmark.py --compare --steps 200`
2. Identify a bottleneck: `uv run python3 -c "import cProfile; ..."`
3. Modify the simulation code with an optimization idea
4. Run the benchmark: `uv run benchmark.py --compare --steps 200`
5. If PASS and faster: git commit, log `keep` in results.tsv
6. If PASS but equal/slower: git reset, log `discard`
7. If FAIL (fingerprint mismatch): git reset, log `discard`, investigate why
8. Repeat

**NEVER STOP**: Run until manually interrupted. If stuck, profile again,
try different approaches, combine previous near-misses.

## Ideas backlog

- Vectorize citizen decision-making with numpy arrays (batch sigmoid, batch random)
- Pre-compute all neighborhoods at init (they're position-dependent, positions change)
- Use __slots__ on agent classes to reduce memory/attribute access overhead
- Flatten condition strings to integer enums (avoid string comparisons)
- Use array-of-structs → struct-of-arrays for agent state
- Consider Cython or numba for the inner loop
- Reduce data collection frequency (collect every N steps instead of every step)
- Profile memory allocation and reduce GC pressure
