# Plan: Cascade ABM Full Parameter Sweep & Manifold Construction

## Background

### What was done before

Built a Mojo GPU batch simulation engine for the Resistance Cascade ABM.
Previous coarse grid search of 187,110 simulations (38.5 min on RTX 3090 at
81 sims/sec) found:

- **Security density dominates**: revolution drops 69.5% to 0% between density
  0.0 and 0.025 (sharp phase transition at ~2%)
- **PP mean** has moderate effect (~2.5x range)
- **Threshold** provides linear control (32% at 1.5, 0% at 5.0)
- **Epsilon** has minimal effect (~2% variation)
- Results robust across 30 seeds (mean std = 0.038)

### Cross-validation status (verified 2026-03-29)

Two-link chain validates correctness with one shared `CpuSim` struct:

- **Link 1**: Python model <-> `CpuSim(use_python=True)` — 45/45 PASS,
  9,000 field checks bit-identical
- **Link 2**: `CpuSim(use_python=False)` <-> GPU kernel — 45/45 PASS,
  9,000 field checks bit-identical

All validation in single file: `cross_validate_chain.mojo`.

Bug fixed during consolidation: LCG arrest path was comparing
`activation_val` (Float32) against `threshold_sig_f64` (Float64 sigmoid).
Changed to `threshold_sig_f32` to match GPU precision.

## Architecture: 1024-agent single-block optimization

### Key insight

NVIDIA GPUs have a **1024 thread-per-block limit**. If total agents <= 1024,
we get one thread per agent in a single block:

- **No per-thread loop** -- perfect 1:1 thread-to-agent mapping
- **barrier() is block-local** -- cheapest possible synchronization
- **All agent data fits in shared memory** -- ~56 bytes/agent x 1024 = 56KB
  (RTX 3090 supports up to 100KB shared per block)
- **Shared memory latency**: ~5 cycles vs ~400 cycles for global memory
  = ~80x faster neighbor reads

### Grid sizing

To fit 1024 agents at max total density 0.9 (citizen 0.8 + security 0.1):

- 0.9 * grid_cells = 1024 => grid_cells = 1138
- **33x33 = 1089 cells** => 0.8 * 1089 = 871 citizens + 0.1 * 1089 = 109 security = 980 max agents

Constants:
```
GRID_W = 33
GRID_H = 33
MAX_AGENTS = 1024
BLOCK_SIZE = 256  (1024 threads exceeded register file -- 256 with ~4 agents/thread)
```

### Measured throughput

Previous (BLOCK_SIZE=256, MAX_AGENTS=1300, 40x40): ~468 sims/sec

Original (33x33, MAX_AGENTS=1024, no shared memory): 1,850 sims/sec (3.9x)

**Current (after autoresearch optimization): ~10,900 sims/sec** (5.96x over 33x33 baseline)

Key optimizations (cumulative):
1. Spatial grid arrest scan for Phase 3: 1,806 -> 6,100 (3.38x)
2. Parallel grid clear + revolution check: -> 6,550 (3.63x)
3. Atomic parallel grid insert: -> 7,700 (4.26x)
4. Citizens-only grid + per-cell security counts: -> 8,830 (4.89x)
5. int16/int8 shared memory (grid_cells, counts, cond): -> 10,900 (6.03x)

## Plan

### Step 1: Full 4D parameter sweep

Sweep all 4 key parameters at 25 points each, 20 seeds per config.

**Parameters:**

| Parameter | Range | Points | Step |
|-----------|-------|--------|------|
| pp_mean | -1.0 to 1.0 | 25 | 0.0833 |
| sec_density | 0.0 to 0.10 | 25 | 0.00417 |
| epsilon | 0.01 to 2.0 | 25 | 0.0829 |
| threshold | 1.0 to 6.0 | 25 | 0.2083 |

**Fixed parameters:**
- citizen_density: 0.7
- max_jail: 100
- vision: 7
- n_steps: 100
- grid: 33x33

**Total: 25^4 x 20 = 7,812,500 simulations**

**Estimated time: ~24 min** (at 10,900 sims/sec, 100 steps = ~2x runtime vs 50)

**Output: SQLite database** via APSW (Python interop from Mojo).
- `simulations` table: per-sim summary (params, max_active, revolution_step, cascade count)
- `model_steps` table: per-step metrics (active, support, oppose, jail, revolution)
- Estimated DB size: ~27 GB (7.8M sims x 100 steps x ~35 bytes/row + indexes)
- WAL mode + deferred indexing for bulk insert performance

Note: sec_density range expanded to 0.10 (previously 0.05) since the smaller
grid now accommodates higher security density within the 1024 agent cap.
At citizen_density=0.7: 762 citizens + 0.1*1089=109 security = 871 total.

### Step 2: Manifold extraction and visualization

Extract 2D manifold slices from the 4D dataset:

1. **6 pairwise manifolds** (C(4,2) = 6): For each pair, average revolution
   probability over the other 2 parameters and 20 seeds. Smooth 25x25 surfaces.

2. **Conditional manifolds**: Fix non-swept parameters at specific values
   (not averaged) to see how the phase transition surface shifts.

3. **3D surface plots** (Plotly `go.Surface`) matching Joe's Schelling blog:
   X, Y = swept params, Z = revolution probability, viridis colorscale.

4. **Interaction analysis**: Check parameter interactions, especially whether
   epsilon has effects conditional on other parameter values.

### Step 3: Optional high-resolution zoom

If coarse sweep reveals interesting structure, run targeted zooms:

| Region | Grid | Seeds | Sims | Time |
|--------|------|-------|------|------|
| sec_density 0.005-0.025 x threshold 1.5-4.0 | 101x101 | 20 | 204,020 | ~1-2 min |
| pp_mean -1.0 to 1.0 x sec_density 0.005-0.025 | 101x101 | 20 | 204,020 | ~1-2 min |

### Output files

```
mojo_cascade/
  manifold_search_gpu.mojo       # GPU simulation: 1024-agent single-block kernel
  manifold_results/              # Output directory
    manifold.db                  # SQLite database (APSW, per-step + summary data)
  cross_validate_chain.mojo      # Both Link 1 and Link 2 validation (one CpuSim struct)
  visualize_manifolds.py         # Plotly 3D surface generation
  charts/
    manifold_*.png               # Static images for report
    manifold_*.html              # Interactive Plotly files
```

## Execution checklist

- [x] Update block_per_sim.mojo: GRID_W/H=33, MAX_AGENTS=1024, BLOCK_SIZE=256
- [x] Add shared memory for agent arrays in kernel (int8/int16 optimized ~23KB)
- [x] Cross-validation chain consolidated (one CpuSim, both links in cross_validate_chain.mojo)
- [x] Create `manifold_search_gpu.mojo` with APSW SQLite output
      - 25^4 parameter grid, 20 seeds per config
      - SQLite output: simulations table (summary) + model_steps table (per-step)
      - Process in batches of 2048 sims (GPU optimal)
- [ ] Update MAX_STEPS to 100 in manifold_search_gpu.mojo and block_per_sim.mojo
- [ ] Build: `cd mojo_cascade && pixi run mojo build manifold_search_gpu.mojo -o manifold_search_gpu`
- [ ] Run: `./manifold_search_gpu` (~24 min)
- [ ] Create `visualize_manifolds.py` with Plotly 3D surface generation
- [ ] Run visualization
- [ ] Review manifolds, decide if zoom sweeps needed
- [ ] Compile showboat report
- [ ] Link report from daily note

## Key technical notes

- MAX_AGENTS=1024, BLOCK_SIZE=256 (256 threads per block, ~4 agents/thread)
- Grid 33x33 = 1089 cells, max 980 agents at 0.9 total density
- citizen_density=0.7 gives 762 citizens, leaving 262 slots for security
- Build from `mojo_cascade/` directory (pixi.toml location)
- `pixi run mojo build` for compilation
- GPU kernel uses UnsafePointer parameters (not LayoutTensor)
- Shared memory ~23KB: s_cond (int8), s_grid_counts (int16), s_sec_counts (int16), s_grid_cells (int16)
- Citizens-only spatial grid + per-cell security counts
- Parallel grid build with atomics, parallel revolution check with tree reduction
- `def main() raises:` required for DeviceContext
- 7.8M sims launched as individual blocks, one block per sim
- Output via APSW SQLite (Python interop): WAL mode, synchronous=OFF, deferred indexes
