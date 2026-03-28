# Plan: Cascade ABM Phase Transition Manifolds

## Background

### What was done before

In the previous session, we built a Mojo GPU batch simulation engine for the
Resistance Cascade ABM and ran a coarse grid search of 187,110 simulations
(38.5 minutes on RTX 3090 at 81 sims/sec). Key findings:

- **Security density dominates**: revolution drops 69.5% to 0% between density
  0.0 and 0.025 (sharp phase transition at ~2%)
- **PP mean** has moderate effect (~2.5x range)
- **Threshold** provides linear control (32% at 1.5, 0% at 5.0)
- **Epsilon** has minimal effect (~2% variation)
- Results robust across 30 seeds (mean std = 0.038)

The existing GPU infrastructure (`grid_search_gpu.mojo`) processes batches of
2048 simulations at 81 sims/sec sustained throughput. The `MAX_AGENTS` buffer
was fixed to 1300 to handle high security densities.

### What Joe is asking for

Joe's blog post at https://www.joehelbing.net/post/schelling demonstrates
**3D manifold visualizations** for evaluating ABMs. For the Schelling model,
he created a 99x99 parameter grid (empty_ratio x similarity_threshold) and
plotted the outcome variable (mean_similarity) as a 3D surface using Plotly.
The resulting "segregation manifold" is an interactive 3D surface plot where:

- **X and Y axes** = two input parameters (the 2D parameter space)
- **Z axis** = the outcome metric (e.g., mean similarity, revolution rate)
- **Color** = mapped to Z value (viridis colorscale)
- The surface reveals **phase transitions** as sharp ridges, cliffs, or
  plateaus in the 3D surface

The blog shows:
1. **3D surface plots** (Plotly `Surface`) of the full parameter manifold
2. **Multiple viewing angles** of the same manifold to show different features
3. **ML surrogate comparison** (ground truth vs XGBoost prediction vs error
   surface side-by-side)
4. **Absolute error surfaces** showing where the phase transitions are hardest
   to predict

Joe wants the same treatment for the Cascade ABM: fine-grained 2D parameter
sweeps visualized as 3D manifolds, revealing the phase transition structure
across parameter pairs.

## Plan

### Step 1: Design the manifold grid searches

Create 6 manifold sweeps covering all pairwise combinations of the 4 key
parameters. Each manifold is a 2D grid with the outcome variable (revolution
probability, averaged over seeds) as the Z axis.

For each manifold, the two non-swept parameters are held at their default
values:
- pp_mean default: 0.0
- sec_density default: 0.02 (near the phase transition)
- epsilon default: 0.5
- threshold default: 2.94444 (original model default)

| # | X axis | Y axis | Grid | x 20 seeds | Sims |
|---|--------|--------|------|------------|------|
| 1 | pp_mean (-1.0 to 1.0) | sec_density (0.0 to 0.05) | 51x51 | 20 | 52,020 |
| 2 | pp_mean (-1.0 to 1.0) | threshold (1.0 to 6.0) | 51x51 | 20 | 52,020 |
| 3 | sec_density (0.0 to 0.05) | threshold (1.0 to 6.0) | 51x51 | 20 | 52,020 |
| 4 | pp_mean (-1.0 to 1.0) | epsilon (0.01 to 2.0) | 51x51 | 20 | 52,020 |
| 5 | sec_density (0.0 to 0.05) | epsilon (0.01 to 2.0) | 51x51 | 20 | 52,020 |
| 6 | epsilon (0.01 to 2.0) | threshold (1.0 to 6.0) | 51x51 | 20 | 52,020 |

**Total: 312,120 simulations**

NOTE: Security density range is capped at 0.05 (not 0.10) because the coarse
search showed zero revolution above 0.025 -- the interesting structure is all
below 0.05. This concentrates resolution where the phase transitions live.

Optionally, add higher-resolution "zoom" manifolds on the most interesting
regions:

| # | X axis | Y axis | Grid | x 20 seeds | Sims |
|---|--------|--------|------|------------|------|
| 7 | pp_mean (-1.0 to 1.0) | sec_density (0.005 to 0.025) | 101x101 | 20 | 204,020 |
| 8 | sec_density (0.005 to 0.025) | threshold (1.5 to 4.0) | 101x101 | 20 | 204,020 |

**With zoom manifolds: 720,160 total simulations**

### Step 2: Modify `grid_search_gpu.mojo` for manifold mode

The current grid_search_gpu.mojo does a full 5D combinatorial sweep. For
manifolds, we need 2D sweeps with fixed defaults for the other parameters.
Two approaches:

**Option A (simpler):** Create a new `manifold_search_gpu.mojo` that takes
the manifold configuration as compile-time constants and runs a single 2D
sweep. Run it 6-8 times, once per manifold.

**Option B (more flexible):** Modify grid_search_gpu.mojo to accept a mode
flag or config that specifies which 2 parameters to sweep and what defaults
to use for the rest. More complex but single binary.

**Recommendation: Option A.** Each manifold is a separate build+run. The
compile times are ~10 seconds and the simplicity is worth it. Can even
template the code so each manifold is just different constants at the top.

Actually, the simplest approach: make a single `manifold_search_gpu.mojo`
with ALL manifolds defined sequentially. It processes one manifold at a time,
writing each to a separate CSV file. The total runtime is the sum of all
manifold runtimes.

### Step 3: Run the simulation

**Estimated timing** (at 468 sims/sec with block-per-sim kernel):

| Configuration | Sims | Time | GPU Hours |
|---------------|------|------|-----------|
| 6 base manifolds (51x51x20) | 312,120 | 11 min | 0.2 hr |
| + 2 zoom manifolds (101x101x20) | 720,160 | 26 min | 0.4 hr |
| Full resolution (99x99x20, all 6) | 1,176,120 | 42 min | 0.7 hr |

NOTE: Previous estimates used 81 sims/sec (one-thread-per-sim). The
block-per-sim kernel with spatial grid achieves 468 sims/sec (5.8x faster).

**Recommendation:** Start with the 6 base manifolds at 51x51 resolution
(~1 hour). If the surfaces look good but need more resolution in the phase
transition zones, add the zoom manifolds (~2.5 hours total). If you want
blog-quality 99x99 resolution on all manifolds, that's ~4 hours.

**Data size:** ~25 MB for base, ~58 MB with zooms, ~94 MB at full resolution.
All well under 100GB limit.

### Step 4: Generate manifold visualizations

Create `visualize_manifolds.py` that reads each manifold CSV and generates:

1. **Interactive 3D surface plots** (Plotly `go.Surface`) for each manifold
   - X, Y = swept parameters
   - Z = revolution probability (averaged over seeds)
   - Color = Z value with viridis or similar colorscale
   - Save as interactive HTML files (like Joe's Schelling blog)
   - Save as static PNG for the report

2. **Multi-angle views** for the most interesting manifolds (pp_mean x
   sec_density, sec_density x threshold) showing the phase transition from
   different perspectives

3. **Seed variance surface** for each manifold -- Z = std(revolution) across
   seeds, showing where the model is stochastic vs deterministic

4. **Composite view** -- small multiples of all 6 manifolds in one figure

Additional metrics to compute as Z-axis alternatives:
- Revolution probability (binary, averaged over seeds)
- Mean active count at final step
- Mean jailed count at final step
- Time to first revolution (if any)

### Step 5: Compile report

Create a showboat report in Basic Memory with:
- All manifold visualizations
- Analysis of phase transition geometry
- Comparison with the coarse grid search findings
- Interactive HTML files linked for exploration

### Output files

```
mojo_cascade/
  manifold_search_gpu.mojo       # GPU simulation code for manifold sweeps
  manifold_results/              # Output directory
    manifold_1_ppmean_secdens.csv
    manifold_2_ppmean_threshold.csv
    manifold_3_secdens_threshold.csv
    manifold_4_ppmean_epsilon.csv
    manifold_5_secdens_epsilon.csv
    manifold_6_epsilon_threshold.csv
    manifold_7_zoom_ppmean_secdens.csv  (optional)
    manifold_8_zoom_secdens_threshold.csv  (optional)
  visualize_manifolds.py         # Plotly 3D surface generation
  charts/
    manifold_*.png               # Static images for report
    manifold_*.html              # Interactive Plotly files
```

## Execution checklist

When ready to run, point Claude Code at this plan and say "execute this plan."

- [ ] Create `manifold_search_gpu.mojo` with all manifold configurations
- [ ] Build: `cd mojo_cascade && pixi run mojo build manifold_search_gpu.mojo -o manifold_search_gpu` (base on block_per_sim.mojo kernel for 5.8x throughput)
- [ ] Create output directory: `mkdir -p manifold_results`
- [ ] Run: `./manifold_search_gpu` (writes CSVs to manifold_results/)
- [ ] Create `visualize_manifolds.py` with Plotly 3D surface generation
- [ ] Run visualization: `uv run --with plotly --with pandas --with kaleido python visualize_manifolds.py`
- [ ] Review manifolds, decide if zoom manifolds are needed
- [ ] If zoom needed, rebuild with zoom manifolds enabled and rerun
- [ ] Compile showboat report with all visualizations
- [ ] Link report from daily note

## Key technical notes

- `MAX_AGENTS` must be 1300 (not 1200) to handle sec_density up to 0.05
  (1120 citizens + 80 security = 1200 agents, barely fits at 1300 buffer)
- Build from `mojo_cascade/` directory (not parent) due to pixi.toml location
- Use `pixi run mojo build` for compilation
- GPU kernel uses UnsafePointer parameters (not LayoutTensor)
- Output CSV has spaces around commas from Mojo's print -- account for this
  in Python parsing with `.strip()`
- Batch size 2048 is optimal for RTX 3090
- `def main() raises:` required for DeviceContext
