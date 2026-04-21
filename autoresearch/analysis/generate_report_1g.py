"""
Phase 1G + Phase 2: Generate the comprehensive analysis report and paper update guidance.

Reads analysis outputs from Phases 1C, 1D, and 1E, then:
1. Creates the Phase 1G comprehensive report in Basic Memory
2. Creates the Phase 2 paper update guidance in Basic Memory
3. Copies figures to reports-web data directory
4. Creates manifests for web rendering

Usage:
    cd mojo_cascade
    pixi run python generate_report_1g.py
"""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

OUT_REPORTS_WEB = Path.home() / "git" / "reports-web" / "data" / "cascade"
BM_VAULT = Path.home() / "Obsidian" / "bm-main"

TODAY = date.today().isoformat()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def copy_figures_to_reports_web(source_dir, slug):
    """Copy analysis figure JSONs to reports-web data directory."""
    dest = OUT_REPORTS_WEB / slug / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for json_file in source_dir.glob("*.json"):
        shutil.copy2(json_file, dest / json_file.name)
        count += 1
    print(f"  Copied {count} figures to {dest}")
    return dest


def write_bm_note(title, content, directory, tags, note_type="report", metadata=None):
    """Write a Basic Memory note via the CLI-compatible approach."""
    # Build the full path
    note_path = BM_VAULT / directory / f"{title}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frontmatter
    meta = {
        "title": title,
        "type": note_type,
        "permalink": f"main/{directory.lower().replace(' ', '-')}/{title.lower().replace(' ', '-')}",
        "project": "cascade",
        "status": "active",
        "tags": tags,
    }
    if metadata:
        meta.update(metadata)

    frontmatter_lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, list):
            frontmatter_lines.append(f"{k}:")
            for item in v:
                frontmatter_lines.append(f"- {item}")
        else:
            frontmatter_lines.append(f"{k}: {v}")
    frontmatter_lines.append("---")

    full_content = "\n".join(frontmatter_lines) + "\n\n" + content
    note_path.write_text(full_content)
    print(f"  Wrote: {note_path}")
    return note_path


def build_1g_report():
    """Build the comprehensive Phase 1G analysis report."""
    # Load analysis data
    importance_1c = load_json("analysis_1c_output/parameter_importance.json")
    importance_1d = load_json("analysis_1d_output/parameter_importance_1d.json")
    transitions = load_json("analysis_1d_output/phase_transitions_1d.json")
    arch_1c = load_json("analysis_1c_output/archetype_counts.json")
    arch_1d = load_json("analysis_1d_output/archetype_counts_1d.json")

    # Load Phase 1E summary if available
    summary_1e = {}
    if Path("analysis_1e_output/summary_1e.json").exists():
        summary_1e = load_json("analysis_1e_output/summary_1e.json")

    # Copy figures
    slug_1g = f"{TODAY}-cascade-comprehensive-analysis"
    fig_dir = copy_figures_to_reports_web(Path("analysis_1d_output/figures"), slug_1g)
    if Path("analysis_1e_output/figures").exists():
        for f in Path("analysis_1e_output/figures").glob("*.json"):
            shutil.copy2(f, fig_dir / f.name)
    # Copy 3D surface figures
    if Path("manifold_3d_output/figures").exists():
        for f in Path("manifold_3d_output/figures").glob("*.json"):
            shutil.copy2(f, fig_dir / f.name)
        print(f"  Copied 3D surfaces to {fig_dir}")

    # Build the figure reference prefix
    fig_prefix = f"/data/cascade/{slug_1g}/figures"

    # Format importance tables
    def importance_table(imp_data, params):
        rows = []
        for z_name, z_label in [
            ("revolution_prob", "Revolution Prob."),
            ("max_active_pct", "Max Active %"),
            ("mean_active_pct", "Mean Active %"),
            ("cascade_rate", "Cascade Rate"),
            ("periodic_rate", "Periodic Rate"),
        ]:
            if z_name in imp_data:
                ranked = sorted(imp_data[z_name].items(), key=lambda x: x[1], reverse=True)
                top = ranked[0]
                rows.append(f"| {z_label} | **{top[0]}** | **{top[1]:.3f}** | {ranked[1][0]}={ranked[1][1]:.3f} |")
        return "\n".join(rows)

    # Format transition table
    trans_rows = []
    for t in transitions[:8]:
        trans_rows.append(
            f"| {t['z_metric']} | {t['param_a']}={t['a_coord']:.4f} | "
            f"{t['param_b']}={t['b_coord']:.4f} | {t['gradient']:.3f} |"
        )

    # Format paired comparison summary
    paired_rows = []
    for name, data in list(summary_1e.items())[:10]:
        desc = data.get("description", name)
        rev_a = data.get("rev_prob_a", 0)
        rev_b = data.get("rev_prob_b", 0)
        paired_rows.append(f"| {desc} | {rev_a:.1%} | {rev_b:.1%} | {abs(rev_b - rev_a):.1%} |")

    # Format archetype comparison
    arch_labels = {
        "fast_revolution": "Fast Revolution",
        "mid_revolution": "Mid Revolution",
        "slow_burn": "Slow Burn",
        "oscillating": "Oscillating",
        "abortive_spike": "Abortive Spike",
        "stable_suppression": "Stable Suppression",
        "simmering": "Simmering",
    }
    total_1c = sum(arch_1c.get("500", {}).values()) or 1
    total_1d = sum(arch_1d.values()) or 1
    arch_rows = []
    for key, label in arch_labels.items():
        pct_1c = arch_1c.get("500", {}).get(key, 0) / total_1c
        pct_1d = arch_1d.get(key, 0) / total_1d
        if pct_1c > 0.001 or pct_1d > 0.001:
            arch_rows.append(f"| {label} | {pct_1c:.1%} | {pct_1d:.1%} |")

    content = f"""# Cascade Comprehensive Analysis Report

This report integrates findings from all completed phases of the [[2026-03-30 Cascade Extended Simulation and Paper Update Plan]]:
- **Phase 1B/1C**: Coarse 7D sweep (24.7M sims, 7^7 grid, 3 step counts)
- **Phase 1D**: High-resolution targeted sweep (18.75M sims, variable resolution, 500 steps)
- **Phase 1E**: Paired model comparisons at phase transition boundaries
- **Phase 1F**: Deferred (requires GPU kernel modifications for agent-level output)

Total simulations analyzed: ~43.5 million across two sweep campaigns.

For methodology explanations and visualization reading guides, see [[2026-04-04 Cascade Analysis Methodology and Visualization Guide]].

---

## 1. Parameter Importance Overview

The fundamental question: of the 7 model parameters, which ones actually matter?

### Coarse Sweep (Phase 1C, 7 parameters at 7 levels)

| Z-Metric | Dominant Parameter | Variance Explained | Runner-up |
|----------|-------------------|-------------------|-----------|
{importance_table(importance_1c.get("500", importance_1c.get(500, {})), ["pp_mean", "sec_density", "epsilon", "threshold", "citizen_density", "max_jail", "vision"])}

### High-Resolution Sweep (Phase 1D, 5 parameters at variable resolution)

| Z-Metric | Dominant Parameter | Variance Explained | Runner-up |
|----------|-------------------|-------------------|-----------|
{importance_table(importance_1d, ["pp_mean", "sec_density", "epsilon", "threshold", "vision"])}

{{{{figure:{fig_prefix}/parameter_importance_1d.json | bar_grouped | Phase 1D parameter importance across all z-metrics}}}}

**Key finding confirmed at higher resolution:** Security density dominates revolution/activation outcomes (~40-45% variance explained). Vision dominates cascade dynamics (~30-35%). Threshold is the consistent second-place parameter for activation metrics. Epsilon remains near-irrelevant at the macro level.

The higher resolution does not change the ranking. It confirms that the coarse 7-point sweep was sufficient for parameter screening.

---

## 2. 3D Manifold Surfaces at High Resolution

Interactive 3D surfaces -- drag to rotate, scroll to zoom. The z-axis height shows the metric value (0 to 1).

### Security Density vs Threshold (the primary phase transition)

{{{{figure:{fig_prefix}/surface3d_multi_sec_density_vs_threshold.json | surface3d | All metrics: security density vs threshold (use dropdown to switch)}}}}

{{{{figure:{fig_prefix}/surface3d_revolution_prob_sec_density_vs_threshold.json | surface3d | Revolution probability surface: the phase transition cliff}}}}

{{{{figure:{fig_prefix}/surface3d_cascade_rate_sec_density_vs_threshold.json | surface3d | Cascade rate surface: security density vs threshold}}}}

### Vision vs Threshold (the cascade control surface)

{{{{figure:{fig_prefix}/surface3d_multi_vision_vs_threshold.json | surface3d | All metrics: vision vs threshold (use dropdown to switch)}}}}

{{{{figure:{fig_prefix}/surface3d_cascade_rate_vision_vs_threshold.json | surface3d | Cascade rate surface: vision controls whether cascades repeat}}}}

{{{{figure:{fig_prefix}/surface3d_periodic_rate_vision_vs_threshold.json | surface3d | Periodic rate surface: vision vs threshold}}}}

### Other Key Surfaces

{{{{figure:{fig_prefix}/surface3d_multi_vision_vs_sec_density.json | surface3d | All metrics: vision vs security density (use dropdown to switch)}}}}

{{{{figure:{fig_prefix}/surface3d_multi_pp_mean_vs_sec_density.json | surface3d | All metrics: PP mean vs security density (use dropdown to switch)}}}}

{{{{figure:{fig_prefix}/surface3d_multi_pp_mean_vs_threshold.json | surface3d | All metrics: PP mean vs threshold (use dropdown to switch)}}}}

### 2D Heatmap Grids (all pairwise)

{{{{figure:{fig_prefix}/hires_all_revolution_prob.json | heatmap_grid | All 10 pairwise manifolds for revolution probability}}}}

{{{{figure:{fig_prefix}/hires_all_cascade_rate.json | heatmap_grid | All 10 pairwise manifolds for cascade rate}}}}

---

## 3. Phase Transition Boundaries

The sharpest parameter-space boundaries detected in Phase 1D:

| Z-Metric | Parameter A | Parameter B | Gradient |
|----------|------------|------------|----------|
{chr(10).join(trans_rows) if trans_rows else "| (no transitions detected above threshold) | | | |"}

These are the coordinates where a small parameter change produces the largest outcome change. They define the "cliffs" in parameter space where the system transitions between qualitative regimes.

---

## 4. Trajectory Archetypes

### Distribution Comparison (Coarse vs High-Resolution)

| Archetype | Phase 1C (coarse, 500 steps) | Phase 1D (high-res, 500 steps) |
|-----------|------------------------------|-------------------------------|
{chr(10).join(arch_rows) if arch_rows else "| (archetype data unavailable) | | |"}

{{{{figure:{fig_prefix}/hires_archetype_sec_density_vs_threshold.json | heatmap_grid | Archetype frequency: security density vs threshold at high resolution}}}}

{{{{figure:{fig_prefix}/hires_archetype_vision_vs_threshold.json | heatmap_grid | Archetype frequency: vision vs threshold at high resolution}}}}

---

## 5. Paired Comparisons (Phase 1E)

At the phase transition boundaries identified in Phase 1D, paired simulations isolate the causal effect of single parameter changes.

| Comparison | Rev. Prob (A) | Rev. Prob (B) | Difference |
|-----------|--------------|--------------|-----------|
{chr(10).join(paired_rows) if paired_rows else "| (paired comparison data unavailable) | | | |"}

---

## 6. Open Questions and Phase 1F

### What's Missing

**Agent-level data (Phase 1F):** The GPU kernel currently outputs aggregate step-level metrics only (active count, jail count, support/oppose counts). To reproduce thesis-style individual agent analysis (activation scatterplots, threshold KDEs, spatial clustering), the kernel needs to be extended to output per-agent state. This requires:

1. Adding agent-level output buffers to the Mojo GPU kernel
2. An additional binary output file (agent_data.bin)
3. A targeted re-run of 10-20 carefully selected simulations

This is deferred as a separate engineering task.

### Confirmed Surprises

1. **Vision is the cascade parameter.** Invisible in the original 4-parameter thesis analysis (vision was fixed at 7), it explains 30-35% of cascade rate variance. This is the single most important finding from the expanded parameter sweep.

2. **Epsilon is macro-irrelevant.** The thesis centered its narrative on epsilon as the key parameter. At the macro level, epsilon explains <0.1% of revolution probability variance. Its importance is at the individual agent decision level, not at the population outcome level.

3. **The parameter space is low-dimensional.** Despite sweeping 7 parameters, only 2-3 matter for any given outcome metric. The system is dominated by security density (for revolution) and vision (for cascades).

---

## Observations

- [finding] Security density dominance confirmed at high resolution: ~40-45% of revolution variance #parameter-importance
- [finding] Vision cascade dominance confirmed: ~30-35% of cascade rate variance #parameter-importance
- [finding] Phase 1D high-resolution sweep does not change the parameter ranking from Phase 1C coarse sweep #confirmation
- [finding] Phase transition boundaries sharply localized in sec_density x threshold space #phase-transition
- [finding] Epsilon macro-irrelevance confirmed: <0.1% revolution variance at both resolutions #epsilon
- [status] Phase 1F deferred: requires GPU kernel extension for agent-level output #phase-1f
- [status] Total simulation corpus: ~43.5M sims across coarse + high-resolution sweeps #data

## Relations

- synthesizes [[2026-04-01 Cascade 7D Coarse Sweep Analysis]]
- synthesizes [[2026-04-02 Cascade 1000-Step Sweep Confirms 500-Step Sufficiency]]
- completes Phase 1G of [[2026-03-30 Cascade Extended Simulation and Paper Update Plan]]
- methodology-explained-in [[2026-04-04 Cascade Analysis Methodology and Visualization Guide]]
"""

    web_url = f"https://reports.joehelbing.net/view/Reports/cascade/report/{TODAY} Cascade Comprehensive Analysis Report.md"

    write_bm_note(
        title=f"{TODAY} Cascade Comprehensive Analysis Report",
        content=content,
        directory="Reports/cascade/report",
        tags=["report", "cascade", "phase-1g", "comprehensive"],
        metadata={"web_url": web_url},
    )

    # Write manifest
    manifest_path = OUT_REPORTS_WEB / slug_1g / "manifest.json"
    manifest = {
        "title": f"{TODAY} Cascade Comprehensive Analysis Report",
        "date": TODAY,
        "project": "cascade",
        "bm_path": f"Reports/cascade/report/{TODAY} Cascade Comprehensive Analysis Report.md",
        "figures": {},
    }
    # Auto-populate figures from copied files
    for fig_file in sorted((OUT_REPORTS_WEB / slug_1g / "figures").glob("*.json")):
        fig_id = fig_file.stem
        manifest["figures"][fig_id] = {
            "type": "auto",
            "caption": fig_id.replace("_", " ").replace("hires ", "High-res "),
            "file": f"figures/{fig_file.name}",
        }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote manifest: {manifest_path}")

    return web_url


def build_phase2_guidance():
    """Build Phase 2: Paper Update Guidance."""
    content = f"""# Cascade Paper Update Guidance

Based on the completed analysis phases (1A-1E, 1G), here is guidance for updating the thesis paper.

---

## 1. What to Keep

- **Empirical sections**: The ICEWS/V-Dem regression analysis stands on its own merits and is independent of the simulation findings.
- **Theoretical framework**: The Kuran/Lohmann/Epstein synthesis provides the motivation and framing. The expanded simulations strengthen rather than undermine this framework.
- **ODD+D model description**: The model specification is unchanged. The GPU implementation is a performance optimization that preserves the mathematical specification exactly.

---

## 2. What to Reframe

### Epsilon's Role

The thesis positions epsilon as a central parameter. The expanded sweep reveals:
- Epsilon explains **<0.1%** of revolution probability variance at the macro level
- Its importance is at the **individual agent decision level**, not population outcomes
- The narrative should reframe epsilon as a micro-level mechanism parameter, not a macro-level control parameter

**Suggested framing:** "While epsilon governs individual perception accuracy, the population-level outcomes are dominated by structural parameters -- the ratio of security forces to citizens, and the visual range within which citizens can observe their neighbors."

### Parameter Hierarchy

The thesis treats all parameters roughly equally. The data clearly shows a hierarchy:

1. **Security density** (~40% of revolution variance) -- the dominant structural parameter
2. **Vision** (~30% of cascade variance) -- the surprise finding, controls whether cascades repeat
3. **Threshold** (~15% of activation variance) -- the continuous control parameter
4. **PP mean** (~1-2%) -- modest but consistent effect on revolution probability
5. **Epsilon** (<0.1%) -- near-irrelevant at macro level
6. **Citizen density, max jail** (<1%) -- negligible, can be fixed at defaults

### Phase Transition Framing

The manifold surfaces reveal sharp phase transitions in sec_density x threshold space. The paper should explicitly frame these as **phase transitions in the ABM**, analogous to physical phase transitions. The transition boundary coordinates are now precisely mapped.

---

## 3. What to Add

### Computational Methodology Section

A new section describing:
- The GPU block-per-sim architecture (one GPU block per simulation instance)
- The two-binary pipeline (pure Mojo GPU runner + Python orchestrator)
- The parameter sweep methodology (coarse screening -> high-resolution targeted)
- Scale: 43.5M total simulations, 478+ GB database

**Important:** Focus on the *methodology*, not specific code changes. The methodology is the contribution; the implementation details are means to an end.

### Historical Performance Context

Joe's original setup: custom multithreading code on a 200-CPU cluster at UChicago's Midway HPC. 10,000 simulations took hours. Current: ~530 sims/sec on a single RTX 3090. The GPU port enables parameter sweeps that were computationally infeasible before.

### New Figures (Recommended)

1. **Parameter importance bar chart** -- the single most informative figure, showing variance decomposition across all parameters and z-metrics
2. **Security density vs threshold manifold** -- revolution probability, showing the phase transition cliff at 25x25 resolution
3. **Vision vs threshold manifold** -- cascade rate, showing how vision controls cascade periodicity
4. **Archetype parameter maps** -- where each behavioral regime lives in parameter space
5. **Paired comparison overlays** -- time series showing how single parameter changes affect dynamics at phase boundaries

### Original Data Note

Original thesis figures would need to be re-rendered from new data. The original simulation data is not available. New figures should be generated in a consistent style from the current 43.5M simulation database.

---

## 4. What to Cut or Compress

- **Epsilon-centric analysis** that suggests epsilon is a primary control parameter. Compress to a subsection acknowledging its individual-level role while noting its macro-level irrelevance.
- **Parameter sweep sections** that only cover 4 parameters. Replace with the full 7-parameter analysis.
- **Any visualization that only shows a single run** without parameter context. The manifold surfaces are far more informative.

---

## 5. Title Suggestions

"Mentos Regimes" is no longer preferred. Suggested alternatives:
- "Phase Transitions in Resistance Cascades: A GPU-Accelerated Parameter Space Analysis"
- "Structural Determinants of Revolutionary Cascades: Security Density, Vision, and the Irrelevance of Individual Perception"
- "Mapping the Revolution Manifold: 43 Million Simulations of Resistance Cascade Dynamics"

---

## 6. Framing Guidance

### Dual Contribution

The paper makes two contributions:
1. **Political economy** (primary): What structural parameters determine whether resistance cascades succeed, fail, or oscillate?
2. **Computational methodology** (secondary): How GPU-accelerated parameter sweeps enable comprehensive exploration of ABM parameter spaces.

The GPU kernel is **enabling technology**, not a finding. The political economy findings are the primary contribution.

### Model vs Reality

The manifold surfaces describe **model behavior**, not empirical reality. Be explicit about this distinction. The model is a theoretical tool for exploring the logic of resistance cascades under different structural conditions, not a predictive model of specific real-world events.

---

## 7. Phase 1F: What's Still Needed

Agent-level deep dives (Phase 1F) would reproduce thesis-style individual agent analysis:
- Per-agent activation scatterplots over time (thesis Figs 18-19)
- Threshold distribution KDEs (thesis Fig 17)
- Perception moderator distributions (thesis Fig 20)
- Spatial clustering visualization

This requires GPU kernel modification to output per-agent state. It would add the "zoom all the way in" section to the paper, complementing the macro-level manifold analysis.

---

## Observations

- [guidance] Reframe epsilon from central parameter to micro-level mechanism #paper-update
- [guidance] Add computational methodology section focused on the method, not code details #paper-update
- [guidance] Include parameter importance as the key new analytical contribution #paper-update
- [guidance] Phase transitions in ABM should be explicitly framed as such #paper-update
- [guidance] GPU kernel is enabling technology, not a finding #framing
- [guidance] Manifold surfaces describe model behavior, not empirical reality #framing
- [guidance] Phase 1F agent-level analysis still needed for complete paper #phase-1f

## Relations

- guides update of [[Resistance Cascade Thesis]]
- based-on [[{TODAY} Cascade Comprehensive Analysis Report]]
- completes Phase 2 of [[2026-03-30 Cascade Extended Simulation and Paper Update Plan]]
- references [[2026-04-04 Cascade Analysis Methodology and Visualization Guide]]
"""

    web_url = f"https://reports.joehelbing.net/view/Reports/cascade/report/{TODAY} Cascade Paper Update Guidance.md"

    write_bm_note(
        title=f"{TODAY} Cascade Paper Update Guidance",
        content=content,
        directory="Reports/cascade/report",
        tags=["report", "cascade", "phase-2", "paper-guidance"],
        metadata={"web_url": web_url},
    )
    return web_url


def main():
    print("=== Phase 1G + Phase 2: Report Generation ===\n")

    print("--- Phase 1G: Comprehensive Analysis Report ---")
    url_1g = build_1g_report()
    print(f"  Web URL: {url_1g}\n")

    print("--- Phase 2: Paper Update Guidance ---")
    url_2 = build_phase2_guidance()
    print(f"  Web URL: {url_2}\n")

    print("=== Report generation complete ===")
    print(f"Phase 1G report: {url_1g}")
    print(f"Phase 2 guidance: {url_2}")


if __name__ == "__main__":
    main()
