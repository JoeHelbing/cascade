"""
Write Phase 1C analysis results as a Basic Memory report with reports-web integration.

Reads the JSON outputs from analysis_1c.py and generates:
1. BM report note in Reports/cascade/report/
2. reports-web manifest with figure references
3. Copies Plotly JSON figures to reports-web data dir

Usage:
    cd mojo_cascade
    pixi run python write_1c_report.py
"""

import json
import shutil
from datetime import date
from pathlib import Path

ANALYSIS_DIR = Path("analysis_1c_output")
FIGURES_DIR = ANALYSIS_DIR / "figures"

TODAY = date.today().isoformat()
SLUG = f"{TODAY}-cascade-7d-coarse-sweep-analysis"
REPORT_TITLE = f"{TODAY} Cascade 7D Coarse Sweep Analysis"

BM_DIR = Path.home() / "Obsidian" / "bm-main" / "Reports" / "cascade" / "report"
WEB_DATA_DIR = Path.home() / "git" / "reports-web" / "data" / "cascade" / SLUG
WEB_FIGURES_DIR = WEB_DATA_DIR / "figures"
BM_PATH_REL = f"Reports/cascade/report/{REPORT_TITLE}.md"
WEB_URL = f"http://127.0.0.1:3939/report/cascade/{SLUG}"

PARAM_LABELS = {
    "pp_mean": "PP Mean",
    "sec_density": "Security Density",
    "epsilon": "Epsilon",
    "threshold": "Threshold",
    "citizen_density": "Citizen Density",
    "max_jail": "Max Jail",
    "vision": "Vision",
}

Z_METRIC_LABELS = {
    "revolution_prob": "Revolution Probability",
    "max_active_pct": "Max Active %",
    "mean_active_pct": "Mean Active %",
    "cascade_rate": "Cascade Rate (2+ peaks)",
    "periodic_rate": "Periodic Rate",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def format_importance_table(importance_data):
    """Format parameter importance as markdown tables."""
    lines = []
    for steps_str, metrics in sorted(importance_data.items(), key=lambda x: int(x[0])):
        lines.append(f"\n#### {steps_str} Steps\n")
        lines.append("| Parameter | Rev. Prob | Max Active % | Mean Active % | Cascade Rate | Periodic |")
        lines.append("|-----------|-----------|-------------|--------------|-------------|----------|")
        for param in ["sec_density", "threshold", "epsilon", "pp_mean",
                       "citizen_density", "max_jail", "vision"]:
            row = f"| {PARAM_LABELS[param]} |"
            for z_name in ["revolution_prob", "max_active_pct", "mean_active_pct",
                           "cascade_rate", "periodic_rate"]:
                val = metrics.get(z_name, {}).get(param, 0.0)
                row += f" {val:.3f} |"
            lines.append(row)
    return "\n".join(lines)


def format_step_comparison(step_data):
    """Format step count comparison as prose."""
    lines = []
    for z_name, comparisons in step_data.items():
        label = Z_METRIC_LABELS.get(z_name, z_name)
        lines.append(f"\n**{label}:**\n")
        for pair, stats in comparisons.items():
            if isinstance(stats, dict) and "correlation" in stats:
                lines.append(f"- {pair}: correlation = {stats['correlation']:.4f}, mean difference = {stats['mean_diff']:.4f}")
                if "new_revolutions" in stats:
                    lines.append(f"  - New revolutions at longer horizon: {stats['new_revolutions']:,} ({stats['pct_new_revolutions']:.1%} of previously non-revolutionary sims)")
    return "\n".join(lines)


def format_findings(findings):
    """Format surprising findings as a list."""
    if not findings:
        return "No new parameters (citizen_density, max_jail, vision) explain >1% of variance in any z-metric at any step count. The original 4 parameters dominate."
    lines = []
    for f in findings[:20]:
        lines.append(f"- {f['note']}")
    return "\n".join(lines)


def format_archetypes(archetype_data):
    """Format archetype distribution tables."""
    lines = []
    for steps_str, counts in sorted(archetype_data.items(), key=lambda x: int(x[0])):
        total = sum(counts.values())
        lines.append(f"\n#### {steps_str} Steps\n")
        lines.append("| Archetype | Count | Fraction |")
        lines.append("|-----------|-------|----------|")
        for arch, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            label = arch.replace("_", " ").title()
            lines.append(f"| {label} | {count:,} | {count/total:.1%} |")
    return "\n".join(lines)


def build_figure_refs():
    """Build figure reference lines and manifest entries from available JSON figures."""
    figures = {}
    refs = []

    json_files = sorted(FIGURES_DIR.glob("*.json"))
    for jf in json_files:
        fig_id = jf.stem
        # Determine figure type from name
        if "manifold" in fig_id:
            fig_type = "heatmap_grid"
        elif "archetype_map" in fig_id:
            fig_type = "heatmap_grid"
        elif "archetype_distribution" in fig_id:
            fig_type = "bar"
        elif "archetype_exemplars" in fig_id:
            fig_type = "line_multi"
        elif "parameter_importance" in fig_id:
            fig_type = "bar_grouped"
        elif "step_comparison" in fig_id:
            fig_type = "scatter"
        else:
            fig_type = "plotly"

        caption = fig_id.replace("_", " ").replace("steps", " steps").title()
        figures[fig_id] = {"type": fig_type, "caption": caption, "data_file": f"figures/{fig_id}.json"}
        refs.append(f"{{{{figure:{fig_id} | {fig_type} | {caption}}}}}")

    return figures, refs


def generate_report():
    """Generate the full BM report markdown."""
    # Load analysis outputs
    importance = load_json(ANALYSIS_DIR / "parameter_importance.json")
    findings = load_json(ANALYSIS_DIR / "surprising_findings.json")
    archetypes = load_json(ANALYSIS_DIR / "archetype_counts.json")

    step_comp = None
    step_comp_path = ANALYSIS_DIR / "step_count_comparison.json"
    if step_comp_path.exists():
        step_comp = load_json(step_comp_path)

    figures_manifest, figure_refs = build_figure_refs()

    # Determine available step counts
    step_counts = sorted(int(k) for k in importance.keys())
    step_str = ", ".join(str(s) for s in step_counts)

    # Build report content
    report = f"""---
title: {REPORT_TITLE}
type: report
permalink: main/reports/cascade/report/{SLUG}
project: cascade
status: active
note_type: report
web_url: {WEB_URL}
tags:
- report
- cascade
- phase-1c
- parameter-sweep
- 7d-analysis
---

# Cascade 7D Coarse Sweep Analysis

Phase 1C analysis of the coarse 7-parameter sweep across the Resistance Cascade ABM. This report identifies which parameters matter, where interesting dynamics occur, and what deserves high-resolution follow-up.

**Sweep configuration:** 7^7 = 823,543 parameter configurations x 10 seeds = 8,235,430 simulations per step count. Step counts: {step_str}. Grid: 33x33 = 1,089 cells. Total simulations analyzed: {8235430 * len(step_counts):,}.

---

## 1. Parameter Importance: Variance Decomposition

For each z-metric, the fraction of total variance explained by each parameter (main effect, averaging over all other parameters).

{{{{figure:parameter_importance | bar_grouped | Parameter importance across step counts and z-metrics}}}}

{format_importance_table(importance)}

### Key Findings

The variance decomposition reveals which parameters drive model behavior. Security density and threshold are expected to dominate based on prior manifold analysis. The question is whether the three new parameters (citizen density, max jail, vision) show non-trivial effects.

---

## 2. Step Count Comparison

Do longer simulations reveal dynamics that short runs miss?

"""
    if step_comp:
        report += format_step_comparison(step_comp)

        # Add scatter plots if they exist
        for jf in sorted(FIGURES_DIR.glob("step_comparison_*.json")):
            fig_id = jf.stem
            report += f"\n\n{{{{figure:{fig_id} | scatter | Z-metric comparison between step counts}}}}\n"
    else:
        report += "\n*Step count comparison requires multiple step counts in the database.*\n"

    report += f"""

---

## 3. Pairwise Manifold Surfaces

All 21 pairwise parameter combinations rendered as heatmaps, with z-metric averaged over all other parameters and seeds. Colorscale 0-1 for cross-comparison.

"""
    # Group manifold figures by z-metric
    manifold_figs = sorted(FIGURES_DIR.glob("manifold_*.json"))
    z_metrics_seen = set()
    for jf in manifold_figs:
        fig_id = jf.stem
        # Extract z-metric and step label
        parts = fig_id.split("_")
        report += f"{{{{figure:{fig_id} | heatmap_grid | {fig_id.replace('_', ' ').title()}}}}}\n\n"

    report += f"""
---

## 4. Surprising Findings: New Parameters

Do citizen_density, max_jail, or vision show non-trivial effects or unexpected interactions?

{format_findings(findings)}

---

## 5. Trajectory Archetypes

Simulations classified into behavioral archetypes based on summary metrics:

- **Fast Revolution:** Revolution in first 20% of steps
- **Mid Revolution:** Revolution between 20-50% of steps
- **Slow Burn:** Revolution after 50% of steps
- **Oscillating:** Multiple cascades (periodic) but no revolution
- **Abortive Spike:** Cascades without revolution, not periodic
- **Stable Suppression:** No cascades, no revolution, low activation
- **Simmering:** No cascades, no revolution, but moderate activation

{format_archetypes(archetypes)}

{{{{figure:archetype_distribution | bar | Archetype distribution across step counts}}}}

"""
    # Add archetype exemplar and map figures
    exemplar_figs = sorted(FIGURES_DIR.glob("archetype_exemplars_*.json"))
    for jf in exemplar_figs:
        fig_id = jf.stem
        report += f"{{{{figure:{fig_id} | line_multi | Example time series for each archetype}}}}\n\n"

    map_figs = sorted(FIGURES_DIR.glob("archetype_map_*.json"))
    for jf in map_figs:
        fig_id = jf.stem
        report += f"{{{{figure:{fig_id} | heatmap_grid | Archetype frequency in parameter space}}}}\n\n"

    report += f"""
---

## 6. Recommendations for Phase 1D

Based on this coarse analysis:

### Parameters to pursue at high resolution
*[To be filled based on actual importance results]*

### Recommended step count
*[To be determined from step count comparison]*

### Interesting parameter boundaries for paired comparisons (Phase 1E)
*[To be identified from manifold surface phase transitions]*

---

## Observations

- [analysis] Phase 1C coarse 7D sweep analysis complete #cascade #phase-1c
- [finding] Parameter importance ranking across {len(step_counts)} step counts #variance-decomposition
- [finding] Trajectory archetype classification across parameter space #temporal-dynamics

## Relations

- continues [[2026-03-30 Cascade Extended Simulation and Paper Update Plan]]
- extends [[2026-03-29 Cascade Manifold Search - 7.8M Simulations Map Phase Transition Surfaces]]
- analyzes cascade 7D coarse parameter sweep data
"""

    return report, figures_manifest


def write_report():
    """Write report, manifest, copy figures, update daily note."""
    report_content, figures_manifest = generate_report()

    # 1. Write BM report
    BM_DIR.mkdir(parents=True, exist_ok=True)
    report_path = BM_DIR / f"{REPORT_TITLE}.md"
    report_path.write_text(report_content)
    print(f"Wrote BM report: {report_path}")

    # 2. Create reports-web manifest + copy figures
    WEB_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": REPORT_TITLE,
        "date": TODAY,
        "project": "cascade",
        "bm_path": BM_PATH_REL,
        "figures": figures_manifest,
    }
    manifest_path = WEB_DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    # Copy figure JSON files
    for jf in sorted(FIGURES_DIR.glob("*.json")):
        dest = WEB_FIGURES_DIR / jf.name
        shutil.copy2(jf, dest)
    n_figs = len(list(FIGURES_DIR.glob("*.json")))
    print(f"Copied {n_figs} figure JSONs to {WEB_FIGURES_DIR}")

    # 3. Update daily note
    daily_dir = Path.home() / "Obsidian" / "bm-main" / "Daily"
    daily_path = daily_dir / f"{TODAY}.md"
    link_line = f"- [ ] Read: [[{REPORT_TITLE}]] ([web]({WEB_URL}))\n"

    if daily_path.exists():
        content = daily_path.read_text()
        if REPORT_TITLE not in content:
            # Find "Notes and Small Todos" section or append
            marker = "## Notes and Small Todos"
            if marker in content:
                idx = content.index(marker) + len(marker)
                # Find the next line after the header
                next_nl = content.index("\n", idx)
                content = content[:next_nl + 1] + link_line + content[next_nl + 1:]
            else:
                content += f"\n{marker}\n{link_line}"
            daily_path.write_text(content)
            print(f"Updated daily note: {daily_path}")
        else:
            print(f"Daily note already contains report link")
    else:
        print(f"Daily note not found at {daily_path} -- add link manually")

    print(f"\nReport URL: {WEB_URL}")
    print(f"Done!")


if __name__ == "__main__":
    write_report()
