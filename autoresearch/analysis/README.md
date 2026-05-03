# Analysis

Post-hoc scripts that turn sweep outputs into figures, reports, and paper-facing interpretation.

## Groups

| Files | Purpose |
|---|---|
| `analysis_1c.py`, `write_1c_report.py` | Phase 1C analysis/report support. |
| `analysis_1d.py` | Phase 1D high-resolution sweep analysis. |
| `analysis_1f.py` | Phase 1F agent-level analysis support. |
| `generate_report_1g.py` | Phase 1G comprehensive analysis/report generation. |
| `analyze_grid_search.py`, `generate_3d_manifolds.py`, `visualize_manifolds.py` | Grid/manifold search analysis and visualization. |
| `generate_explainer_figures.py`, `regenerate_figures.py` | Paper/report figure generation. |
| `compare_outputs.py`, `visualize_comparison.py` | Historical GPU/Python comparison and cross-validation visualization artifacts. |
| `load_step_metrics.py` | Shared loading helper for step-metric datasets. |

This directory should contain interpretation and visualization logic, not the canonical correctness gate. Verification-specific logic belongs in `../validation/` and is summarized in `../../docs/verification/`. Start with `pixi run validate-cpu`, `pixi run validate-gpu`, or `pixi run validate` for the live pipeline.
