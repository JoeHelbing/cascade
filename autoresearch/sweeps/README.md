# Sweeps

Scripts that execute parameter-space simulations and write result datasets.

| File | Purpose |
|---|---|
| `run_7d_sweep.py` | Coarse 7D parameter sweep. |
| `run_1d_sweep.py` | High-resolution targeted Phase 1D sweep. |
| `run_1e_paired.py` | Paired-comparison sweep. |
| `generate_agent_params.py` | Generates `../configs/agent_sim_params.json` for Phase 1F. |
| `run_phase1f.py` | Runs Phase 1F agent-level output batches. |
| `run_agent_sims.py` | Lower-level agent simulation runner support. |

Generated result directories are intentionally ignored by git. Keep reusable parameter selections in `../configs/` and paper-facing interpretation in `../analysis/` or `../../docs/`.
