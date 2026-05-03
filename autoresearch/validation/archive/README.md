# Validation Archive

This directory preserves validation-adjacent files that are **not** the live python -> mojo_cpu -> mojo_gpu validation pipeline.

Nothing here was deleted; files were moved out of `autoresearch/validation/` so the root validation directory can show the canonical chain clearly.

## Directories

| Directory | Contents | Status |
|---|---|---|
| `historical/` | Old cross-validation, capture/replay, oscillation, seed-picking, animation, and showboat scripts. | Kept for provenance/debugging. Some scripts contain stale paths that assume they live in the validation root. |
| `generated/` | Local trace, CSV, HTML, pickle, and cache outputs from earlier runs. | Regenerable artifacts; not required to understand or run the canonical pipeline. |

## Canonical pipeline lives one level up

Start with:

```bash
pixi run validate-cpu
pixi run validate-gpu
pixi run validate
```

The live files are:

- `../picked_seeds.json`
- `../run_python_trace.py`
- `../compare_bitexact.py`
- `../compare_mojo_cpu.py` (secondary aggregate helper)
- `../run_pipeline.py`
- `../../../mojo_cpu.mojo`
- `../../../mojo_gpu.mojo`
