# Verification Pipeline

This page is the repo-local map for understanding and extending Cascade verification.

## Chain of trust

The model has three implementation surfaces:

```text
original_python/ Mesa reference
    ↓ CPU trace / bit-exact comparison
mojo_cpu.mojo --rng python
    ↓ same CPU implementation, RNG provider switch only
mojo_cpu.mojo --rng gpu
    ↓ aggregate/fingerprint comparison
mojo_gpu.mojo throughput kernel
```

Validate one change boundary at a time. Do not use a new Python reimplementation
of the GPU algorithm as the source of truth; that only validates the
reimplementation. The original Mesa model remains the semantic reference.

## One live validation script

The canonical validation chain is folded into one script:

```text
autoresearch/validation/run_pipeline.py
```

Use:

```bash
pixi run validate-cpu   # Mesa trace -> mojo_cpu --rng python -> bit-exact compare
pixi run validate-gpu   # build/run mojo_gpu and check aggregate output shape
pixi run validate       # both boundaries in order
```

The validation root intentionally contains only:

- `run_pipeline.py` — the single live validation-chain script.
- `picked_seeds.json` — canonical CPU seed/config set.
- `README.md` — quick local guide.

Older split helpers such as `run_python_trace.py`, `compare_bitexact.py`, and
`compare_mojo_cpu.py` are preserved under
`autoresearch/validation/archive/historical/`; their live logic is now inside
`run_pipeline.py`.

## CPU verification: Mesa -> Mojo CPU

`run_pipeline.py --stage cpu` does the full CPU boundary inline:

1. Runs the original Mesa model for `picked_seeds.json`.
2. Writes `python_trace.parquet` and `python_model_trace.parquet` as generated artifacts.
3. Builds `mojo_cpu.mojo`.
4. Runs `build/mojo_cpu --rng python > mojo_cpu_bitexact.csv`.
5. Compares Mesa and Mojo per-agent rows bit-for-bit on tracked columns.

Known current result from project memory: the CPU port reached bit-exact
validation for the picked-seed set: `96,320 / 96,320` citizen rows matched Mesa
bit-for-bit.

## GPU verification: Mojo CPU boundary -> Mojo GPU

GPU validation is currently less formal than CPU validation. The intended bridge
is `mojo_cpu.mojo --rng gpu`: the same CPU simulation implementation runs with
only its RNG provider switched from Python/Mesa RNG to the GPU-compatible LCG
provider.

The current `validate-gpu` gate builds/runs `mojo_gpu.mojo`, captures
`autoresearch/validation/mojo_gpu_output.txt`, and asserts the output contains
45 aggregate `Sim ...` lines. This is a smoke/fingerprint-style gate, not yet a
full CPU/GPU equality gate.

## Archive and non-canonical files

Historical and exploratory validation scripts live under
`autoresearch/validation/archive/historical/`. Generated artifacts from local
runs live under `autoresearch/validation/archive/generated/` when cleaned from
the root. Nothing was deleted.

## Gaps to resolve before paper-grade verification

1. Make `mojo_cpu --rng gpu` and `mojo_gpu` match on documented validation cases.
2. Add a clean GPU trace or stable aggregate fixture analogous to the CPU bit-exact path.
3. Replace or wrap hardcoded GPU comparison parameters with a documented config file.
4. Preserve the one-file validation interface in `autoresearch/validation/run_pipeline.py`.
