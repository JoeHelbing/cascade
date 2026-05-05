# Verification Pipeline

This page is the repo-local map for understanding and extending Cascade verification.

## Chain of trust

The model has three implementation surfaces:

```text
python-core-simulation/cascade_core.py
    ↓ canonical Parquet trace / SHA256 bit-exact comparison
mojo_cpu.mojo --rng python
    ↓ same CPU implementation, GPU-compatible RNG mode
mojo_cpu.mojo --rng gpu
    ↓ Parquet aggregate/fingerprint comparison
mojo_gpu.mojo throughput kernel
```

Validate one change boundary at a time. The current CPU correctness basis is the readable `python-core-simulation/cascade_core.py` reference. The archived Mesa/original-Python path remains provenance for the model semantics.

## One live validation script

The canonical validation chain is folded into one script:

```text
autoresearch/validation/run_pipeline.py
```

Use:

```bash
pixi run validate-cpu   # python-core Parquet -> mojo_cpu --rng python Parquet -> SHA256 match
pixi run validate-gpu   # mojo_cpu --rng gpu + mojo_gpu Parquet aggregate gate
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

## CPU verification: Python core -> Mojo CPU

`run_pipeline.py --stage cpu` does the full CPU boundary inline:

1. Runs unit regressions for Python-core mechanics and a small Mojo CPU CLI bit check.
2. Runs `python-core-simulation/cascade_core.py` for every picked seed.
3. Writes `python_core_trace.parquet` with `sim_id`, `seed`, config metadata, the trace row schema, and raw `*_bits` columns for Float64 fields.
4. Runs `build/mojo_cpu --rng python` once per picked seed.
5. Writes `mojo_cpu_python_rng_trace.parquet` in the same canonical layout.
6. Compares the two Parquet SHA256 digests.

A matching digest proves the canonical artifacts have identical rows, metadata, nulls, booleans, numeric values, and IEEE-754 Float64 bit patterns.

## GPU verification: Mojo CPU boundary -> Mojo GPU

GPU validation is currently less formal than CPU validation. The intended bridge
is `mojo_cpu.mojo --rng gpu`: the same CPU simulation implementation runs with
only its RNG provider switched from Python/Mesa RNG to the GPU-compatible LCG
provider.

The current `validate-gpu` gate builds/runs `mojo_gpu.mojo`, captures
`autoresearch/validation/mojo_gpu_output.txt`, and writes both a trace-level SHA gate and aggregate artifacts:

- `mojo_cpu_mojo_rng_state_trace.parquet` from `mojo_cpu --rng mojo`.
- `mojo_gpu_state_trace.parquet` from `mojo_gpu --trace-validation`.
- `mojo_cpu_gpu_rng_trace.parquet` and `mojo_cpu_gpu_rng_aggregate.parquet` for the CPU-side aggregate bridge.
- `mojo_gpu_aggregate.parquet` for the 45 aggregate `Sim ...` GPU lines.

The per-agent state-trace SHA must match for the no-security GPU comparison seed. The broader 45-case GPU sweep remains aggregate/tolerance-based because only the no-security subset has a supported CPU bridge.

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
