# Sakana SVD hash parity debug flow

Run from the repository root.

## 1. Emit the Python-side checkpoint report

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

This writes:

- `tmp/sakana_parity/python_sample_trace.json`
- `tmp/sakana_parity/python_components/trinity_svf_components.safetensors`
- `tmp/sakana_parity/python_components/trinity_svf_scale_offsets.safetensors`
- `tmp/sakana_parity/python_components/trinity_svf_debug_manifest.json`

The report now separates two concepts:

- **stored reference hash**: the historical value in `sakana_python_reference_manifest.json`;
- **current Python baseline hash**: the value produced by the current Python/PyTorch environment.

If the script prints `reference_hash_reproducible: False`, do **not** expect Elixir
or freshly recomputed Python SVD components to match the stored `600be6...` hash.
That means the stored hash is provenance-sensitive to the original SVD component
basis.

## 1a. Strict historical reproduction, when original SVD weights are available

If you have the original Python `svd_weights.pt`, run:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --svd-weights path/to/svd_weights.pt \
  --strict-reference-hash \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

Only use strict stored-reference assertions after the Python report itself says
`reference_hash_reproducible: True`.

## 2. Emit the Elixir-side checkpoint report

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

This writes native Nx SVD variants and semantic Python-component reconstruction
variants. The Elixir tracer snapshots intermediate tensors to `Nx.BinaryBackend`
before reconstruction so EXLA donated buffers cannot crash the report.

For semantic Python components, the tracer now emits both host/BinaryBackend and
device/EXLA variants. Use the host/BinaryBackend semantic variant for strict
functional parity with the current Python report; use the device variant to
inspect runtime CUDA numerical drift. CUDA/EXLA may use different fp32 GEMM
semantics than PyTorch CPU, so a device variant can have a larger zero-offset
error and a different `bf16` hash even when the formula and V layout are right.

Native variants are expected to differ when the SVD basis differs. Semantic
Python-component variants isolate formula, V/Vh layout, orientation, final `bf16`
cast, raw-byte hashing, and compute backend.

## 3. Compare both reports

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

For opt-in exact checks:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-reference \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

or:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

## 4. Run the focused test with diagnostics enabled

```bash
TRINITY_SVD_PARITY_OUT=tmp/sakana_parity/elixir_from_test.json \
TRINITY_PYTHON_PARITY_REPORT=tmp/sakana_parity/python_sample_trace.json \
TRINITY_PYTHON_COMPONENTS_DIR=tmp/sakana_parity/python_components \
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only slow_qwen_svd --trace
```

Default behavior verifies shapes, offsets, zero-offset reconstruction sanity, and
Python-component V-layout handling without requiring a non-reproducible stored
hash. Strict byte-level checks are explicit:

- `TRINITY_STRICT_REFERENCE_HASH=1` requires a semantic component variant to
  match the stored manifest hash. Use only after Python itself reproduces it.
- `TRINITY_STRICT_CURRENT_PYTHON_HASH=1` requires a semantic component variant
  to match the current Python baseline hash.
- `TRINITY_STRICT_NATIVE_SVD_HASH=1` requires native Nx SVD to match the stored
  Python hash. This is expected to fail when native SVD produces a different but
  valid singular-vector basis.
