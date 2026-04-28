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

The script prints whether `torch.svd` or `torch.linalg.svd` matched the expected
reference hash. Keep `--model-torch-dtype float32` for reference parity. Using
`auto` can load Qwen weights as `bf16` on some Transformers installations, which
is useful as a diagnostic variant but is not the dtype used to generate the
stored Python reference hash.

## 2. Emit the Elixir-side checkpoint report

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --components-dir tmp/sakana_parity/python_components \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

This writes native Nx SVD variants and semantic Python-component reconstruction
variants. The Elixir tracer snapshots intermediate tensors to `Nx.BinaryBackend`
before reconstruction so EXLA donated buffers cannot crash the report.  Native variants are expected to differ if the SVD basis differs;
semantic Python-component variants should isolate formula, orientation, final
`bf16` cast, and raw-byte hashing.

## 3. Compare both reports

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

## 4. Run the focused test with diagnostics enabled

```bash
TRINITY_SVD_PARITY_OUT=tmp/sakana_parity/elixir_from_test.json \
TRINITY_PYTHON_COMPONENTS_DIR=tmp/sakana_parity/python_components \
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only slow_qwen_svd --trace
```

To require the native Nx SVD path to match the Python hash, set
`TRINITY_STRICT_NATIVE_SVD_HASH=1`. That mode is intentionally separate because
native SVD basis differences can be visible after non-zero singular-value
scaling.
