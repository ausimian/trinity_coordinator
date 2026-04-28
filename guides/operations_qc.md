# Operations And Quality Gates

This guide lists the checks that should be run before changing the parity or
runtime lanes.

## Fast Regression Suite

```bash
XLA_TARGET=cuda12 mix test
```

Current expected result:

```text
1 doctest, 159 tests, 0 failures, 25 excluded
```

The excluded tests include slow Qwen/SVD gates and expensive artifact export
paths.

## Static Checks

```bash
mix format --check-formatted
mix credo --strict
mix dialyzer
```

Python script syntax:

```bash
python3 -m py_compile priv/sakana_trinity/scripts/*.py
```

Docs:

```bash
mix docs
```

Docs should build without undefined-reference warnings.

## Functional Parity Gate

Generate reports:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --model-torch-dtype float32 \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

```bash
XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
  --semantic-only \
  --device-semantic-only \
  --preferred-layout-only \
  --source-from-python-stage \
  --components-dir tmp/sakana_parity/python_components \
  --python-report tmp/sakana_parity/python_sample_trace.json \
  --stage-dir tmp/sakana_parity/elixir_stages \
  --out tmp/sakana_parity/elixir_sample_trace.json
```

Run strict functional comparison:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This is the required parity gate.

When validating the full selected tensor set, generate Python with
`--all-selected-tensors --svd-weights path/to/svd_weights.pt`, then add
`--all-selected-tensors` to the Elixir command above. Add
`--selected-source-regex 'model\.layers\.26\.'` for the current bounded replay
gate. That mode reads
`trinity_svf_all_selected_stage_debug.safetensors` and fails strict stage
tolerances if any required stage fails for any replayed selected tensor.
Embedding and LM-head tensors require a chunked large-tensor gate before they
should be enforced through Elixir; the monolithic stage replay can otherwise
stall or exhaust GPU memory.

The additional Elixir flags are intentional for commit-loop parity work:

- `--source-from-python-stage` avoids loading Qwen only to fetch the sample
  source tensor.
- `--preferred-layout-only` skips known-wrong V-layout diagnostics.
- `--device-semantic-only` avoids a large host CPU matrix multiply and still
  emits stage checks from the EXLA semantic variant.

## Byte-Match Gate

Use only when intentionally working on exact final bytes:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-current-python \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

This is expected to fail in the current state.

## Expensive Gates

Adapted coordinator smoke against the canonical import from Phase 2:

```bash
XLA_TARGET=cuda12 mix trinity.hitl.adapted \
  --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
```

This gate must print:

```text
Artifact status: complete
Artifact layout: checkpoint_directory
Selected tensor count: 9
Selected singular value count: 9216
adapted Qwen vector shape: {1, 1024}
adapted route logits shape: {1, 10}
adapted agent logits shape: {7}
adapted role logits shape: {3}
```

It proves the imported artifact can patch Qwen, load the router head, and route
a fixed transcript on CUDA. It does not replace router trace parity, which still
needs a Python side-by-side trace for tokenization, hidden vector, and logits.

Qwen-focused tests:

```bash
XLA_TARGET=cuda12 mix test --only qwen --trace
```

Slow SVD tests:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only slow_qwen_svd --trace
```

Expensive export gate:

```bash
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs \
  --only expensive_qwen_svd --trace
```

These are not part of every local commit loop.

## Expected XLA Warnings

EXLA/CUDA may print:

```text
ptxas warning : Registers are spilled to local memory
```

This does not mean VRAM overflow. It means a compiled GPU kernel used more
registers than available and some values were placed in CUDA local memory. It is
often a compile/runtime performance concern, not a parity failure by itself.

Use `--semantic-only` during semantic parity work to avoid the slow native Nx
SVD compilation path. For the fastest sample parity loop, pair it with
`--source-from-python-stage --preferred-layout-only --device-semantic-only`.

## Commit Checklist

- [ ] `mix format --check-formatted`
- [ ] `python3 -m py_compile priv/sakana_trinity/scripts/*.py`
- [ ] `XLA_TARGET=cuda12 mix test`
- [ ] `mix credo --strict`
- [ ] `mix dialyzer`
- [ ] `mix docs`
- [ ] parity runtime check when parity code changed:
  `compare_sakana_parity_reports.py --strict-stage-tolerances`
- [ ] README and guides updated when behavior or standards change.
