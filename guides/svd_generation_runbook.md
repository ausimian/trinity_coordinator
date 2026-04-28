# SVD Generation Runbook

This runbook covers the two long-running ways to produce all-selected Python
SVD parity inputs.

## Purpose

The all-selected parity gate needs Python `U/S/V` components for every selected
Qwen tensor. There are two useful sources:

1. the original supplemental decomposer under
   `docs/priv/trinity_code_submission/decompose_model.py`;
2. the parity harness's explicit
   `--decompose-all-selected-if-missing` path.

Both paths should write to `tmp/` and must not modify
`docs/priv/trinity_code_submission`.

## Runtime Warning

Both jobs are expensive. They load `Qwen/Qwen3-0.6B` and run large CPU SVDs. If
you run them at the same time, limit BLAS threads so they do not fight for every
CPU core:

```bash
export THREADS=4
```

Increase or decrease this based on available RAM and CPU cores.

## Terminal 1: Original Supplemental `svd_weights.pt`

The supplemental script imports `fire`. If it is missing, install it in the
Python environment used for the run:

```bash
python3 -m pip install --user fire
```

Run:

```bash
THREADS=4 \
OUT_ROOT=tmp/sakana_parity/original_submission_svd \
priv/sakana_trinity/scripts/run_original_submission_svd_weights.sh
```

This writes:

```text
tmp/sakana_parity/original_submission_svd/Qwen_Qwen3-0.6B/svd_weights.pt
tmp/sakana_parity/original_submission_svd/decompose_model.log
tmp/sakana_parity/original_submission_svd/python_sample_trace.json
tmp/sakana_parity/original_submission_svd/python_components/
tmp/sakana_parity/original_submission_svd/elixir_sample_trace.json
tmp/sakana_parity/original_submission_svd/elixir_stages/
tmp/sakana_parity/original_submission_svd/compare.log
```

Set `RUN_PARITY_AFTER=0` to only generate `svd_weights.pt` and skip the parity
replay:

```bash
RUN_PARITY_AFTER=0 \
THREADS=4 \
OUT_ROOT=tmp/sakana_parity/original_submission_svd \
priv/sakana_trinity/scripts/run_original_submission_svd_weights.sh
```

## Terminal 2: Explicit All-Selected Recompute

Run:

```bash
THREADS=4 \
OUT_ROOT=tmp/sakana_parity/expensive_all_selected_decompose \
priv/sakana_trinity/scripts/run_expensive_all_selected_decompose.sh
```

This writes:

```text
tmp/sakana_parity/expensive_all_selected_decompose/python_decompose_all_selected.log
tmp/sakana_parity/expensive_all_selected_decompose/python_sample_trace.json
tmp/sakana_parity/expensive_all_selected_decompose/python_components/
tmp/sakana_parity/expensive_all_selected_decompose/elixir_sample_trace.json
tmp/sakana_parity/expensive_all_selected_decompose/elixir_stages/
tmp/sakana_parity/expensive_all_selected_decompose/compare.log
```

## What To Report Back

For each terminal, report:

- whether the command exited zero;
- the final `current_python_baseline` line;
- the `reference_hash_reproducible` line;
- the final comparator summary from `compare.log`;
- any failed required stage lines.

If a run fails, provide the relevant log:

```text
decompose_model.log
python_all_selected.log
python_decompose_all_selected.log
elixir_all_selected.log
compare.log
```

Do not paste stage safetensors. They can be very large; provide paths and log
summaries first.
