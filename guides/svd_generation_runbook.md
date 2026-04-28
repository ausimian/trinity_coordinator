# SVD Generation Runbook

Use this runbook when validating the full selected-tensor Sakana parity gate.
The Python commands are run through `uv`; do not use `pip` for this workflow.

## What To Run First

Run the original supplemental decomposer first. This is the correct first path
because it creates the historical-style `svd_weights.pt` file from the
unmodified source script:

```text
docs/priv/trinity_code_submission/decompose_model.py
```

The script below treats that directory as read-only input and writes everything
under `tmp/`.

```bash
cd ~/p/g/n/trinity_coordinator
priv/sakana_trinity/scripts/run_original_submission_svd_weights.sh
```

That one command does all of this:

1. uses `uv run` with pinned Python deps;
2. runs the original supplemental `decompose_model.py`;
3. writes `tmp/sakana_parity/original_submission_svd/Qwen_Qwen3-0.6B/svd_weights.pt`;
4. runs Python all-selected parity from that `.pt`;
5. runs Elixir all-selected replay;
6. runs `compare_sakana_parity_reports.py --strict-stage-tolerances`.

Do not start the second job until this one finishes or fails.

## What To Send Back After Run 1

Send these lines or files:

```text
tmp/sakana_parity/original_submission_svd/decompose_model.log
tmp/sakana_parity/original_submission_svd/python_all_selected.log
tmp/sakana_parity/original_submission_svd/elixir_all_selected.log
tmp/sakana_parity/original_submission_svd/compare.log
```

If the command succeeds, the most important file is:

```text
tmp/sakana_parity/original_submission_svd/compare.log
```

Also send:

```text
tmp/sakana_parity/original_submission_svd/Qwen_Qwen3-0.6B/svd_weights.pt
```

as a path only. Do not paste or upload the `.pt` unless asked.

## Only If Needed: Independent Expensive Recompute

Run this second command only if the first path fails or if we need an
independent comparison that does not use `svd_weights.pt`.

```bash
cd ~/p/g/n/trinity_coordinator
priv/sakana_trinity/scripts/run_expensive_all_selected_decompose.sh
```

That one command:

1. uses `uv run` with pinned Python deps;
2. recomputes all selected SVD components directly in
   `debug_sakana_parity_sample.py`;
3. runs Elixir all-selected replay;
4. runs `compare_sakana_parity_reports.py --strict-stage-tolerances`.

Send back:

```text
tmp/sakana_parity/expensive_all_selected_decompose/python_decompose_all_selected.log
tmp/sakana_parity/expensive_all_selected_decompose/elixir_all_selected.log
tmp/sakana_parity/expensive_all_selected_decompose/compare.log
```

## Expected Cost

These are long CPU SVD jobs over Qwen matrices. The scripts cap BLAS threads to
4 internally. Leave that alone for the first run; we can tune it later if the
machine is idle or overloaded.

## Success Criteria

The run is useful if it reaches the comparator.

The comparator success line is:

```text
strict stage tolerances passed by exiting with status 0
```

If it exits non-zero, send the final 80 lines of `compare.log` and the first
failed required stage line.
