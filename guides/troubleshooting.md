# Troubleshooting

This guide covers common failure modes in the current Qwen/Sakana parity lane.

## The Elixir Parity Task Is Slow

Use semantic-only mode:

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

Without `--semantic-only`, the task runs native Nx SVD diagnostics. Native SVD
can trigger expensive XLA compilation and ptxas register-spill warnings. That is
not needed for semantic Python-component debugging.

The additional fast-loop flags address separate sources of wasted time:

- `--source-from-python-stage` avoids loading Qwen just to recover the sample
  source tensor.
- `--preferred-layout-only` skips `nx`/`vh` layout diagnostics once `torch_v`
  has been established.
- `--device-semantic-only` avoids a large host CPU matrix multiply and uses EXLA
  for the one required reconstruction path.

## Python Does Not Match The Historical Hash

If Python prints:

```text
reference_hash_reproducible: False
```

then do not require Elixir to match the stored historical hash. Use the current
Python readback baseline instead.

To pursue historical reproduction, provide the original `svd_weights.pt`:

```bash
python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
  --svd-weights path/to/original/svd_weights.pt \
  --strict-reference-hash \
  --out tmp/sakana_parity/python_sample_trace.json \
  --write-components-dir tmp/sakana_parity/python_components
```

## Wrong V Layout

Symptoms:

- `torch_v` has low zero-offset error;
- `nx` or `vh` has very large zero-offset error, around `1.6` for the current
  sample.

Interpretation:

- legacy `torch.svd` `V` requires `V.T` during reconstruction;
- the Elixir semantic layout should be `v_layout: :torch_v`.

Do not try to make the wrong layout match.

Once this is established, use `--preferred-layout-only` in routine parity runs.

## Source Or Final Shape Mismatch

The sample has different source and final orientations:

```text
source_shape: [3072, 1024]
sample_reconstructed_shape: [1024, 3072]
```

If shape checks fail, inspect whether the code is comparing Python source
orientation against Bumblebee target orientation. The parity code should orient
tensors explicitly before hashing or comparing.

## Stage Checks Fail At Source Or Offsets

If these fail:

```text
stage.source_f32
stage.offsets_f32
stage.scaled_s
```

then it is probably not a backend numeric issue. Check:

- source tensor key;
- Qwen model dtype and loading path;
- offset span;
- router vector file and hash;
- safetensors key names;
- source orientation.

## Stage Checks Fail At Reconstruction

If exact input stages pass but reconstruction exceeds tolerance, check:

- `v_layout`;
- normalization formula;
- dtype cast timing;
- whether offsets were cast to singular-value dtype;
- final/source orientation;
- backend used for host and device variants.

Do not immediately loosen tolerances. First inspect top diffs from the
comparator output.

## Final Hash Differs But Stage Tolerances Pass

This is the current known state.

Interpretation:

- the mathematical port is functionally correct under declared tolerances;
- exact final `bf16` bytes still differ;
- byte-match work should focus on reconstruction accumulation and rounding.

Run:

```bash
python3 priv/sakana_trinity/scripts/compare_sakana_parity_reports.py \
  --strict-stage-tolerances \
  --top-diffs 10 \
  tmp/sakana_parity/python_sample_trace.json \
  tmp/sakana_parity/elixir_sample_trace.json
```

Use the top-diff output to inspect the largest f32 and `bf16` differences.

## EXLA Donated Buffer Errors

Errors like:

```text
Buffer has been deleted or donated.
```

usually mean diagnostic code tried to read an EXLA tensor after a compiled
operation consumed it. The parity tracer snapshots diagnostic tensors to
`Nx.BinaryBackend` to avoid this. If this recurs, make sure any safetensors
readback used for diagnostics is immediately transferred to `Nx.BinaryBackend`.

## Missing CUDA

Check:

```bash
XLA_TARGET=cuda12 mix run -e 'IO.inspect(EXLA.Client.get_supported_platforms())'
```

If CUDA is missing, verify:

- NVIDIA driver;
- `nvidia-smi`;
- `XLA_TARGET=cuda12`;
- EXLA dependency target;
- environment isolation, especially shells launched without CUDA env vars.

## XLA_TARGET Rejected At Compile Time

`xla 0.10.x` (which EXLA 0.12+ uses) accepts:

```text
cpu, cuda, cuda12, cuda13, rocm, tpu
```

Anything else is rejected at compile time. The most common error mode
is a stale shell export like `XLA_TARGET=cuda14`. Set:

```bash
export XLA_TARGET=cuda12
```

`cuda12` is the canonical recommended default for CUDA hosts even when
the host installed toolkit is 13.x — the `XLA_TARGET` controls which
prebuilt XLA artifact is fetched; mismatched host CUDA installations
are tolerated by EXLA via dynamic loading. Use `cuda13` when you
specifically want the cuda13 prebuilt.

### Automatic preflight

The project surfaces unsupported targets automatically via a Mix
preflight that runs from `mix.exs` before any compilation step. An
operator whose shell exports an unsupported `XLA_TARGET` will see a
single readable line instead of an EXLA stacktrace:

```text
** (Mix.Error) XLA_TARGET="cuda14" is not accepted by the bundled xla 0.10.x.
Accepted values: "cpu", "cuda", "cuda12", "cuda13", "rocm", "tpu".
Recommended for CUDA hosts: export XLA_TARGET=cuda12.
Recommended for CPU hosts: unset XLA_TARGET (or use cpu).
The bundled xla rejects unrecognised targets at compile time, so EXLA
cannot compile until XLA_TARGET is corrected.
```

This fires for `mix compile`, `mix test`, `mix deps.compile`,
`mix deps.update`, `mix credo`, `mix dialyzer`, `mix docs`, and any
other task that evaluates `mix.exs`. To bypass it for one command,
prefix the invocation:

```bash
XLA_TARGET=cuda12 mix help
```

### Manual preflight (alternative)

`mix trinity.env.check` is the operator-facing preflight task; it
performs the same validation and additionally accepts
`--require TARGET` and `--artifact-dir DIR` options:

```bash
mix trinity.env.check
```


## Artifact Fetch Failed

### `cannot load pin "priv/sakana_trinity/artifact_pin.json"`

The pin file is committed to the repo. If a fresh clone reports this,
something is unusual — the pin should always be present. Check
`git status` and reset if needed:

```bash
git checkout HEAD -- priv/sakana_trinity/artifact_pin.json
```

### `trinity.artifact.fetch failed for ...: {:checksum_mismatch, expected, actual}`

A file was downloaded but its SHA-256 did not match the pinned value.
Most often a partial download or a corrupt cache entry. Clear the
file from the HuggingFace cache and retry:

```bash
rm -rf ~/.cache/huggingface/hub/datasets--nshkrdotcom--trinity-coordinator-adapted-qwen3-0.6b
mix trinity.artifact.fetch
```

If the mismatch persists after a clean re-download, file an issue; the
pin and the published bundle disagree.

### `:enoent` or HTTP-level error during fetch

HuggingFace was unreachable. Retry. For chronic outages or air-gapped
hosts, use the GitHub Release fallback documented in
[Artifact Distribution §3](artifact_distribution.md).

### `offline mode and file not cached`

You invoked `mix trinity.artifact.fetch --offline` or
`HF_HUB_OFFLINE=1` against a cold cache. Drop the offline flag once
to warm the cache; subsequent offline runs hit the cache.

## EMLX / Apple Silicon

### `Profile :emlx requires backend Elixir.EMLX.Backend which is not loaded`

The `:emlx` runtime profile is the Apple Silicon lane. EMLX is an
**optional** dependency — the project does not pull it on Linux/CUDA
hosts. To use the profile, add the dep to your parent application:

```elixir
# mix.exs
def deps do
  [
    {:trinity_coordinator, "~> 0.1"},   # (it hasn not been published to hex, use github or path dep)
    {:emlx, "~> 0.3"}                   # <-- add this
  ]
end
```

then `mix deps.get`. The next `mix run examples/qwen_router_prompt_eval.exs --runtime-profile emlx`
should pick up `EMLX.Backend` cleanly.

### Exporter raises `:unaccepted_backend` on Apple

The exporter validates that each adapted tensor was materialised on a
backend the runtime profile accepts. If you ran the export with
`--runtime-profile cuda_exla` on an Apple host, the validation will
reject the resulting EMLX-shaped tensors. Re-run with the correct
profile:

```bash
mix trinity.sakana.export_adapted --runtime-profile emlx \
  --svd-compute-type f32 \
  --out priv/sakana_trinity/adapted_qwen3_0_6b_layer26
```

The `--svd-compute-type f32` flag is recommended on EMLX because the
thin-SVD path uses `Nx.LinAlg.svd`'s default implementation (which
goes through `eigh`); doing that work in f32 keeps the small-σ tail
precise.

### EMLX OOM on the embedder SVD

The Qwen3-0.6B embedder is `151_936 × 1024`. Before two fixes
landed, the SVD of this matrix tried to materialise a full `m × m`
U matrix — about 92 GB. The fix landed in two places:

1. **EMLX v0.3.0** routed `Nx.LinAlg.svd/2` with `full_matrices?: false`
   through Nx's default implementation instead of MLX's native SVD
   (which always allocates the full U). Commit `3482b79`, Paulo
   Valente, "fix: use nx-defined implementation for non-full svd
   computation".
2. **Nx main commit `6424c89`**
   ([PR #1753](https://github.com/elixir-nx/nx/pull/1753)) refactored
   the default thin-SVD path itself to keep the working set bounded
   by `min(m, n)²`. Both EMLX and EXLA benefit from this fix.

`trinity_coordinator` pins the post-#1753 Nx (see `mix.exs`), so a
user who runs `mix trinity.sakana.export_adapted --runtime-profile emlx`
on Apple Silicon with `{:emlx, "~> 0.3"}` in their parent app gets
the bounded-memory path automatically. If you see an OOM, confirm
your Nx version: it should be 0.12.x or later.
