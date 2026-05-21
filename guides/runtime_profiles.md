# Runtime Profiles

`trinity_coordinator` ships with a small set of named **runtime
profiles** that bundle a backend choice, a default coordinator SLM,
and a set of validation expectations into one keyword. Profiles are
defined in `TrinityCoordinator.RuntimeProfile` and resolved by name
or struct.

A profile answers three questions in one place:

1. **Which Nx backend?** (`:nx_backend` — e.g. `{EXLA.Backend, client: :cuda}`,
   `{EMLX.Backend, device: :gpu}`, `Nx.BinaryBackend`.)
2. **Is CUDA required?** (`:require_cuda?` — gates `Runtime.put_cuda_backend!/0`
   so callers can opt out cleanly.)
3. **What is the default SLM coordinator profile?** (`:default_slm_profile`.)

Plus metadata for downstream callers (`:export_svd?`, `:large_svd?`,
`:qwen_runtime?`, `:artifact_runtime?`) and operator-facing
`:notes` / `:warnings`.

## The Built-In Profiles

### `:cuda_exla` (default)

Linux/CUDA happy path. Used for everything the project does today.

- `nx_backend: {EXLA.Backend, client: :cuda}`
- `require_cuda?: true`
- `default_slm_profile: :qwen_coordinator`

Requires `XLA_TARGET=cuda12` and a working EXLA CUDA stack.

### `:emlx`

Apple Silicon (MLX-backed) lane. Brings up `EMLX.Backend, device: :gpu`
when the optional `:emlx` dep is present.

- `nx_backend: {EMLX.Backend, device: :gpu}`
- `require_cuda?: false`
- `default_slm_profile: :qwen_coordinator`

EMLX is an optional dependency. To use this profile, add it to your
parent application's `mix.exs`:

```elixir
{:emlx, "~> 0.3"}
```

Then:

```bash
mix deps.get
mix run examples/qwen_router_prompt_eval.exs --runtime-profile emlx \
  --snapshot examples/fixtures/qwen_router_prompt_eval_logits.json \
  --determinism-runs 2
```

#### EMLX-specific Caveats

- **Thin SVD memory footprint.** Nx main as of commit `6424c89` (Paulo
  Valente, [PR #1753](https://github.com/elixir-nx/nx/pull/1753))
  refactored `Nx.LinAlg.svd/2` with `full_matrices?: false` so it does
  not materialise the full `m × m` U on the Qwen3-0.6B embedder
  (where `m = 151_936`, i.e. (92 GB of U under the old path).
  This fix is in the Nx version that `trinity_coordinator` pins to.
  Both EMLX and EXLA benefit from this change.
- **`--svd-compute-type f32`.** Recommended on Apple. The thin-SVD
  path uses an `eigh` decomposition under the hood; doing that work
  in f32 keeps the small-σ tail precise.
- **Backend label.** When the exporter validates per-tensor backend
  during the SVD reconstruction step, it accepts the
  `"EMLX.Backend"` label as well as `"EXLA.Backend<cuda:"`. No code
  changes needed for the user.
- **Bumblebee Qwen3 support.** Bumblebee is git-pinned to a Qwen3-
  supporting commit (post-v0.7.0 main). EMLXAxon
  ([github.com/elixir-nx/emlx](https://github.com/elixir-nx/emlx)) has
  independently validated Qwen3-0.6B loading through the EMLX backend.
  Paulo Valente confirmed on 2026-05-21 that running with the bare
  EMLX backend (no `EMLXAxon.rewrite/1`) successfully exports and
  passes 37/37 on the prompt eval.
- **bf16 round-trip.** The bundle is bf16 safetensors. EMLX accepts
  bf16 natively (`{:bf, 16}` ↔ MLX `bfloat16`). No quantisation or
  type cast required.

### `:cpu_binary`

Pure-Elixir CPU fallback. Useful for unit tests and for quick
sanity-checks on machines without any GPU.

- `nx_backend: Nx.BinaryBackend`
- `require_cuda?: false`
- `default_slm_profile: :qwen_coordinator`

Expect order-of-magnitude slower latencies than CUDA or EMLX. Not
intended for production use.

### `:tiny_gpt2`

Synthetic profile used in tests; not for real workloads. Skip unless
you're writing tests.

### `{:custom, BackendMod, opts}`

Tuple-shaped profile for anyone wiring up a backend that does not have
a built-in name. The runtime calls `Nx.global_default_backend({BackendMod, opts})`
when this profile is selected.

## Which Mix Tasks Accept `--runtime-profile`?

After the Phase D refactor:

- `mix trinity.sakana.export_adapted --runtime-profile <name>` — pick
  the backend used to run the SVD/SVF pipeline.
- `mix trinity.sakana.router_trace --runtime-profile <name>` — trace a
  routing call with the selected backend.
- `mix trinity.sakana.large_tensor_chunks --runtime-profile <name>` —
  chunked tensor work (default: CUDA for back-compat).
- `mix trinity.sakana.parity_sample --runtime-profile <name>` —
  parity sampling.
- `mix trinity.hitl.adapted --runtime-profile <name>`, also
  `trinity.hitl.base_qwen`, `trinity.hitl.gpu`, `trinity.hitl.head_route`.
- `mix run examples/qwen_router_prompt_eval.exs --runtime-profile <name>`.
- `mix run examples/local_coordinator_route.exs --runtime-profile <name>`.
- `mix run examples/mock_orchestration_trace.exs --runtime-profile <name>`.

`--runtime-profile cuda_exla` is the default for every task, so no
existing CUDA workflow needs a flag.

## Backend Selection In Library Code

`TrinityCoordinator.Sakana.Coordinator.load/1` accepts:

```elixir
TrinityCoordinator.Sakana.Coordinator.load(
  runtime_profile: :emlx,
  artifact_dir: "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
)
```

For finer-grained overrides — for example, picking a non-named backend
without writing a `{:custom, ...}` profile — you can pass the
backend tuple directly:

```elixir
TrinityCoordinator.Sakana.Coordinator.load(
  runtime_profile: :emlx,                  # for require_cuda? = false
  backend: {EMLX.Backend, device: :cpu}    # but use CPU device
)
```

The `:backend` and `:require_cuda` keys are compatibility overrides
that pre-date the profile system; they remain supported.

## Choosing A Profile

| You have… | Use |
|---|---|
| NVIDIA GPU + CUDA-12 toolchain + Linux | `:cuda_exla` (default) |
| Apple Silicon (M-series) | `:emlx` + add `{:emlx, "~> 0.3"}` to your deps |
| No GPU; want to run unit tests / quick sanity checks | `:cpu_binary` |
| Some other backend (e.g. Torchx, custom NIF) | `{:custom, BackendMod, opts}` |

## Verifying A Profile

```bash
mix trinity.env.check
```

reports the current `XLA_TARGET` and any artifact-directory issues
without loading EXLA. For richer per-profile validation, the
`RuntimeProfile.compatibility_probe/1` family of functions returns a
structured report indicating whether the profile's expected backend is
loadable in this process, whether the artifact path exists, and so on.

## References

- `TrinityCoordinator.RuntimeProfile` — the module that defines and
  resolves profiles.
- `TrinityCoordinator.Sakana.Coordinator.load/1` — the canonical load
  entry point.
- [Artifact Distribution](artifact_distribution.md) — how to fetch /
  publish the bundle.
- [Troubleshooting](troubleshooting.md) — common failure modes by
  symptom.
