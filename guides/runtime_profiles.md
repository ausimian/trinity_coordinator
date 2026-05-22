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

## Per-Profile Snapshot Fixtures And Margin Floors

`examples/qwen_router_prompt_eval.exs` supports a per-profile snapshot
fixture lane and per-profile margin defaults so a non-CUDA backend can
land its own empirical floors without rewriting or lowering the
canonical CUDA snapshot.

Resolution order for the `--snapshot` flag (Phase 5):

1. **Explicit `--snapshot path`** — wins unconditionally; existing CI
   flows pinning the canonical CUDA fixture keep working without
   reinterpretation.
2. **`examples/fixtures/runtime_profiles/<profile>/qwen_router_prompt_eval_logits.json`**
   — picked up automatically when the per-profile file is present.
3. **`nil`** — no snapshot drift check (the same default behaviour as
   before Phase 5). To pin against the CUDA snapshot, pass
   `--snapshot examples/fixtures/qwen_router_prompt_eval_logits.json`
   explicitly. We deliberately do **not** fall through to the legacy
   fixture path automatically: that would silently enable a strict
   6dp logits byte-equivalence check for operators who did not opt in.

Margin floor resolution (`--min-agent-margin` / `--min-role-margin`):

1. **Explicit CLI flag** — wins.
2. **`RuntimeProfile.default_margins(profile)`** — every built-in
   profile inherits the canonical CUDA defaults (`agent: 0.24`,
   `role: 1.06`) unless overridden via
   `RuntimeProfile.override_default_margins/2` (e.g. for a future
   `:emily` profile that wants `agent: 0.33`, `role: 0.82`).
3. **Module-level defaults** (legacy fallback in the eval script).

## Validating With Emily (Apple Silicon, MLX, Research)

[Emily](https://hex.pm/packages/emily) is the open-source MLX-backed Nx
backend used by the Apple-side research validation pass. The canonical
Apple lane is `:emlx`; `:emily` is intentionally **not** a built-in
profile today. Apple-side operators who want to validate an export
against Emily use the `{:custom, Emily.Backend, []}` route instead.

To run the prompt eval through Emily without adding a new profile:

```elixir
# config/runtime.exs or in your iex session
profile =
  TrinityCoordinator.RuntimeProfile.resolve({:custom, Emily.Backend, []})
  |> TrinityCoordinator.RuntimeProfile.override_default_margins(
       agent: 0.33,
       role: 0.82
     )
```

```bash
# Parent app's mix.exs (do NOT add this to trinity_coordinator's own mix.exs):
#   {:emily, "~> 0.4", only: [:dev, :test]}
mix deps.get

mix trinity.sakana.export_adapted \
  --force \
  --svd-compute-type f32 \
  --runtime-profile '{:custom, Emily.Backend, []}' \
  --out tmp/emily_adapted_qwen3_0_6b_layer26

mix run examples/qwen_router_prompt_eval.exs \
  --runtime-profile '{:custom, Emily.Backend, []}' \
  --artifact-dir tmp/emily_adapted_qwen3_0_6b_layer26 \
  --determinism-runs 2 \
  --min-agent-margin 0.33 \
  --min-role-margin 0.82
```

Per ausimian's validation pass on 2026-05-21:

- 0/37 drift on the decision-stable fields (`agent_id`, `role_id`,
  `token_count`, `transcript_hash`).
- 37/37 differ on `route_hash` (6dp logit drift — expected on a
  different kernel stack).
- The empirical worst margins were `agent: 0.417` (`two_assistant_turns`)
  and `role: 1.029` (`escalate_to_human`); 80% floors are therefore
  `0.33` / `0.82`.
- Phase 1 (lazy-backend timing sync) makes `decompose_elapsed_ms`
  report real GPU wall time on Emily / EMLX instead of the host-side
  dispatch cost of an unmaterialised future.

To ship Emily as a first-class built-in profile in a future release,
add a `def resolve(:emily) do ... end` clause that mirrors the `:emlx`
clause but with `nx_backend: Emily.Backend` and the
`override_default_margins/2` already applied. Phase 4 of the
[Emily-validation response checklist](https://github.com/nshkrdotcom/trinity_coordinator)
documents the rationale for keeping it validation-only for now.

## References

- `TrinityCoordinator.RuntimeProfile` — the module that defines and
  resolves profiles.
- `TrinityCoordinator.Sakana.Coordinator.load/1` — the canonical load
  entry point.
- [Artifact Distribution](artifact_distribution.md) — how to fetch /
  publish the bundle.
- [Troubleshooting](troubleshooting.md) — common failure modes by
  symptom.
