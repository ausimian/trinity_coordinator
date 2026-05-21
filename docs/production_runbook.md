# Production Deployment Runbook

This runbook covers operating `trinity_coordinator` in service. It assumes
you have a working CUDA host and the canonical adapted artifact bundle.

The companion documents are:

- [`docs/agent_slot_provider_mapping.md`](agent_slot_provider_mapping.md) —
  why slot 0's training label is not a provider binding.
- [`docs/bumblebee_unpin_playbook.md`](bumblebee_unpin_playbook.md) — the
  15-minute job to take when a Bumblebee Hex release lands.
- [`guides/operations_qc.md`](../guides/operations_qc.md) — quality gates
  for code changes (not for service operation).

## 1. Pre-deploy environment check

```bash
XLA_TARGET=cuda12 mix trinity.env.check
```

A pre-flight task that validates `XLA_TARGET` against the bundled `xla`
dependency's accepted values and (optionally) that `--artifact-dir`
exists with a `manifest.json`. Fails fast with a single readable line
before EXLA loads.

Common output:

```text
trinity.env.check: ok / xla_target=cuda12
```

Known gotcha: `XLA_TARGET=cuda13` is rejected at compile time by
`xla 0.9.1`. Use `cuda12`.

## 2. Canonical artifact

Path: `priv/sakana_trinity/adapted_qwen3_0_6b_layer26/`.

Contents:

```text
manifest.json
router_head.safetensors
checkpoints/*.safetensors
```

The router head sha256 invariant (Phase 3 promoted) is
`7ff2db0e…6c09be`. Validate with:

```bash
XLA_TARGET=cuda12 mix trinity.env.check --artifact-dir \
  priv/sakana_trinity/adapted_qwen3_0_6b_layer26
```

The dir is gitignored by design: it is a generated artifact, not source.
Copy it from a blessed bundle on first install.

## 3. Provider pool selection

The router emits an agent slot id (`0..6`). The slot label in the Sakana
checkpoint (`gpt-5`, `gemini-2.5-pro`, ...) is training metadata, not a
provider binding.

Read [`docs/agent_slot_provider_mapping.md`](agent_slot_provider_mapping.md)
for the full contract. Three pools ship:

| Pool | Used when | Slot 0..6 all map to |
|---|---|---|
| `:default` | safe default for service runs | `:openai` provider, `gpt-4o-mini` model |
| `:mock` | `--mock-provider` runs and CI | `:mock` provider |
| `:gemini_cli_asm` | Gemini via the ASM CLI lane | `:asm` provider, `gemini-3.1-flash-lite-preview` |

A custom pool example lives in
[`docs/agent_slot_provider_mapping.md`](agent_slot_provider_mapping.md)
under "Configuring a custom pool". That is the only way to make slot 0
actually call a `gpt-5` API; there is no implicit binding.

## 4. Budgets

All budget options default to `nil` (unbounded). On exceed the orchestrator
returns `{:error, {:budget_exceeded, kind, details}}` and emits a
`:run_failed` trace event with the same kind and details.

### `:max_wall_time_ms`

- Protects against: runaway wall time.
- Reference for a 5-turn loop: `30_000` to `60_000`.
- Checkpoint: `:turn_start`.
- Details: `%{limit_ms, elapsed_ms, checkpoint, turn}`.

### `:max_provider_calls`

- Protects against: runaway dispatches.
- Reference for a 5-turn loop: `5` (one dispatch per turn) up to `10`
  if you expect verifier-revision retries.
- Checkpoint: `:before_dispatch`.
- Semantics: `:max_provider_calls = N` allows exactly N dispatches; the
  (N+1)th attempt aborts.
- Details: `%{limit, observed, checkpoint}` (`observed` is `N+1` at
  fail time).

### `:max_provider_latency_ms`

- Protects against: one slow dispatch dragging down the SLA.
- Reference: `30_000` for hosted LLMs; `60_000` for ASM CLI lane.
- Checkpoint: `:after_dispatch`.
- Details: `%{limit_ms, observed_ms, checkpoint, turn}`.

### `:max_verifier_revisions`

- Protects against: a verifier that keeps rejecting forever.
- Reference: `3` for a 5-turn loop.
- Checkpoint: `:after_verifier_revision`.
- Counted: only Verifier dispatches whose status is not `:accepted`.
- Details: `%{limit, observed, checkpoint, turn}`.

### `:max_estimated_cost_usd`

- Protects against: spend overrun.
- Reference: depends entirely on your pool's per-call cost.
- Requires `:cost_estimator_fn` to actually fire.
- Checkpoint: `:after_dispatch`.
- Details: `%{limit_usd, observed_usd, checkpoint, turn}`.

`:cost_estimator_fn` signature is `(dispatch_map) :: float()` where
`dispatch_map` includes `:provider`, `:provider_model`, `:response_text`,
`:mode`, `:provider_latency_ms`. The orchestrator deliberately does not
ship a pricing table; you supply the function with your current vendor
pricing.

Without a `:cost_estimator_fn`, setting `:max_estimated_cost_usd`
non-nil triggers a one-shot `Logger.warning/1` per `run_loop/4` call
that the budget will not fire.

## 5. Trace persistence and rotation

`Orchestrator.run_loop/4` accepts `:trace` with a keyword payload:

```elixir
trace: [
  enabled: true,
  sink: {:jsonl, "/var/log/trinity/run_2026-05-20.jsonl"},
  run_id: "deploy_42",
  content: :hash
]
```

`content: :hash` (default) redacts free-text content; `content: :full`
keeps it (for debug only — do not enable in production unless your
storage is locked down). The redactor scrubs any `api_key`,
`authorization`, `password`, `secret`, or `token` map key recursively.

Suggested rotation: rotate JSONL files daily by `run_id` or by date.
Each line is a single `Event` record.

Trace events emitted per turn (in order):

| Event | Meaning |
|---|---|
| `:run_started` | Once per `run_loop/4` call |
| `:turn_started` | Once per turn |
| `:slm_extracted` | Hidden-state vector ready |
| `:route_selected` | Coordinator head produced agent + role |
| `:provider_called` (status=`started`) | Dispatch attempt |
| `:provider_called` (status=`ok` or `error`) | Dispatch outcome |
| `:turn_completed` | Includes `:route_decision` map (Phase 11) |
| `:run_completed` | Once on verifier acceptance |
| `:run_failed` | Once on any error path (incl. budget) |

The `:route_decision` field in `:turn_completed` carries a JSON-safe map
with `:agent_id`, `:role_id`, `:role_name`, `:margins`,
`:selection_modes`, `:transcript_hash`, and `:artifact_identity`.

## 6. Failure-mode triage

| Error tuple | Meaning | Likely cause | Action |
|---|---|---|---|
| `{:error, :coordinator_load_error, msg}` | Coordinator could not load | Wrong `XLA_TARGET`, missing artifact dir, or EXLA not built for CUDA | `mix trinity.env.check --artifact-dir <path>` |
| `{:error, :missing_slm_context}` | No SLM context supplied | Caller passed neither `:slm_context` nor `:extractor_fn` | Supply one (test or production) |
| `{:error, :verifier_before_worker_response}` | Verifier role selected before any Worker dispatched | Prompt biases verifier on turn 0 | Re-prompt with a non-verifier opener or call `RoleInjector.role_atom("Worker")` first |
| `{:error, {:provider_dispatch_failed, reason, latency_ms}}` | Provider returned an error | Auth, network, vendor outage | Inspect `reason`; retry if transient |
| `{:error, {:budget_exceeded, :wall_time, _}}` | `:max_wall_time_ms` hit | Loop took too long | Raise the limit or shrink the prompt |
| `{:error, {:budget_exceeded, :provider_calls, _}}` | `:max_provider_calls` hit | Loop tried more dispatches than budget | Raise or accept partial answer |
| `{:error, {:budget_exceeded, :provider_latency_ms, _}}` | One dispatch was too slow | Slow vendor, slow network, or the vendor is rate-limiting | Raise limit, switch pool, or backoff |
| `{:error, {:budget_exceeded, :verifier_revisions, _}}` | Verifier rejected `:max_verifier_revisions + 1` times | Bad prompt, bad worker, or genuinely-hard task | Raise limit, change prompt, or escalate |
| `{:error, {:budget_exceeded, :estimated_cost_usd, _}}` | Cost exceeded | Cumulative dispatches above limit | Raise limit or shrink loop |
| `{:error, :max_turns_reached}` | Loop hit `:max_turns` with no verifier accept and no latest worker response | Misconfigured roles or prompt | Inspect trace |

`:max_turns_reached` is distinct from `:max_turns_latest_worker_response`
(`:ok` result): the latter happens when the loop hits the turn cap with
a partial answer; the former when no Worker ever produced an answer.

## 7. Secrets handling

The governed-authority path in
[`README.md` § "Running The Router"](../README.md#running-the-router)
is the production way to ship credentials. The orchestrator code does
not read `System.get_env/1` directly; runtime env reads live in
`config/runtime.exs` or a `Config.Provider`.

Trace output records provider/model labels, opaque refs, hashes, and
fixed redaction markers, never materialized secret values.

## 8. Updating the artifact in place

```bash
# Stop the runtime (your supervisor / systemd unit).
sudo systemctl stop trinity-coordinator

# Copy new artifact.
rsync -av --delete /path/to/new/adapted_qwen3_0_6b_layer26/ \
  priv/sakana_trinity/adapted_qwen3_0_6b_layer26/

# Validate.
XLA_TARGET=cuda12 mix trinity.env.check --artifact-dir \
  priv/sakana_trinity/adapted_qwen3_0_6b_layer26

XLA_TARGET=cuda12 mix trinity.hitl.adapted

XLA_TARGET=cuda12 mix run examples/qwen_router_prompt_eval.exs \
  --snapshot examples/fixtures/qwen_router_prompt_eval_logits.json \
  --determinism-runs 2

# Restart.
sudo systemctl start trinity-coordinator
```

If the eval script's default margin floors fire (Phase 11 Option D —
`--min-agent-margin 0.24`, `--min-role-margin 1.06`), the new artifact
materially changed routing confidence. Compare to the previous
snapshot before promoting.

## 9. Observability surface

Recommended sink today: JSONL file plus a logrotate or daily file
naming convention. Each line is one `Event` record.

A Telemetry bridge is on the roadmap (not landed). When it lands,
`trinity_coordinator` will emit Telemetry events that mirror the JSONL
event schema, so external Phoenix or Telegraf consumers can subscribe
without parsing files.

## 10. Rollback

```bash
sudo systemctl stop trinity-coordinator
rsync -av --delete /path/to/previous/adapted_qwen3_0_6b_layer26/ \
  priv/sakana_trinity/adapted_qwen3_0_6b_layer26/
XLA_TARGET=cuda12 mix trinity.env.check
XLA_TARGET=cuda12 mix trinity.hitl.adapted
sudo systemctl start trinity-coordinator
```

Confirm:

```bash
XLA_TARGET=cuda12 mix run examples/qwen_router_prompt_eval.exs \
  --snapshot examples/fixtures/qwen_router_prompt_eval_logits.json \
  --determinism-runs 2
```

37/37 PASS is the expected result.

## See also

- [`docs/agent_slot_provider_mapping.md`](agent_slot_provider_mapping.md)
- [`docs/bumblebee_unpin_playbook.md`](bumblebee_unpin_playbook.md)
- [`guides/provider_service_hardening.md`](../guides/provider_service_hardening.md)
- [`CHANGELOG.md`](../CHANGELOG.md)
