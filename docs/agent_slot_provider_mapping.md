# Agent Slot ↔ Provider Mapping

This document is the canonical reference for one of the easiest things to
misunderstand about TRINITY: **what an agent slot is, and how it relates to
the model the system actually calls.**

If you only read one paragraph, read this:

> The Sakana TRINITY router emits an **agent slot id** (`0..6`). The labels
> attached to those slots in the Sakana checkpoint (`gpt-5`, `claude-sonnet-4-...`,
> `gemini-2.5-pro`, `deepseek-ai/DeepSeek-R1-...`, `google/gemma-3-27b-it`,
> `Qwen/Qwen3-32B (reasoning)`, `Qwen/Qwen3-32B (direct)`) are **historical
> training metadata**, not provider bindings. They tell you what model was in
> that slot when Sakana trained the router. They **do not** tell `trinity_coordinator`
> which live model to call. That decision is made by the **provider pool**.

The rest of this document expands on that.

## The seven agent slots

The router head shape is `[10, 1024]`, decomposed as 7 agent logits + 3 role
logits. The 7-agent decomposition is fixed by the Sakana checkpoint we
imported; the slot labels are stored in the manifest under
`python_semantic_manifest.routing.agent_labels`:

| Slot | Sakana label | What it means |
|---|---|---|
| 0 | `gpt-5` | A large general reasoning slot at training time. |
| 1 | `claude-sonnet-4-20250514` | A large reasoning/writing slot. |
| 2 | `gemini-2.5-pro` | A long-context slot. |
| 3 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | A distilled-reasoning slot. |
| 4 | `google/gemma-3-27b-it` | A mid-size general slot. |
| 5 | `Qwen/Qwen3-32B (reasoning)` | A reasoning-mode Qwen slot. |
| 6 | `Qwen/Qwen3-32B (direct)` | A direct-mode Qwen slot. |

The router's job is to pick one of these slots for each turn. **What happens
to the call after slot selection is up to the provider pool.**

## The provider pool boundary

`TrinityCoordinator.ProviderPool` (`lib/trinity_coordinator/provider_pool.ex`)
defines named pools that map each `agent_id` to an explicit `{provider, model}`
pair. Three pools ship with the project:

| Pool | Used when | Slot 0..6 all map to |
|---|---|---|
| `:default` | safe default for service runs | `:openai` provider, `gpt-4o-mini` model |
| `:mock` | `--mock-provider` / `--mock` runs and CI | `:mock` provider |
| `:gemini_cli_asm` | live Gemini-via-ASM lane | `:asm` provider, `gemini-3.1-flash-lite-preview` model |

The default pool deliberately collapses all seven slots to a single inexpensive
model. **No live deployment should assume slot 0 means "call GPT-5" — none of
the default pools call any of the named Sakana labels.**

If you want a deployment where slot 0 calls a real `gpt-5` API, you must
configure a custom provider pool that says so. There is no implicit binding.

## The legacy compatibility map

`TrinityCoordinator.AgentPool` also carries a legacy `@legacy_agents` map
(see `lib/trinity_coordinator/agent_pool.ex`) that predates provider pools.
It maps every slot to `{provider: :openai, model: "gpt-4o-mini"}` for callers
that still pass `agents:` directly. It exists for backward compatibility; new
code should use named provider pools.

## Configuring a custom pool

```elixir
# In config/runtime.exs or via Config.Provider:
config :trinity_coordinator, :provider_pools, [
  default: [
    [id: 0, name: :gpt5,        provider: :openai,    model: "gpt-5"],
    [id: 1, name: :claude,      provider: :anthropic, model: "claude-sonnet-4-20250514"],
    [id: 2, name: :gemini,      provider: :gemini,    model: "gemini-2.5-pro"],
    [id: 3, name: :deepseek,    provider: :openai_compatible, model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"],
    [id: 4, name: :gemma,       provider: :openai_compatible, model: "google/gemma-3-27b-it"],
    [id: 5, name: :qwen_reason, provider: :openai_compatible, model: "Qwen/Qwen3-32B"],
    [id: 6, name: :qwen_direct, provider: :openai_compatible, model: "Qwen/Qwen3-32B"]
  ]
]
```

This is the **only** way to make slot 0 actually call GPT-5. The router does
not configure provider routing for you.

## How to read a trace

JSONL trace events include:

- `selected_agent_id` — the slot id (0..6) the router returned.
- `provider` — the provider the pool resolved that slot to.
- `model` — the model id the pool resolved that slot to.
- `selected_agent_label` (if present) — the Sakana checkpoint label,
  metadata-only.

Always read `provider` and `model` to know what was actually called. Reading
`selected_agent_label` alone is misleading.

## Safety assertions

Tests under `test/trinity_coordinator/provider_pool_test.exs` enforce:

- The `:default` and `:mock` pools each declare exactly 7 entries with ids
  0..6, no duplicates, no gaps.
- The `:mock` pool's provider is `:mock` for every slot.
- The `:default` pool's models are explicit (string-typed, non-nil).

These assertions prevent a future refactor from accidentally re-introducing
"slot label → live model" coupling.

## Why this matters

The most likely production incident in this part of the system is a misread
label. Someone sees `agent 0 = gpt-5` in a Sakana fixture, assumes the system
calls GPT-5, and is then surprised when the bill shows `gpt-4o-mini` (or a
custom deployment shows none of the above). The router and the provider pool
are deliberately decoupled. This document exists to keep them that way.
