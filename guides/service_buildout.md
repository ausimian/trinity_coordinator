# Service Buildout Plan

The parity work is a foundation. The service goal is to run the final small
local coordinator model and integrate it with real LLM provider calls.

## Target Runtime

The target runtime loop is:

1. Receive a transcript.
2. Load or reuse `:qwen_sakana_adapted`.
3. Extract the penultimate hidden-state vector.
4. Route through the imported Sakana head.
5. Select agent and role.
6. Inject the role prompt.
7. Dispatch to a configured provider.
8. Persist a trace.
9. Continue until verifier acceptance or budget exhaustion.

Compatibility mode should mirror the supplemental Python loop:

- split logits into seven agent logits and three role logits;
- preserve raw role order `solver`, `thinker`, `verifier`;
- expose paper-facing `Worker` as an alias for raw `solver`, not a new role id;
- support stochastic softmax sampling for trace reproduction;
- support deterministic argmax for tests and operator debugging;
- allow a thinker response to set `<suggested_role>solver</suggested_role>` or
  `<suggested_role>verifier</suggested_role>` for the next turn.

## What Is Already In Place

- Core extractor/head/orchestrator contracts.
- Qwen base profile.
- Adapted Qwen profile hook.
- Router vector conversion.
- SVD/SVF reconstruction mechanics.
- Artifact loader and manifest validation.
- OpenAI-compatible provider boundary.
- Trace hashing and JSONL helpers.

## What Remains Before Service Use

The implementation order is:

1. all-selected tensor stage parity - complete for bounded layer-26 replay;
2. canonical Python semantic bundle import - complete;
3. adapted Qwen coordinator profile validation - complete;
4. fixed-transcript router trace parity - complete;
5. runtime service loop with trace persistence and provider adapters.

Router trace parity is now the gate that caught the square k/v orientation bug:
the adapted profile loaded and emitted logits, but Python and Elixir disagreed
until square Qwen layer kernels were transposed by semantic source path. Keep
that trace in the release checklist.

Complete adapted artifact validation:

- promote or copy the validated `tmp/sakana_parity/adapted_artifacts_from_python`
  bundle to the default `priv/sakana_trinity/adapted_qwen3_0_6b_layer26` runtime
  path when ready;
- keep `mix trinity.hitl.adapted --artifact-dir ...` as the canonical smoke
  command for adapted routing;
- keep `mix trinity.sakana.router_trace --python-report ...` as the Python
  parity gate before service changes.

Strengthen provider integration:

- define provider pools for the intended deployment;
- replace mock-heavy checks with explicit credential-gated smoke tests;
- record provider model, endpoint, latency, and error class in traces;
- make provider failure behavior explicit.

The checkpoint agent order is:

```text
0 gpt-5
1 claude-sonnet-4-20250514
2 gemini-2.5-pro
3 deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
4 google/gemma-3-27b-it
5 Qwen/Qwen3-32B (reasoning)
6 Qwen/Qwen3-32B (direct)
```

Any production pool may remap providers, but imported checkpoint logits only
have a defensible interpretation when this order is explicit.

Build service ergonomics:

- add a runnable service/demo command for the adapted coordinator;
- keep model loading warm across requests;
- expose route diagnostics without dumping sensitive prompt content by default;
- provide a minimal config story for CUDA, profile, provider pool, and budgets.

Persist traces:

- transcript hash;
- profile name;
- backend labels;
- hidden-state shape;
- route logits summary;
- selected agent and role;
- provider request metadata;
- verifier decision;
- error/retry metadata.

## Provider Boundary Status

The provider boundary exists, but the project should still be described as
provider-integration-in-progress. Tests can verify routing without pretending to
call real external models.

Production-like provider tests should be explicitly gated by credentials and
budget. A missing API key should produce a clear skipped/gated result, not a
fake pass.

## Operational Profile

The operational profile should prefer:

- one loaded local Qwen coordinator per runtime;
- CUDA-backed tensors for the SLM path;
- persisted adapted artifacts loaded from a complete manifest;
- real provider calls only through configured pools;
- JSONL traces for later audit.

## Success Criteria

The service foundation is complete when:

- `--strict-stage-tolerances` remains green for the semantic parity sample;
- canonical adapted artifact export succeeds and resumes reliably;
- `:qwen_sakana_adapted` loads and routes a real transcript on CUDA;
- a demo command runs the adapted coordinator through the provider boundary;
- provider calls are either real and traceable or explicitly gated;
- docs describe the exact operator commands and expected outputs.
