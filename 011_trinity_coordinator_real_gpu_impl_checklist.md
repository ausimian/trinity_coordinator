# TRINITY Real-GPU Implementation Checklist

Date started: 2026-04-25

Active stream: stream:qwen

## Legend

- [ ] not done
- [x] done
- `stream:` marks which implementation doc this item belongs to

## Stream 1: Sep-CMA-ES Training

- [x] stream:sep_cma_es define missing end-to-end `SepCMAES.train/4` API
- [x] stream:sep_cma_es fix candidate/model state reward aggregation and metadata
- [x] stream:sep_cma_es add seeded sampling determinism and reproducibility tests
- [x] stream:sep_cma_es add recombination/best-candidate ranking logic
- [x] stream:sep_cma_es add evaluator adapter and one-generation integration test
- [x] stream:sep_cma_es add checkpointed run trace and stop conditions
- [ ] stream:sep_cma_es gate provider-backed eval path and document
- [ ] stream:sep_cma_es update docs and README usage examples

## Stream 2: Coordination Head Variants

- [x] stream:head_variants add variant option parsing with validation
- [x] stream:head_variants support `:linear` default without behavior change
- [x] stream:head_variants add `:block_diagonal` head implementation and tests
- [x] stream:head_variants add `:sparse` head implementation and tests
- [x] stream:head_variants add variant routing metadata and parameter-count helpers
- [x] stream:head_variants add demo flags and docs updates

## Stream 3: Trace Persistence

- [x] stream:trace_persistence add `Trace.Event` schema and validation
- [x] stream:trace_persistence add deterministic hashing helpers for vectors/transcripts/params
- [x] stream:trace_persistence add JSONL sink and redaction helpers
- [x] stream:trace_persistence integrate trace context into orchestrator lifecycle
- [x] stream:trace_persistence integrate provider metadata emission in orchestrator/agent flow
- [x] stream:trace_persistence add trace persistence tests and example output

## Stream 4: Configurable Provider Pools

- [x] stream:provider_pools finish `ProviderPool.Spec` validation and normalization checks
- [x] stream:provider_pools integrate pool size and specs in orchestrator defaults
- [x] stream:provider_pools add `openai_compatible` adapter contract tests with base URL
- [x] stream:provider_pools document runtime config and sample profiles
- [x] stream:provider_pools keep pool checklists and docs in sync

## Stream 5: Benchmark Harnesses

- [x] stream:benchmark implement dataset loader and fixture validation
- [x] stream:benchmark add feature extraction utility using real extractor
- [x] stream:benchmark implement separability suite metrics and report output
- [x] stream:benchmark implement routing accuracy suite and confusion matrix output
- [x] stream:benchmark implement turn-budget suite metrics
- [x] stream:benchmark add `mix trinity.benchmark` task and CLI docs

## Stream 6: Provider Smoke Tests

- [x] stream:provider_smoke add env/budget guard helpers (`TRINITY_ENABLE_PROVIDER_TESTS`, budget)
- [x] stream:provider_smoke add credential parsing and skip behavior tests
- [x] stream:provider_smoke add redacted trace persistence requirements for provider calls
- [x] stream:provider_smoke add single integration smoke test scaffold (gated by env)
- [x] stream:provider_smoke add multi-turn smoke command coverage and safety limits

## Stream 7: Production Qwen Profile Readiness

- [x] stream:qwen profile confirm dependency support path without blockers
- [x] stream:qwen add automated profile probe for module compatibility
- [x] stream:qwen define explicit pending/unsupported behavior in profile contract
- [ ] stream:qwen add docs section and command examples for transition to production profile
- [ ] stream:qwen run compatibility checks in CI-friendly sequence
