# Bumblebee Unpin Playbook

This playbook is the 15-minute job to take when a Bumblebee Hex release
lands that includes Qwen3 support at or after commit
`0fd8114cf5429af9236f100f3350986e9d823c02`.

Until then, `mix.exs` pins Bumblebee to that commit on
`elixir-nx/bumblebee`, and `mix trinity.gates --include-hex-build` treats
the `hex_build_advisory` step as non-blocking by design.

This playbook re-promotes that gate to blocking once Bumblebee is
unpinned.

## 1. When to trigger

Trigger when both are true:

- `mix hex.info bumblebee` shows a published version that lists Qwen3
  in its CHANGELOG.
- The published version is at or beyond Bumblebee commit
  `0fd8114cf5429af9236f100f3350986e9d823c02`.

If only the first is true, audit the diff between the pinned commit and
the published version before proceeding. If material differences exist,
defer this playbook and file a Bumblebee issue.

## 2. The change set (single PR)

### 2.1 `mix.exs`

Replace the git-pinned spec:

```elixir
# Before:
{:bumblebee,
 github: "elixir-nx/bumblebee",
 ref: "0fd8114cf5429af9236f100f3350986e9d823c02",
 override: true},

# After (pick the published version):
{:bumblebee, "~> X.Y"},
```

(`override: true` is no longer needed because the Hex release will be
the unambiguous source.)

### 2.2 `lib/mix/tasks/trinity.gates.ex`

Flip the `hex_build` tuple. In `optional_gates/1`, change:

```elixir
# Before:
[{:hex_build, "mix", ["hex.build", "--unpack"], false, true}]

# After:
[{:hex_build, "mix", ["hex.build", "--unpack"], true, false}]
```

The 4th tuple element is `blocking?`; the 5th is `advisory?`. Flipping
both makes the gate blocking (and not advisory).

Update the moduledoc to remove the "advisory" note for `hex_build`.

### 2.3 `CHANGELOG.md`

Add a follow-up entry:

```markdown
## YYYY-MM-DD

### Changed
- Unpinned Bumblebee to Hex release `X.Y` (was a git pin against
  `0fd8114cf5…6c09be`).
- `mix trinity.gates --include-hex-build` `hex_build` step is now
  blocking again; the previous advisory mode is no longer needed.
```

## 3. Verification (in order)

```bash
cd /home/home/p/g/n/trinity_coordinator
export XLA_TARGET=cuda12

mix deps.unlock bumblebee
mix deps.get

mix test
# Expect: same test count as pre-unpin, 0 failures.

mix trinity.gates --include-hex-build
# Expect: exit 0, all 7 steps passing (including hex_build, now blocking).
```

Canonical smokes:

```bash
mix trinity.hitl.adapted
# Expect: shape contract output matching README "Adapted Coordinator Smoke".

mix run examples/qwen_router_prompt_eval.exs \
  --snapshot examples/fixtures/qwen_router_prompt_eval_logits.json \
  --determinism-runs 2
# Expect: 37/37 PASS.
```

If the snapshot fails on a substantive logit drift, the new Bumblebee
loader changed Qwen3 weight conversion. Investigate before promoting.

## 4. Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `mix deps.get` reports older Bumblebee than expected | Hex cache stale | `mix hex.update bumblebee` then retry |
| 37-case eval fails after upgrade | Loader changed something material | Diff loader code in new Bumblebee; isolate; consider a one-minor pin while you patch |
| `mix hex.build --unpack` still fails | Another git pin you didn't know about | `grep -n "github:\|git:" mix.exs`; resolve each |
| Router head sha drift on `mix trinity.sakana.router_trace` | Bumblebee changed checkpoint conversion | Compare hashes; if drift is expected, capture the new fixture and re-baseline; if not, escalate |

## 5. Rollback

```bash
git revert <unpin commit>
mix deps.get
XLA_TARGET=cuda12 mix test
```

The previous `mix.exs` git pin is the canonical fallback. It is
known-good against the test suite at the time this playbook was
written.

## See also

- [`CHANGELOG.md`](../CHANGELOG.md) — current Bumblebee-pin reasoning
  is in the 2026-05-20 entry.
- [`docs/production_runbook.md`](production_runbook.md) — §1 environment
  check; §8 in-place artifact update.
- `~/jb/docs/20260519/sakana/appendix/G_post_checklist_review_and_amendments.md`
  §G.2.6 — original advisory-demotion decision record.
