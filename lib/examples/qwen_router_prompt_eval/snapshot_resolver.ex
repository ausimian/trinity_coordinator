defmodule Examples.QwenRouterPromptEval.SnapshotResolver do
  @moduledoc """
  Resolves the snapshot fixture path used by `examples/qwen_router_prompt_eval.exs`.

  Resolution order, in priority:

    1. If the operator explicitly passes `--snapshot <path>`, that path
       wins unconditionally; the resolver returns it back unmodified so
       existing CI flows pinning the canonical CUDA snapshot keep
       working without reinterpretation.
    2. Otherwise, probe
       `examples/fixtures/runtime_profiles/<profile>/qwen_router_prompt_eval_logits.json`
       — per-profile snapshot lane (Phase 5). Lets a future `:emily`
       (or any other profile) ship its own backend-specific snapshot
       without lowering or rewriting the canonical CUDA fixture.
    3. Otherwise, return `nil`. The eval entry point treats `nil` as
       "no snapshot drift check", which is **the same default behaviour
       as before Phase 5** — if you want to pin against the canonical
       CUDA snapshot, pass `--snapshot examples/fixtures/qwen_router_prompt_eval_logits.json`
       explicitly. We deliberately do NOT fall through to the legacy
       fixture path automatically: on CUDA the snapshot is a strict
       6dp logits byte-equivalence check and quietly enabling it for
       operators who did not opt in would change observable behaviour.

  Test-side `:base_dir` hook lets `Examples.QwenRouterSnapshotResolverTest`
  exercise the resolution without touching the live `examples/fixtures/`
  tree.
  """

  @doc """
  Resolves the snapshot path for a given runtime profile and optional explicit override.

  ## Options

    * `:base_dir` — root used when probing the per-profile path.
      Defaults to the current working directory; only test code
      should ever pass this.
  """
  @spec resolve(atom(), String.t() | nil, keyword()) :: String.t() | nil
  def resolve(profile_name, explicit, opts \\ [])

  def resolve(_profile_name, explicit, _opts) when is_binary(explicit), do: explicit

  def resolve(profile_name, nil, opts) when is_atom(profile_name) do
    base_dir = Keyword.get(opts, :base_dir, File.cwd!())

    per_profile =
      Path.join([
        base_dir,
        "examples",
        "fixtures",
        "runtime_profiles",
        Atom.to_string(profile_name),
        "qwen_router_prompt_eval_logits.json"
      ])

    if File.regular?(per_profile), do: per_profile, else: nil
  end
end
