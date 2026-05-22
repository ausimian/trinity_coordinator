defmodule Examples.QwenRouterSnapshotResolverTest do
  @moduledoc """
  Phase 5 — per-profile snapshot fixture resolution.

  When the operator does not pass `--snapshot`, the prompt-eval entry
  point falls through:

    1. `examples/fixtures/runtime_profiles/<profile>/qwen_router_prompt_eval_logits.json`
       if it exists (per-profile snapshot lane)
    2. `nil` — no snapshot drift check (preserves pre-Phase-5 behaviour
       when no per-profile fixture is shipped)

  When the operator explicitly passes `--snapshot path`, that path wins
  unmodified so existing CI flows keep working unchanged.
  """

  use ExUnit.Case, async: true

  alias Examples.QwenRouterPromptEval.SnapshotResolver

  describe "resolve/3" do
    test "explicit override returns the override path even if no fallback exists" do
      assert SnapshotResolver.resolve(:cuda_exla, "/tmp/explicit.json") == "/tmp/explicit.json"
    end

    test "no override returns the per-profile path when it exists" do
      dir =
        Path.join(System.tmp_dir!(), "qwen_snap_resolver_#{System.unique_integer([:positive])}")

      File.mkdir_p!(dir)
      on_exit(fn -> File.rm_rf!(dir) end)

      profile_path =
        Path.join([
          dir,
          "examples",
          "fixtures",
          "runtime_profiles",
          "fake",
          "qwen_router_prompt_eval_logits.json"
        ])

      File.mkdir_p!(Path.dirname(profile_path))
      File.write!(profile_path, "{}")

      resolved = SnapshotResolver.resolve(:fake, nil, base_dir: dir)

      assert resolved == profile_path
    end

    test "no override + no per-profile file returns nil (does NOT auto-fall-back to the legacy CUDA fixture)" do
      dir =
        Path.join(System.tmp_dir!(), "qwen_snap_resolver_#{System.unique_integer([:positive])}")

      File.mkdir_p!(dir)
      on_exit(fn -> File.rm_rf!(dir) end)

      # Seed a legacy-shaped fixture; the resolver must ignore it.
      legacy_path =
        Path.join([dir, "examples", "fixtures", "qwen_router_prompt_eval_logits.json"])

      File.mkdir_p!(Path.dirname(legacy_path))
      File.write!(legacy_path, "{}")

      resolved = SnapshotResolver.resolve(:cuda_exla, nil, base_dir: dir)

      assert resolved == nil,
             "must not opt operators into the snapshot check who did not pass --snapshot"
    end

    test "no override + no fixtures at all returns nil so the eval can no-op the snapshot check" do
      dir =
        Path.join(System.tmp_dir!(), "qwen_snap_resolver_#{System.unique_integer([:positive])}")

      File.mkdir_p!(dir)
      on_exit(fn -> File.rm_rf!(dir) end)

      assert SnapshotResolver.resolve(:cuda_exla, nil, base_dir: dir) == nil
    end
  end
end
