defmodule XlaTargetValidatorTest do
  @moduledoc """
  Tests for the XLA_TARGET preflight validator.

  These tests manipulate `XLA_TARGET` via `System.put_env/2` and
  `System.delete_env/1`. Per AGENTS.md, tests may manipulate environment
  variables for config-boundary checks, which is exactly what this is. The suite
  is `async: false` because `System.put_env/2` is process-global and would
  otherwise race with other tests that read XLA_TARGET.
  """

  use ExUnit.Case, async: false

  setup do
    original = System.get_env("XLA_TARGET")

    on_exit(fn ->
      case original do
        nil -> System.delete_env("XLA_TARGET")
        value -> System.put_env("XLA_TARGET", value)
      end
    end)

    :ok
  end

  describe "supported_xla_targets/0" do
    test "matches the bundled xla 0.10.x acceptance list exactly" do
      assert XlaTargetValidator.supported_xla_targets() ==
               ["cpu", "cuda", "cuda12", "cuda13", "rocm", "tpu"]
    end
  end

  describe "recommended_xla_target/0" do
    test "is cuda12 (the canonical CUDA lane for the current dep stack)" do
      assert XlaTargetValidator.recommended_xla_target() == "cuda12"
    end
  end

  describe "validate!/0 — accepting cases" do
    test "accepts unset XLA_TARGET" do
      System.delete_env("XLA_TARGET")
      assert :ok = XlaTargetValidator.validate!()
    end

    test "accepts empty XLA_TARGET (treated as unset)" do
      System.put_env("XLA_TARGET", "")
      assert :ok = XlaTargetValidator.validate!()
    end

    test "accepts every value in supported_xla_targets/0" do
      for target <- XlaTargetValidator.supported_xla_targets() do
        System.put_env("XLA_TARGET", target)
        assert :ok = XlaTargetValidator.validate!(), "expected #{inspect(target)} to be accepted"
      end
    end
  end

  describe "validate!/0 — rejecting cases" do
    test "accepts cuda13 (xla 0.10.x added it; previously rejected under xla 0.9.x)" do
      System.put_env("XLA_TARGET", "cuda13")
      assert :ok = XlaTargetValidator.validate!()
    end

    test "rejects an unrecognised CUDA suffix and names cuda12 as the recommended remediation" do
      System.put_env("XLA_TARGET", "cuda14")

      try do
        XlaTargetValidator.validate!()
        flunk("expected validate!/0 to raise on XLA_TARGET=cuda14")
      rescue
        e in Mix.Error ->
          msg = Exception.message(e)
          assert String.contains?(msg, "cuda14")
          assert String.contains?(msg, "cuda12")
          assert String.contains?(msg, "not accepted by the bundled xla 0.10.x")
          assert String.contains?(msg, "guides/troubleshooting.md")
      end
    end

    test "rejects a garbage value and surfaces the actual value in the message" do
      System.put_env("XLA_TARGET", "garbage_target_xyz")

      try do
        XlaTargetValidator.validate!()
        flunk("expected validate!/0 to raise on a garbage target")
      rescue
        e in Mix.Error ->
          assert String.contains?(Exception.message(e), "garbage_target_xyz")
      end
    end

    test "rejects case mismatch (xla is case-sensitive)" do
      System.put_env("XLA_TARGET", "CUDA12")

      try do
        XlaTargetValidator.validate!()
        flunk("expected validate!/0 to raise on CUDA12 (wrong case)")
      rescue
        e in Mix.Error ->
          assert String.contains?(Exception.message(e), "CUDA12")
      end
    end

    test "rejects whitespace-padded value (xla does not trim)" do
      System.put_env("XLA_TARGET", "cuda12 ")

      try do
        XlaTargetValidator.validate!()
        flunk("expected validate!/0 to raise on a padded target")
      rescue
        e in Mix.Error ->
          assert String.contains?(Exception.message(e), "cuda12 ")
      end
    end
  end

  describe "raw_xla_target/0" do
    test "returns the literal env value (no trimming, no normalisation)" do
      System.put_env("XLA_TARGET", "cuda12 ")
      assert XlaTargetValidator.raw_xla_target() == "cuda12 "
    end

    test "returns nil when XLA_TARGET is unset" do
      System.delete_env("XLA_TARGET")
      assert XlaTargetValidator.raw_xla_target() == nil
    end
  end

  describe "repo_under_mix_deps?/1" do
    test "detects a project loaded from a parent deps directory" do
      assert XlaTargetValidator.repo_under_mix_deps?("/tmp/fresh/deps/trinity_coordinator")
      assert XlaTargetValidator.repo_under_mix_deps?("/tmp/fresh/deps/foo/deps/bar")
    end

    test "does not classify ordinary workspace paths as deps-managed" do
      refute XlaTargetValidator.repo_under_mix_deps?("/home/home/p/g/n/trinity_coordinator")
      refute XlaTargetValidator.repo_under_mix_deps?("/tmp/workspace/trinity_coordinator")
    end
  end

  describe "validate_root_project!/1" do
    test "skips invalid XLA_TARGET when this project is loaded as a dependency" do
      System.put_env("XLA_TARGET", "cuda14")

      assert :ok =
               XlaTargetValidator.validate_root_project!(
                 "/tmp/parent_project/deps/trinity_coordinator"
               )
    end

    test "rejects invalid XLA_TARGET when this project is the root" do
      System.put_env("XLA_TARGET", "cuda14")

      try do
        XlaTargetValidator.validate_root_project!("/tmp/root_project/trinity_coordinator")
        flunk("expected validate_root_project!/1 to raise for a root project")
      rescue
        e in Mix.Error ->
          assert String.contains?(Exception.message(e), "cuda14")
      end
    end
  end
end
