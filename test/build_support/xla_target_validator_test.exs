defmodule XlaTargetValidatorTest do
  @moduledoc """
  Tests for the XLA_TARGET preflight validator.

  These tests manipulate `XLA_TARGET` via `System.put_env/2` and
  `System.delete_env/1`. Per AGENTS.md, tests may manipulate
  environment variables for config-boundary checks, which is exactly
  what this is. The suite is `async: false` because `System.put_env/2`
  is process-global and would otherwise race with other tests that
  read XLA_TARGET.

  The validator module itself lives at
  `build_support/xla_target_validator.exs` and is loaded eagerly by
  `mix.exs`; by the time the test suite runs, it is already loaded
  into the Code server.
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
    test "matches the bundled xla 0.9.x acceptance list exactly" do
      assert XlaTargetValidator.supported_xla_targets() ==
               ["cpu", "cuda", "cuda12", "rocm", "tpu"]
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
    test "rejects cuda13 (the canonical reported failure) and names cuda12 as remediation" do
      System.put_env("XLA_TARGET", "cuda13")

      try do
        XlaTargetValidator.validate!()
        flunk("expected validate!/0 to raise on XLA_TARGET=cuda13")
      rescue
        e in Mix.Error ->
          msg = Exception.message(e)
          assert String.contains?(msg, "cuda13")
          assert String.contains?(msg, "cuda12")
          assert String.contains?(msg, "not accepted by the bundled xla 0.9.x")
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
end
