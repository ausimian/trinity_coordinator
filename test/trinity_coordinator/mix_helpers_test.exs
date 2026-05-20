defmodule TrinityCoordinator.MixHelpersTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.MixHelpers

  describe "format_load_error/1" do
    test "unwraps coordinator_load_error with binary message" do
      reason = {:coordinator_load_error, "EXLA CUDA platform is not available"}
      assert MixHelpers.format_load_error(reason) == "EXLA CUDA platform is not available"
    end

    test "falls back to inspect/1 for any other shape" do
      assert MixHelpers.format_load_error(:enoent) == ":enoent"
      assert MixHelpers.format_load_error({:error, :missing}) == "{:error, :missing}"
    end

    test "does not treat non-binary inner message as a binary" do
      reason = {:coordinator_load_error, %{file: "x"}}
      formatted = MixHelpers.format_load_error(reason)
      assert is_binary(formatted)
      assert String.contains?(formatted, "coordinator_load_error")
    end
  end

  describe "load_coordinator!/1" do
    test "raises Mix.Error when the artifact directory does not exist" do
      missing =
        Path.join(
          System.tmp_dir!(),
          "trinity_mix_helpers_missing_#{System.unique_integer([:positive])}"
        )

      refute File.exists?(missing)

      raised =
        try do
          MixHelpers.load_coordinator!(artifact_dir: missing, require_cuda: false)
          flunk("expected Mix.Error")
        rescue
          e in Mix.Error -> e
        end

      message = Exception.message(raised)
      assert String.contains?(message, "coordinator load failed")
    end
  end
end
