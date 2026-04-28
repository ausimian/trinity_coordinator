defmodule TrinityCoordinator.SLMProfileArtifactTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Hitl.Adapted
  alias TrinityCoordinator.SLMProfile

  test "adapted qwen profile patches backbone tensors without injecting routing head into SLM params" do
    profile = SLMProfile.qwen_sakana_adapted()

    assert profile.name == :qwen_sakana_adapted
    assert profile.adapted_artifact_dir != nil
    assert profile.artifact_patch_options[:patch_router_head] == false
    assert profile.expected_hidden_size == 1024
  end

  test "adapted HITL task accepts an explicit canonical artifact directory" do
    opts =
      Adapted.parse_args!([
        "--artifact-dir",
        "tmp/sakana_parity/adapted_artifacts_from_python",
        "--compare-path",
        "decoder.blocks.26.self_attention.query.kernel",
        "--message",
        "Route this fixed transcript."
      ])

    assert opts.artifact_dir == "tmp/sakana_parity/adapted_artifacts_from_python"
    assert opts.compare_path == "decoder.blocks.26.self_attention.query.kernel"
    assert opts.message == "Route this fixed transcript."
  end

  test "adapted HITL task defaults to the canonical priv artifact directory" do
    opts = Adapted.parse_args!([])

    assert opts.artifact_dir == "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
    assert opts.compare_path == "decoder.blocks.26.self_attention.query.kernel"
    assert opts.message =~ "Select a TRINITY role"
  end
end
