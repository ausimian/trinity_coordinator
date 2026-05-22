defmodule TrinityCoordinator.ArtifactPinTest do
  @moduledoc """
  Pins the committed `priv/sakana_trinity/artifact_pin.json` against the
  canonical published HF dataset so that `mix trinity.artifact.fetch`
  works on a fresh clone without operator edits.

  History: the pin file shipped with `repo_id: "your-org/..."` as a
  placeholder; that crashed every fresh-clone onboarding flow with
  `:unauthorized` because the `your-org/...` repo does not exist on
  the Hub. The clean-room test on 2026-05-21 surfaced this.
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.ArtifactFetch

  @committed_pin_path "priv/sakana_trinity/artifact_pin.json"

  test "committed artifact_pin.json points at the canonical nshkrdotcom dataset" do
    pin = ArtifactFetch.load_pin!(@committed_pin_path)

    assert pin.repo_id == "nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b",
           """
           The committed artifact_pin.json must point at the published HF
           dataset, not a placeholder. If you are forking the project, fork
           the dataset too and update both pin.repo_id and the corresponding
           SHA-256 values. Do NOT commit `your-org/` placeholders.
           """
  end

  test "committed artifact_pin.json revision is a published tag" do
    pin = ArtifactFetch.load_pin!(@committed_pin_path)

    assert pin.revision == "v1.0.0",
           "pin revision must match a tag published on the HF dataset"
  end

  test "committed artifact_pin.json lists exactly 11 files (manifest + router_head + 9 checkpoints)" do
    pin = ArtifactFetch.load_pin!(@committed_pin_path)

    assert length(pin.files) == 11,
           "pin must enumerate every file in the artifact bundle (manifest.json, router_head.safetensors, 9 checkpoint safetensors)"

    paths = Enum.map(pin.files, & &1.path)
    assert "manifest.json" in paths
    assert "router_head.safetensors" in paths
    assert Enum.any?(paths, &String.starts_with?(&1, "checkpoints/"))
  end
end
