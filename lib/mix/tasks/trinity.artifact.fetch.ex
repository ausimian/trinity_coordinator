defmodule Mix.Tasks.Trinity.Artifact.Fetch do
  @shortdoc "Downloads and SHA-verifies the adapted Qwen3 artifact bundle from HuggingFace"
  @moduledoc """
  Downloads the Sakana-adapted Qwen3 artifact bundle from the configured
  HuggingFace dataset repository and verifies each file against the pinned
  SHA-256 manifest.

  This is the canonical onboarding step for a fresh clone. The bundle is
  ~624 MB, gitignored, and *not* present in the cloned repo. Without this
  task (or an equivalent local export run), `mix run examples/qwen_router_prompt_eval.exs`
  cannot load the router because the artifact directory will be missing.

  ## Usage

      mix trinity.artifact.fetch
      mix trinity.artifact.fetch --dest custom/path
      mix trinity.artifact.fetch --pin custom_pin.json
      mix trinity.artifact.fetch --offline

  ## Options

    * `--pin PATH` - path to the pinned artifact descriptor JSON.
      Defaults to `priv/sakana_trinity/artifact_pin.json`.

    * `--dest PATH` - destination directory for the materialised bundle.
      Defaults to `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`.

    * `--offline` - consult the local HuggingFace cache only; do not hit
      the network. Useful for air-gapped CI. Defaults to off.

    * `--help` - print this message and exit.

  ## Cache behaviour

  Downloads land in the standard `~/.cache/huggingface/` cache (configurable
  via `HfHub`'s `cache_dir`). Files already present at the destination with
  the correct SHA-256 are not re-downloaded, so repeat invocations are
  cheap.

  ## Test seam

  The task reads `Process.get(:trinity_artifact_fetch_downloader)` for an
  optional test downloader override. Production callers do not need to set
  this; it exists so test suites can stub HuggingFace without requiring
  network access.
  """

  use Mix.Task

  alias TrinityCoordinator.ArtifactFetch

  @switches [
    pin: :string,
    dest: :string,
    offline: :boolean,
    help: :boolean
  ]

  @impl Mix.Task
  def run(argv) do
    {opts, _, _} = OptionParser.parse(argv, strict: @switches)

    if Keyword.get(opts, :help, false) do
      print_help()
      :ok
    else
      do_fetch(opts)
    end
  end

  defp do_fetch(opts) do
    pin_path = Keyword.get(opts, :pin, ArtifactFetch.default_pin_path())
    dest = Keyword.get(opts, :dest, ArtifactFetch.default_dest())
    offline? = Keyword.get(opts, :offline, false)

    pin = load_pin_or_raise!(pin_path)

    fetch_opts =
      [
        dest: dest,
        offline_mode: offline?
      ]
      |> maybe_inject_test_downloader()

    ArtifactFetch.fetch!(pin, fetch_opts)

    Mix.shell().info(
      "trinity.artifact.fetch: ok — #{length(pin.files)} files into #{dest} " <>
        "(repo #{pin.repo_id} @ #{pin.revision})"
    )

    :ok
  end

  defp load_pin_or_raise!(pin_path) do
    ArtifactFetch.load_pin!(pin_path)
  rescue
    e in [File.Error, ArgumentError] ->
      Mix.raise(
        "trinity.artifact.fetch: cannot load pin #{inspect(pin_path)}: " <>
          Exception.message(e)
      )
  end

  defp maybe_inject_test_downloader(opts) do
    case Process.get(:trinity_artifact_fetch_downloader) do
      nil -> opts
      fun when is_function(fun, 1) -> Keyword.put(opts, :downloader, fun)
    end
  end

  defp print_help do
    Mix.shell().info("""
    mix trinity.artifact.fetch — download and SHA-verify the adapted-Qwen3 bundle

    Options:
      --pin PATH        Pinned artifact descriptor (default: priv/sakana_trinity/artifact_pin.json)
      --dest PATH       Destination directory      (default: priv/sakana_trinity/adapted_qwen3_0_6b_layer26)
      --offline         Use HuggingFace cache only; do not hit the network
      --help            Print this message

    Run with no options for the recommended onboarding flow.
    """)
  end
end
