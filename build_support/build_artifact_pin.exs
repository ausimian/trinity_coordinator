# Build/refresh priv/sakana_trinity/artifact_pin.json from the bundle's
# own manifest.json. Run from project root:
#
#     mix run build_support/build_artifact_pin.exs \
#       --manifest priv/sakana_trinity/adapted_qwen3_0_6b_layer26/manifest.json \
#       --bundle-dir priv/sakana_trinity/adapted_qwen3_0_6b_layer26 \
#       --repo-id nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b \
#       --revision v1.0.0 \
#       --out priv/sakana_trinity/artifact_pin.json
#
# The script only reads files; it does not perform any network operation.

defmodule BuildArtifactPin do
  @moduledoc false

  @switches [
    manifest: :string,
    bundle_dir: :string,
    repo_id: :string,
    revision: :string,
    out: :string
  ]

  def run(argv) do
    {opts, _, _} = OptionParser.parse(argv, strict: @switches)

    manifest_path = require_opt!(opts, :manifest)
    bundle_dir    = require_opt!(opts, :bundle_dir)
    repo_id       = require_opt!(opts, :repo_id)
    revision      = require_opt!(opts, :revision)
    out_path      = require_opt!(opts, :out)

    manifest = manifest_path |> File.read!() |> Jason.decode!()

    selected =
      manifest
      |> Map.fetch!("selected_tensors")
      |> Enum.map(fn entry ->
        rel = Map.fetch!(entry, "checkpoint_path")
        sha = Map.fetch!(entry, "checkpoint_sha256")
        %{path: rel, sha256: sha}
      end)

    files =
      [
        %{
          path: "manifest.json",
          sha256: sha256_file!(manifest_path)
        },
        %{
          path: Map.fetch!(manifest, "router_head_artifact"),
          sha256: Map.fetch!(manifest, "router_head_sha256")
        }
      ] ++ selected

    pin = %{
      version: 1,
      repo_id: repo_id,
      revision: revision,
      manifest_sha256: sha256_file!(manifest_path),
      files: files
    }

    # Validate each path exists in the bundle
    Enum.each(files, fn %{path: rel} ->
      full = Path.join(bundle_dir, rel)

      unless File.exists?(full) do
        raise "expected bundle file missing on disk: #{full}"
      end
    end)

    out_path |> Path.dirname() |> File.mkdir_p!()
    File.write!(out_path, Jason.encode!(pin, pretty: true) <> "\n")

    IO.puts("wrote #{out_path}: #{length(files)} files, revision #{revision}")
  end

  defp require_opt!(opts, key) do
    Keyword.get(opts, key) ||
      raise "--#{String.replace(Atom.to_string(key), "_", "-")} is required"
  end

  defp sha256_file!(path) do
    :sha256
    |> :crypto.hash_init()
    |> hash_file(path)
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  defp hash_file(ctx, path) do
    File.stream!(path, [], 65_536)
    |> Enum.reduce(ctx, fn chunk, c -> :crypto.hash_update(c, chunk) end)
  end
end

BuildArtifactPin.run(System.argv())
