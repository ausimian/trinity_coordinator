defmodule TrinityCoordinator.ArtifactFetch.Pin do
  @moduledoc """
  Pinned descriptor for the project's adapted-artifact bundle.

  The pin file is committed to the repo. It carries:

    * `:repo_id` — HuggingFace dataset repo holding the bundle
    * `:revision` — the bundle revision (tag, branch, or sha)
    * `:manifest_sha256` — sha256 of the bundle's own `manifest.json`,
      so the pin and the manifest cannot silently disagree
    * `:files` — list of `%{path: rel_path, sha256: hex}` entries,
      one per file in the bundle

  The pin is regenerated from the bundle's manifest with
  `build_support/build_artifact_pin.exs`.
  """

  @enforce_keys [:version, :repo_id, :revision, :manifest_sha256, :files]
  defstruct [:version, :repo_id, :revision, :manifest_sha256, :files]

  @type entry :: %{required(:path) => String.t(), required(:sha256) => String.t()}
  @type t :: %__MODULE__{
          version: integer(),
          repo_id: String.t(),
          revision: String.t(),
          manifest_sha256: String.t(),
          files: [entry()]
        }

  @supported_version 1

  @required_keys ~w(version repo_id revision files)

  @doc """
  Loads a pin descriptor from a JSON file on disk.

  Raises:

    * `File.Error` — pin file does not exist
    * `ArgumentError` — pin file is missing a required key or carries an
      unsupported `:version`
  """
  @spec load_pin!(Path.t()) :: t()
  def load_pin!(path) do
    raw = File.read!(path)
    decoded = Jason.decode!(raw)

    Enum.each(@required_keys, fn k ->
      unless Map.has_key?(decoded, k) do
        raise ArgumentError,
              "trinity.artifact pin #{inspect(path)} missing required key #{inspect(k)}"
      end
    end)

    version = Map.fetch!(decoded, "version")

    unless version == @supported_version do
      raise ArgumentError,
            "trinity.artifact pin #{inspect(path)} carries unsupported pin version " <>
              inspect(version) <> "; this codebase understands version #{@supported_version}"
    end

    files =
      Enum.map(Map.fetch!(decoded, "files"), fn %{"path" => p, "sha256" => s} ->
        %{path: p, sha256: s}
      end)

    %__MODULE__{
      version: version,
      repo_id: Map.fetch!(decoded, "repo_id"),
      revision: Map.fetch!(decoded, "revision"),
      manifest_sha256: Map.get(decoded, "manifest_sha256", ""),
      files: files
    }
  end
end
