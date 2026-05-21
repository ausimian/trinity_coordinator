defmodule TrinityCoordinator.ArtifactFetch do
  @moduledoc """
  Fetches the Sakana-adapted Qwen3 artifact bundle from a HuggingFace
  dataset repository, validates each file's SHA-256 against the project's
  pinned manifest, and writes the files to the bundle directory.

  ## Why this exists

  The full 624 MB adapted-artifact bundle is gitignored — it is generated
  output, not source. A fresh `git clone` therefore arrives without the
  bundle. This module is the canonical entry point that turns a fresh
  clone into a runnable state, in one command:

      mix trinity.artifact.fetch

  Under the hood it reads `priv/sakana_trinity/artifact_pin.json` for the
  HuggingFace `repo_id`, `revision`, and per-file expected SHA-256, then
  uses `HfHub.Download.hf_hub_download/1` to fetch each file with
  `verify_checksum: true` and `expected_sha256: <pinned>`. Files already
  present at the destination with the correct sha256 are skipped, so
  repeat invocations are cheap.

  ## Testing seam

  Every public entry point accepts an optional `:downloader` callback so
  tests can stub network access without depending on Bypass or other
  HTTP fixtures. The default downloader is `&default_download/1` which
  delegates to `HfHub.Download.hf_hub_download/1`.
  """

  alias TrinityCoordinator.ArtifactFetch.Pin

  @default_pin_path "priv/sakana_trinity/artifact_pin.json"
  @default_dest "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"

  @typedoc "Downloader callback signature."
  @type downloader :: (keyword() -> {:ok, Path.t()} | {:error, term()})

  @doc """
  Default location for the project's pinned artifact descriptor.
  """
  @spec default_pin_path() :: Path.t()
  def default_pin_path, do: @default_pin_path

  @doc """
  Default destination for the materialised artifact bundle.
  """
  @spec default_dest() :: Path.t()
  def default_dest, do: @default_dest

  @doc """
  Loads the pinned descriptor from disk.

  See `TrinityCoordinator.ArtifactFetch.Pin.load_pin!/1`.
  """
  @spec load_pin!(Path.t()) :: Pin.t()
  def load_pin!(path \\ @default_pin_path), do: Pin.load_pin!(path)

  @doc """
  Fetches every file listed in `pin` into `:dest`.

  ## Options

    * `:dest` — destination directory. Defaults to `default_dest/0`.
    * `:downloader` — function used for the HuggingFace download.
      Defaults to `&default_download/1`. The function receives a
      keyword list with at least `:repo_id`, `:filename`, `:revision`,
      `:repo_type`, `:verify_checksum`, `:expected_sha256`, and
      `:offline_mode` and must return `{:ok, path}` or `{:error, reason}`.
    * `:offline_mode` — when `true`, the downloader is asked to consult
      the local HuggingFace cache only (no network). Defaults to `false`.
    * `:progress_callback` — optional `(bytes_downloaded, total)`
      function passed straight through to the downloader.

  Returns `:ok`; raises with a descriptive message on any failure.
  """
  @spec fetch!(Pin.t(), keyword()) :: :ok
  def fetch!(%Pin{} = pin, opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        dest: @default_dest,
        downloader: &default_download/1,
        offline_mode: false,
        progress_callback: nil
      )

    dest = Keyword.fetch!(opts, :dest)
    File.mkdir_p!(dest)

    Enum.each(pin.files, &fetch_one!(&1, pin, dest, opts))

    :ok
  end

  @doc """
  Default downloader. Delegates to `HfHub.Download.hf_hub_download/1`.

  This is the in-process default; tests inject their own `:downloader`
  callback through `fetch!/2`.
  """
  @spec default_download(keyword()) :: {:ok, Path.t()} | {:error, term()}
  def default_download(args) do
    HfHub.Download.hf_hub_download(args)
  end

  defp fetch_one!(%{path: rel, sha256: expected_sha}, %Pin{} = pin, dest, opts) do
    target = Path.join(dest, rel)

    if file_has_expected_sha?(target, expected_sha) do
      :ok
    else
      File.mkdir_p!(Path.dirname(target))

      download_args =
        [
          repo_id: pin.repo_id,
          filename: rel,
          repo_type: :dataset,
          revision: pin.revision,
          verify_checksum: true,
          expected_sha256: expected_sha,
          offline_mode: Keyword.fetch!(opts, :offline_mode)
        ]
        |> maybe_add(:progress_callback, Keyword.get(opts, :progress_callback))

      case Keyword.fetch!(opts, :downloader).(download_args) do
        {:ok, cached_path} ->
          File.cp!(cached_path, target)
          :ok

        {:error, reason} ->
          raise "trinity.artifact.fetch failed for #{rel}: #{inspect(reason)}"
      end
    end
  end

  defp maybe_add(args, _key, nil), do: args
  defp maybe_add(args, key, value), do: Keyword.put(args, key, value)

  defp file_has_expected_sha?(path, expected_sha) do
    case File.stat(path) do
      {:ok, _} ->
        actual = compute_sha256(path)
        actual == expected_sha

      {:error, _} ->
        false
    end
  end

  defp compute_sha256(path) do
    File.stream!(path, 65_536)
    |> Enum.reduce(:crypto.hash_init(:sha256), fn chunk, ctx ->
      :crypto.hash_update(ctx, chunk)
    end)
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end
end
