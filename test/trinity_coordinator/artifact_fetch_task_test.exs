defmodule Mix.Tasks.Trinity.Artifact.FetchTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.Trinity.Artifact.Fetch
  alias TrinityCoordinator.ArtifactFetch.Pin

  @tmp_prefix "trinity_artifact_fetch_task_test"

  setup do
    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Quiet)
    on_exit(fn -> Mix.shell(previous_shell) end)

    tmp = unique_tmp_dir()
    File.mkdir_p!(tmp)
    on_exit(fn -> File.rm_rf(tmp) end)
    {:ok, tmp: tmp}
  end

  test "prints --help and returns :ok without invoking the downloader", %{tmp: tmp} do
    pin_path = write_synthetic_pin!(tmp, ["manifest.json"])

    parent = self()

    downloader = fn _ ->
      send(parent, :downloader_invoked)
      {:ok, "/should/not/be/used"}
    end

    Process.put(:trinity_artifact_fetch_downloader, downloader)
    on_exit(fn -> Process.delete(:trinity_artifact_fetch_downloader) end)

    assert :ok = Fetch.run(["--help", "--pin", pin_path, "--dest", Path.join(tmp, "dest")])

    refute_received :downloader_invoked
  end

  test "fetches into the given --dest using the injected downloader", %{tmp: tmp} do
    pin_path = write_synthetic_pin!(tmp, ["manifest.json"])
    cache = Path.join(tmp, "cache")
    File.mkdir_p!(cache)
    File.write!(Path.join(cache, "manifest.json"), "manifest-bytes")

    downloader = fn args ->
      {:ok, Path.join(cache, args[:filename])}
    end

    Process.put(:trinity_artifact_fetch_downloader, downloader)
    on_exit(fn -> Process.delete(:trinity_artifact_fetch_downloader) end)

    dest = Path.join(tmp, "dest")

    assert :ok = Fetch.run(["--pin", pin_path, "--dest", dest])
    assert File.read!(Path.join(dest, "manifest.json")) == "manifest-bytes"
  end

  test "--offline forwards offline_mode through to the downloader", %{tmp: tmp} do
    pin_path = write_synthetic_pin!(tmp, ["manifest.json"])
    cache = Path.join(tmp, "cache")
    File.mkdir_p!(cache)
    File.write!(Path.join(cache, "manifest.json"), "x")

    parent = self()

    downloader = fn args ->
      send(parent, {:downloader_args, args})
      {:ok, Path.join(cache, args[:filename])}
    end

    Process.put(:trinity_artifact_fetch_downloader, downloader)
    on_exit(fn -> Process.delete(:trinity_artifact_fetch_downloader) end)

    assert :ok =
             Fetch.run([
               "--pin",
               pin_path,
               "--dest",
               Path.join(tmp, "dest"),
               "--offline"
             ])

    assert_receive {:downloader_args, args}
    assert args[:offline_mode] == true
  end

  test "fails with Mix.Error when the pin file does not exist", %{tmp: tmp} do
    missing = Path.join(tmp, "no_such_pin.json")

    assert_raise Mix.Error, fn ->
      Fetch.run(["--pin", missing])
    end
  end

  defp write_synthetic_pin!(tmp, file_paths, opts \\ []) do
    pin_path = Path.join(tmp, "pin.json")

    pin =
      %Pin{
        version: 1,
        repo_id: Keyword.get(opts, :repo_id, "owner/repo"),
        revision: Keyword.get(opts, :revision, "v1"),
        manifest_sha256: "deadbeef",
        files: Enum.map(file_paths, fn p -> %{path: p, sha256: "aa"} end)
      }
      |> Map.from_struct()
      |> Jason.encode!()

    File.write!(pin_path, pin)
    pin_path
  end

  defp unique_tmp_dir do
    Path.join([System.tmp_dir!(), "#{@tmp_prefix}-#{System.unique_integer([:positive])}"])
  end
end
