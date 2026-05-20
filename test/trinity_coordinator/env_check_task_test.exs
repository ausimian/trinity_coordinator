defmodule Mix.Tasks.Trinity.Env.CheckTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.Trinity.Env.Check

  setup do
    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Quiet)
    on_exit(fn -> Mix.shell(previous_shell) end)

    original = System.get_env("XLA_TARGET")

    on_exit(fn ->
      case original do
        nil -> System.delete_env("XLA_TARGET")
        v -> System.put_env("XLA_TARGET", v)
      end
    end)

    :ok
  end

  test "accepts a recognised XLA_TARGET" do
    System.put_env("XLA_TARGET", "cuda12")
    assert :ok = Check.run([])
  end

  test "rejects an unrecognised XLA_TARGET" do
    System.put_env("XLA_TARGET", "cuda13")
    assert_message(fn -> Check.run([]) end, "is not one of")
  end

  test "accepts an unset XLA_TARGET" do
    System.delete_env("XLA_TARGET")
    assert :ok = Check.run([])
  end

  test "fails when --require is passed but XLA_TARGET is unset" do
    System.delete_env("XLA_TARGET")
    assert_message(fn -> Check.run(["--require", "cuda12"]) end, "is not set but --require")
  end

  test "fails when --require mismatches the active XLA_TARGET" do
    System.put_env("XLA_TARGET", "cpu")
    assert_message(fn -> Check.run(["--require", "cuda12"]) end, "but --require")
  end

  test "passes when --require matches" do
    System.put_env("XLA_TARGET", "cuda12")
    assert :ok = Check.run(["--require", "cuda12"])
  end

  test "fails when --artifact-dir does not exist" do
    System.put_env("XLA_TARGET", "cuda12")

    missing =
      Path.join(
        System.tmp_dir!(),
        "trinity_env_check_missing_#{System.unique_integer([:positive])}"
      )

    refute File.exists?(missing)

    assert_message(fn -> Check.run(["--artifact-dir", missing]) end, "does not exist")
  end

  test "fails when --artifact-dir exists but manifest.json is missing" do
    System.put_env("XLA_TARGET", "cuda12")

    dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_env_check_partial_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    assert_message(fn -> Check.run(["--artifact-dir", dir]) end, "is missing")
  end

  test "passes when --artifact-dir contains manifest.json" do
    System.put_env("XLA_TARGET", "cuda12")

    dir =
      Path.join(
        System.tmp_dir!(),
        "trinity_env_check_ok_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "manifest.json"), "{}")
    on_exit(fn -> File.rm_rf!(dir) end)

    assert :ok = Check.run(["--artifact-dir", dir])
  end

  defp assert_message(fun, needle) do
    fun.()
    flunk("expected Mix.Error with message containing #{inspect(needle)}")
  rescue
    e in Mix.Error ->
      message = Exception.message(e)

      assert String.contains?(message, needle),
             "expected Mix.Error message to contain #{inspect(needle)}, got: #{message}"
  end
end
