defmodule Mix.Tasks.Trinity.GatesTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.Trinity.Gates

  setup do
    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Quiet)
    on_exit(fn -> Mix.shell(previous_shell) end)

    on_exit(fn -> Gates.__clear_command_runner__() end)

    tmp = Path.join(System.tmp_dir!(), "trinity_gates_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    on_exit(fn -> File.rm_rf!(tmp) end)

    {:ok, tmp: tmp}
  end

  test "runs the baseline gates in order and writes a summary when all pass", %{tmp: tmp} do
    {:ok, log} = Agent.start_link(fn -> [] end)

    Gates.__set_command_runner__(fn cmd, args ->
      Agent.update(log, &[{cmd, args} | &1])
      {"all good\n", 0}
    end)

    summary = Path.join(tmp, "summary.json")
    Gates.run(["--summary-out", summary])

    invocations = Agent.get(log, &Enum.reverse/1)
    assert {"mix", ["format", "--check-formatted"]} = Enum.at(invocations, 0)
    assert {"mix", ["compile", "--warnings-as-errors"]} = Enum.at(invocations, 1)
    assert {"mix", ["test"]} = Enum.at(invocations, 2)
    assert {"mix", ["credo", "--strict"]} = Enum.at(invocations, 3)
    assert {"mix", ["dialyzer"]} = Enum.at(invocations, 4)
    assert {"mix", ["docs", "--warnings-as-errors"]} = Enum.at(invocations, 5)

    parsed = Jason.decode!(File.read!(summary))
    assert parsed["schema_version"] == 1
    assert parsed["ok"] == true
    assert parsed["release_grade?"] == true
    assert length(parsed["steps"]) == 6
    assert Enum.all?(parsed["steps"], fn s -> s["exit_status"] == 0 end)
  end

  test "--fast skips dialyzer and docs", %{tmp: tmp} do
    {:ok, log} = Agent.start_link(fn -> [] end)

    Gates.__set_command_runner__(fn cmd, args ->
      Agent.update(log, &[{cmd, args} | &1])
      {"ok\n", 0}
    end)

    Gates.run(["--fast", "--summary-out", Path.join(tmp, "fast.json")])

    invs = Agent.get(log, &Enum.reverse/1) |> Enum.map(fn {_, args} -> List.first(args) end)
    refute "dialyzer" in invs
    refute "docs" in invs

    parsed = Jason.decode!(File.read!(Path.join(tmp, "fast.json")))
    assert parsed["fast?"] == true
    assert parsed["release_grade?"] == false
  end

  test "stops on first failed blocking gate and raises Mix.Error", _ctx do
    Gates.__set_command_runner__(fn _cmd, args ->
      # Fail at step 3 (mix test).
      case args do
        ["test"] -> {"FAILED\n", 1}
        _ -> {"ok\n", 0}
      end
    end)

    fun = fn -> Gates.run([]) end

    raised =
      try do
        fun.()
        flunk("expected Mix.Error")
      rescue
        e in Mix.Error -> e
      end

    assert String.contains?(Exception.message(raised), "trinity.gates:")
    assert String.contains?(Exception.message(raised), "test")
  end

  test "--include-hex-build runs hex.build as advisory and does NOT fail the wrapper", %{
    tmp: tmp
  } do
    Gates.__set_command_runner__(fn _cmd, args ->
      case args do
        ["hex.build", "--unpack"] -> {"Dependencies excluded: bumblebee\n", 1}
        _ -> {"ok\n", 0}
      end
    end)

    summary_path = Path.join(tmp, "advisory.json")

    # Must NOT raise even though hex.build returned 1.
    Gates.run(["--include-hex-build", "--summary-out", summary_path])

    parsed = Jason.decode!(File.read!(summary_path))
    assert parsed["ok"] == true

    hex_step = Enum.find(parsed["steps"], &(&1["name"] == "hex_build"))
    assert hex_step["advisory"] == true
    assert hex_step["blocking"] == false
    assert hex_step["exit_status"] == 1
  end

  test "--include-parity-check requires --python-report and --elixir-report", _ctx do
    Gates.__set_command_runner__(fn _, _ -> {"ok\n", 0} end)

    raised =
      try do
        Gates.run(["--include-parity-check"])
        flunk("expected Mix.Error")
      rescue
        e in Mix.Error -> e
      end

    assert String.contains?(Exception.message(raised), "--python-report")
  end

  test "--include-parity-check forwards both report paths to trinity.parity.check", %{tmp: _tmp} do
    {:ok, log} = Agent.start_link(fn -> [] end)

    Gates.__set_command_runner__(fn _cmd, args ->
      Agent.update(log, &[args | &1])
      {"ok\n", 0}
    end)

    Gates.run([
      "--include-parity-check",
      "--python-report",
      "py.json",
      "--elixir-report",
      "el.json"
    ])

    invs = Agent.get(log, &Enum.reverse/1)

    assert Enum.any?(invs, fn args ->
             args ==
               [
                 "trinity.parity.check",
                 "--python-report",
                 "py.json",
                 "--elixir-report",
                 "el.json"
               ]
           end)
  end
end
