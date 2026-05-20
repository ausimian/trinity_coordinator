defmodule Mix.Tasks.Trinity.Parity.CheckTest do
  use ExUnit.Case, async: false

  alias Mix.Tasks.Trinity.Parity.Check

  setup do
    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Quiet)
    on_exit(fn -> Mix.shell(previous_shell) end)

    tmp = Path.join(System.tmp_dir!(), "parity_check_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    on_exit(fn -> File.rm_rf!(tmp) end)

    py_report = Path.join(tmp, "python_report.json")
    el_report = Path.join(tmp, "elixir_report.json")
    File.write!(py_report, ~s|{"status":"ok"}|)
    File.write!(el_report, ~s|{"status":"ok"}|)

    {:ok, tmp: tmp, py_report: py_report, el_report: el_report}
  end

  test "raises when --python-report is missing", _ctx do
    assert_message(fn -> Check.run([]) end, "--python-report is required")
  end

  test "raises when --elixir-report does not exist", %{py_report: py} do
    assert_message(
      fn ->
        Check.run([
          "--python-report",
          py,
          "--elixir-report",
          "/tmp/definitely_missing_#{System.unique_integer([:positive])}.json"
        ])
      end,
      "is not a regular file"
    )
  end

  test "succeeds and writes summary when fake comparator exits 0", %{
    tmp: tmp,
    py_report: py,
    el_report: el
  } do
    fake = make_fake_comparator!(tmp, 0, "ok-from-fake")
    summary = Path.join(tmp, "summary.json")

    # Stub the script path by swapping the real script aside? No — easier: invoke
    # the task with --python pointing at a wrapper that prints + exits, and
    # forge a temporary comparator script at the expected relative path.
    # Instead, use a shim: redirect via setting --python to our fake interpreter
    # that ignores its first arg.
    Check.run([
      "--python",
      fake,
      "--python-report",
      py,
      "--elixir-report",
      el,
      "--summary-out",
      summary
    ])

    assert File.regular?(summary)
    parsed = Jason.decode!(File.read!(summary))
    assert parsed["schema_version"] == 1
    assert parsed["exit_status"] == 0
    assert parsed["ok"] == true
    assert parsed["python_report"] == py
    assert parsed["elixir_report"] == el
    assert is_integer(parsed["duration_ms"])
    assert String.contains?(parsed["stdout_tail"] || "", "ok-from-fake")
  end

  test "raises Mix.Error on non-zero comparator status", %{
    tmp: tmp,
    py_report: py,
    el_report: el
  } do
    fake = make_fake_comparator!(tmp, 5, "fake failure body")

    assert_message(
      fn ->
        Check.run([
          "--python",
          fake,
          "--python-report",
          py,
          "--elixir-report",
          el
        ])
      end,
      "parity comparator failed"
    )
  end

  test "forwards --strict-current-python and --top-diffs to the comparator", %{
    tmp: tmp,
    py_report: py,
    el_report: el
  } do
    # Fake interpreter records its argv into a file so we can assert what was forwarded.
    record = Path.join(tmp, "argv.txt")

    fake = """
    #!/usr/bin/env bash
    printf '%s\\n' "$@" > #{record}
    echo "ok"
    exit 0
    """

    fake_path = Path.join(tmp, "fake_record.sh")
    File.write!(fake_path, fake)
    File.chmod!(fake_path, 0o755)

    Check.run([
      "--python",
      fake_path,
      "--python-report",
      py,
      "--elixir-report",
      el,
      "--strict-current-python",
      "--top-diffs",
      "7"
    ])

    forwarded = File.read!(record) |> String.split("\n", trim: true)
    assert "priv/sakana_trinity/scripts/compare_sakana_parity_reports.py" in forwarded
    assert "--strict-stage-tolerances" in forwarded
    assert "--strict-current-python" in forwarded
    assert "--top-diffs" in forwarded
    assert "7" in forwarded
  end

  # --- helpers ---

  defp make_fake_comparator!(tmp, exit_code, stdout_msg) do
    path = Path.join(tmp, "fake_#{exit_code}.sh")

    File.write!(path, """
    #!/usr/bin/env bash
    echo "#{stdout_msg}"
    exit #{exit_code}
    """)

    File.chmod!(path, 0o755)
    path
  end

  defp assert_message(fun, needle) do
    fun.()
    flunk("expected Mix.Error with message containing #{inspect(needle)}")
  rescue
    e in Mix.Error ->
      msg = Exception.message(e)
      assert String.contains?(msg, needle), "expected #{inspect(needle)}, got: #{msg}"
  end
end
