defmodule Mix.Tasks.Trinity.Parity.Check do
  @shortdoc "Wraps the Python comparator for Sakana parity reports"
  @moduledoc """
  Wraps `priv/sakana_trinity/scripts/compare_sakana_parity_reports.py` as a
  first-class Mix task with structured-summary output.

  ## Usage

      mix trinity.parity.check \\
        --python-report path/to/python_report.json \\
        --elixir-report path/to/elixir_report.json

  ## Options

    * `--python-report PATH` (required)
    * `--elixir-report PATH` (required)
    * `--strict-stage-tolerances` (default true) — passes `--strict-stage-tolerances`
      to the comparator; the default required functional-correctness gate.
    * `--no-strict-stage-tolerances` — disable the default strict-tolerance gate.
    * `--strict-current-python` (default false) — passes `--strict-current-python`
      to the comparator; opt-in.
    * `--strict-reference` (default false) — passes `--strict-reference` to the
      comparator; only meaningful when the original `svd_weights.pt` is available.
    * `--top-diffs N` — forwarded as `--top-diffs N`.
    * `--summary-out PATH` — write a structured JSON summary at PATH.
    * `--python PATH` — Python interpreter to invoke (default `python3`).

  Both `--python-report` and `--elixir-report` must exist before the comparator
  is invoked; the task fails fast with a readable `Mix.raise/1` line otherwise,
  rather than letting the Python script emit a stack trace.

  The wrapper streams the comparator's stdout/stderr to the operator on
  success and on failure, then `Mix.raise/1`s on non-zero exit status so
  downstream gates can detect failure without parsing log text.

  When `--summary-out` is supplied, the JSON written there has shape:

      {
        "schema_version": 1,
        "exit_status": 0,
        "wrapper_options": { ... },
        "comparator_args": [...],
        "python_report": "...",
        "elixir_report": "...",
        "duration_ms": 123,
        "ok": true,
        "stdout_tail": "...",
        "stderr_tail": "..."
      }
  """

  use Mix.Task

  @comparator_script "priv/sakana_trinity/scripts/compare_sakana_parity_reports.py"
  @summary_schema_version 1
  @tail_bytes 4096

  @impl Mix.Task
  def run(argv) do
    {opts, _argv, invalid} =
      OptionParser.parse(argv,
        strict: [
          python_report: :string,
          elixir_report: :string,
          strict_stage_tolerances: :boolean,
          strict_current_python: :boolean,
          strict_reference: :boolean,
          top_diffs: :integer,
          summary_out: :string,
          python: :string
        ]
      )

    if invalid != [] do
      Mix.raise("invalid options: " <> inspect(invalid))
    end

    python_report = require_path!(opts, :python_report)
    elixir_report = require_path!(opts, :elixir_report)
    python_bin = Keyword.get(opts, :python, "python3")
    summary_out = Keyword.get(opts, :summary_out)

    comparator_args = build_comparator_args(opts, python_report, elixir_report)

    ensure_script_present!()

    started_ms = System.monotonic_time(:millisecond)

    {output, exit_status} =
      System.cmd(python_bin, [@comparator_script | comparator_args], stderr_to_stdout: true)

    duration_ms = System.monotonic_time(:millisecond) - started_ms

    IO.write(output)

    if summary_out do
      write_summary!(summary_out, %{
        exit_status: exit_status,
        ok: exit_status == 0,
        python_report: python_report,
        elixir_report: elixir_report,
        comparator_args: comparator_args,
        duration_ms: duration_ms,
        stdout_tail: tail_bytes(output, @tail_bytes),
        stderr_tail: nil,
        wrapper_options: wrapper_options_snapshot(opts)
      })
    end

    if exit_status == 0 do
      :ok
    else
      Mix.raise(
        "parity comparator failed (exit #{exit_status}); see output above. " <>
          "args=#{inspect(comparator_args)}"
      )
    end
  end

  defp require_path!(opts, key) do
    case Keyword.fetch(opts, key) do
      {:ok, path} when is_binary(path) and path != "" ->
        unless File.regular?(path) do
          Mix.raise("--#{String.replace(to_string(key), "_", "-")} #{path} is not a regular file")
        end

        path

      _ ->
        Mix.raise("--#{String.replace(to_string(key), "_", "-")} is required")
    end
  end

  defp ensure_script_present! do
    unless File.regular?(@comparator_script) do
      Mix.raise("expected comparator at #{@comparator_script}; is the repo checkout complete?")
    end
  end

  defp build_comparator_args(opts, python_report, elixir_report) do
    strict_tolerances? = Keyword.get(opts, :strict_stage_tolerances, true)

    base = [
      python_report,
      elixir_report
    ]

    extras =
      [
        {strict_tolerances?, "--strict-stage-tolerances"},
        {Keyword.get(opts, :strict_current_python, false), "--strict-current-python"},
        {Keyword.get(opts, :strict_reference, false), "--strict-reference"}
      ]
      |> Enum.filter(fn {flag, _} -> flag end)
      |> Enum.map(fn {_flag, opt} -> opt end)

    top_diffs =
      case Keyword.get(opts, :top_diffs) do
        nil -> []
        n when is_integer(n) and n >= 0 -> ["--top-diffs", Integer.to_string(n)]
      end

    base ++ extras ++ top_diffs
  end

  defp write_summary!(path, payload) do
    File.mkdir_p!(Path.dirname(path))

    body =
      payload
      |> Map.put(:schema_version, @summary_schema_version)
      |> Jason.encode!(pretty: true)

    File.write!(path, body)
  end

  defp tail_bytes(binary, n) when is_binary(binary) and is_integer(n) and n > 0 do
    size = byte_size(binary)

    if size <= n do
      binary
    else
      binary_part(binary, size - n, n)
    end
  end

  defp tail_bytes(_, _), do: nil

  defp wrapper_options_snapshot(opts) do
    Enum.into(opts, %{}, fn {k, v} -> {Atom.to_string(k), v} end)
  end
end
