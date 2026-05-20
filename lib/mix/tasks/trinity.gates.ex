defmodule Mix.Tasks.Trinity.Gates do
  @shortdoc "Runs the AGENTS.md quality gate matrix as a single command"
  @moduledoc """
  Runs the canonical AGENTS.md quality gate matrix in order and reports a
  machine-readable summary.

  The fixed gate sequence is:

    1. `mix format --check-formatted`
    2. `mix compile --warnings-as-errors`
    3. `mix test`
    4. `mix credo --strict`
    5. `mix dialyzer`
    6. `mix docs --warnings-as-errors`

  Optional add-ons:

    * `--include-parity-check` — runs `mix trinity.parity.check` with the
      paths from `--python-report` and `--elixir-report` after the baseline
      gates pass.
    * `--include-hex-build` — runs `mix hex.build --unpack` in **advisory**
      mode. Verified 2026-05-20: `mix hex.build --unpack` exits 1 because
      `mix.exs` pins `:bumblebee` to a GitHub ref + override and Hex refuses
      packages whose declared deps include a non-Hex source. The wrapper
      captures the exit code and emits a diagnostic line
      `hex_build_advisory: pass|fail` but does **not** fail the wrapper on
      a non-zero exit. Blocking semantics return once Bumblebee is unpinned.
      Source: `docs/20260519/sakana/appendix/G_post_checklist_review_and_amendments.md` §G.2.6.

  Stop-on-first-failure: the wrapper exits non-zero as soon as one blocking
  gate fails, except advisory gates (`--include-hex-build`) which are reported
  but never block.

  ## Usage

      mix trinity.gates
      mix trinity.gates --summary-out tmp/trinity_gates_summary.json
      mix trinity.gates --skip-dialyzer --skip-docs
      mix trinity.gates --include-parity-check \\
        --python-report tmp/sakana_parity/python.json \\
        --elixir-report tmp/sakana_parity/elixir.json
      mix trinity.gates --include-hex-build

  Options:

    * `--skip-dialyzer` — skip step 5 (Dialyzer; slow locally).
    * `--skip-docs` — skip step 6 (`mix docs --warnings-as-errors`).
    * `--fast` — equivalent to `--skip-dialyzer --skip-docs`. Marked
      non-release in the summary.
    * `--include-parity-check` — see above. Requires `--python-report` and
      `--elixir-report`.
    * `--python-report PATH` — only used with `--include-parity-check`.
    * `--elixir-report PATH` — only used with `--include-parity-check`.
    * `--include-hex-build` — see above. Advisory.
    * `--summary-out PATH` — write structured JSON summary at PATH
      (schema_version 1).

  ## Summary JSON

  Each step records: name, args, exit_status, duration_ms, blocking,
  advisory, and a tail-bounded `output` for diagnostics. The top-level
  `ok` flag reflects whether all blocking gates passed.
  """

  use Mix.Task

  @summary_schema_version 1
  @tail_bytes 4096

  @baseline_gates [
    {:format, "mix", ["format", "--check-formatted"], true, false},
    {:compile, "mix", ["compile", "--warnings-as-errors"], true, false},
    {:test, "mix", ["test"], true, false},
    {:credo, "mix", ["credo", "--strict"], true, false},
    {:dialyzer, "mix", ["dialyzer"], true, false},
    {:docs, "mix", ["docs", "--warnings-as-errors"], true, false}
  ]

  @impl Mix.Task
  def run(argv) do
    {opts, _argv, invalid} =
      OptionParser.parse(argv,
        strict: [
          skip_dialyzer: :boolean,
          skip_docs: :boolean,
          fast: :boolean,
          include_parity_check: :boolean,
          python_report: :string,
          elixir_report: :string,
          include_hex_build: :boolean,
          summary_out: :string
        ]
      )

    if invalid != [] do
      Mix.raise("invalid options: " <> inspect(invalid))
    end

    fast? = Keyword.get(opts, :fast, false)
    skip_dialyzer? = fast? or Keyword.get(opts, :skip_dialyzer, false)
    skip_docs? = fast? or Keyword.get(opts, :skip_docs, false)

    summary_out = Keyword.get(opts, :summary_out)
    started_ms = System.monotonic_time(:millisecond)

    gates = filter_baseline_gates(skip_dialyzer?, skip_docs?)
    optional_gates = optional_gates(opts)

    {step_results, blocking_failed?} = run_steps(gates ++ optional_gates)

    total_duration_ms = System.monotonic_time(:millisecond) - started_ms

    summary = %{
      schema_version: @summary_schema_version,
      ok: not blocking_failed?,
      fast?: fast?,
      release_grade?: not fast? and not skip_dialyzer? and not skip_docs?,
      total_duration_ms: total_duration_ms,
      steps: step_results
    }

    if summary_out, do: write_summary!(summary_out, summary)

    if blocking_failed? do
      failed = Enum.filter(step_results, fn s -> s.blocking and s.exit_status != 0 end)
      ids = Enum.map_join(failed, ", ", & &1.name)
      Mix.raise("trinity.gates: blocking gate(s) failed: " <> ids)
    end

    :ok
  end

  defp filter_baseline_gates(skip_dialyzer?, skip_docs?) do
    @baseline_gates
    |> Enum.reject(fn {name, _, _, _, _} ->
      (skip_dialyzer? and name == :dialyzer) or (skip_docs? and name == :docs)
    end)
  end

  defp optional_gates(opts) do
    parity =
      if Keyword.get(opts, :include_parity_check, false) do
        python_report =
          Keyword.get(opts, :python_report) ||
            Mix.raise("--include-parity-check requires --python-report PATH")

        elixir_report =
          Keyword.get(opts, :elixir_report) ||
            Mix.raise("--include-parity-check requires --elixir-report PATH")

        [
          {:parity_check, "mix",
           [
             "trinity.parity.check",
             "--python-report",
             python_report,
             "--elixir-report",
             elixir_report
           ], true, false}
        ]
      else
        []
      end

    hex_build =
      if Keyword.get(opts, :include_hex_build, false) do
        [{:hex_build, "mix", ["hex.build", "--unpack"], false, true}]
      else
        []
      end

    parity ++ hex_build
  end

  defp run_steps(gates) do
    Enum.reduce(gates, {[], false}, fn {name, cmd, args, blocking?, advisory?}, {acc, failed?} ->
      stop_early? = blocking? and failed?

      if stop_early? do
        {acc ++ [skipped_step(name, cmd, args, blocking?, advisory?)], failed?}
      else
        result = run_single_step(name, cmd, args, blocking?, advisory?)
        next_failed = failed? or (blocking? and result.exit_status != 0)
        {acc ++ [result], next_failed}
      end
    end)
  end

  defp run_single_step(name, cmd, args, blocking?, advisory?) do
    Mix.shell().info("\n[trinity.gates] step=#{name} cmd=#{cmd} args=#{inspect(args)}")
    started = System.monotonic_time(:millisecond)
    {out, status} = command_runner().(cmd, args)
    duration = System.monotonic_time(:millisecond) - started

    if advisory? do
      label = if status == 0, do: "pass", else: "fail"
      Mix.shell().info("hex_build_advisory: #{label} (exit=#{status})")
    end

    Mix.shell().info("[trinity.gates] step=#{name} exit=#{status} duration_ms=#{duration}")

    %{
      name: Atom.to_string(name),
      cmd: cmd,
      args: args,
      exit_status: status,
      duration_ms: duration,
      blocking: blocking?,
      advisory: advisory?,
      output: tail_bytes(out, @tail_bytes),
      skipped: false
    }
  end

  defp skipped_step(name, cmd, args, blocking?, advisory?) do
    %{
      name: Atom.to_string(name),
      cmd: cmd,
      args: args,
      exit_status: nil,
      duration_ms: 0,
      blocking: blocking?,
      advisory: advisory?,
      output: nil,
      skipped: true
    }
  end

  defp write_summary!(path, summary) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(summary, pretty: true))
    Mix.shell().info("[trinity.gates] summary written: #{path}")
  end

  defp tail_bytes(bin, n) when is_binary(bin) do
    size = byte_size(bin)

    if size <= n do
      bin
    else
      binary_part(bin, size - n, n)
    end
  end

  defp tail_bytes(_, _), do: nil

  # The runner can be overridden in tests by setting
  # `:persistent_term.put({__MODULE__, :command_runner}, fun)`. In production,
  # we shell out via System.cmd/3.
  defp command_runner do
    case :persistent_term.get({__MODULE__, :command_runner}, nil) do
      nil -> &default_runner/2
      fun when is_function(fun, 2) -> fun
    end
  end

  defp default_runner(cmd, args) do
    System.cmd(cmd, args, stderr_to_stdout: true)
  end

  @doc false
  def __set_command_runner__(fun) when is_function(fun, 2) do
    :persistent_term.put({__MODULE__, :command_runner}, fun)
  end

  @doc false
  def __clear_command_runner__ do
    :persistent_term.erase({__MODULE__, :command_runner})
  end
end
