defmodule Mix.Tasks.Trinity.Env.Check do
  @shortdoc "Validates the build/runtime environment before TRINITY tasks load EXLA"
  @moduledoc """
  Fails fast with an actionable message when the local environment cannot
  support the TRINITY coordinator before EXLA loads.

  Two classes of failure are checked:

  1. `XLA_TARGET` is set to a value not accepted by the current `xla` dependency.
     The recognised targets for the bundled `xla 0.9.x` are `cpu`, `cuda`,
     `cuda12`, `rocm`, and `tpu`. A contributor whose shell exports e.g.
     `XLA_TARGET=cuda13` cannot even compile `:exla`; this check tells them why
     before any other Mix task is attempted.
  2. The artifact directory is missing when the operator is running a
     Qwen-dependent task and the directory is passed explicitly via
     `--artifact-dir`. Missing artifact directories are otherwise reported by
     `TrinityCoordinator.Sakana.Coordinator.load/1`, but only after the heavy
     dependency chain has loaded.

  ## Usage

      mix trinity.env.check
      mix trinity.env.check --artifact-dir priv/sakana_trinity/adapted_qwen3_0_6b_layer26
      mix trinity.env.check --require cpu
      mix trinity.env.check --require cuda12

  ## Options

    * `--artifact-dir DIR` - check that the named artifact directory exists and
      contains `manifest.json`. Optional; when omitted only environment checks
      run.
    * `--require TARGET` - require `XLA_TARGET` to be set to exactly the given
      value. When omitted, the task accepts any of the recognised targets or
      the unset state.

  The task accepts explicit options only; it does not read its inputs from
  process environment variables outside the explicit `XLA_TARGET` check.

  ## Exit behaviour

  The task uses `Mix.raise/1` on failure, producing a single readable line
  and a non-zero exit status without a `MatchError` stacktrace.
  """

  use Mix.Task

  @recognised_xla_targets ~w(cpu cuda cuda12 rocm tpu)

  @impl Mix.Task
  def run(argv) do
    {opts, _argv, _invalid} =
      OptionParser.parse(argv,
        strict: [artifact_dir: :string, require: :string],
        aliases: [a: :artifact_dir, r: :require]
      )

    xla_target = xla_target_from_env(argv)
    required = Keyword.get(opts, :require)
    artifact_dir = Keyword.get(opts, :artifact_dir)

    check_xla_target!(xla_target, required)
    check_artifact_dir!(artifact_dir)

    if not silent?() do
      Mix.shell().info("trinity.env.check: ok")
      Mix.shell().info("  xla_target=#{xla_target || "(unset)"}")

      if artifact_dir do
        Mix.shell().info("  artifact_dir=#{artifact_dir}")
      end
    end

    :ok
  end

  defp silent?, do: Mix.shell() == Mix.Shell.Quiet

  # `argv` is reserved for tests/dry runs that want to inject an explicit
  # `XLA_TARGET` value. In normal operation the value is read once from
  # `System.get_env/1` at task entry, since reading the *build-time* env that
  # has already been frozen into the dep tree is the only sensible read.
  defp xla_target_from_env(_argv) do
    case System.get_env("XLA_TARGET") do
      nil -> nil
      "" -> nil
      v -> v
    end
  end

  defp check_xla_target!(nil, nil), do: :ok

  defp check_xla_target!(nil, required) when is_binary(required) do
    Mix.raise(
      "trinity.env.check: XLA_TARGET is not set but --require #{inspect(required)} was passed. " <>
        "Export XLA_TARGET=#{required} before running."
    )
  end

  defp check_xla_target!(value, nil) when is_binary(value) do
    if value in @recognised_xla_targets do
      :ok
    else
      Mix.raise(
        "trinity.env.check: XLA_TARGET=#{inspect(value)} is not one of #{inspect(@recognised_xla_targets)}. " <>
          "The bundled xla dependency rejects unrecognised targets at compile time. " <>
          "Recommended: unset XLA_TARGET (CPU) or export XLA_TARGET=cuda12 for the canonical CUDA path."
      )
    end
  end

  defp check_xla_target!(value, required) when is_binary(value) and is_binary(required) do
    cond do
      value == required ->
        :ok

      value not in @recognised_xla_targets ->
        check_xla_target!(value, nil)

      true ->
        Mix.raise(
          "trinity.env.check: XLA_TARGET=#{inspect(value)} but --require #{inspect(required)} was passed. " <>
            "Re-export XLA_TARGET=#{required} before continuing."
        )
    end
  end

  defp check_artifact_dir!(nil), do: :ok

  defp check_artifact_dir!(dir) when is_binary(dir) do
    manifest = Path.join(dir, "manifest.json")

    cond do
      not File.dir?(dir) ->
        Mix.raise(
          "trinity.env.check: artifact directory #{inspect(dir)} does not exist or is not a directory. " <>
            "Either copy a blessed artifact bundle into it or follow guides/artifacts_and_export.md."
        )

      not File.regular?(manifest) ->
        Mix.raise(
          "trinity.env.check: artifact directory #{inspect(dir)} exists but #{inspect(manifest)} is missing. " <>
            "The directory may be partially populated."
        )

      true ->
        :ok
    end
  end
end
