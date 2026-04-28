defmodule Mix.Tasks.Trinity.Sakana.ParitySample do
  @moduledoc """
  Emits an incremental JSON report for the Sakana Python-reference SVD sample.

      XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
        --out tmp/sakana_parity/elixir_sample_trace.json

  To compare against Python semantic components, first run the companion Python
  script and pass the directory/report it writes:

      python3 priv/sakana_trinity/scripts/debug_sakana_parity_sample.py \
        --out tmp/sakana_parity/python_sample_trace.json \
        --write-components-dir tmp/sakana_parity/python_components

      XLA_TARGET=cuda12 mix trinity.sakana.parity_sample \
        --components-dir tmp/sakana_parity/python_components \
        --python-report tmp/sakana_parity/python_sample_trace.json \
        --out tmp/sakana_parity/elixir_sample_trace.json
  """

  use Mix.Task

  alias TrinityCoordinator.Sakana.ParityTrace

  @shortdoc "Emits Sakana SVD sample parity diagnostics"
  @default_out "tmp/sakana_parity/elixir_sample_trace.json"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          out: :string,
          components_dir: :string,
          python_report: :string,
          router_vector: :string,
          reference: :string,
          no_cuda: :boolean
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    report =
      ParityTrace.sample_report!(
        router_vector_path:
          Keyword.get(
            opts,
            :router_vector,
            "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"
          ),
        reference_manifest_path:
          Keyword.get(
            opts,
            :reference,
            "priv/sakana_trinity/reference/sakana_python_reference_manifest.json"
          ),
        components_dir: Keyword.get(opts, :components_dir),
        python_report_path: Keyword.get(opts, :python_report),
        require_cuda: not Keyword.get(opts, :no_cuda, false)
      )

    out = Keyword.get(opts, :out, @default_out)
    :ok = ParityTrace.write_json!(out, report)

    Mix.shell().info("Wrote Elixir parity report: #{out}")
    print_hash_summary(report)
  end

  defp print_hash_summary(report) do
    expected = get_in(report, ["reference", "expected_bf16_sha256"])
    python_current = get_in(report, ["python_current_baseline", "observed_bf16_sha256"])

    python_reproducible =
      get_in(report, ["python_current_baseline", "expected_hash_reproducible"])

    Mix.shell().info("Stored Python bf16 hash: #{expected}")

    if python_current do
      Mix.shell().info(
        "Current Python baseline hash: #{python_current} reproducible_stored=#{inspect(python_reproducible)}"
      )
    else
      Mix.shell().info("Current Python baseline hash: (no --python-report supplied)")
    end

    report
    |> Map.get("native_elixir_svd_variants", [])
    |> Enum.each(fn variant ->
      Mix.shell().info(
        "native #{variant["label"]}: #{variant["observed_bf16_sha256"]} match_stored=#{variant["matches_expected"]} match_python=#{variant["matches_python_current"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
      )
    end)

    case Map.get(report, "semantic_python_component_variants") do
      variants when is_list(variants) ->
        Enum.each(variants, fn variant ->
          Mix.shell().info(
            "semantic #{variant["label"]}: #{variant["observed_bf16_sha256"]} match_stored=#{variant["matches_expected"]} match_python=#{variant["matches_python_current"]} zero_error=#{variant["zero_offset_max_abs_error_vs_source"]}"
          )
        end)

      nil ->
        Mix.shell().info("No Python semantic component directory was supplied.")

      other ->
        Mix.shell().info("Semantic component status: #{inspect(other)}")
    end
  end
end
