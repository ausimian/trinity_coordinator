defmodule Mix.Tasks.Trinity.Sakana.ImportPython do
  @moduledoc """
  Convert a Python semantic Sakana/TRINITY export bundle into the canonical
  Elixir runtime artifact layout.

      XLA_TARGET=cuda12 mix trinity.sakana.import_python \
        --source-dir priv/sakana_trinity/artifacts/exported \
        --manifest trinity_sakana_export_manifest.json \
        --reference priv/sakana_trinity/reference/sakana_python_reference_manifest.json \
        --out priv/sakana_trinity/adapted_qwen3_0_6b_layer26_from_python \
        --force
  """

  use Mix.Task

  alias TrinityCoordinator.Sakana.PythonImporter

  @shortdoc "Imports Python semantic Sakana artifacts into canonical Elixir artifacts"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          source_dir: :string,
          manifest: :string,
          reference: :string,
          out: :string,
          force: :boolean,
          resume: :boolean,
          no_load_qwen: :boolean,
          json: :boolean
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    source_dir = Keyword.get(opts, :source_dir) || Mix.raise("--source-dir is required")
    out_dir = Keyword.get(opts, :out) || Mix.raise("--out is required")
    json? = Keyword.get(opts, :json, false)

    import_opts = [
      source_dir: source_dir,
      manifest: Keyword.get(opts, :manifest, "trinity_sakana_export_manifest.json"),
      reference_manifest: Keyword.get(opts, :reference),
      out_dir: out_dir,
      force: Keyword.get(opts, :force, false),
      resume: Keyword.get(opts, :resume, false),
      load_qwen: not Keyword.get(opts, :no_load_qwen, false),
      progress: progress_fun(json?)
    ]

    print_summary(import_opts)

    case PythonImporter.import_bundle(import_opts) do
      {:ok, manifest} ->
        if json? do
          IO.puts(Jason.encode!(normalize_for_json(%{status: :ok, manifest: manifest})))
        else
          IO.puts("Python semantic import complete")
          IO.puts("  Output directory: #{out_dir}")
          IO.puts("  Status: #{manifest["status"]}")
          IO.puts("  Selected tensor count: #{manifest["selected_tensor_count"]}")
          IO.puts("  Selected singular values: #{manifest["selected_singular_value_count"]}")
          IO.puts("  Router head shape: #{inspect(manifest["router_head_shape"])}")
        end

      {:error, reason} ->
        Mix.raise("Python semantic import failed: #{inspect(reason)}")
    end
  end

  defp print_summary(opts) do
    IO.puts("TRINITY Python Semantic Import")
    IO.puts("  Source dir: #{Keyword.fetch!(opts, :source_dir)}")
    IO.puts("  Manifest: #{Keyword.fetch!(opts, :manifest)}")
    IO.puts("  Reference: #{inspect(Keyword.get(opts, :reference_manifest))}")
    IO.puts("  Output dir: #{Keyword.fetch!(opts, :out_dir)}")
    IO.puts("  Force: #{Keyword.fetch!(opts, :force)}")
    IO.puts("  Resume: #{Keyword.fetch!(opts, :resume)}")
    IO.puts("  Load Qwen targets: #{Keyword.fetch!(opts, :load_qwen)}")
  end

  defp progress_fun(true), do: fn event -> IO.puts(Jason.encode!(normalize_for_json(event))) end
  defp progress_fun(false), do: fn _event -> :ok end

  defp normalize_for_json(value) when is_map(value) do
    Map.new(value, fn {key, val} -> {to_string(key), normalize_for_json(val)} end)
  end

  defp normalize_for_json(value) when is_list(value), do: Enum.map(value, &normalize_for_json/1)
  defp normalize_for_json(value) when is_tuple(value), do: Tuple.to_list(value)
  defp normalize_for_json(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_for_json(value), do: value
end
