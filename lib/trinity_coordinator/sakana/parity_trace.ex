defmodule TrinityCoordinator.Sakana.ParityTrace do
  @moduledoc """
  Incremental parity tracing for the Sakana/Python SVD sample hash.

  The native Elixir path recomputes SVD with Nx. The Python reference hash was
  produced from Python/PyTorch SVD components. For non-zero singular-value
  offsets, different valid SVD bases can reconstruct the original tensor with
  zero offsets but diverge after per-singular-value scaling. This module emits a
  compact JSON report so the native path, imported Python-component path, dtype
  choices, orientation choices, and final tensor bytes can be compared side by
  side.
  """

  alias TrinityCoordinator.{Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, SVD}

  @router_vector_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"
  @reference_manifest_path "priv/sakana_trinity/reference/sakana_python_reference_manifest.json"
  @scale_count 9_216
  @hidden_size 1_024
  @output_count 10
  @component_file "trinity_svf_components.safetensors"
  @scale_file "trinity_svf_scale_offsets.safetensors"

  @type report :: map()

  @doc "Builds the complete native and optional Python-component parity report."
  @spec sample_report!(keyword()) :: report()
  def sample_report!(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        router_vector_path: @router_vector_path,
        reference_manifest_path: @reference_manifest_path,
        components_dir: nil,
        require_cuda: true
      )

    if opts[:require_cuda] do
      Runtime.put_cuda_backend!()
    end

    reference = load_json!(opts[:reference_manifest_path])
    sample = Map.fetch!(reference, "sample_adapted_tensor")

    {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    selected = qwen_selected_tensors(model_info)
    sample_entry = sample_entry!(selected, sample)

    vector = SVD.load_router_vector!(opts[:router_vector_path])
    split = SVD.split_router_vector(vector, @scale_count, @hidden_size, @output_count)
    offsets = sample_offsets(split.scale_offsets, sample)

    source_tensor = orient_to_shape!(sample_entry.tensor, sample["source_shape"], sample["elixir_name"])

    native_variants = native_variants(source_tensor, offsets, sample)
    semantic = maybe_semantic_component_report(opts[:components_dir], source_tensor, sample)

    %{
      "schema" => "trinity_sakana_svd_parity_trace.v1",
      "generated_at_utc" => DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601(),
      "paths" => %{
        "router_vector" => opts[:router_vector_path],
        "reference_manifest" => opts[:reference_manifest_path],
        "components_dir" => opts[:components_dir]
      },
      "reference" => %{
        "source_name" => sample["source_name"],
        "elixir_name" => sample["elixir_name"],
        "offset_start" => sample["offset_start"],
        "offset_end" => sample["offset_end"],
        "source_shape" => sample["source_shape"],
        "sample_reconstructed_shape" => sample["sample_reconstructed_shape"],
        "expected_bf16_sha256" => sample["sample_reconstructed_bf16_sha256"],
        "expected_bf16_min" => sample["sample_reconstructed_bf16_min"],
        "expected_bf16_max" => sample["sample_reconstructed_bf16_max"]
      },
      "selection" => %{
        "selected_tensor_count" => length(selected),
        "selected_singular_value_count" => SVD.singular_value_count(selected),
        "sample_elixir_shape" => shape_list(sample_entry.tensor),
        "sample_source_oriented_shape" => shape_list(source_tensor),
        "sample_source_type" => inspect(Nx.type(sample_entry.tensor)),
        "sample_source_backend" => Runtime.tensor_backend(sample_entry.tensor)
      },
      "router_vector" => tensor_summary(vector, prefix_count: 8),
      "scale_offsets" => tensor_summary(offsets, prefix_count: 16),
      "source_tensor" => tensor_summary(source_tensor, prefix_count: 16),
      "native_elixir_svd_variants" => native_variants,
      "semantic_python_component_variants" => semantic
    }
  end

  @doc "Writes a parity report as pretty JSON."
  @spec write_json!(String.t(), report()) :: :ok
  def write_json!(path, report) when is_binary(path) and is_map(report) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(normalize_json(report), pretty: true))
    :ok
  end

  defp native_variants(source_tensor, offsets, sample) do
    [
      %{label: "native_nx_f32_svd_offsets_singular_final_bf16", compute_type: :f32, offset_type: :singular},
      %{label: "native_nx_source_svd_offsets_singular_final_bf16", compute_type: :source, offset_type: :singular},
      %{label: "native_nx_f32_svd_offsets_f32_final_bf16", compute_type: :f32, offset_type: :f32},
      %{label: "native_nx_f32_svd_offsets_source_final_bf16", compute_type: :f32, offset_type: :source}
    ]
    |> Enum.map(fn config -> native_variant(source_tensor, offsets, sample, config) end)
  end

  defp native_variant(source_tensor, offsets, sample, config) do
    decomp = SVD.decompose_tensor(source_tensor, compute_type: config.compute_type)
    typed_offsets = cast_offsets(offsets, config.offset_type, source_tensor, decomp)
    zero_offsets = Nx.broadcast(0.0, Nx.shape(decomp.s)) |> Nx.as_type(Nx.type(decomp.s))
    zero_reconstruct = SVD.reconstruct(decomp, zero_offsets)
    sample_reconstruct = SVD.reconstruct(decomp, typed_offsets) |> Nx.as_type(:bf16)
    final = orient_to_shape!(sample_reconstruct, sample["sample_reconstructed_shape"], config.label)
    expected = sample["sample_reconstructed_bf16_sha256"]
    observed = Artifact.tensor_sha256(final)

    %{
      "label" => config.label,
      "svd_provider" => "elixir_nx",
      "compute_type" => Atom.to_string(config.compute_type),
      "offset_type" => Atom.to_string(config.offset_type),
      "u" => tensor_summary(decomp.u, prefix_count: 4, include_alt_hashes: false),
      "s" => singular_summary(decomp.s, typed_offsets),
      "v" => tensor_summary(decomp.v, prefix_count: 4, include_alt_hashes: false),
      "zero_offset_max_abs_error_vs_source" => max_abs_error(zero_reconstruct, svd_source_tensor(source_tensor, config.compute_type)),
      "final" => tensor_summary(final, prefix_count: 16),
      "observed_bf16_sha256" => observed,
      "expected_bf16_sha256" => expected,
      "matches_expected" => observed == expected
    }
  end

  defp maybe_semantic_component_report(nil, _source_tensor, _sample), do: nil
  defp maybe_semantic_component_report("", _source_tensor, _sample), do: nil

  defp maybe_semantic_component_report(components_dir, source_tensor, sample) do
    component_path = Path.join(components_dir, @component_file)
    scale_path = Path.join(components_dir, @scale_file)

    if File.exists?(component_path) and File.exists?(scale_path) do
      components = Safetensors.read!(component_path)
      scales = Safetensors.read!(scale_path)
      safe_key = sanitize_python_key(sample["source_name"])
      u = fetch_tensor!(components, "svd.U.#{safe_key}")
      s = fetch_tensor!(components, "svd.S.#{safe_key}")
      v = fetch_tensor!(components, "svd.V.#{safe_key}")
      offsets = fetch_tensor!(scales, "svf.scale_offsets.#{safe_key}")

      [:nx, :torch_v]
      |> Enum.map(fn layout -> safe_semantic_variant(layout, u, s, v, offsets, source_tensor, sample) end)
    else
      %{
        "error" => "missing_semantic_component_files",
        "component_path" => component_path,
        "scale_path" => scale_path
      }
    end
  end

  defp safe_semantic_variant(layout, u, s, v, offsets, source_tensor, sample) do
    semantic_variant(layout, u, s, v, offsets, source_tensor, sample)
  rescue
    e ->
      %{
        "label" => "semantic_python_components_v_layout_#{layout}",
        "svd_provider" => "python_components_safetensors",
        "v_layout" => Atom.to_string(layout),
        "error" => Exception.message(e),
        "matches_expected" => false
      }
  end

  defp semantic_variant(layout, u, s, v, offsets, source_tensor, sample) do
    decomp = %{u: u, s: s, v: v}
    typed_offsets = Nx.as_type(offsets, Nx.type(s))
    zero_offsets = Nx.broadcast(0.0, Nx.shape(s)) |> Nx.as_type(Nx.type(s))
    zero_reconstruct = SVD.reconstruct(decomp, zero_offsets, v_layout: layout)
    sample_reconstruct = SVD.reconstruct(decomp, typed_offsets, v_layout: layout) |> Nx.as_type(:bf16)
    final = orient_to_shape!(sample_reconstruct, sample["sample_reconstructed_shape"], "semantic_#{layout}")
    expected = sample["sample_reconstructed_bf16_sha256"]
    observed = Artifact.tensor_sha256(final)

    %{
      "label" => "semantic_python_components_v_layout_#{layout}",
      "svd_provider" => "python_components_safetensors",
      "v_layout" => Atom.to_string(layout),
      "u" => tensor_summary(u, prefix_count: 4, include_alt_hashes: false),
      "s" => singular_summary(s, typed_offsets),
      "v" => tensor_summary(v, prefix_count: 4, include_alt_hashes: false),
      "offsets" => tensor_summary(offsets, prefix_count: 16),
      "zero_offset_max_abs_error_vs_source" => max_abs_error(zero_reconstruct, Nx.as_type(source_tensor, :f32)),
      "final" => tensor_summary(final, prefix_count: 16),
      "observed_bf16_sha256" => observed,
      "expected_bf16_sha256" => expected,
      "matches_expected" => observed == expected
    }
  end

  defp singular_summary(s, offsets) do
    scaled_s = Nx.multiply(s, Nx.add(Nx.as_type(offsets, Nx.type(s)), 1))

    %{
      "singular_values" => tensor_summary(s, prefix_count: 16),
      "typed_offsets" => tensor_summary(offsets, prefix_count: 16),
      "scaled_s" => tensor_summary(scaled_s, prefix_count: 16),
      "sum_s" => scalar(Nx.sum(Nx.as_type(s, :f32))),
      "sum_scaled_s" => scalar(Nx.sum(Nx.as_type(scaled_s, :f32))),
      "normalization" => scalar(Nx.divide(Nx.sum(Nx.as_type(s, :f32)), Nx.sum(Nx.as_type(scaled_s, :f32))))
    }
  end

  defp qwen_selected_tensors(model_info) do
    SVD.decomposable_tensor_entries(
      model_info.params,
      path_filter: SVD.layer_index_filter([26])
    )
  end

  defp sample_entry!(selected, sample) do
    Enum.find(selected, &(&1.path == sample["elixir_name"])) ||
      raise ArgumentError, "sample tensor #{inspect(sample["elixir_name"])} was not selected"
  end

  defp sample_offsets(scale_offsets, sample) do
    offset_start = sample["offset_start"]
    singular_values = sample["offset_end"] - sample["offset_start"]
    Nx.slice(scale_offsets, [offset_start], [singular_values])
  end

  defp cast_offsets(offsets, :singular, _source_tensor, decomp), do: Nx.as_type(offsets, Nx.type(decomp.s))
  defp cast_offsets(offsets, :f32, _source_tensor, _decomp), do: Nx.as_type(offsets, :f32)
  defp cast_offsets(offsets, :source, source_tensor, _decomp), do: Nx.as_type(offsets, Nx.type(source_tensor))

  defp svd_source_tensor(tensor, :source), do: tensor
  defp svd_source_tensor(tensor, :f32), do: Nx.as_type(tensor, :f32)

  defp orient_to_shape!(%Nx.Tensor{} = tensor, shape_list, label) when is_list(shape_list) do
    target = List.to_tuple(shape_list)

    cond do
      Nx.shape(tensor) == target ->
        tensor

      tuple_size(Nx.shape(tensor)) == 2 and Nx.shape(Nx.transpose(tensor)) == target ->
        Nx.transpose(tensor)

      true ->
        raise ArgumentError,
              "cannot orient #{inspect(label)} from #{inspect(Nx.shape(tensor))} to #{inspect(target)}"
    end
  end

  defp max_abs_error(left, right) do
    left = Nx.as_type(left, :f32)
    right = Nx.as_type(right, :f32)

    left
    |> Nx.subtract(right)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> scalar()
  end

  defp tensor_summary(tensor, opts \\ []) do
    opts = Keyword.validate!(opts, prefix_count: 8, include_alt_hashes: true)
    tensor_f32 = Nx.as_type(tensor, :f32)
    size = Nx.size(tensor)
    prefix_count = min(size, opts[:prefix_count])

    base = %{
      "shape" => shape_list(tensor),
      "type" => inspect(Nx.type(tensor)),
      "backend" => Runtime.tensor_backend(tensor),
      "size" => size,
      "sha256" => Artifact.tensor_sha256(tensor),
      "min" => scalar(Nx.reduce_min(tensor_f32)),
      "max" => scalar(Nx.reduce_max(tensor_f32)),
      "sum" => scalar(Nx.sum(tensor_f32)),
      "prefix_f32" => prefix_f32(tensor, prefix_count)
    }

    if opts[:include_alt_hashes] do
      Map.merge(base, %{
        "sha256_as_f32" => Artifact.tensor_sha256(Nx.as_type(tensor, :f32)),
        "sha256_as_bf16" => Artifact.tensor_sha256(Nx.as_type(tensor, :bf16))
      })
    else
      base
    end
  end

  defp prefix_f32(_tensor, 0), do: []

  defp prefix_f32(tensor, count) do
    tensor
    |> Nx.as_type(:f32)
    |> Nx.reshape({Nx.size(tensor)})
    |> Nx.slice([0], [count])
    |> Nx.to_flat_list()
  end

  defp scalar(tensor), do: tensor |> Nx.to_number() |> finite_float()

  defp finite_float(value) when is_float(value) do
    cond do
      value != value -> "nan"
      true -> value
    end
  end

  defp finite_float(value), do: value

  defp shape_list(tensor), do: Nx.shape(tensor) |> Tuple.to_list()

  defp fetch_tensor!(map, key) do
    case Map.fetch(map, key) do
      {:ok, %Nx.Tensor{} = tensor} -> tensor
      _ -> raise ArgumentError, "missing tensor #{inspect(key)}; available keys: #{inspect(Map.keys(map))}"
    end
  end

  defp sanitize_python_key(source_name) do
    source_name
    |> String.replace("/", "__")
    |> String.replace(~r/[^0-9A-Za-z_.-]/, "__")
  end

  defp load_json!(path) do
    path
    |> File.read!()
    |> Jason.decode!()
  end

  defp normalize_json(value) when is_map(value) do
    Map.new(value, fn {key, val} -> {to_string(key), normalize_json(val)} end)
  end

  defp normalize_json(value) when is_list(value), do: Enum.map(value, &normalize_json/1)
  defp normalize_json(value) when is_tuple(value), do: Tuple.to_list(value)
  defp normalize_json(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_json(value), do: value
end
