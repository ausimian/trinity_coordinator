defmodule TrinityCoordinator.Sakana.SVD do
  @moduledoc """
  Elixir/Nx implementation of the SVD/SVF artifact mechanics used by TRINITY.

  This module intentionally handles the math and artifact loading only. It does
  not call providers and does not claim score reproduction.
  """

  @router_vector_key "trinity_router_es_vector"

  @type decomposition :: %{
          required(:u) => Nx.Tensor.t(),
          required(:s) => Nx.Tensor.t(),
          required(:v) => Nx.Tensor.t(),
          required(:shape) => tuple()
        }

  @type tensor_entry :: %{
          required(:path) => String.t(),
          required(:segments) => [term()],
          required(:tensor) => Nx.Tensor.t()
        }

  @doc "Returns true when a tensor matches Sakana's matrix-selection rule."
  def decomposable_tensor?(%Nx.Tensor{} = tensor) do
    shape = Nx.shape(tensor)
    tuple_size(shape) > 1 and Enum.all?(Tuple.to_list(shape), &(&1 > 1))
  end

  def decomposable_tensor?(_), do: false

  @doc "Runs a reduced SVD over one tensor."
  def decompose_tensor(%Nx.Tensor{} = tensor, opts \\ []) do
    opts = Keyword.validate!(opts, full_matrices?: false)
    {u, s, v} = svd_tuple!(tensor, opts)
    %{u: u, s: s, v: v, shape: Nx.shape(tensor)}
  end

  @doc """
  Reconstructs a tensor from SVD components and Sakana-style scale offsets.

  With zero offsets, this reconstructs the source tensor. With non-zero offsets,
  it applies `S * (1 + scale_offsets)` and Sakana's singular-value sum
  normalization.
  """
  def reconstruct(%{u: u, s: s, v: v}, scale_offsets) do
    scale_offsets = Nx.as_type(scale_offsets, Nx.type(s))
    scaled_s = Nx.multiply(s, Nx.add(scale_offsets, 1))
    normalization = Nx.divide(Nx.sum(s), Nx.sum(scaled_s))

    u
    |> Nx.multiply(Nx.reshape(scaled_s, {1, Nx.axis_size(scaled_s, 0)}))
    |> Nx.dot(v)
    |> Nx.multiply(normalization)
  end

  @doc "Flattens nested tensor containers into stable string paths."
  def flatten_tensors(container) do
    container
    |> flatten_tensor_entries()
    |> Enum.map(fn %{path: path, tensor: tensor} -> {path, tensor} end)
  end

  @doc "Flattens nested tensor containers with both stable paths and original segments."
  def flatten_tensor_entries(container) do
    container
    |> do_flatten([])
    |> Enum.sort_by(fn %{path: path} -> path end)
  end

  @doc "Returns decomposable tensors from a nested params container."
  def decomposable_tensors(container, opts \\ []) do
    container
    |> decomposable_tensor_entries(opts)
    |> Enum.map(fn %{path: path, tensor: tensor} -> {path, tensor} end)
  end

  @doc "Returns decomposable tensor entries from a nested params container."
  def decomposable_tensor_entries(container, opts \\ []) do
    opts = Keyword.validate!(opts, path_filter: nil)
    filter = opts[:path_filter]

    container
    |> flatten_tensor_entries()
    |> Enum.filter(fn %{path: path, tensor: tensor} ->
      decomposable_tensor?(tensor) and path_matches?(path, filter)
    end)
  end

  @doc """
  Returns the path filter equivalent to Sakana's `opt_layer_indices`.

  Bumblebee uses paths such as `decoder.blocks.26...`; Sakana's PyTorch code
  keeps all non-transformer tensors and only the requested transformer layers.
  """
  def layer_index_filter(nil), do: nil

  def layer_index_filter(indices) when is_list(indices) do
    indices = Enum.map(indices, &to_string/1)

    fn path ->
      not String.contains?(path, "decoder.blocks.") or
        Enum.any?(indices, &String.contains?(path, "decoder.blocks.#{&1}."))
    end
  end

  @doc "Counts singular values that would be consumed by selected tensors."
  def singular_value_count(tensors) when is_list(tensors) do
    Enum.reduce(tensors, 0, fn item, acc ->
      tensor = tensor_from_item!(item)
      shape = tensor |> Nx.shape() |> Tuple.to_list()
      acc + Enum.min(shape)
    end)
  end

  @doc "Produces a deterministic manifest for selected tensors."
  def tensor_manifest(tensors) when is_list(tensors) do
    Enum.map(tensors, fn item ->
      {path, tensor} = path_and_tensor_from_item!(item)

      %{
        path: path,
        shape: Nx.shape(tensor),
        singular_values: tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min(),
        type: Nx.type(tensor)
      }
    end)
  end

  @doc "Decomposes selected tensor entries and records scale-vector spans."
  def decompose_tensors(tensors) when is_list(tensors) do
    tensors
    |> Enum.map(fn item ->
      {path, segments, tensor} = path_segments_and_tensor_from_item!(item)
      count = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()

      %{
        path: path,
        segments: segments,
        tensor: tensor,
        decomposition: decompose_tensor(tensor),
        singular_values: count
      }
    end)
  end

  @doc "Reconstructs selected decompositions by consuming scale offsets in order."
  def reconstruct_tensors(decompositions, scale_offsets) when is_list(decompositions) do
    {reconstructed, offset} =
      Enum.map_reduce(decompositions, 0, fn item, offset ->
        count = item.singular_values

        offsets =
          scale_offsets
          |> Nx.slice([offset], [count])
          |> Nx.as_type(Nx.type(item.tensor))

        tensor =
          item.decomposition
          |> reconstruct(offsets)
          |> Nx.as_type(Nx.type(item.tensor))

        {%{path: item.path, segments: Map.get(item, :segments), tensor: tensor}, offset + count}
      end)

    if offset != Nx.size(scale_offsets) do
      raise ArgumentError,
            "scale offset size mismatch: consumed #{offset}, got #{Nx.size(scale_offsets)}"
    end

    %{tensors: reconstructed, offset_count: offset}
  end

  @doc "Decomposes and reconstructs selected tensors with Sakana scale offsets."
  def adapt_tensors(tensors, scale_offsets) when is_list(tensors) do
    tensors
    |> decompose_tensors()
    |> reconstruct_tensors(scale_offsets)
  end

  @doc "Puts adapted tensor entries back into a nested params container."
  def put_tensor_entries(%Axon.ModelState{} = state, tensor_entries)
      when is_list(tensor_entries) do
    %{state | data: put_tensor_entries(state.data, tensor_entries)}
  end

  def put_tensor_entries(container, tensor_entries) when is_list(tensor_entries) do
    Enum.reduce(tensor_entries, container, fn entry, acc ->
      segments = Map.get(entry, :segments)

      if not is_list(segments) or segments == [] do
        raise ArgumentError,
              "adapted tensor entry #{inspect(entry.path)} does not include path segments"
      end

      put_tensor_at_segments(acc, segments, entry.tensor)
    end)
  end

  @doc "Loads the raw Sakana router ES vector from safetensors."
  def load_router_vector!(path, tensor_name \\ @router_vector_key) do
    path
    |> Safetensors.read!()
    |> Map.fetch!(tensor_name)
  end

  @doc "Splits a raw ES vector into SVF scale offsets and linear head weights."
  def split_router_vector(vector, scale_count, hidden_size, output_count)
      when is_integer(scale_count) and scale_count >= 0 and
             is_integer(hidden_size) and hidden_size > 0 and
             is_integer(output_count) and output_count > 0 do
    size = Nx.size(vector)
    head_count = hidden_size * output_count
    expected_size = scale_count + head_count

    if size != expected_size do
      raise ArgumentError,
            "router vector size mismatch: expected #{expected_size}, got #{size}"
    end

    scale_offsets = Nx.slice(vector, [0], [scale_count])

    head_weights =
      vector
      |> Nx.slice([scale_count], [head_count])
      |> Nx.reshape({output_count, hidden_size})

    %{
      scale_offsets: scale_offsets,
      head_weights: head_weights,
      scale_count: scale_count,
      head_count: head_count
    }
  end

  @doc "Loads Sakana's linear head weights into an Axon dense head state."
  def put_linear_head_weights(params, head_weights, layer_name \\ "routing_head") do
    {output_count, hidden_size} = Nx.shape(head_weights)

    kernel =
      head_weights
      |> Nx.transpose()
      |> Nx.as_type(:f32)

    bias = Nx.broadcast(0.0, {output_count})

    expected_kernel_shape = {hidden_size, output_count}

    case get_in(params.data, [layer_name, "kernel"]) do
      %Nx.Tensor{} = existing ->
        if Nx.shape(existing) == expected_kernel_shape do
          put_in(params.data[layer_name], %{
            "kernel" => kernel,
            "bias" => bias
          })
        else
          raise ArgumentError,
                "head kernel shape mismatch: expected #{inspect(expected_kernel_shape)}, got #{inspect(Nx.shape(existing))}"
        end

      _ ->
        raise ArgumentError, "missing Axon dense layer #{inspect(layer_name)} kernel"
    end
  end

  defp do_flatten(%Nx.Tensor{} = tensor, path) do
    [%{path: format_path(path), segments: path, tensor: tensor}]
  end

  defp do_flatten(%Axon.ModelState{} = state, path), do: do_flatten(state.data, path)

  defp do_flatten(map, path) when is_map(map) do
    map
    |> Enum.flat_map(fn {key, value} -> do_flatten(value, path ++ [key]) end)
  end

  defp do_flatten(list, path) when is_list(list) do
    list
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> do_flatten(value, path ++ [index]) end)
  end

  defp do_flatten(tuple, path) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> do_flatten(value, path ++ [index]) end)
  end

  defp do_flatten(_other, _path), do: []

  defp format_path(path) do
    path
    |> Enum.map_join(".", fn
      key when is_atom(key) -> Atom.to_string(key)
      key -> to_string(key)
    end)
  end

  defp path_matches?(_path, nil), do: true
  defp path_matches?(path, filter) when is_function(filter, 1), do: filter.(path)
  defp path_matches?(path, pattern) when is_binary(pattern), do: String.contains?(path, pattern)

  defp svd_tuple!(tensor, opts) do
    svd = Function.capture(Nx.LinAlg, :svd, 2)
    result = svd.(tensor, full_matrices?: opts[:full_matrices?])

    case result do
      {%Nx.Tensor{} = u, %Nx.Tensor{} = s, %Nx.Tensor{} = v} ->
        {u, s, v}

      other ->
        raise ArgumentError,
              "expected Nx.LinAlg.svd/2 to return {u, s, v}, got: #{inspect(other)}"
    end
  end

  defp tensor_from_item!({_path, %Nx.Tensor{} = tensor}), do: tensor
  defp tensor_from_item!(%{tensor: %Nx.Tensor{} = tensor}), do: tensor

  defp path_and_tensor_from_item!({path, %Nx.Tensor{} = tensor}), do: {path, tensor}

  defp path_and_tensor_from_item!(%{path: path, tensor: %Nx.Tensor{} = tensor}),
    do: {path, tensor}

  defp path_segments_and_tensor_from_item!({path, %Nx.Tensor{} = tensor}), do: {path, nil, tensor}

  defp path_segments_and_tensor_from_item!(%{
         path: path,
         segments: segments,
         tensor: %Nx.Tensor{} = tensor
       }),
       do: {path, segments, tensor}

  defp put_tensor_at_segments(_container, [], %Nx.Tensor{} = tensor), do: tensor

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_map(container) do
    if Map.has_key?(container, segment) do
      Map.update!(container, segment, &put_tensor_at_segments(&1, rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing map segment #{inspect(segment)}"
    end
  end

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_list(container) and is_integer(segment) do
    if segment >= 0 and segment < length(container) do
      List.update_at(container, segment, &put_tensor_at_segments(&1, rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing list index #{segment}"
    end
  end

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_tuple(container) and is_integer(segment) do
    if segment >= 0 and segment < tuple_size(container) do
      put_elem(container, segment, put_tensor_at_segments(elem(container, segment), rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing tuple index #{segment}"
    end
  end

  defp put_tensor_at_segments(_container, [segment | _rest], %Nx.Tensor{}) do
    raise ArgumentError, "cannot descend into path segment #{inspect(segment)}"
  end
end
