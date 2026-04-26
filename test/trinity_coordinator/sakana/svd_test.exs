defmodule TrinityCoordinator.Sakana.SVDTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{CoordinationHead, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.SVD

  @router_vector_path "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"

  test "decomposes and reconstructs a matrix with Sakana normalization" do
    matrix =
      Nx.tensor(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 10.0]
        ],
        type: :f32
      )

    decomposition = SVD.decompose_tensor(matrix)
    zeros = Nx.broadcast(0.0, Nx.shape(decomposition.s))
    reconstructed = SVD.reconstruct(decomposition, zeros)

    max_error =
      reconstructed
      |> Nx.subtract(matrix)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert Nx.shape(decomposition.u) == {3, 3}
    assert Nx.shape(decomposition.s) == {3}
    assert Nx.shape(decomposition.v) == {3, 3}
    assert max_error < 1.0e-3
  end

  @tag :integration
  test "decomposition and reconstruction preserve CUDA backend" do
    Runtime.put_cuda_backend!()

    matrix = Nx.iota({4, 3}, type: :f32) |> Nx.backend_transfer({EXLA.Backend, client: :cuda})
    decomposition = SVD.decompose_tensor(matrix)
    zeros = Nx.broadcast(0.0, Nx.shape(decomposition.s))
    reconstructed = SVD.reconstruct(decomposition, zeros)

    assert Runtime.tensor_backend(reconstructed) =~ "EXLA.Backend<cuda:"
  end

  test "selects only matrix-like tensors and flattens paths deterministically" do
    container = %{
      z: Nx.iota({2}, type: :f32),
      a: %{
        singleton: Nx.iota({2, 1}, type: :f32),
        matrix: Nx.iota({2, 3}, type: :f32)
      },
      b: [Nx.iota({3, 2}, type: :f32)]
    }

    flattened = SVD.flatten_tensors(container)
    selected = SVD.decomposable_tensors(container)

    assert Enum.map(flattened, &elem(&1, 0)) == ["a.matrix", "a.singleton", "b.0", "z"]
    assert Enum.map(selected, &elem(&1, 0)) == ["a.matrix", "b.0"]
    assert SVD.singular_value_count(selected) == 4

    entries = SVD.decomposable_tensor_entries(container)
    assert Enum.map(entries, & &1.segments) == [[:a, :matrix], [:b, 0]]
  end

  test "loads and splits the Sakana router vector safetensors artifact" do
    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    assert Nx.shape(vector) == {19_456}
    assert Nx.shape(split.scale_offsets) == {9216}
    assert Nx.shape(split.head_weights) == {10, 1024}
    assert split.scale_count == 9216
    assert split.head_count == 10_240
  end

  test "loads Sakana head weights into the linear Axon head" do
    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
    updated = SVD.put_linear_head_weights(params, split.head_weights)

    assert Nx.shape(updated.data["routing_head"]["kernel"]) == {1024, 10}
    assert Nx.shape(updated.data["routing_head"]["bias"]) == {10}

    route = CoordinationHead.route(model, updated, Nx.broadcast(0.01, {1, 1024}), 7, 3)

    assert Nx.shape(route.logits) == {1, 10}
    assert Nx.shape(route.agent_logits) == {7}
    assert Nx.shape(route.role_logits) == {3}
  end

  test "applies scale offsets to selected decomposed tensors in deterministic order" do
    container = %{
      "layer.with.dots" => %{"kernel" => Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :f32)},
      b: [Nx.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]], type: :f32)]
    }

    tensors = SVD.decomposable_tensor_entries(container)

    zero_offsets = Nx.broadcast(0.0, {4})
    zero_adapted = SVD.adapt_tensors(tensors, zero_offsets)

    assert zero_adapted.offset_count == 4
    assert Enum.map(zero_adapted.tensors, & &1.path) == ["b.0", "layer.with.dots.kernel"]

    first_error =
      zero_adapted.tensors
      |> hd()
      |> Map.fetch!(:tensor)
      |> Nx.subtract(hd(tensors).tensor)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert first_error < 1.0e-3

    nonzero_offsets = Nx.tensor([0.1, -0.1, 0.2, -0.2], type: :f32)
    nonzero_adapted = SVD.adapt_tensors(tensors, nonzero_offsets)

    assert Enum.map(nonzero_adapted.tensors, &Nx.shape(&1.tensor)) == [{3, 2}, {2, 2}]
    assert Enum.all?(nonzero_adapted.tensors, &(Nx.type(&1.tensor) == {:f, 32}))

    updated = SVD.put_tensor_entries(container, zero_adapted.tensors)
    assert Nx.shape(updated.b |> hd()) == {3, 2}
    assert Nx.shape(updated["layer.with.dots"]["kernel"]) == {2, 2}
  end

  @tag :qwen
  test "selects Qwen SVF tensors for Sakana layer 26 on CUDA" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected =
      SVD.decomposable_tensors(model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    assert SVD.singular_value_count(selected) == 9216

    assert Enum.any?(selected, fn {path, _tensor} ->
             String.contains?(path, "decoder.blocks.26.")
           end)

    assert Enum.any?(selected, fn {path, _tensor} ->
             not String.contains?(path, "decoder.blocks.")
           end)

    {_path, tensor} = hd(selected)
    assert Runtime.tensor_backend(tensor) =~ "EXLA.Backend<cuda:"

    manifest = SVD.tensor_manifest(selected)
    assert Enum.any?(manifest, &(&1.singular_values > 0))
    assert Enum.all?(manifest, &is_binary(&1.path))
  end

  @tag :qwen
  test "maps a representative Qwen layer 26 tensor to its Sakana scale-offset span on CUDA" do
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected =
      SVD.decomposable_tensors(model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    {offset_start, {_path, tensor}} =
      Enum.reduce_while(selected, 0, fn {path, tensor} = item, offset ->
        if String.contains?(path, "decoder.blocks.26.self_attention.query") do
          {:halt, {offset, item}}
        else
          count = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()
          {:cont, offset + count}
        end
      end)

    singular_values = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()
    offsets = Nx.slice(split.scale_offsets, [offset_start], [singular_values])

    assert offset_start >= 0
    assert offset_start + singular_values <= split.scale_count
    assert Nx.shape(offsets) == {singular_values}
    assert Runtime.tensor_backend(tensor) =~ "EXLA.Backend<cuda:"
  end

  @tag :expensive_qwen_svd
  test "fully reconstructs and reinserts all Sakana-selected Qwen SVF tensors on CUDA" do
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    assert {:ok, {model_info, _tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    selected =
      SVD.decomposable_tensor_entries(model_info.params,
        path_filter: SVD.layer_index_filter([26])
      )

    selected_shapes = Enum.map(selected, &Nx.shape(&1.tensor))
    selected_types = Enum.map(selected, &Nx.type(&1.tensor))

    adapted = SVD.adapt_tensors(selected, split.scale_offsets)
    updated_params = SVD.put_tensor_entries(model_info.params, adapted.tensors)

    updated =
      SVD.decomposable_tensor_entries(updated_params,
        path_filter: SVD.layer_index_filter([26])
      )

    assert SVD.singular_value_count(selected) == 9216
    assert adapted.offset_count == 9216
    assert length(adapted.tensors) == length(selected)
    assert Enum.map(adapted.tensors, & &1.path) == Enum.map(selected, & &1.path)
    assert Enum.map(adapted.tensors, &Nx.shape(&1.tensor)) == selected_shapes
    assert Enum.map(adapted.tensors, &Nx.type(&1.tensor)) == selected_types
    assert Enum.map(updated, &Nx.shape(&1.tensor)) == selected_shapes

    assert adapted.tensors |> hd() |> Map.fetch!(:tensor) |> Runtime.tensor_backend() =~
             "EXLA.Backend<cuda:"
  end

  @tag :qwen
  test "routes a real Qwen hidden vector through the Sakana linear head on CUDA" do
    Runtime.put_cuda_backend!()

    vector = SVD.load_router_vector!(@router_vector_path)
    split = SVD.split_router_vector(vector, 9216, 1024, 10)

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())
    params = SVD.put_linear_head_weights(params, split.head_weights)

    assert {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)

    assert {:ok, metadata} =
             TrinityCoordinator.Extractor.extract_penultimate_hidden_state_with_metadata(
               model_info,
               tokenizer,
               [%{"role" => "user", "content" => "Route this request."}]
             )

    route = CoordinationHead.route(model, params, metadata.vector, 7, 3)

    assert metadata.vector_shape == {1, 1024}
    assert Runtime.tensor_backend(metadata.vector) =~ "EXLA.Backend<cuda:"
    assert Runtime.tensor_backend(route.logits) =~ "EXLA.Backend<cuda:"
    assert Nx.shape(route.logits) == {1, 10}
  end
end
