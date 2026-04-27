defmodule TrinityCoordinator.Sakana.PythonImporterTest do
  use ExUnit.Case, async: false

  alias TrinityCoordinator.Sakana.{Artifact, ExportSpec, PythonImporter}

  test "imports a synthetic Python semantic bundle into canonical artifacts" do
    source_dir = unique_tmp_dir("python_source")
    out_dir = unique_tmp_dir("python_out")
    File.rm_rf!(out_dir)

    on_exit(fn ->
      File.rm_rf(source_dir)
      File.rm_rf(out_dir)
    end)

    components_path = Path.join(source_dir, "trinity_svf_components.safetensors")
    scales_path = Path.join(source_dir, "trinity_svf_scale_offsets.safetensors")
    head_path = Path.join(source_dir, "trinity_router_head.safetensors")
    manifest_path = Path.join(source_dir, "trinity_sakana_export_manifest.json")

    u = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    s = Nx.tensor([1.0, 2.0], type: :f32)
    v = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)
    offsets = Nx.tensor([0.0, 0.0], type: :f32)
    head = Nx.iota({4, 2}, type: :f32)

    Safetensors.write!(components_path, %{
      "svd.U.model.embed_tokens.weight" => u,
      "svd.S.model.embed_tokens.weight" => s,
      "svd.V.model.embed_tokens.weight" => v
    })

    Safetensors.write!(scales_path, %{"svf.scale_offsets.model.embed_tokens.weight" => offsets})
    Safetensors.write!(head_path, %{"trinity.router_head.linear.weight" => head})

    python_manifest = %{
      "format" => "trinity_sakana_safetensors_export",
      "components_path" => Path.basename(components_path),
      "scale_offsets_path" => Path.basename(scales_path),
      "router_head_path" => Path.basename(head_path),
      "source_vector_sha256" => "synthetic",
      "selected_tensors" => [
        %{
          "source_name" => "model.embed_tokens.weight",
          "elixir_name" => "embedder.token_embedding.kernel",
          "shape" => [2, 2],
          "singular_values" => 2,
          "offset_start" => 0,
          "offset_end" => 2
        }
      ]
    }

    File.write!(manifest_path, Jason.encode!(python_manifest))

    spec = %ExportSpec{
      name: :synthetic_python_import,
      base_model_repo: "synthetic",
      bumblebee_module: Bumblebee.Text.Gpt2,
      architecture: :base,
      hidden_size: 2,
      num_agents: 1,
      num_roles: 3,
      selected_layer_indices: [],
      scale_offset_count: 2,
      source_vector_tensor: "synthetic",
      router_head_tensor_key: Artifact.router_head_tensor_key(),
      source_vector_path: "synthetic",
      out_dir: out_dir,
      xla_target: "host",
      export_backend: "test"
    }

    assert {:ok, manifest} =
             PythonImporter.import_bundle(
               source_dir: source_dir,
               manifest: Path.basename(manifest_path),
               out_dir: out_dir,
               force: true,
               load_qwen: false,
               spec: spec
             )

    assert manifest["status"] == "complete"
    assert manifest["export_complete"] == true
    assert manifest["selected_tensor_count"] == 1
    assert manifest["router_head_shape"] == [4, 2]

    assert {:ok, loaded_manifest} = Artifact.load_manifest(out_dir)
    assert loaded_manifest["artifact_layout"] == Artifact.artifact_layout_single_file()

    head_tensor = Artifact.load_router_head!(out_dir, manifest: loaded_manifest)
    assert Nx.shape(head_tensor) == {4, 2}

    tensors = Artifact.load_adapted_tensors!(out_dir, manifest: loaded_manifest)
    adapted = Map.fetch!(tensors, "embedder.token_embedding.kernel")

    assert Nx.shape(adapted) == {2, 2}
    assert Nx.all_close(adapted, Nx.tensor([[1.0, 0.0], [0.0, 2.0]], type: :f32), atol: 1.0e-6)
  end

  defp unique_tmp_dir(prefix) do
    Path.join(System.tmp_dir!(), "#{prefix}_#{System.unique_integer([:positive])}")
    |> tap(&File.mkdir_p!/1)
  end
end
