defmodule TrinityCoordinator.Test.SakanaTinyArtifactFactory do
  @moduledoc """
  Test/dev helper that produces (and refreshes) the tiny synthetic Sakana
  artifact at `test/fixtures/sakana_tiny_artifact/`.

  The tiny artifact exercises the canonical manifest + router-head load path
  without requiring Qwen3-0.6B or CUDA:

    * `hidden_size:  8`
    * `num_agents:   2`
    * `num_roles:    3`
    * `output_count: 5`  (head shape `[5, 8]`, 40 × f32 = 160 bytes payload)

  Compiled in `:test` only via `elixirc_paths(:test)`.

  ## Regenerating

      mix run --no-start -e \\
        'TrinityCoordinator.Test.SakanaTinyArtifactFactory.refresh!()'

  Then `git add test/fixtures/sakana_tiny_artifact/`.
  """

  @fixture_dir Path.join(["test", "fixtures", "sakana_tiny_artifact"])
  @router_head_file "router_head.safetensors"
  @router_head_tensor_key "trinity_router_head"
  @hidden_size 8
  @num_agents 2
  @num_roles 3
  @output_count 5

  @doc "Returns the on-disk fixture directory (relative to repo root)."
  @spec fixture_dir() :: String.t()
  def fixture_dir, do: @fixture_dir

  @doc "Returns the canonical hidden_size used by the tiny fixture."
  @spec hidden_size() :: pos_integer()
  def hidden_size, do: @hidden_size

  @doc "Returns the canonical {num_agents, num_roles, hidden_size}."
  @spec dimensions() :: {pos_integer(), pos_integer(), pos_integer()}
  def dimensions, do: {@num_agents, @num_roles, @hidden_size}

  @doc """
  Returns the deterministic router-head tensor (shape `{5, 8}` f32). Same value
  on every call.
  """
  @spec router_head_tensor() :: Nx.Tensor.t()
  def router_head_tensor do
    # Deterministic structured weights: each row gets its own offset so that
    # different inputs map to different argmaxes when the tests want to probe
    # routing behavior.
    1..@output_count
    |> Enum.flat_map(fn row ->
      Enum.map(1..@hidden_size, fn col -> row * 0.1 + col * 0.01 end)
    end)
    |> Nx.tensor(type: :f32)
    |> Nx.reshape({@output_count, @hidden_size})
  end

  @doc """
  Builds the manifest map (Elixir terms, before JSON encoding).
  """
  @spec manifest(String.t()) :: map()
  def manifest(router_head_sha256) when is_binary(router_head_sha256) do
    %{
      "artifact_version" => 1,
      "status" => "complete",
      "created_at" => "2026-05-20T00:00:00Z",
      "updated_at" => "2026-05-20T00:00:00Z",
      "base_model_repo" => "synthetic/tiny-router",
      "bumblebee_module" => "Bumblebee.Text.Synthetic",
      "architecture" => "for_routing_only",
      "xla_target" => "cpu",
      "export_backend" => "tiny_synthetic_factory",
      "export_complete" => true,
      "importer" => "TrinityCoordinator.Test.SakanaTinyArtifactFactory",
      "python_manifest_path" => nil,
      "reference_manifest_path" => nil,
      "source_vector_path" => "synthetic/tiny_source_vector",
      "source_vector_tensor" => "tiny_components",
      "source_vector_shape" => [0],
      "source_vector_sha256" =>
        "0000000000000000000000000000000000000000000000000000000000000000",
      "scale_offset_count" => 0,
      "router_head_shape" => [@output_count, @hidden_size],
      "router_head_artifact" => @router_head_file,
      "router_head_tensor_key" => @router_head_tensor_key,
      "router_head_sha256" => router_head_sha256,
      "adapted_tensors_artifact" => "adapted_tensors.safetensors",
      "selected_tensor_count" => 0,
      "selected_singular_value_count" => 0,
      "selected_tensors" => [],
      "artifact_layout" => "checkpoint_directory",
      "source_split" => %{
        "hidden_size" => @hidden_size,
        "output_count" => @output_count,
        "scale_count" => 0
      },
      "split" => %{
        "head_count" => @output_count * @hidden_size,
        "scale_count" => 0
      }
    }
  end

  @doc """
  Writes the tiny safetensors + manifest into `dir`. Returns the sha256 of the
  safetensors file. Creates the directory if needed.
  """
  @spec write!(String.t()) :: String.t()
  def write!(dir) when is_binary(dir) do
    File.mkdir_p!(dir)

    tensor = router_head_tensor()
    safetensors_path = Path.join(dir, @router_head_file)
    write_safetensors!(safetensors_path, tensor)

    head_sha = sha256_hex(File.read!(safetensors_path))

    manifest_path = Path.join(dir, "manifest.json")
    File.write!(manifest_path, Jason.encode!(manifest(head_sha), sort_keys: true))

    head_sha
  end

  @doc """
  Regenerates the committed tiny fixture at `#{@fixture_dir}` relative to the
  current working directory. Used by maintainers; the test suite uses
  `write!/1` against a temp directory.
  """
  @spec refresh!() :: String.t()
  def refresh!, do: write!(Path.expand(@fixture_dir, File.cwd!()))

  defp write_safetensors!(path, %Nx.Tensor{} = tensor) do
    [out_dim, hidden_dim] = Nx.shape(tensor) |> Tuple.to_list()
    payload = Nx.to_binary(tensor)
    payload_size = byte_size(payload)

    header_map = %{
      @router_head_tensor_key => %{
        "dtype" => "F32",
        "shape" => [out_dim, hidden_dim],
        "data_offsets" => [0, payload_size]
      }
    }

    header_json = Jason.encode!(header_map, sort_keys: true)
    # safetensors spec: 8-byte little-endian header length + JSON + payload
    header_len = byte_size(header_json)

    body =
      <<header_len::little-unsigned-integer-size(64)>> <>
        header_json <> payload

    File.write!(path, body)
  end

  defp sha256_hex(bin) when is_binary(bin) do
    :crypto.hash(:sha256, bin) |> Base.encode16(case: :lower)
  end
end
