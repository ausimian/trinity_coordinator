defmodule TrinityCoordinator.Sakana.Coordinator do
  @moduledoc """
  High-level loader for the artifact-driven TRINITY coordinator.

  It returns a single struct-like map containing:

    * the Qwen model_info/tokenizer used for hidden-state extraction,
    * the standalone Axon routing-head model and params,
    * artifact manifest metadata,
    * inferred `num_agents`, `num_roles`, and hidden size.

  Provider LLM calls are not performed here.
  """

  alias TrinityCoordinator.{Extractor, Runtime, RuntimeProfile, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, Head}

  @type t :: %{
          required(:model_info) => map(),
          required(:tokenizer) => map(),
          required(:routing_model) => Axon.t(),
          required(:routing_params) => struct(),
          required(:manifest) => map(),
          required(:artifact_dir) => String.t(),
          required(:num_agents) => pos_integer(),
          required(:num_roles) => pos_integer(),
          required(:hidden_size) => pos_integer(),
          required(:backend) => term(),
          required(:runtime_profile) => TrinityCoordinator.RuntimeProfile.t()
        }

  @doc """
  Loads the Sakana-adapted Qwen backbone and routing head.

  Options:

    * `:artifact_dir` - defaults to `Artifact.default_output_dir/0`.
    * `:num_roles` - defaults to `3`.
    * `:runtime_profile` - a `TrinityCoordinator.RuntimeProfile` name or struct.
      Defaults to `:cuda_exla` (canonical production lane). The profile
      determines the Nx backend tuple and whether CUDA must be present.
    * `:backend` - compatibility override. When supplied, overrides the
      backend derived from `:runtime_profile`.
    * `:require_cuda` - compatibility override. When supplied, overrides the
      profile's `require_cuda?` flag.

  Backward compatibility: callers that pass only `:backend` and
  `:require_cuda` continue to behave exactly as before — the defaults match the
  previous CUDA-shaped defaults.
  """
  @spec load(keyword()) :: {:ok, t()} | {:error, term()}
  def load(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        artifact_dir: Artifact.default_output_dir(),
        num_roles: 3,
        runtime_profile: :cuda_exla,
        backend: nil,
        require_cuda: nil
      )

    profile = RuntimeProfile.resolve(opts[:runtime_profile])
    backend = opts[:backend] || profile.nx_backend

    require_cuda? =
      case opts[:require_cuda] do
        nil -> profile.require_cuda?
        b when is_boolean(b) -> b
      end

    if require_cuda? do
      Runtime.put_cuda_backend!()
    end

    slm_profile =
      SLMProfile.qwen_coordinator()
      |> Map.put(:adapted_artifact_dir, opts[:artifact_dir])
      |> Map.put(:artifact_patch_options,
        patch_router_head: false,
        allow_incomplete: false,
        cast_tensors: true
      )

    with {:ok, {model_info, tokenizer}} <- SLMProfile.load_profile(slm_profile),
         {:ok, manifest} <- Artifact.load_manifest(opts[:artifact_dir]),
         head_weights <- Artifact.load_router_head!(opts[:artifact_dir], manifest: manifest),
         {:ok, head_state} <-
           Head.build_routing_state(head_weights,
             num_roles: opts[:num_roles],
             backend: backend
           ),
         :ok <-
           Head.assert_shape_invariants!(head_state, manifest) do
      {:ok,
       %{
         model_info: model_info,
         tokenizer: tokenizer,
         routing_model: head_state.model,
         routing_params: head_state.params,
         manifest: manifest,
         artifact_dir: opts[:artifact_dir],
         num_agents: head_state.num_agents,
         num_roles: head_state.num_roles,
         hidden_size: head_state.hidden_size,
         backend: backend,
         runtime_profile: profile
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      {:error, {:coordinator_load_error, Exception.message(e)}}
  end

  @doc """
  Extracts a vector with the adapted model and routes it through the artifact head.
  """
  def route_messages(%{} = coordinator, messages) when is_list(messages) do
    with {:ok, extraction} <-
           Extractor.extract_penultimate_hidden_state_with_metadata(
             coordinator.model_info,
             coordinator.tokenizer,
             messages
           ) do
      # EXLA may donate the route input during the head forward pass. Keep a
      # host snapshot for router trace diagnostics before passing the CUDA
      # tensor into Axon.
      vector_snapshot = Nx.backend_transfer(extraction.vector, Nx.BinaryBackend)

      route_input =
        if coordinator.backend do
          Nx.backend_transfer(vector_snapshot, coordinator.backend)
        else
          vector_snapshot
        end

      route =
        TrinityCoordinator.CoordinationHead.route(
          coordinator.routing_model,
          coordinator.routing_params,
          route_input,
          coordinator.num_agents,
          coordinator.num_roles
        )

      extraction =
        extraction
        |> Map.put(:vector, route_input)
        |> Map.put(:vector_snapshot, vector_snapshot)

      {:ok, %{extraction: extraction, route: route}}
    end
  end
end
