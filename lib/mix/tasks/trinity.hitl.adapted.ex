defmodule Mix.Tasks.Trinity.Hitl.Adapted do
  @moduledoc """
  HITL gate: load the adapted Qwen coordinator, prove a selected tensor differs
  from the base Qwen tensor, and route a live hidden vector.

      XLA_TARGET=cuda12 mix trinity.hitl.adapted

  Requires a complete canonical artifact directory:
  `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Coordinator, SVD}

  @shortdoc "HITL adapted-Qwen coordinator route check"
  @default_compare_path "decoder.blocks.26.self_attention.query.kernel"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    {opts, rest, errors} = OptionParser.parse(args, strict: [compare_path: :string])
    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    compare_path = Keyword.get(opts, :compare_path, @default_compare_path)

    HITL.banner("TRINITY HITL ADAPTED COORDINATOR CHECK")
    Runtime.put_cuda_backend!()

    {:ok, {base_info, _base_tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    {:ok, coordinator} = Coordinator.load()

    HITL.kv("Artifact dir", coordinator.artifact_dir)
    HITL.kv("Artifact status", coordinator.manifest["status"])
    HITL.kv("Selected tensor count", coordinator.manifest["selected_tensor_count"])
    HITL.kv("Hidden size", coordinator.hidden_size)
    HITL.kv("Num agents", coordinator.num_agents)
    HITL.kv("Num roles", coordinator.num_roles)

    prove_tensor_patch!(base_info.params, coordinator.model_info.params, compare_path)

    {:ok, routed} =
      Coordinator.route_messages(coordinator, [
        %{"role" => "user", "content" => "Select a TRINITY role for this reasoning task."}
      ])

    HITL.ensure_shape!(routed.extraction.vector_shape, {1, 1_024}, "adapted Qwen vector")
    HITL.ensure_cuda_tensor!(routed.extraction.vector, "adapted Qwen vector")
    HITL.ensure_shape!(routed.route.logits, {1, 10}, "adapted route logits")
    HITL.ensure_cuda_tensor!(routed.route.logits, "adapted route logits")
    HITL.kv("Agent id", routed.route.agent_id)
    HITL.kv("Role id", routed.route.role_id)
    HITL.kv("Role name", HITL.role_name(routed.route.role_id))

    HITL.pass("TRINITY HITL ADAPTED COORDINATOR CHECK")
  end

  defp prove_tensor_patch!(base_params, adapted_params, path) do
    base_entries = SVD.flatten_tensor_entries(base_params) |> Map.new(&{&1.path, &1.tensor})
    adapted_entries = SVD.flatten_tensor_entries(adapted_params) |> Map.new(&{&1.path, &1.tensor})

    base = Map.fetch!(base_entries, path)
    adapted = Map.fetch!(adapted_entries, path)

    max_diff =
      Nx.subtract(Nx.as_type(adapted, :f32), Nx.as_type(base, :f32))
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    HITL.kv("Adapted tensor compare path", path)
    HITL.kv("Base tensor backend", Runtime.tensor_backend(base))
    HITL.kv("Adapted tensor backend", Runtime.tensor_backend(adapted))
    HITL.kv("Adapted tensor max_abs_diff", max_diff)

    unless max_diff > 0.0 do
      raise "adapted tensor equals base tensor at #{path}"
    end
  end
end
