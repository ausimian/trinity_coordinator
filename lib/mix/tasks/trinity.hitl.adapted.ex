defmodule Mix.Tasks.Trinity.Hitl.Adapted do
  @moduledoc """
  HITL gate: load the adapted Qwen coordinator, prove a selected tensor differs
  from the base Qwen tensor, and route a live hidden vector.

      XLA_TARGET=cuda12 mix trinity.hitl.adapted

  By default this uses the complete canonical artifact directory:
  `priv/sakana_trinity/adapted_qwen3_0_6b_layer26`.

  To validate a freshly imported Python semantic bundle before promoting it to
  `priv/`, pass:

      XLA_TARGET=cuda12 mix trinity.hitl.adapted \
        --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Runtime, SLMProfile}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator, SVD}

  @shortdoc "HITL adapted-Qwen coordinator route check"
  @default_compare_path "decoder.blocks.26.self_attention.query.kernel"
  @default_message "Select a TRINITY role for this reasoning task."

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    opts = parse_args!(args)

    HITL.banner("TRINITY HITL ADAPTED COORDINATOR CHECK")
    Runtime.put_cuda_backend!()

    {:ok, {base_info, _base_tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    {:ok, coordinator} = Coordinator.load(artifact_dir: opts.artifact_dir)

    HITL.kv("Artifact dir", coordinator.artifact_dir)
    HITL.kv("Artifact status", coordinator.manifest["status"])
    HITL.kv("Artifact layout", coordinator.manifest["artifact_layout"])
    HITL.kv("Artifact export complete", coordinator.manifest["export_complete"])
    HITL.kv("Selected tensor count", coordinator.manifest["selected_tensor_count"])

    HITL.kv(
      "Selected singular value count",
      coordinator.manifest["selected_singular_value_count"]
    )

    HITL.kv("Hidden size", coordinator.hidden_size)
    HITL.kv("Num agents", coordinator.num_agents)
    HITL.kv("Num roles", coordinator.num_roles)

    ensure_manifest_contract!(coordinator.manifest)
    prove_tensor_patch!(base_info.params, coordinator.model_info.params, opts.compare_path)

    {:ok, routed} =
      Coordinator.route_messages(coordinator, [
        %{"role" => "user", "content" => opts.message}
      ])

    HITL.ensure_shape!(routed.extraction.vector_shape, {1, 1_024}, "adapted Qwen vector")
    HITL.ensure_cuda_tensor!(routed.extraction.vector, "adapted Qwen vector")
    HITL.ensure_shape!(routed.route.logits, {1, 10}, "adapted route logits")
    HITL.ensure_cuda_tensor!(routed.route.logits, "adapted route logits")
    HITL.ensure_shape!(routed.route.agent_logits, {7}, "adapted agent logits")
    HITL.ensure_cuda_tensor!(routed.route.agent_logits, "adapted agent logits")
    HITL.ensure_shape!(routed.route.role_logits, {3}, "adapted role logits")
    HITL.ensure_cuda_tensor!(routed.route.role_logits, "adapted role logits")
    HITL.kv("Agent id", routed.route.agent_id)
    HITL.kv("Role id", routed.route.role_id)
    HITL.kv("Role name", HITL.role_name(routed.route.role_id))
    HITL.kv("Agent logits", HITL.short_logits(routed.route.agent_logits))
    HITL.kv("Role logits", HITL.short_logits(routed.route.role_logits))

    HITL.pass("TRINITY HITL ADAPTED COORDINATOR CHECK")
  end

  @doc false
  def parse_args!(args) do
    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          artifact_dir: :string,
          compare_path: :string,
          message: :string
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    %{
      artifact_dir: Keyword.get(opts, :artifact_dir, Artifact.default_output_dir()),
      compare_path: Keyword.get(opts, :compare_path, @default_compare_path),
      message: Keyword.get(opts, :message, @default_message)
    }
  end

  defp ensure_manifest_contract!(manifest) do
    HITL.assert!(manifest["status"] == "complete", {:invalid_artifact_status, manifest["status"]})
    HITL.assert!(manifest["export_complete"] == true, :artifact_export_incomplete)

    HITL.assert!(
      manifest["selected_tensor_count"] == 9,
      {:invalid_selected_tensor_count, manifest["selected_tensor_count"]}
    )

    HITL.assert!(
      manifest["selected_singular_value_count"] == 9_216,
      {:invalid_selected_singular_value_count, manifest["selected_singular_value_count"]}
    )
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
