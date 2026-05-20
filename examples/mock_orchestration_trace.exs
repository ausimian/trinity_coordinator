defmodule Examples.MockOrchestrationTrace do
  @moduledoc false

  alias TrinityCoordinator.{HITL, Orchestrator, Runtime, StateManager, Trace}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator}

  @default_artifact_dir "tmp/sakana_parity/adapted_artifacts_from_python"
  @default_prompt "Select a TRINITY role for this reasoning task."
  @default_trace_path "tmp/examples/mock_orchestration_trace.jsonl"

  def main(argv) do
    Application.ensure_all_started(:trinity_coordinator)

    {opts, rest, errors} =
      argv
      |> normalize_argv()
      |> OptionParser.parse(
        strict: [
          artifact_dir: :string,
          max_turns: :integer,
          prompt: :string,
          trace_out: :string
        ]
      )

    unless rest == [], do: raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: raise("Invalid options: #{inspect(errors)}")

    context = %{
      artifact_dir: Keyword.get(opts, :artifact_dir, @default_artifact_dir),
      max_turns: Keyword.get(opts, :max_turns, 5),
      prompt: Keyword.get(opts, :prompt, @default_prompt),
      trace_path: Keyword.get(opts, :trace_out, @default_trace_path)
    }

    ensure_manifest!(Artifact.manifest_path(context.artifact_dir))
    run_example!(context)
  end

  defp normalize_argv(["--" | rest]), do: rest
  defp normalize_argv(argv), do: argv

  defp ensure_manifest!(manifest_path) do
    unless File.exists?(manifest_path) do
      raise """
      Missing adapted artifact manifest: #{manifest_path}

      Build or import the canonical artifact bundle first, then rerun:

          XLA_TARGET=cuda12 mix run examples/mock_orchestration_trace.exs -- \\
            --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
      """
    end
  end

  defp run_example!(context) do
    HITL.banner("TRINITY EXAMPLE: MOCK ORCHESTRATION TRACE")
    Runtime.put_cuda_backend!()
    File.mkdir_p!(Path.dirname(context.trace_path))
    File.rm(context.trace_path)

    coordinator =
      TrinityCoordinator.MixHelpers.load_coordinator!(artifact_dir: context.artifact_dir)

    {:ok, pid} = StateManager.start_link([%{role: "user", content: context.prompt}])

    result =
      Orchestrator.run_loop(
        pid,
        coordinator.routing_model,
        coordinator.routing_params,
        max_turns: context.max_turns,
        num_agents: coordinator.num_agents,
        num_roles: coordinator.num_roles,
        slm_context: {coordinator.model_info, coordinator.tokenizer},
        mock_agent_fn: &mock_agent/3,
        provider_pool: :mock,
        trace: [
          enabled: true,
          sink: {:jsonl, context.trace_path},
          run_id: "examples_mock_orchestration",
          content: :hash
        ]
      )

    print_report(context, coordinator, result)
  end

  defp mock_agent(:worker, messages, metadata) do
    IO.puts(
      "mock_provider role=Worker agent_id=#{metadata.agent_id} messages=#{length(messages)}"
    )

    {:ok, "Result: 6 * 7 = 42."}
  end

  defp mock_agent(:thinker, messages, metadata) do
    IO.puts(
      "mock_provider role=Thinker agent_id=#{metadata.agent_id} messages=#{length(messages)}"
    )

    {:ok,
     "<suggestion>Ask the solver for the concrete answer.</suggestion><suggested_role>solver</suggested_role>"}
  end

  defp mock_agent(:verifier, messages, metadata) do
    IO.puts(
      "mock_provider role=Verifier agent_id=#{metadata.agent_id} messages=#{length(messages)}"
    )

    {:ok, "ACCEPT: mock verifier accepted the latest Worker response."}
  end

  defp print_report(context, coordinator, result) do
    events = read_trace_events(context.trace_path)
    manifest_path = Artifact.manifest_path(context.artifact_dir)
    manifest_hash = Artifact.file_sha256!(manifest_path)

    IO.puts("""

    Input
      prompt: #{context.prompt}
      transcript_hash: #{Trace.Hash.messages([%{role: "user", content: context.prompt}])}

    Artifact
      dir: #{context.artifact_dir}
      manifest_path: #{manifest_path}
      manifest_sha256: #{manifest_hash}
      status: #{coordinator.manifest["status"]}
      layout: #{coordinator.manifest["artifact_layout"]}

    Run
      max_turns: #{context.max_turns}
      provider_mode: mock
      result: #{inspect(result)}
      trace_path: #{context.trace_path}
      event_count: #{length(events)}

    Trace Summary
    #{trace_summary(events)}

    Boundary
      live_provider_calls: none
      purpose: prove the adapted coordinator can drive the orchestrator, role injection, mock provider boundary, verifier termination, and JSONL trace persistence.
    """)
  end

  defp read_trace_events(path) do
    path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(&Jason.decode!/1)
  end

  defp trace_summary(events) do
    events
    |> Enum.map(&summarize_event/1)
    |> Enum.reject(&is_nil/1)
    |> Enum.map_join("\n", &("  " <> &1))
  end

  defp summarize_event(%{"event" => "run_started"} = event) do
    "run_started provider_mode=#{event["provider_mode"]} agents=#{event["num_agents"]} roles=#{event["num_roles"]}"
  end

  defp summarize_event(%{"event" => "slm_extracted"} = event) do
    "turn=#{event["turn"]} slm_extracted vector_shape=#{inspect(event["vector_shape"])} backend=#{event["vector_backend"]}"
  end

  defp summarize_event(%{"event" => "route_selected"} = event) do
    "turn=#{event["turn"]} route_selected agent=#{event["selected_agent"]} role=#{event["selected_role"]} raw_role=#{event["raw_selected_role"]} override=#{event["role_override_from_thinker"]}"
  end

  defp summarize_event(%{"event" => "provider_called", "status" => "started"} = event) do
    "turn=#{event["turn"]} provider_started mode=#{event["provider_mode"]} role=#{event["selected_role_name"]}"
  end

  defp summarize_event(%{"event" => "provider_called", "status" => status} = event) do
    response = event["response_hash"] || event["error"]

    "turn=#{event["turn"]} provider_#{status} latency_ms=#{event["provider_latency_ms"]} response_hash=#{response}"
  end

  defp summarize_event(%{"event" => "turn_completed"} = event) do
    "turn=#{event["turn"]} turn_completed verifier=#{event["verifier_status"]} final=#{event["final"]}"
  end

  defp summarize_event(%{"event" => "run_completed"} = event) do
    "run_completed status=#{event["final_status"]} response_hash=#{event["response_hash"]}"
  end

  defp summarize_event(%{"event" => "run_failed"} = event) do
    "run_failed reason=#{event["reason"]}"
  end

  defp summarize_event(_event), do: nil
end

Examples.MockOrchestrationTrace.main(System.argv())
