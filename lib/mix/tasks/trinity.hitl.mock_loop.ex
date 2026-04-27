defmodule Mix.Tasks.Trinity.Hitl.MockLoop do
  @moduledoc """
  HITL gate: run the adapted coordinator through the orchestrator with mocked LLM calls.

      XLA_TARGET=cuda12 mix trinity.hitl.mock_loop --trace-out tmp/trinity_mock_trace.jsonl

  This intentionally performs no live provider calls. By default it writes a
  hash-redacted JSONL trace and validates that the trace file exists.
  """

  use Mix.Task

  alias TrinityCoordinator.{HITL, Orchestrator, StateManager}
  alias TrinityCoordinator.Sakana.Coordinator

  @shortdoc "HITL adapted coordinator mock-orchestrator check"
  @default_trace_path "tmp/trinity_mock_trace.jsonl"

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [trace_out: :string, trace_content: :string, run_id: :string]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    trace_path = Keyword.get(opts, :trace_out, @default_trace_path)
    trace_content = parse_trace_content(Keyword.get(opts, :trace_content, "hash"))
    run_id = Keyword.get(opts, :run_id, "hitl_mock")

    File.mkdir_p!(Path.dirname(trace_path))
    File.rm(trace_path)

    HITL.banner("TRINITY HITL MOCK ORCHESTRATOR LOOP")
    HITL.kv("Trace path", trace_path)
    HITL.kv("Trace content", trace_content)

    {:ok, coordinator} = Coordinator.load()

    {:ok, pid} =
      StateManager.start_link([
        %{role: "user", content: "Solve a tiny arithmetic task: compute 6 * 7."}
      ])

    turn_counter = :counters.new(1, [])

    mock_agent_fn = fn role, messages, metadata ->
      :counters.add(turn_counter, 1, 1)
      turn = :counters.get(turn_counter, 1)

      HITL.kv("Mock turn #{turn}", %{
        role: role,
        agent_id: metadata.agent_id,
        messages: length(messages)
      })

      case role do
        :verifier -> {:ok, "ACCEPT: The current answer is complete enough for the smoke test."}
        :thinker -> {:ok, "Plan: multiply 6 by 7 and ask a verifier to check it."}
        :worker -> {:ok, "Result: 6 * 7 = 42."}
        _ -> {:ok, "Proceed."}
      end
    end

    result =
      Orchestrator.run_loop(
        pid,
        coordinator.routing_model,
        coordinator.routing_params,
        max_turns: 5,
        num_agents: coordinator.num_agents,
        num_roles: coordinator.num_roles,
        slm_context: {coordinator.model_info, coordinator.tokenizer},
        mock_agent_fn: mock_agent_fn,
        provider_pool: :mock,
        trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: run_id, content: trace_content]
      )

    turns = :counters.get(turn_counter, 1)
    HITL.kv("Mock turns executed", turns)
    HITL.kv("Loop result", result)
    HITL.kv("Trace path", trace_path)

    unless turns > 0 do
      raise "mock loop did not execute any provider turn"
    end

    unless File.exists?(trace_path) do
      raise "trace file was not written: #{trace_path}"
    end

    trace_events = trace_event_names(trace_path)
    HITL.kv("Trace events", trace_events)

    unless Enum.member?(trace_events, "slm_extracted") and
             Enum.member?(trace_events, "route_selected") do
      raise "trace file did not include extraction and route events"
    end

    case result do
      {:ok, _response} ->
        HITL.kv("Termination", "Verifier ACCEPT")

      {:error, :max_turns_reached} ->
        HITL.kv("Termination", "max_turns_reached after successful mock dispatch")

      {:error, reason} ->
        raise "mock loop failed: #{inspect(reason)}"
    end

    HITL.pass("TRINITY HITL MOCK ORCHESTRATOR LOOP")
  end

  defp parse_trace_content("full"), do: :full
  defp parse_trace_content(:full), do: :full
  defp parse_trace_content(_), do: :hash

  defp trace_event_names(path) do
    path
    |> File.read!()
    |> String.split("\n", trim: true)
    |> Enum.map(fn line ->
      case Jason.decode(line) do
        {:ok, %{"event" => event}} -> event
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end
end
