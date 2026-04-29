defmodule TrinityCoordinator.OrchestratorMockTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.{CoordinationHead, Orchestrator, StateManager}

  test "runs verifier ACCEPT path through mock provider without live credentials" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 0.0, 10.0])

    extractor_fn = fn _messages, _slm_context ->
      {:ok,
       %{
         vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
         vector_shape: {1, input_dim},
         hidden_state_shape: {1, 2, input_dim},
         input_shapes: %{"input_ids" => {1, 2}}
       }}
    end

    mock_agent_fn = fn :verifier, messages, metadata ->
      assert metadata.agent_id == 0
      assert hd(messages).role == "system"
      {:ok, "ACCEPT: smoke-test verifier accepted."}
    end

    {:ok, pid} =
      StateManager.start_link([
        %{role: "user", content: "check this"},
        %{role: "assistant", content: "Candidate answer ready for verification."}
      ])

    assert {:ok, "ACCEPT: smoke-test verifier accepted."} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 3,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock
             )

    messages = StateManager.get_messages(pid)
    assert List.last(messages).content =~ "ACCEPT"
  end

  test "mock worker path executes provider turn and then reaches max turns" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 10.0, 0.0, 0.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    counter = :counters.new(1, [])

    mock_agent_fn = fn :worker, _messages ->
      :counters.add(counter, 1, 1)
      {:ok, "Result: one worker turn."}
    end

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "do work"}])

    assert {:ok, "Result: one worker turn."} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock
             )

    assert :counters.get(counter, 1) == 1
  end

  test "thinker suggestion overrides the next raw route and max turns returns latest worker result" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 10.0, 0.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.broadcast(0.0, {1, input_dim}),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    parent = self()

    mock_agent_fn = fn
      :thinker, _messages, _metadata ->
        send(parent, {:role_called, :thinker})

        {:ok,
         """
         <suggestion>Ask the solver for the concrete answer.</suggestion>
         <suggested_role>solver</suggested_role>
         """}

      :worker, _messages, _metadata ->
        send(parent, {:role_called, :worker})
        {:ok, "Result: suggested worker produced 42."}
    end

    trace_path =
      Path.join(System.tmp_dir!(), "trinity_thinker_#{System.unique_integer([:positive])}.jsonl")

    on_exit(fn -> File.rm(trace_path) end)

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "solve"}])

    assert {:ok, "Result: suggested worker produced 42."} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 2,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock,
               trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: "thinker_unit"]
             )

    assert_receive {:role_called, :thinker}
    assert_receive {:role_called, :worker}

    route_events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)
      |> Enum.filter(&(&1["event"] == "route_selected"))

    assert Enum.map(route_events, & &1["selected_role"]) == ["Thinker", "Worker"]

    override_event = Enum.at(route_events, 1)
    assert override_event["role_override_from_thinker"] == true
    assert override_event["raw_selected_role"] == "Thinker"
  end

  test "verifier before any worker response terminates explicitly without dispatch" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 0.0, 10.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.broadcast(0.0, {1, input_dim}),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    mock_agent_fn = fn _role, _messages, _metadata ->
      flunk("verifier-before-worker must not dispatch a provider")
    end

    trace_path =
      Path.join(
        System.tmp_dir!(),
        "trinity_early_verifier_#{System.unique_integer([:positive])}.jsonl"
      )

    on_exit(fn -> File.rm(trace_path) end)

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "check"}])

    assert {:error, :verifier_before_worker_response} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 3,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock,
               trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: "early_verifier_unit"]
             )

    events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)

    assert Enum.any?(
             events,
             &(&1["event"] == "route_selected" and &1["selected_role"] == "Verifier")
           )

    assert Enum.any?(
             events,
             &(&1["event"] == "run_failed" and &1["reason"] == "verifier_before_worker_response")
           )

    refute Enum.any?(
             events,
             &(&1["event"] == "provider_called" and &1["status"] in ["started", "ok"])
           )
  end

  test "provider failure is traced and never becomes a successful pass" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 10.0, 0.0, 0.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.broadcast(0.0, {1, input_dim}),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    mock_agent_fn = fn :worker, _messages, _metadata -> {:error, :mock_provider_failed} end

    trace_path =
      Path.join(
        System.tmp_dir!(),
        "trinity_provider_error_#{System.unique_integer([:positive])}.jsonl"
      )

    on_exit(fn -> File.rm(trace_path) end)

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "do work"}])

    assert {:error, :mock_provider_failed} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock,
               trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: "provider_error_unit"]
             )

    events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)

    assert Enum.any?(events, &(&1["event"] == "provider_called" and &1["status"] == "started"))

    error_event =
      Enum.find(events, &(&1["event"] == "provider_called" and &1["status"] == "error"))

    assert is_map(error_event)
    assert is_integer(error_event["provider_latency_ms"])

    assert Enum.any?(
             events,
             &(&1["event"] == "run_failed" and &1["reason"] == "mock_provider_failed")
           )

    refute Enum.any?(events, &(&1["event"] == "run_completed"))
  end

  test "default runtime role order preserves imported Python checkpoint order" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 10.0, 0.0, 0.0])

    extractor_fn = fn _messages ->
      %{
        vector: Nx.broadcast(0.0, {1, input_dim}),
        vector_shape: {1, input_dim},
        hidden_state_shape: {1, 2, input_dim},
        input_shapes: %{}
      }
    end

    mock_agent_fn = fn role, _messages, metadata ->
      assert role == :worker
      assert metadata.role_name == "Worker"
      assert metadata.agent_id == 0
      {:ok, "raw role 0 reached Worker/solver compatibility path"}
    end

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "do work"}])

    assert {:ok, "raw role 0 reached Worker/solver compatibility path"} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock
             )
  end

  test "mock loop emits extraction route provider and verifier trace events" do
    input_dim = 4
    num_agents = 1
    num_roles = 3

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())
    params = force_role_bias(params, [0.0, 0.0, 0.0, 10.0])

    extractor_fn = fn _messages, _slm_context ->
      {:ok,
       %{
         vector: Nx.tensor([[0.0, 0.0, 0.0, 0.0]], type: :f32),
         vector_shape: {1, input_dim},
         hidden_state_shape: {1, 2, input_dim},
         input_shapes: %{"input_ids" => {1, 2}}
       }}
    end

    mock_agent_fn = fn :verifier, _messages, _metadata ->
      {:ok, "ACCEPT: trace-backed verifier accepted."}
    end

    trace_path =
      Path.join(System.tmp_dir!(), "trinity_trace_#{System.unique_integer([:positive])}.jsonl")

    on_exit(fn -> File.rm(trace_path) end)

    {:ok, pid} =
      StateManager.start_link([
        %{role: "user", content: "check this"},
        %{role: "assistant", content: "Candidate answer ready for trace verification."}
      ])

    assert {:ok, _} =
             Orchestrator.run_loop(pid, model, params,
               max_turns: 1,
               num_agents: num_agents,
               num_roles: num_roles,
               slm_context: :mock_context,
               extractor_fn: extractor_fn,
               mock_agent_fn: mock_agent_fn,
               provider_pool: :mock,
               route_opts: [return_probabilities: true],
               trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: "unit", content: :hash]
             )

    events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)

    assert Enum.any?(events, &(&1["event"] == "slm_extracted"))

    route_event = Enum.find(events, &(&1["event"] == "route_selected"))
    assert route_event["agent_selection_mode"] == "argmax"
    assert route_event["role_selection_mode"] == "argmax"
    assert is_list(route_event["agent_probabilities"])
    assert is_list(route_event["role_probabilities"])

    provider_event =
      Enum.find(events, &(&1["event"] == "provider_called" and &1["status"] == "ok"))

    assert provider_event["provider_mode"] == "mock"
    assert provider_event["mock"] == true
    assert provider_event["selected_role_name"] == "Verifier"
    assert is_integer(provider_event["provider_latency_ms"])

    turn_event = Enum.find(events, &(&1["event"] == "turn_completed"))
    assert turn_event["verifier_parse_status"] == "accepted"
    assert turn_event["verifier_status"] == "accepted"
    assert turn_event["final"] == true
  end

  defp force_role_bias(%Axon.ModelState{} = params, values) do
    bias = Nx.tensor(values, type: :f32)
    kernel = Nx.broadcast(0.0, {4, length(values)})

    put_in(params.data["routing_head"], %{
      "kernel" => kernel,
      "bias" => bias
    })
  end
end
