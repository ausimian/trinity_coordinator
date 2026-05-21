defmodule TrinityCoordinator.OrchestratorBudgetsTest do
  require Logger

  @moduledoc """
  Phase 10 budget enforcement tests. Use mock extractor + mock provider so
  the loop runs without Qwen/CUDA.
  """
  use ExUnit.Case, async: false

  alias TrinityCoordinator.{CoordinationHead, Orchestrator, StateManager}

  alias TrinityCoordinator.Sakana.Head

  defp build_mock_routing do
    # Bias agent 0 + role 0 (Worker) so the loop dispatches to Worker first.
    # Build a (2 agents + 3 roles) = 5-output head with positive weights only
    # on the agent 0 and role 0 rows so argmax is deterministic.
    num_agents = 2
    num_roles = 3
    output_count = num_agents + num_roles
    hidden = 8

    weights =
      0..(output_count - 1)
      |> Enum.flat_map(fn row ->
        # Agent 0 (row 0) and Role 0 (row 2 = num_agents+0) get +10.0; others 0.0.
        bias_row =
          cond do
            row == 0 -> 10.0
            row == num_agents -> 10.0
            true -> 0.0
          end

        List.duplicate(bias_row, hidden)
      end)
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({output_count, hidden})

    model = CoordinationHead.build_model(hidden, num_agents, num_roles)
    {init_fn, _predict} = Axon.build(model)
    params = init_fn.(Nx.template({1, hidden}, :f32), Axon.ModelState.empty())

    params = Head.put_head_weights!(params, weights)
    {model, params}
  end

  defp mock_extractor(_messages, _ctx) do
    {:ok, %{vector: Nx.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], type: :f32)}}
  end

  # Provider that always says ACCEPT so verifier role terminates immediately.
  defp mock_acceptor(_role, _messages), do: {:ok, "ACCEPT"}

  # Provider that never accepts so the loop runs many turns.
  defp mock_runaway(_role, _messages), do: {:ok, "Worker answer: thinking..."}

  setup do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "test prompt"}])
    {:ok, pid: pid, routing: build_mock_routing()}
  end

  test "no budgets specified: existing behavior preserved (loop runs to verifier ACCEPT)", %{
    pid: pid,
    routing: {model, params}
  } do
    {result, _} =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 3,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: &mock_acceptor/2,
        num_agents: 2,
        num_roles: 3
      )
      |> case do
        {:ok, text} -> {:ok, text}
        other -> {other, nil}
      end

    # mock_acceptor returns ACCEPT; verifier role consumes it.
    assert result == :ok
  end

  test "max_provider_calls: 2 allows exactly two dispatches before aborting", %{
    pid: pid,
    routing: {model, params}
  } do
    counter = :counters.new(1, [:atomics])

    counted_runaway =
      counted_provider(counter, fn _role, _messages -> {:ok, "Worker answer: thinking..."} end)

    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 50,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: counted_runaway,
        num_agents: 2,
        num_roles: 3,
        max_provider_calls: 2
      )

    assert {:error, {:budget_exceeded, :provider_calls, details}} = result
    assert details.limit == 2
    # With observed > limit semantics: limit=2, observed=3 at fail time,
    # and exactly 2 actual provider dispatches happened before the abort.
    assert details.observed == 3
    assert details.checkpoint == :before_dispatch
    assert :counters.get(counter, 1) == 2
  end

  test "max_provider_calls: 1 allows exactly one dispatch", %{
    pid: pid,
    routing: {model, params}
  } do
    counter = :counters.new(1, [:atomics])

    counted_runaway =
      counted_provider(counter, fn _role, _messages -> {:ok, "Worker answer: thinking..."} end)

    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 50,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: counted_runaway,
        num_agents: 2,
        num_roles: 3,
        max_provider_calls: 1
      )

    assert {:error, {:budget_exceeded, :provider_calls, details}} = result
    assert details.limit == 1
    assert details.observed == 2
    assert :counters.get(counter, 1) == 1
  end

  test "max_wall_time_ms budget enforced (0ms forces immediate fail at turn_start)", %{
    pid: pid,
    routing: {model, params}
  } do
    # 0ms budget: elapsed_ms(t0) >= 0 is always true at turn_start, so the
    # very first budget check inside the loop fires before any role dispatch.
    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 50,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: &mock_runaway/2,
        num_agents: 2,
        num_roles: 3,
        max_wall_time_ms: 0
      )

    assert {:error, {:budget_exceeded, :wall_time, details}} = result
    assert details.limit_ms == 0
    assert details.checkpoint == :turn_start
  end

  test "check_budgets/3 directly: returns :ok when no budgets set" do
    run_ctx = %{
      budgets: %{
        max_wall_time_ms: nil,
        max_provider_calls: nil,
        max_verifier_revisions: nil,
        max_estimated_cost_usd: nil
      },
      counters: %{started_monotonic_ms: System.monotonic_time(:millisecond)}
    }

    assert :ok = Orchestrator.check_budgets(run_ctx, :test_checkpoint, %{})
  end

  # ---- Phase 11 helpers ----------------------------------------------------

  # Wraps an arity-2 mock provider so each call bumps the supplied counter.
  defp counted_provider(counter, fun) when is_function(fun, 2) do
    fn role, messages ->
      :counters.add(counter, 1, 1)
      fun.(role, messages)
    end
  end

  defp slow_provider(sleep_ms) do
    fn _role, _messages ->
      Process.sleep(sleep_ms)
      {:ok, "Worker answer: slow"}
    end
  end

  # ---- Phase 11.2: max_provider_latency_ms --------------------------------

  test "max_provider_latency_ms aborts after a slow dispatch", %{
    pid: pid,
    routing: {model, params}
  } do
    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 5,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: slow_provider(40),
        num_agents: 2,
        num_roles: 3,
        max_provider_latency_ms: 10
      )

    assert {:error, {:budget_exceeded, :provider_latency_ms, details}} = result
    assert details.limit_ms == 10
    assert details.observed_ms >= 40
    assert details.checkpoint == :after_dispatch
    assert details.turn == 0
  end

  test "max_provider_latency_ms is permissive when dispatch is fast", %{
    pid: pid,
    routing: {model, params}
  } do
    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 3,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: &mock_acceptor/2,
        num_agents: 2,
        num_roles: 3,
        max_provider_latency_ms: 10_000
      )

    # mock_acceptor accepts on first verifier; here agent 0/role 0 is biased to
    # Worker, so the loop reaches max_turns with the latest-worker-response
    # fallback. Either {:ok, _} or {:error, :max_turns_reached} is acceptable;
    # what matters is that the latency budget did NOT fire.
    refute match?({:error, {:budget_exceeded, :provider_latency_ms, _}}, result)
  end

  # ---- Phase 11.3: max_verifier_revisions ---------------------------------

  test "max_verifier_revisions aborts after the budgeted number of rejections", %{
    pid: pid
  } do
    # Build a routing that strongly biases role 2 (Verifier) so we exercise the
    # verifier-revision counter without depending on the Worker→Verifier dance.
    num_agents = 2
    num_roles = 3
    output_count = num_agents + num_roles
    hidden = 8

    weights =
      0..(output_count - 1)
      |> Enum.flat_map(fn row ->
        bias_row =
          cond do
            row == 0 -> 10.0
            # Bias role 2 = Verifier (row index = num_agents + 2).
            row == num_agents + 2 -> 10.0
            true -> 0.0
          end

        List.duplicate(bias_row, hidden)
      end)
      |> Nx.tensor(type: :f32)
      |> Nx.reshape({output_count, hidden})

    model = CoordinationHead.build_model(hidden, num_agents, num_roles)
    {init_fn, _predict} = Axon.build(model)
    params = init_fn.(Nx.template({1, hidden}, :f32), Axon.ModelState.empty())
    params = Head.put_head_weights!(params, weights)

    # Seed the state with a worker response so the verifier role is allowed to
    # dispatch on turn 0.
    StateManager.append_assistant(pid, "Worker answer: bad")

    # Mock provider returns a verifier rejection (no ACCEPT token).
    rejecting = fn _role, _messages -> {:ok, "Verifier review: not yet, please revise."} end

    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 20,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: rejecting,
        num_agents: num_agents,
        num_roles: num_roles,
        max_verifier_revisions: 2
      )

    assert {:error, {:budget_exceeded, :verifier_revisions, details}} = result
    assert details.limit == 2
    assert details.observed >= 3
    assert details.checkpoint == :after_verifier_revision
  end

  # ---- Phase 11.4: max_estimated_cost_usd ---------------------------------

  test "max_estimated_cost_usd fires when cost_estimator_fn is supplied", %{
    pid: pid,
    routing: {model, params}
  } do
    fixed_cost = fn _dispatch -> 0.50 end

    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 50,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: fn _role, _messages -> {:ok, "Worker answer: thinking..."} end,
        num_agents: 2,
        num_roles: 3,
        max_estimated_cost_usd: 1.0,
        cost_estimator_fn: fixed_cost
      )

    assert {:error, {:budget_exceeded, :estimated_cost_usd, details}} = result
    assert details.limit_usd == 1.0
    # With observed > limit and $0.50/call: after 3 dispatches observed=1.50;
    # 1.50 > 1.00 fires the abort.
    assert details.observed_usd >= 1.0
    assert details.checkpoint == :after_dispatch
  end

  test "max_estimated_cost_usd without cost_estimator_fn logs a warning once", %{
    pid: pid,
    routing: {model, params}
  } do
    import ExUnit.CaptureLog

    original_level = Logger.level()
    Logger.configure(level: :warning)
    on_exit(fn -> Logger.configure(level: original_level) end)

    log =
      capture_log(fn ->
        Orchestrator.run_loop(pid, model, params,
          max_turns: 3,
          extractor_fn: &mock_extractor/2,
          mock_agent_fn: fn _role, _messages -> {:ok, "Worker answer: thinking..."} end,
          num_agents: 2,
          num_roles: 3,
          max_estimated_cost_usd: 1.0
        )
      end)

    assert log =~ "max_estimated_cost_usd was set"
    assert log =~ "cost_estimator_fn"
    # One-shot: ensure the substring appears at most once across multiple turns.
    occurrences = log |> String.split("max_estimated_cost_usd was set") |> length()
    # split returns N+1 pieces for N matches.
    assert occurrences == 2
  end

  # ---- Phase 11.5: RouteDecision in :turn_completed -----------------------

  test "turn_completed events include a route_decision map", %{
    pid: pid,
    routing: {model, params}
  } do
    trace_path =
      Path.join(
        System.tmp_dir!(),
        "orch_budget_phase11_rd_" <> Integer.to_string(System.unique_integer([:positive]))
      )

    on_exit(fn -> File.rm(trace_path) end)

    Orchestrator.run_loop(pid, model, params,
      max_turns: 1,
      extractor_fn: &mock_extractor/2,
      mock_agent_fn: &mock_acceptor/2,
      num_agents: 2,
      num_roles: 3,
      trace: [enabled: true, sink: {:jsonl, trace_path}, run_id: "phase11_rd"]
    )

    events =
      trace_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.map(&Jason.decode!/1)

    turn_completed = Enum.find(events, &(&1["event"] == "turn_completed"))
    assert turn_completed != nil, "expected at least one turn_completed event"

    rd = turn_completed["route_decision"]
    assert is_map(rd)
    assert is_integer(rd["agent_id"])
    assert is_integer(rd["role_id"])
    assert is_binary(rd["role_name"])
    assert is_map(rd["margins"])
    assert is_map(rd["selection_modes"])
    assert is_binary(rd["transcript_hash"])
  end
end
