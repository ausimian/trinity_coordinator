defmodule TrinityCoordinator.OrchestratorBudgetsTest do
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

  test "max_provider_calls budget terminates loop early with structured error", %{
    pid: pid,
    routing: {model, params}
  } do
    result =
      Orchestrator.run_loop(pid, model, params,
        max_turns: 50,
        extractor_fn: &mock_extractor/2,
        mock_agent_fn: &mock_runaway/2,
        num_agents: 2,
        num_roles: 3,
        max_provider_calls: 2
      )

    assert {:error, {:budget_exceeded, :provider_calls, details}} = result
    assert details.limit == 2
    assert details.observed >= 2
    assert details.checkpoint == :before_dispatch
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
end
