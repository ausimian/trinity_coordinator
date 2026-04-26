defmodule TrinityCoordinator.OrchestratorTest do
  use ExUnit.Case
  alias TrinityCoordinator.{CoordinationHead, Extractor, Orchestrator, StateManager}

  defmodule TestAgentAdapter do
    @behaviour TrinityCoordinator.AgentPool.Adapter

    @impl true
    def call(_spec, messages, _opts) do
      if has_verifier_prompt?(messages) do
        {:ok, "ACCEPT"}
      else
        {:ok, "partial progress"}
      end
    end

    defp has_verifier_prompt?(messages) do
      Enum.any?(messages, fn message ->
        role = message[:role]
        content = message[:content] || message["content"] || ""
        role == "system" and String.contains?(content, "Check the current solution")
      end)
    end
  end

  test "runs loop until max turns when verifier never accepts" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

    assert Orchestrator.run_loop(pid, model, params,
      max_turns: 3,
      agent_pool_opts: [adapter: TestAgentAdapter]
    ) == {:error, :missing_slm_context}
  end

  @tag :integration
  test "runs loop with a real tiny SLM extraction context" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(32, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())
    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    result =
      Orchestrator.run_loop(
        pid,
        model,
        params,
        max_turns: 5,
        slm_context: {model_info, tokenizer},
        agent_pool_opts: [adapter: TestAgentAdapter]
      )

    assert result in [{:ok, "ACCEPT"}, {:error, :max_turns_reached}]
  end
end
