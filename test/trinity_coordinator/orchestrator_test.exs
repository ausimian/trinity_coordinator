defmodule TrinityCoordinator.OrchestratorTest do
  use ExUnit.Case
  alias TrinityCoordinator.{CoordinationHead, Extractor, Orchestrator, Runtime, StateManager}

  test "requires an SLM context before routing" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

    assert Orchestrator.run_loop(pid, model, params, max_turns: 3) ==
             {:error, :missing_slm_context}
  end

  @tag :integration
  test "runs real SLM extraction and router forward pass before provider dispatch" do
    Runtime.put_cuda_backend!()

    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(32, 5, 3)
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
        max_turns: 1,
        num_agents: 5,
        num_roles: 3,
        slm_context: {model_info, tokenizer},
        agent_pool_opts: [openai_api_key: nil]
      )

    assert result == {:error, :missing_openai_api_key}
  end
end
