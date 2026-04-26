defmodule TrinityCoordinator.OrchestratorTest do
  use ExUnit.Case
  alias TrinityCoordinator.{CoordinationHead, Extractor, Orchestrator, StateManager}

  test "runs loop until max turns or ACCEPT" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    model = CoordinationHead.build_model(1024, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 1024}, :f32), Axon.ModelState.empty())

    result = Orchestrator.run_loop(pid, params, 5)

    # Should either find ACCEPT naturally through the mocked verifier loop, or timeout
    assert result == {:ok, "ACCEPT"} or result == {:error, :max_turns_reached}
  end

  @tag :integration
  test "runs loop with a real tiny SLM extraction context" do
    {:ok, pid} = StateManager.start_link([%{role: "user", content: "Solve this"}])

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    model = CoordinationHead.build_model(32, 7, 3)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, 32}, :f32), Axon.ModelState.empty())

    result = Orchestrator.run_loop(pid, params, 5, {model_info, tokenizer})
    assert result in [{:ok, "ACCEPT"}, {:error, :max_turns_reached}]
  end
end
