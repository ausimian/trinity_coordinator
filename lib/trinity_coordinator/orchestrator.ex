defmodule TrinityCoordinator.Orchestrator do
  @moduledoc """
  The main loop that orchestrates the TRINITY framework.
  """
  alias TrinityCoordinator.{
    AgentPool,
    CoordinationHead,
    Extractor,
    RoleInjector,
    StateManager
  }

  @roles %{0 => "Thinker", 1 => "Worker", 2 => "Verifier"}

  @doc """
  Runs the multi-agent loop until ACCEPT is reached or max_turns is exceeded.
  """
  def run_loop(pid, params, max_turns \\ 5) do
    do_run_loop(pid, params, 0, max_turns, nil)
  end

  @doc """
  Runs the multi-agent loop with a preloaded SLM model/tokenizer context.
  """
  def run_loop(pid, params, max_turns, slm_context) do
    do_run_loop(pid, params, 0, max_turns, slm_context)
  end

  defp do_run_loop(_pid, _params, turns, max_turns, _slm_context) when turns >= max_turns do
    {:error, :max_turns_reached}
  end

  defp do_run_loop(pid, params, turns, max_turns, slm_context) do
    messages = StateManager.get_messages(pid)

    tensor = maybe_extract_tensor(messages, slm_context)

    {agent_id, role_id} = CoordinationHead.forward(params, tensor)
    role_name = Map.get(@roles, role_id, "Worker")

    injected_messages = RoleInjector.inject_role(messages, role_name)

    {:ok, response_text} = AgentPool.call_agent(agent_id, injected_messages)

    StateManager.append_assistant(pid, response_text)

    if role_name == "Verifier" and response_text == "ACCEPT" do
      {:ok, response_text}
    else
      do_run_loop(pid, params, turns + 1, max_turns, slm_context)
    end
  end

  defp maybe_extract_tensor(_messages, nil), do: Nx.broadcast(0.5, {1, 1024})

  defp maybe_extract_tensor(messages, {model_info, tokenizer})
       when is_map(model_info) and is_map(tokenizer) do
    case Extractor.extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages) do
      {:ok, tensor} -> tensor
      _ -> Nx.broadcast(0.5, {1, 1024})
    end
  end

  defp maybe_extract_tensor(messages, %{model_info: model_info, tokenizer: tokenizer}) do
    case Extractor.extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages) do
      {:ok, tensor} -> tensor
      _ -> Nx.broadcast(0.5, {1, 1024})
    end
  end

  defp maybe_extract_tensor(_messages, _), do: Nx.broadcast(0.5, {1, 1024})
end
