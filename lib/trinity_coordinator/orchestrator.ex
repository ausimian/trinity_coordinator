defmodule TrinityCoordinator.Orchestrator do
  @moduledoc """
  Orchestrates a real TRINITY multi-turn routing loop.
  """
  alias TrinityCoordinator.{
    AgentPool,
    CoordinationHead,
    Extractor,
    RoleInjector,
    StateManager
  }

  @roles %{0 => "Thinker", 1 => "Worker", 2 => "Verifier"}
  @default_max_turns 5

  @doc """
  Run loop with keyword options:

  - `:max_turns` – stop after this many turns if no termination.
  - `:slm_context` – `{model_info, tokenizer}` for real extraction.
  - `:stop_token` – verifier termination token (default `"ACCEPT"`).
  - `:agent_pool_opts` – custom options passed through to `AgentPool`.
  - `:roles` – optional role-map for index->name decoding.
  - `:num_agents` – number of agent logits in the coordination head.
  - `:num_roles` – number of role logits in the coordination head.
  """
  def run_loop(pid, model, params, opts \\ [])

  def run_loop(pid, model, params, opts) when is_list(opts) do
    max_turns = Keyword.get(opts, :max_turns, @default_max_turns)
    slm_context = Keyword.get(opts, :slm_context)
    stop_token = Keyword.get(opts, :stop_token, "ACCEPT")
    roles = Keyword.get(opts, :roles, @roles)
    agent_pool_opts = Keyword.get(opts, :agent_pool_opts, [])
    num_agents = Keyword.get(opts, :num_agents, AgentPool.agent_count())
    num_roles = Keyword.get(opts, :num_roles, 3)

    case validate_loop_input(pid, model, params) do
      {:ok, _} ->
        do_run_loop(
          pid,
          model,
          params,
          0,
          max_turns,
          slm_context,
          %{
            roles: roles,
            stop_token: stop_token,
            agent_pool_opts: agent_pool_opts,
            num_agents: num_agents,
            num_roles: num_roles
          }
        )

      error ->
        error
    end
  end

  def run_loop(pid, model, params, max_turns) when is_integer(max_turns) do
    run_loop(pid, model, params, max_turns: max_turns)
  end

  def run_loop(pid, model, params, max_turns, slm_context) when is_integer(max_turns) do
    run_loop(pid, model, params,
      max_turns: max_turns,
      slm_context: slm_context
    )
  end

  defp validate_loop_input(pid, model, params) do
    cond do
      not is_pid(pid) -> {:error, :invalid_state_pid}
      model == nil -> {:error, :invalid_model}
      params == nil -> {:error, :invalid_params}
      true -> {:ok, :ok}
    end
  end

  defp do_run_loop(_pid, _model, _params, turn, max_turns, _slm_context, _ctx)
       when turn >= max_turns do
    {:error, :max_turns_reached}
  end

  defp do_run_loop(
         pid,
         model,
         params,
         turn,
         max_turns,
         slm_context,
         ctx
       ) do
    messages = StateManager.get_messages(pid)

    with {:ok, penultimate} <- extract_router_tensor(messages, slm_context),
         {agent_id, role_id} <-
           CoordinationHead.forward(model, params, penultimate, ctx.num_agents, ctx.num_roles),
         role_name when is_binary(role_name) <- Map.get(ctx.roles, role_id, "Worker"),
         injected_messages <- RoleInjector.inject_role(messages, role_name),
         {:ok, response_text} <-
           AgentPool.call_agent(agent_id, injected_messages, ctx.agent_pool_opts) do
      StateManager.append_assistant(pid, response_text)

      if verifier_accept?(role_name, response_text, ctx.stop_token) do
        {:ok, response_text}
      else
        do_run_loop(pid, model, params, turn + 1, max_turns, slm_context, ctx)
      end
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, :unexpected_orchestrator_state}
    end
  end

  defp verifier_accept?(role_name, response_text, stop_token) when role_name == "Verifier" do
    response_text
    |> String.trim()
    |> String.upcase()
    |> String.starts_with?(String.upcase(stop_token))
  end

  defp verifier_accept?(_role_name, _response_text, _stop_token), do: false

  defp extract_router_tensor(_messages, nil), do: {:error, :missing_slm_context}

  defp extract_router_tensor(messages, {model_info, tokenizer}) do
    Extractor.extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages)
  end

  defp extract_router_tensor(messages, %{model_info: model_info, tokenizer: tokenizer}) do
    extract_router_tensor(messages, {model_info, tokenizer})
  end

  defp extract_router_tensor(_messages, _context), do: {:error, :invalid_slm_context}
end
