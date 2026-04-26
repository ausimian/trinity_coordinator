defmodule TrinityCoordinator.AgentPool do
  @moduledoc """
  Provider dispatch for selected coordinator agents.

  The orchestrator depends on this module for real LLM calls.
  """

  @agents %{
    0 => %{provider: :openai, model: "gpt-4o-mini"},
    1 => %{provider: :openai, model: "claude-3-5-sonnet"},
    2 => %{provider: :openai, model: "gemini-1.5-flash"},
    3 => %{provider: :openai, model: "deepseek-chat"},
    4 => %{provider: :openai, model: "llama-3.1-70b"}
  }

  defstruct [:agent_id, :provider, :model, :messages, :response]

  @doc """
  Routes the message list to the mapped provider for the selected agent.
  """
  def call_agent(agent_id, messages, opts \\ []) do
    with {:ok, messages} <- normalize_messages(messages),
         {:ok, spec} <- fetch_agent_spec(agent_id),
         {:ok, adapter} <- adapter_for(spec.provider, opts),
         {:ok, response} <-
           adapter.call(spec, messages, opts) do
      {:ok, response}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp fetch_agent_spec(agent_id) when is_integer(agent_id) do
    case @agents[agent_id] do
      nil -> {:error, {:unknown_agent, agent_id}}
      spec -> {:ok, spec}
    end
  end

  defp adapter_for(provider, opts) do
    case Keyword.get(opts, :adapter) do
      nil -> adapter_from_provider(provider)
      adapter -> {:ok, adapter}
    end
  end

  defp adapter_from_provider(:openai), do: {:ok, TrinityCoordinator.AgentPool.OpenAI}
  defp adapter_from_provider(_), do: {:error, :unsupported_provider}

  defp normalize_messages(messages) when is_list(messages) do
    normalized =
      Enum.map(messages, fn message ->
        role = Map.get(message, :role, Map.get(message, "role"))
        content = Map.get(message, :content, Map.get(message, "content"))

    if is_binary(role) and is_binary(content) do
      %{role: role, content: content}
    else
      {:error, {:invalid_message, message}}
    end
      end)

    case Enum.find(normalized, &match?({:error, _}, &1)) do
      nil -> {:ok, normalized}
      {:error, reason} -> {:error, reason}
    end
  end

  defp normalize_messages(_), do: {:error, :invalid_messages}
end
