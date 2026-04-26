defmodule TrinityCoordinator.StateManager do
  @moduledoc """
  Manages the state of the conversation transcript for the TRINITY framework.
  Stores messages in a format compatible with HuggingFace chat templates.
  """
  use Agent

  @type role :: String.t()
  @type message :: %{role: role, content: String.t()}
  @type messages :: [message()]

  @allowed_roles ["system", "user", "assistant"]

  @doc """
  Starts the state store with an initial transcript.
  """
  def start_link(initial_messages \\ []) do
    Agent.start_link(fn -> normalize_messages!(initial_messages) end)
  end

  @doc """
  Returns the current transcript.
  """
  def get_messages(pid) do
    Agent.get(pid, & &1)
  end

  @doc """
  Appends a message to the transcript.
  """
  def append_message(pid, role, content) do
    Agent.update(pid, fn messages ->
      messages ++ [normalize_message!(%{role: role, content: content})]
    end)
  end

  def append_user(pid, content), do: append_message(pid, "user", content)
  def append_assistant(pid, content), do: append_message(pid, "assistant", content)
  def append_system(pid, content), do: append_message(pid, "system", content)

  defp normalize_messages!(messages) when is_list(messages) do
    Enum.map(messages, &normalize_message!/1)
  end

  defp normalize_messages!(_messages) do
    raise ArgumentError, "messages must be a list of maps"
  end

  defp normalize_message!(%{role: role, content: content}) do
    validate_message!(%{role: role, content: content}, "role/content keys")
  end

  defp normalize_message!(%{"role" => role, "content" => content}) do
    validate_message!(%{role: role, content: content}, "string role/content keys")
  end

  defp normalize_message!(message) do
    raise ArgumentError, "unsupported message format: #{inspect(message)}"
  end

  defp validate_message!(%{role: role, content: content}, origin) do
    unless is_binary(role) and is_binary(content) do
      raise ArgumentError,
            "invalid message #{origin}: role and content must be strings, got #{inspect(%{role: role, content: content})}"
    end

    unless role in @allowed_roles do
      raise ArgumentError, "unsupported message role #{inspect(role)}"
    end

    %{role: role, content: content}
  end
end
