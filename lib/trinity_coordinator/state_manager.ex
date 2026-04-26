defmodule TrinityCoordinator.StateManager do
  @moduledoc """
  Manages the state of the conversation transcript for the TRINITY framework.
  Stores messages in a format compatible with HuggingFace chat templates.
  """
  use Agent

  def start_link(initial_messages \\ []) do
    Agent.start_link(fn -> initial_messages end)
  end

  def get_messages(pid) do
    Agent.get(pid, & &1)
  end

  def append_message(pid, role, content) do
    Agent.update(pid, fn messages ->
      messages ++ [%{role: role, content: content}]
    end)
  end

  def append_user(pid, content), do: append_message(pid, "user", content)
  def append_assistant(pid, content), do: append_message(pid, "assistant", content)
  def append_system(pid, content), do: append_message(pid, "system", content)
end
