defmodule TrinityCoordinator.AgentPool do
  @moduledoc """
  Executes HTTP requests to the selected Agent (LLM).
  """

  @agents %{
    0 => "gpt-4",
    1 => "claude-3-5-sonnet",
    2 => "gemini-pro",
    3 => "deepseek-coder",
    4 => "llama-3-8b"
  }

  @doc """
  Mocks a call to an LLM provider based on the chosen agent_id.
  In a real scenario, this uses Req to hit the respective API.
  """
  def call_agent(agent_id, messages) do
    agent_name = Map.get(@agents, agent_id, "unknown-model")

    system_msg = Enum.find(messages, &(&1[:role] == "system" or &1.role == "system"))

    # If it's a Verifier, it outputs ACCEPT 30% of the time, else REVISE
    response_text =
      if system_msg && Map.get(system_msg, :content, "") =~ "ACCEPT" do
        if :rand.uniform() > 0.7, do: "ACCEPT", else: "REVISE"
      else
        "This is a mocked response from #{agent_name}."
      end

    {:ok, response_text}
  end
end
