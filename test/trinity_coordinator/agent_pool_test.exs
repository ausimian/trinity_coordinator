defmodule TrinityCoordinator.AgentPoolTest do
  use ExUnit.Case
  alias TrinityCoordinator.AgentPool

  test "calls agent and returns response" do
    messages = [%{role: "user", content: "Hi"}]
    {:ok, response} = AgentPool.call_agent(0, messages)

    assert response == "This is a mocked response from gpt-4."
  end

  test "verifier agent can return ACCEPT or REVISE" do
    messages = [
      %{role: "system", content: "Check the current solution. Output ACCEPT..."},
      %{role: "user", content: "Hi"}
    ]

    {:ok, response} = AgentPool.call_agent(1, messages)

    assert response in ["ACCEPT", "REVISE"]
  end
end
