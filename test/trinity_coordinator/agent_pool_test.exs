defmodule TrinityCoordinator.AgentPoolTest do
  use ExUnit.Case
  alias TrinityCoordinator.AgentPool

  test "exposes the default seven-agent provider pool" do
    assert AgentPool.agent_count() == 7
    assert map_size(AgentPool.agent_specs()) == 7
  end

  test "routes known agents to the real provider boundary and requires credentials" do
    messages = [%{role: "user", content: "Hi"}]

    assert {:error, :missing_openai_api_key} =
             AgentPool.call_agent(0, messages, openai_api_key: nil)
  end

  test "unknown agent ids fail fast" do
    messages = [%{role: "user", content: "Hi"}]

    assert {:error, {:unknown_agent, 99}} = AgentPool.call_agent(99, messages)
  end

  test "invalid message payloads fail before provider dispatch" do
    messages = [%{role: "user", text: "Hi"}]

    assert {:error, {:invalid_message, %{role: "user", text: "Hi"}}} =
             AgentPool.call_agent(0, messages)
  end
end
