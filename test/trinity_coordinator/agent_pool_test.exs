defmodule TrinityCoordinator.AgentPoolTest do
  use ExUnit.Case
  alias TrinityCoordinator.AgentPool

  defmodule TestAdapter do
    @behaviour TrinityCoordinator.AgentPool.Adapter

    @impl true
    def call(_spec, messages, _opts) do
      {:ok, "Test adapter response for #{length(messages)} messages"}
    end
  end

  test "returns a mapped response from a provider adapter" do
    messages = [%{role: "user", content: "Hi"}]
    {:ok, response} = AgentPool.call_agent(0, messages, adapter: TestAdapter)

    assert response == "Test adapter response for 1 messages"
  end

  test "unknown agent ids fail fast" do
    messages = [%{role: "user", content: "Hi"}]

    assert {:error, {:unknown_agent, 99}} = AgentPool.call_agent(99, messages, adapter: TestAdapter)
  end
end
