defmodule TrinityCoordinator.AgentPool.Adapter do
  @moduledoc """
  Behaviour for provider adapters that can execute LLM calls.
  """

  @callback call(
              agent_spec :: map(),
              messages :: list(map()),
              opts :: keyword()
            ) :: {:ok, String.t()} | {:error, term()}
end
