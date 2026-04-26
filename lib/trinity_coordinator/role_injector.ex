defmodule TrinityCoordinator.RoleInjector do
  @moduledoc """
  Injects role-specific system prompts into the conversation transcript.
  """

  @roles %{
    "Thinker" =>
      "Analyze the current state and provide high-level guidance, plans, or critiques.",
    "Worker" => "Execute the next step of the plan. Write code, math, or concrete text.",
    "Verifier" =>
      "Check the current solution. Output ACCEPT if it is complete and correct, or REVISE with a diagnosis if it is flawed."
  }

  @doc """
  Prepends a system prompt to the list of messages based on the given role.
  """
  def inject_role(messages, role) do
    system_prompt = Map.get(@roles, role, "You are a helpful assistant.")
    [%{role: "system", content: system_prompt}] ++ messages
  end
end
