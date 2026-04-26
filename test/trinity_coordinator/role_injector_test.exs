defmodule TrinityCoordinator.RoleInjectorTest do
  use ExUnit.Case
  alias TrinityCoordinator.RoleInjector

  test "injects Thinker role correctly" do
    messages = [%{role: "user", content: "Hello"}]
    injected = RoleInjector.inject_role(messages, "Thinker")

    assert length(injected) == 2
    assert Enum.at(injected, 0).role == "system"
    assert Enum.at(injected, 0).content =~ "Analyze the current state"
    assert Enum.at(injected, 1).content == "Hello"
  end

  test "injects Worker role correctly" do
    messages = []
    injected = RoleInjector.inject_role(messages, "Worker")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content =~ "Execute the next step"
  end

  test "injects Verifier role correctly" do
    messages = []
    injected = RoleInjector.inject_role(messages, "Verifier")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content =~ "Check the current solution"
  end

  test "defaults to a helpful assistant if role is unknown" do
    messages = []
    injected = RoleInjector.inject_role(messages, "UnknownRole")

    assert length(injected) == 1
    assert Enum.at(injected, 0).content == "You are a helpful assistant."
  end
end
