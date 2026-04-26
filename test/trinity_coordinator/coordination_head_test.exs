defmodule TrinityCoordinator.CoordinationHeadTest do
  use ExUnit.Case
  alias TrinityCoordinator.CoordinationHead

  test "builds model and routes correctly given random params" do
    input_dim = 10
    num_agents = 3
    num_roles = 2

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    # Create a dummy input tensor {batch_size=1, hidden_dim=10}
    tensor = Nx.broadcast(0.5, {1, input_dim})

    {agent_id, role_id} = CoordinationHead.forward(model, params, tensor, num_agents, num_roles)

    # We can't know the exact predicted ID since weights are random,
    # but we can ensure they fall within the correct index bounds.
    assert is_integer(agent_id)
    assert agent_id >= 0 and agent_id < num_agents

    assert is_integer(role_id)
    assert role_id >= 0 and role_id < num_roles
  end
end
