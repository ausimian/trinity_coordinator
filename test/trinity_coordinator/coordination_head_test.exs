defmodule TrinityCoordinator.CoordinationHeadTest do
  use ExUnit.Case
  alias TrinityCoordinator.CoordinationHead

  test "builds model and returns bounded route details from a real Axon forward pass" do
    input_dim = 10
    num_agents = 3
    num_roles = 2

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)
    {init_fn, _predict_fn} = Axon.build(model)
    params = init_fn.(Nx.template({1, input_dim}, :f32), Axon.ModelState.empty())

    tensor = Nx.broadcast(0.5, {1, input_dim})

    route = CoordinationHead.route(model, params, tensor, num_agents, num_roles)

    assert Nx.shape(route.logits) == {1, num_agents + num_roles}
    assert Nx.shape(route.agent_logits) == {num_agents}
    assert Nx.shape(route.role_logits) == {num_roles}

    assert is_integer(route.agent_id)
    assert route.agent_id >= 0 and route.agent_id < num_agents

    assert is_integer(route.role_id)
    assert route.role_id >= 0 and route.role_id < num_roles
  end

  test "builds combined one-hot labels for agent and role supervision" do
    labels = CoordinationHead.build_labels([0, 2], [1, 0], 3, 2)

    assert Nx.shape(labels) == {2, 5}
    assert Nx.to_flat_list(labels) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
  end

  @tag :integration
  test "trains the real Axon coordination head on tensors and routes with trained parameters" do
    input_dim = 4
    num_agents = 2
    num_roles = 2

    model = CoordinationHead.build_model(input_dim, num_agents, num_roles)

    features =
      Nx.tensor(
        [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ],
        type: :f32
      )

    labels = CoordinationHead.build_labels([0, 1, 0, 1], [0, 0, 1, 1], num_agents, num_roles)

    trained_state =
      CoordinationHead.train_supervised(model, features, labels,
        num_agents: num_agents,
        num_roles: num_roles,
        epochs: 40,
        learning_rate: 0.1,
        compiler: EXLA
      )

    route =
      CoordinationHead.route(
        model,
        trained_state,
        Nx.slice(features, [0, 0], [1, input_dim]),
        num_agents,
        num_roles
      )

    assert route.agent_id == 0
    assert route.role_id == 0
    assert inspect(route.logits) =~ "EXLA.Backend<"
  end
end
