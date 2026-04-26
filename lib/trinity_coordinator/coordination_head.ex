defmodule TrinityCoordinator.CoordinationHead do
  @moduledoc """
  The linear neural network layer that maps the extracted hidden state
  to Agent and Role selections using Axon.
  """

  @doc """
  Builds the Axon model structure.
  """
  def build_model(input_dim \\ 1024, num_agents \\ 7, num_roles \\ 3) do
    total_outputs = num_agents + num_roles

    Axon.input("hidden_state", shape: {nil, input_dim})
    |> Axon.dense(total_outputs, name: "routing_head")
  end

  @doc """
  Runs the forward pass given the network parameters and the 2D tensor {batch, hidden_dim}.
  Returns a tuple of {agent_id, role_id}.
  """
  def forward(params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    input_dim = Nx.axis_size(penultimate_tensor, 1)
    model = build_model(input_dim, num_agents, num_roles)

    # Run inference without compiler JITing for single step (or in reality, you'd use Axon.Loop / defn)
    logits = Axon.predict(model, params, %{"hidden_state" => penultimate_tensor})

    # Squeeze batch size so it's a 1D tensor of shape {total_outputs}
    logits_1d = Nx.squeeze(logits, axes: [0])

    agent_logits = Nx.slice(logits_1d, [0], [num_agents])
    role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

    agent_id = Nx.argmax(agent_logits) |> Nx.to_number()
    role_id = Nx.argmax(role_logits) |> Nx.to_number()

    {agent_id, role_id}
  end
end
