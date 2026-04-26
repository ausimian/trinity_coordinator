defmodule TrinityCoordinator.CoordinationHead do
  @moduledoc """
  A linear coordination head that maps SLM hidden states to agent/role logits.
  """

  @doc "Builds the Axon model structure."
  def build_model(input_dim \\ 1024, num_agents \\ 7, num_roles \\ 3) do
    total_outputs = num_agents + num_roles

    Axon.input("hidden_state", shape: {nil, input_dim})
    |> Axon.dense(total_outputs, name: "routing_head")
  end

  @doc "Returns raw logits as a rank-2 tensor with shape {batch, num_agents+num_roles}."
  def output_logits(model, params, penultimate_tensor) do
    Axon.predict(model, params, %{"hidden_state" => penultimate_tensor})
  end

  @doc "Runs the forward pass and returns `{agent_id, role_id}`."
  def forward(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    logits = output_logits(model, params, penultimate_tensor)

    output_dim = num_agents + num_roles
    logits_shape = Nx.shape(logits)

    with {batch, dim} when is_integer(batch) and is_integer(dim) <- logits_shape,
         ^output_dim <- dim do
      :ok
    else
      {_batch, _dim} ->
        raise ArgumentError,
              "coordination head must output #{output_dim} logits, got shape #{inspect(logits_shape)}"

      _ ->
        raise ArgumentError,
              "invalid coordination head output shape #{inspect(logits_shape)}"
    end

    # Squeeze batch size so it's a 1D tensor of shape {output_dim}
    logits_1d = Nx.squeeze(logits, axes: [0])

    agent_logits = Nx.slice(logits_1d, [0], [num_agents])
    role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

    agent_id = Nx.to_number(Nx.argmax(agent_logits))
    role_id = Nx.argmax(role_logits) |> Nx.to_number()

    if not is_integer(agent_id) or not is_integer(role_id) do
      raise ArgumentError, "invalid argmax output from coordination head"
    end

    {agent_id, role_id}
  end
end
