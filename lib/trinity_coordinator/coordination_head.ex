defmodule TrinityCoordinator.CoordinationHead do
  @moduledoc """
  A linear coordination head that maps SLM hidden states to agent/role logits.

  The head follows the TRINITY formulation: a single affine projection maps the
  SLM hidden state to `num_agents + num_roles` logits. The first `num_agents`
  logits select the downstream model, and the final `num_roles` logits select
  the role.
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

  @doc """
  Runs the real Axon forward pass and returns route details.
  """
  def route(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    logits = output_logits(model, params, penultimate_tensor)
    validate_logits!(logits, num_agents, num_roles)

    logits_1d = Nx.squeeze(logits, axes: [0])

    agent_logits = Nx.slice(logits_1d, [0], [num_agents])
    role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

    agent_id = Nx.to_number(Nx.argmax(agent_logits))
    role_id = Nx.to_number(Nx.argmax(role_logits))

    if not is_integer(agent_id) or not is_integer(role_id) do
      raise ArgumentError, "invalid argmax output from coordination head"
    end

    %{
      agent_id: agent_id,
      role_id: role_id,
      logits: logits,
      agent_logits: agent_logits,
      role_logits: role_logits
    }
  end

  @doc "Runs the forward pass and returns `{agent_id, role_id}`."
  def forward(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    route = route(model, params, penultimate_tensor, num_agents, num_roles)
    {route.agent_id, route.role_id}
  end

  @doc """
  Builds a combined one-hot label tensor for supervised head training.

  Each label row is `[agent_one_hot, role_one_hot]`.
  """
  def build_labels(agent_ids, role_ids, num_agents \\ 7, num_roles \\ 3)
      when is_list(agent_ids) and is_list(role_ids) do
    if length(agent_ids) != length(role_ids) do
      raise ArgumentError, "agent_ids and role_ids must have the same length"
    end

    agent_ids
    |> Enum.zip(role_ids)
    |> Enum.map(fn {agent_id, role_id} ->
      validate_label_id!(agent_id, num_agents, :agent_id)
      validate_label_id!(role_id, num_roles, :role_id)

      one_hot(agent_id, num_agents) ++ one_hot(role_id, num_roles)
    end)
    |> Nx.tensor(type: :f32)
  end

  @doc """
  Trains the coordination head with real Axon/Polaris supervised optimization.

  This is the direct supervised path described in the paper appendix: the SLM is
  frozen, extracted hidden-state vectors are provided as `features`, and only
  the lightweight routing head is trained.
  """
  def train_supervised(model, features, labels, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        num_agents: 7,
        num_roles: 3,
        epochs: 20,
        learning_rate: 0.01,
        compiler: EXLA,
        log: 0,
        initial_model_state: Axon.ModelState.empty()
      )

    validate_training_tensors!(features, labels, opts[:num_agents], opts[:num_roles])

    data = [{%{"hidden_state" => features}, labels}]
    optimizer = Polaris.Optimizers.adam(learning_rate: opts[:learning_rate])
    loss = supervised_loss(opts[:num_agents], opts[:num_roles])

    run_opts = [epochs: opts[:epochs]]

    run_opts =
      if opts[:compiler], do: Keyword.put(run_opts, :compiler, opts[:compiler]), else: run_opts

    model
    |> Axon.Loop.trainer(loss, optimizer, log: opts[:log])
    |> Axon.Loop.run(data, opts[:initial_model_state], run_opts)
  end

  defp supervised_loss(num_agents, num_roles) do
    fn y_true, y_pred ->
      batch_size = Nx.axis_size(y_true, 0)

      agent_true = Nx.slice(y_true, [0, 0], [batch_size, num_agents])
      role_true = Nx.slice(y_true, [0, num_agents], [batch_size, num_roles])

      agent_pred = Nx.slice(y_pred, [0, 0], [batch_size, num_agents])
      role_pred = Nx.slice(y_pred, [0, num_agents], [batch_size, num_roles])

      agent_loss =
        Axon.Losses.categorical_cross_entropy(agent_true, agent_pred,
          from_logits: true,
          reduction: :mean
        )

      role_loss =
        Axon.Losses.categorical_cross_entropy(role_true, role_pred,
          from_logits: true,
          reduction: :mean
        )

      Nx.add(agent_loss, role_loss)
    end
  end

  defp validate_logits!(logits, num_agents, num_roles) do
    output_dim = num_agents + num_roles
    logits_shape = Nx.shape(logits)

    case logits_shape do
      {1, ^output_dim} ->
        :ok

      {_batch, ^output_dim} ->
        raise ArgumentError,
              "coordination routing expects a single example, got shape #{inspect(logits_shape)}"

      {_batch, _dim} ->
        raise ArgumentError,
              "coordination head must output #{output_dim} logits, got shape #{inspect(logits_shape)}"

      _ ->
        raise ArgumentError, "invalid coordination head output shape #{inspect(logits_shape)}"
    end
  end

  defp validate_training_tensors!(features, labels, num_agents, num_roles) do
    output_dim = num_agents + num_roles

    case {Nx.shape(features), Nx.shape(labels)} do
      {{batch_size, _input_dim}, {batch_size, ^output_dim}} when batch_size > 0 ->
        :ok

      {feature_shape, label_shape} ->
        raise ArgumentError,
              "invalid training tensor shapes, got features #{inspect(feature_shape)} and labels #{inspect(label_shape)}"
    end
  end

  defp validate_label_id!(id, limit, _name) when is_integer(id) and id >= 0 and id < limit,
    do: :ok

  defp validate_label_id!(id, limit, name) do
    raise ArgumentError, "#{name} must be an integer in 0..#{limit - 1}, got #{inspect(id)}"
  end

  defp one_hot(index, size) do
    Enum.map(0..(size - 1), fn
      ^index -> 1.0
      _ -> 0.0
    end)
  end
end
