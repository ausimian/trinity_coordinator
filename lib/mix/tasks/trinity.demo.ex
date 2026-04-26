defmodule Mix.Tasks.Trinity.Demo do
  @moduledoc """
  Demonstrates the real GPU-backed TRINITY router path.

      XLA_TARGET=cuda12 mix trinity.demo
  """

  use Mix.Task

  alias TrinityCoordinator.{CoordinationHead, Extractor, Runtime}

  @shortdoc "Runs a real GPU-backed TRINITY router demonstration"

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("app.start")
    Logger.configure(level: :error)
    :logger.set_primary_config(:level, :error)

    info("TRINITY Coordinator GPU demo")
    info("==============================")
    info("")

    Runtime.put_cuda_backend!()

    platforms = Runtime.supported_platforms()
    info("1. EXLA runtime")
    info("   XLA_TARGET: #{System.get_env("XLA_TARGET", "(unset)")}")
    info("   Supported platforms: #{inspect(platforms)}")
    info("")

    info("2. Loading real SLM and tokenizer")
    repo = {:hf, "hf-internal-testing/tiny-random-gpt2"}
    {:ok, {model_info, tokenizer}} = Extractor.load_slm_model(repo, Bumblebee.Text.Gpt2, :base)
    info("   Repository: #{inspect(repo)}")
    info("   Model module: #{inspect(Bumblebee.Text.Gpt2)}")
    info("")

    messages = [%{"role" => "user", "content" => "Find a strategy for a short algebra proof."}]

    info("3. Formatting transcript and running real SLM forward pass")

    {:ok, metadata} =
      Extractor.extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages)

    info("   Transcript:")
    info(indent(metadata.transcript, 6))
    info("   Tokenizer input shapes: #{inspect(metadata.input_shapes)}")
    info("   Final hidden-state tensor shape: #{inspect(metadata.hidden_state_shape)}")
    info("   Second-to-last token vector shape: #{inspect(metadata.vector_shape)}")
    info("   Vector backend: #{Runtime.tensor_backend(metadata.vector)}")
    info("")

    training_batches = [
      [%{"role" => "user", "content" => "Solve a symbolic algebra problem."}],
      [%{"role" => "user", "content" => "Write and debug a small function."}],
      [%{"role" => "user", "content" => "Check whether this proof is complete."}],
      [%{"role" => "user", "content" => "Plan a multi-step reasoning solution."}]
    ]

    info("4. Extracting real SLM vectors for supervised head training")

    {:ok, features} =
      Extractor.extract_batch_penultimate_hidden_states(model_info, tokenizer, training_batches)

    info("   Training examples: #{length(training_batches)}")
    info("   Feature tensor shape: #{inspect(Nx.shape(features))}")
    info("   Feature backend: #{Runtime.tensor_backend(features)}")
    info("")

    num_agents = 3
    num_roles = 3
    labels = CoordinationHead.build_labels([0, 1, 2, 0], [1, 1, 2, 0], num_agents, num_roles)

    info("5. Training real Axon coordination head")
    model = CoordinationHead.build_model(Nx.axis_size(features, 1), num_agents, num_roles)

    trained_state =
      CoordinationHead.train_supervised(model, features, labels,
        num_agents: num_agents,
        num_roles: num_roles,
        epochs: 30,
        learning_rate: 0.05,
        compiler: EXLA
      )

    info("   Head input dimension: #{Nx.axis_size(features, 1)}")

    info(
      "   Output logits: #{num_agents + num_roles} (#{num_agents} agents + #{num_roles} roles)"
    )

    info("   Trained state: #{inspect(trained_state)}")
    info("")

    info("6. Routing the original transcript")
    route = CoordinationHead.route(model, trained_state, metadata.vector, num_agents, num_roles)
    info("   Logits backend: #{Runtime.tensor_backend(route.logits)}")
    info("   Agent logits: #{inspect_rounded(route.agent_logits)}")
    info("   Role logits: #{inspect_rounded(route.role_logits)}")
    info("   Selected agent id: #{route.agent_id}")
    info("   Selected role id: #{route.role_id} (#{role_name(route.role_id)})")
    info("")

    info(
      "Demo complete: real Bumblebee SLM forward pass -> second-to-last hidden-state vector ->"
    )

    info("real Axon training -> real Axon routing forward pass, all on EXLA CUDA.")
  end

  defp inspect_rounded(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&Float.round(&1, 4))
    |> inspect()
  end

  defp role_name(0), do: "Thinker"
  defp role_name(1), do: "Worker"
  defp role_name(2), do: "Verifier"
  defp role_name(_), do: "Unknown"

  defp indent(text, spaces) do
    prefix = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map_join("\n", &(prefix <> &1))
  end

  defp info(message), do: Mix.shell().info(message)
end
