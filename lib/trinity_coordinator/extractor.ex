defmodule TrinityCoordinator.Extractor do
  @moduledoc """
  Handles extracting specific information from the Small Language Model's outputs.
  """
  import Nx.Defn

  @default_slm_repo {:hf, "hf-internal-testing/tiny-random-gpt2"}
  @default_slm_architecture :base
  @default_slm_module Bumblebee.Text.Gpt2

  @doc """
  Extracts the hidden state vector corresponding to the second-to-last token
  in the sequence. The input is expected to be a tensor of shape:
  {batch_size, sequence_length, hidden_dimension}.
  """
  defn extract_penultimate_hidden_state(hidden_states) do
    # Ensure it's a tensor (if a tuple is passed, take the last layer)
    # However, defn requires all inputs to be numerical/tensors.
    # We assume the caller passes the final layer's tensor.

    {batch, seq_len, hidden_dim} = Nx.shape(hidden_states)

    # We want to slice starting at token index `seq_len - 2`
    # and we want exactly 1 token.
    sliced = Nx.slice(hidden_states, [0, seq_len - 2, 0], [batch, 1, hidden_dim])

    # Remove the sequence_length dimension to return {batch, hidden_dim}
    Nx.squeeze(sliced, axes: [1])
  end

  @doc """
  Loads a lightweight SLM and tokenizer for real hidden-state extraction.

  The defaults are intentionally tiny so this can run as a smoke test in local
  environments; swap these options for your target SLM repository and module.
  """
  def load_slm_model(
        slm_repo \\ @default_slm_repo,
        slm_module \\ @default_slm_module,
        architecture \\ @default_slm_architecture
      ) do
    with {:ok, model_info} <-
           Bumblebee.load_model(slm_repo, module: slm_module, architecture: architecture),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(slm_repo) do
      {:ok, {model_info, tokenizer}}
    end
  end

  @doc """
  Runs one non-generative forward pass and returns the penultimate token's hidden
  state vector suitable for routing.
  """
  def extract_penultimate_hidden_state_from_messages(
        messages,
        slm_repo \\ @default_slm_repo,
        slm_module \\ @default_slm_module,
        architecture \\ @default_slm_architecture
      ) do
    with {:ok, {model_info, tokenizer}} <- load_slm_model(slm_repo, slm_module, architecture) do
      extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages)
    end
  end

  @doc """
  Runs one non-generative forward pass and returns the penultimate token's hidden
  state vector suitable for routing.
  """
  def extract_penultimate_hidden_state_from_texts(model_info, tokenizer, messages) do
    transcript = format_messages(messages)
    inputs = Bumblebee.apply_tokenizer(tokenizer, transcript)
    outputs = Axon.predict(model_info.model, model_info.params, inputs)
    hidden_states = extract_hidden_states(outputs)

    hidden_state = extract_last_layer_hidden_state(hidden_states)

    if hidden_state == nil do
      {:error, :missing_hidden_state}
    else
      {:ok, extract_penultimate_hidden_state(hidden_state)}
    end
  end

  defp extract_hidden_states(outputs) do
    Map.get(outputs, :hidden_state, Map.get(outputs, "hidden_state")) ||
      Map.get(outputs, :hidden_states, Map.get(outputs, "hidden_states"))
  end

  defp extract_last_layer_hidden_state(hidden_states) do
    case hidden_states do
      tensor = %Nx.Tensor{} ->
        tensor

      {_, _} = tuple when is_tuple(tuple) ->
        tuple
        |> Tuple.to_list()
        |> Enum.find_value(fn
          %Axon.None{} -> nil
          value -> value
        end)

      layers when is_list(layers) ->
        layers
        |> Enum.reverse()
        |> Enum.find_value(fn
          %Axon.None{} -> nil
          value -> value
        end)

      _ ->
        nil
    end
  end

  defp format_messages(messages) when is_list(messages) do
    messages
    |> Enum.map_join("\n", fn message ->
      role = Map.get(message, "role", Map.get(message, :role, "unknown"))
      content = Map.get(message, "content", Map.get(message, :content, ""))
      "#{role}: #{content}"
    end)
  end

  defp format_messages(_), do: ""
end
