defmodule TrinityCoordinator.ExtractorTest do
  use ExUnit.Case
  alias TrinityCoordinator.{Extractor, Runtime}

  test "formats structured messages into a deterministic transcript" do
    messages = [
      %{role: "system", content: "Coordinate carefully."},
      %{"role" => "user", "content" => "Solve this."}
    ]

    assert Extractor.format_messages(messages) ==
             {:ok, "system: Coordinate carefully.\nuser: Solve this."}
  end

  test "extracts the penultimate hidden state from a 3D tensor" do
    # Create a deterministic tensor with shape {1, 5, 10} (batch_size=1, seq_len=5, hidden_dim=10)
    # Nx.iota creates a tensor filled with 0, 1, 2, ...
    tensor = Nx.iota({1, 5, 10})

    # The 2nd-to-last token is at index 3 (0-indexed).
    # Since hidden_dim is 10, its values should start at 30 (3 * 10).

    extracted = Extractor.extract_penultimate_hidden_state(tensor)

    assert Nx.shape(extracted) == {1, 10}

    flat_list = Nx.to_flat_list(extracted)
    assert hd(flat_list) == 30
    assert List.last(flat_list) == 39
  end

  test "falls back to the final token when sequence length is one" do
    tensor = Nx.iota({1, 1, 4})

    extracted = Extractor.extract_penultimate_hidden_state(tensor)

    assert Nx.shape(extracted) == {1, 4}
    assert Nx.to_flat_list(extracted) == [0, 1, 2, 3]
  end

  @tag :integration
  test "extracts the penultimate hidden state from a real tiny SLM" do
    Runtime.put_cuda_backend!()

    messages = [%{"role" => "user", "content" => "Hello world"}]

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    assert {:ok, vector} =
             Extractor.extract_penultimate_hidden_state_from_texts(
               model_info,
               tokenizer,
               messages
             )

    assert Nx.shape(vector) |> Tuple.to_list() == [1, 32]
    assert Runtime.tensor_backend(vector) =~ "EXLA.Backend<cuda:"
  end

  @tag :integration
  @tag :qwen
  test "extracts qwen coordinator hidden state on CUDA" do
    Runtime.put_cuda_backend!()

    messages = [%{"role" => "user", "content" => "Route this short request."}]

    assert {:ok, {model_info, tokenizer}} =
             TrinityCoordinator.SLMProfile.load_profile(:qwen_coordinator)

    assert {:ok, metadata} =
             Extractor.extract_penultimate_hidden_state_with_metadata(
               model_info,
               tokenizer,
               messages
             )

    assert metadata.vector_shape == {1, 1024}
    assert metadata.hidden_state_shape |> Tuple.to_list() |> List.last() == 1024
    assert Runtime.tensor_backend(metadata.vector) =~ "EXLA.Backend<cuda:"
  end

  @tag :integration
  test "extracts real batch vectors and metadata from a real tiny SLM" do
    Runtime.put_cuda_backend!()

    message_batches = [
      [%{"role" => "user", "content" => "Classify this math problem."}],
      [%{"role" => "user", "content" => "Classify this code problem."}]
    ]

    assert {:ok, {model_info, tokenizer}} =
             Extractor.load_slm_model(
               {:hf, "hf-internal-testing/tiny-random-gpt2"},
               Bumblebee.Text.Gpt2,
               :base
             )

    assert {:ok, metadata} =
             Extractor.extract_penultimate_hidden_state_with_metadata(
               model_info,
               tokenizer,
               hd(message_batches)
             )

    assert metadata.transcript == "user: Classify this math problem."
    assert metadata.hidden_state_shape |> Tuple.to_list() |> List.last() == 32
    assert metadata.vector_shape == {1, 32}
    assert Runtime.tensor_backend(metadata.vector) =~ "EXLA.Backend<cuda:"

    assert {:ok, batch} =
             Extractor.extract_batch_penultimate_hidden_states(
               model_info,
               tokenizer,
               message_batches
             )

    assert Nx.shape(batch) == {2, 32}
    assert Runtime.tensor_backend(batch) =~ "EXLA.Backend<cuda:"
  end
end
