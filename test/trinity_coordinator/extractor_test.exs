defmodule TrinityCoordinator.ExtractorTest do
  use ExUnit.Case
  alias TrinityCoordinator.Extractor

  test "extracts the penultimate hidden state from a 3D tensor" do
    # Create a mock tensor with shape {1, 5, 10} (batch_size=1, seq_len=5, hidden_dim=10)
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

  @tag :integration
  test "extracts the penultimate hidden state from a real tiny SLM" do
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
  end
end
