defmodule TrinityCoordinator.SLMProfileTest do
  use ExUnit.Case

  alias TrinityCoordinator.{Runtime, SLMProfile}

  test "tiny profile exposes runtime-ready metadata" do
    profile = SLMProfile.tiny_gpt2()

    assert profile.name == :tiny_gpt2
    assert profile.status == :ready
    assert match?({:hf, _}, profile.repo)
    assert profile.module == Bumblebee.Text.Gpt2
    assert profile.architecture == :base
    assert is_integer(profile.expected_hidden_size) and profile.expected_hidden_size > 0
  end

  test "qwen profile is explicitly tagged for production intent" do
    profile = SLMProfile.qwen_coordinator()

    assert profile.name == :qwen_coordinator
    assert profile.status == :ready
    assert profile.expected_hidden_size == 1024
    assert is_tuple(profile.repo)
    assert profile.module == Bumblebee.Text.Qwen3
  end

  test "compatibility_probe reports supported modules for ready profiles" do
    {:ok, probe} = SLMProfile.compatibility_probe(:tiny_gpt2)

    assert probe.status == :compatible
    assert :"Elixir.Bumblebee.Text.Gpt2" in probe.supported_text_modules
  end

  @tag :qwen
  test "compatibility_probe reports qwen as compatible" do
    {:ok, probe} = SLMProfile.compatibility_probe(:qwen_coordinator)

    assert probe.status == :compatible
    assert Bumblebee.Text.Qwen3 in probe.supported_text_modules
  end

  @tag :integration
  @tag :qwen
  test "loads qwen profile through load_profile/1" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:qwen_coordinator)
    assert model_info.spec.hidden_size == 1024
    assert tokenizer != nil
  end

  @tag :integration
  test "loads tiny profile through load_profile/1" do
    Runtime.put_cuda_backend!()

    assert {:ok, {model_info, tokenizer}} = SLMProfile.load_profile(:tiny_gpt2)
    assert is_map(model_info)
    assert tokenizer != nil
  end

  test "load_profile resolves only known profile names" do
    assert {:error, {:unknown_profile, :does_not_exist}} =
             SLMProfile.load_profile(:does_not_exist)
  end

  test "load_profile validates malformed profiles" do
    assert {:error, :invalid_profile} = SLMProfile.load_profile(%{name: :unknown})
  end
end
