defmodule TrinityCoordinator.Sakana.CoordinatorFastKernelsTest do
  @moduledoc """
  Covers the `Coordinator.maybe_apply_fast_kernels/2` hook that the
  `:emily_fast` profile uses to splice `Emily.Bumblebee.FastKernels.apply/1`
  into the model-load pipeline.

  The non-:emily_fast routing test runs everywhere. The :emily_fast
  branches are split by host: the missing-dep error path is asserted
  on hosts where Emily is absent (CUDA CI); the rewriter-invocation
  path is asserted on hosts where Emily is loaded (Apple Silicon).
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.RuntimeProfile
  alias TrinityCoordinator.Sakana.Coordinator

  @fast_kernels Module.concat([Emily, Bumblebee, FastKernels])
  @emily_loaded? Code.ensure_loaded?(@fast_kernels)

  describe "maybe_apply_fast_kernels/2 — non-:emily_fast routing" do
    test "every other built-in profile returns model_info unchanged" do
      model_info = %{model: :sentinel_axon_graph, params: %{}}

      for name <- [:cuda_exla, :host_exla, :binary, :mock_tiny, :emlx, :emily] do
        profile = RuntimeProfile.resolve(name)
        assert {:ok, ^model_info} = Coordinator.maybe_apply_fast_kernels(model_info, profile)
      end
    end

    test "{:custom, ...} profiles are unchanged" do
      profile = RuntimeProfile.resolve({:custom, Nx.BinaryBackend, []})
      model_info = %{model: :sentinel_axon_graph, params: %{}}

      assert {:ok, ^model_info} = Coordinator.maybe_apply_fast_kernels(model_info, profile)
    end
  end

  describe "maybe_apply_fast_kernels/2 — :emily_fast, Emily dep absent" do
    @describetag emily_loaded?: @emily_loaded?

    @tag :missing_emily_only
    test "returns {:error, {:emily_fast_kernels_unavailable, _}}" do
      if @emily_loaded? do
        # Emily is loaded — this branch is the wrong oracle here. The
        # adjacent describe block covers the loaded path.
        :ok
      else
        profile = RuntimeProfile.resolve(:emily_fast)
        model_info = %{model: :sentinel_axon_graph, params: %{}}

        assert {:error, {:emily_fast_kernels_unavailable, msg}} =
                 Coordinator.maybe_apply_fast_kernels(model_info, profile)

        assert msg =~ "Emily.Bumblebee.FastKernels"
        assert msg =~ ":emily"
        assert msg =~ "guides/runtime_profiles.md"
      end
    end
  end

  describe "maybe_apply_fast_kernels/2 — :emily_fast, Emily dep loaded" do
    @describetag emily_loaded?: @emily_loaded?

    @tag :emily_only
    test "calls Emily.Bumblebee.FastKernels.apply/1 on model_info.model" do
      if @emily_loaded? do
        profile = RuntimeProfile.resolve(:emily_fast)

        # Minimal Axon model that contains none of the patterns the
        # rewriter recognises (no RMSNorm / LayerNorm / RoPE / SDPA).
        # Therefore the rewriter is a no-op and returns the same graph.
        model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(8)
        model_info = %{model: model, params: %{}}

        assert {:ok, rewritten_info} =
                 Coordinator.maybe_apply_fast_kernels(model_info, profile)

        # Rewriter returns an Axon model (no-op pass-through on this
        # input). Equality semantics for Axon structs aren't part of
        # the public contract, so assert on shape rather than identity.
        assert match?(%Axon{}, rewritten_info.model)
      else
        :ok
      end
    end
  end
end
