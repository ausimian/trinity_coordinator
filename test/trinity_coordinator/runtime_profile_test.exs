defmodule TrinityCoordinator.RuntimeProfileTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.RuntimeProfile

  describe "resolve/1" do
    test ":cuda_exla requires CUDA and points at EXLA cuda client" do
      p = RuntimeProfile.resolve(:cuda_exla)
      assert p.name == :cuda_exla
      assert p.require_cuda? == true
      assert p.nx_backend == {EXLA.Backend, client: :cuda}
      assert p.qwen_runtime? == true
      assert p.export_svd? == true
      assert p.large_svd? == true
    end

    test ":host_exla uses EXLA host client and does not require CUDA" do
      p = RuntimeProfile.resolve(:host_exla)
      assert p.require_cuda? == false
      assert p.nx_backend == {EXLA.Backend, client: :host}
    end

    test ":binary uses Nx.BinaryBackend and disables qwen runtime + svd export" do
      p = RuntimeProfile.resolve(:binary)
      assert p.nx_backend == Nx.BinaryBackend
      assert p.require_cuda? == false
      assert p.qwen_runtime? == false
      assert p.export_svd? == false
    end

    test ":mock_tiny is a synthetic-coordinator profile" do
      p = RuntimeProfile.resolve(:mock_tiny)
      assert p.qwen_runtime? == false
      assert p.artifact_runtime? == true
      assert p.default_slm_profile == :tiny_synthetic
    end

    test ":emlx is descriptive and includes warnings about backend presence" do
      p = RuntimeProfile.resolve(:emlx)
      assert match?([_ | _], p.warnings)
      assert match?([_ | _], p.notes)
    end

    test "passes %RuntimeProfile{} structs through unchanged" do
      orig = %RuntimeProfile{name: :hand_rolled, nx_backend: Nx.BinaryBackend}
      assert RuntimeProfile.resolve(orig) == orig
    end

    test "{:custom, backend, opts} produces a custom-named struct" do
      p = RuntimeProfile.resolve({:custom, EXLA.Backend, [client: :host]})
      assert p.name == :custom
      assert p.nx_backend == {EXLA.Backend, client: :host}
    end

    test "unknown profile raises with a helpful message" do
      raised =
        try do
          RuntimeProfile.resolve(:bogus)
        rescue
          e -> e
        end

      assert %ArgumentError{} = raised
      msg = Exception.message(raised)
      assert String.contains?(msg, ":bogus")
      assert String.contains?(msg, ":cuda_exla")
    end

    test "builtin_names/0 lists all the names resolve/1 accepts as atoms" do
      for name <- RuntimeProfile.builtin_names() do
        assert %RuntimeProfile{} = RuntimeProfile.resolve(name)
      end
    end
  end
end
