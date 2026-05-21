defmodule TrinityCoordinator.RuntimeProfileTest do
  use ExUnit.Case, async: false

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

    test ":emlx maps to EMLX.Backend with device: :gpu and does not require CUDA" do
      p = RuntimeProfile.resolve(:emlx)
      assert p.name == :emlx
      assert p.nx_backend == {EMLX.Backend, device: :gpu}
      assert p.require_cuda? == false
      assert p.qwen_runtime? == true
      assert p.export_svd? == true
      assert p.default_slm_profile == :qwen_coordinator
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

  describe "put_default_backend!/1" do
    test ":binary profile sets Nx.BinaryBackend without raising" do
      original = Nx.default_backend()
      on_exit(fn -> Nx.global_default_backend(original) end)

      :ok = RuntimeProfile.put_default_backend!(:binary)
      assert Nx.default_backend() == {Nx.BinaryBackend, []}
    end

    test "profile whose backend module is not loaded raises an informative error" do
      synthetic = %RuntimeProfile{
        name: :synthetic_missing,
        nx_backend: {:"Elixir.NoSuchBackend.Missing", []},
        require_cuda?: false
      }

      raised =
        try do
          RuntimeProfile.put_default_backend!(synthetic)
        rescue
          e -> e
        end

      msg = Exception.message(raised)

      assert String.contains?(msg, ":synthetic_missing") or
               String.contains?(msg, "NoSuchBackend")
    end

    test ":emlx profile raises an EMLX-specific error when EMLX.Backend is not loaded" do
      # On CUDA hosts EMLX.Backend is absent (optional dep). Confirm the
      # raise message names the dep so the operator knows what to add.
      raised =
        try do
          RuntimeProfile.put_default_backend!(:emlx)
        rescue
          e -> e
        else
          _ ->
            # If we get here, EMLX is loaded on this host; that is also acceptable.
            nil
        end

      if raised do
        msg = Exception.message(raised)
        assert String.contains?(msg, "EMLX")
      end
    end
  end

  describe "accepts_backend_label?/2" do
    test ":cuda_exla accepts EXLA.Backend<cuda:N> labels" do
      profile = RuntimeProfile.resolve(:cuda_exla)
      assert RuntimeProfile.accepts_backend_label?(profile, "EXLA.Backend<cuda:0>")
      refute RuntimeProfile.accepts_backend_label?(profile, "EMLX.Backend")
      refute RuntimeProfile.accepts_backend_label?(profile, "Nx.BinaryBackend")
    end

    test ":host_exla accepts EXLA.Backend<host:N> labels" do
      profile = RuntimeProfile.resolve(:host_exla)
      assert RuntimeProfile.accepts_backend_label?(profile, "EXLA.Backend<host:0>")
      refute RuntimeProfile.accepts_backend_label?(profile, "EXLA.Backend<cuda:0>")
    end

    test ":emlx accepts EMLX.Backend labels" do
      profile = RuntimeProfile.resolve(:emlx)
      assert RuntimeProfile.accepts_backend_label?(profile, "EMLX.Backend")
      refute RuntimeProfile.accepts_backend_label?(profile, "EXLA.Backend<cuda:0>")
    end

    test ":binary accepts Nx.BinaryBackend labels" do
      profile = RuntimeProfile.resolve(:binary)
      assert RuntimeProfile.accepts_backend_label?(profile, "Nx.BinaryBackend")
      refute RuntimeProfile.accepts_backend_label?(profile, "EXLA.Backend<cuda:0>")
    end

    test "{:custom, mod, opts} accepts strings beginning with module label" do
      profile = RuntimeProfile.resolve({:custom, Nx.BinaryBackend, []})
      assert RuntimeProfile.accepts_backend_label?(profile, "Nx.BinaryBackend")
    end
  end
end
