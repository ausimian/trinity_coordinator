defmodule TrinityCoordinator.Runtime.BackendLabelTest do
  @moduledoc """
  Phase 3 (backend label → backend recovery) — pins the contract for
  `TrinityCoordinator.Runtime.BackendLabel.from_label/1`.

  This helper is the inverse of `TrinityCoordinator.Runtime.tensor_backend/1`:
  given a label string, it returns the `{:ok, backend}` tuple suitable for
  `Nx.backend_transfer/2` (where `backend` is either a module or
  `{module, opts}`). For an unknown label it returns
  `{:error, {:unknown_backend_label, label}}` so callers can decide
  whether to log + fall back or to refuse.
  """

  use ExUnit.Case, async: false

  alias TrinityCoordinator.Runtime.BackendLabel
  require Logger

  describe "from_label/1 — known labels" do
    test "EXLA cuda label returns {EXLA.Backend, client: :cuda}" do
      assert {:ok, {EXLA.Backend, client: :cuda}} =
               BackendLabel.from_label("EXLA.Backend<cuda:0>")
    end

    test "EXLA host label returns {EXLA.Backend, client: :host}" do
      assert {:ok, {EXLA.Backend, client: :host}} =
               BackendLabel.from_label("EXLA.Backend<host:0>")
    end

    test "Nx.BinaryBackend label returns Nx.BinaryBackend" do
      assert {:ok, Nx.BinaryBackend} = BackendLabel.from_label("Nx.BinaryBackend")
    end

    test "EMLX.Backend label returns {EMLX.Backend, device: :gpu}" do
      # Apple lane — this is Phase 3's headline new coverage. Today the
      # private silent-fallback chains return Nx.BinaryBackend, which
      # would coerce Apple-resident tensors back to BinaryBackend during
      # alignment. With Phase 3 the EMLX label round-trips to its own
      # backend so the transfer becomes a no-op (Apple → Apple) instead
      # of a silent host transfer.
      assert {:ok, {EMLX.Backend, device: :gpu}} =
               BackendLabel.from_label("EMLX.Backend")
    end

    test "EMLX.Backend with a trailing device suffix also resolves" do
      assert {:ok, {EMLX.Backend, device: :gpu}} =
               BackendLabel.from_label("EMLX.Backend<gpu>")
    end
  end

  describe "from_label/1 — unknown labels" do
    test "returns {:error, {:unknown_backend_label, label}} for arbitrary strings" do
      assert {:error, {:unknown_backend_label, "Emily.Backend"}} =
               BackendLabel.from_label("Emily.Backend")

      assert {:error, {:unknown_backend_label, "SomeFutureBackend"}} =
               BackendLabel.from_label("SomeFutureBackend")

      assert {:error, {:unknown_backend_label, "unknown"}} =
               BackendLabel.from_label("unknown")
    end
  end

  describe "from_label!/1 — same as from_label/1 with safe fallback + warning" do
    setup do
      previous = Logger.level()
      Logger.configure(level: :warning)
      on_exit(fn -> Logger.configure(level: previous) end)
      :ok
    end

    test "returns Nx.BinaryBackend for unknown labels (preserves today's silent-fallback semantics, plus a Logger.warning)" do
      import ExUnit.CaptureLog

      logged =
        capture_log(fn ->
          assert BackendLabel.from_label!("Emily.Backend") == Nx.BinaryBackend
        end)

      assert logged =~ "unknown backend label"
      assert logged =~ "Emily.Backend"
      assert logged =~ "Nx.BinaryBackend"
    end

    test "round-trips known labels without logging" do
      import ExUnit.CaptureLog

      logged =
        capture_log(fn ->
          assert {EXLA.Backend, client: :cuda} = BackendLabel.from_label!("EXLA.Backend<cuda:0>")

          assert {EXLA.Backend, client: :host} =
                   BackendLabel.from_label!("EXLA.Backend<host:0>")

          assert Nx.BinaryBackend == BackendLabel.from_label!("Nx.BinaryBackend")
          assert {EMLX.Backend, device: :gpu} = BackendLabel.from_label!("EMLX.Backend")
        end)

      refute logged =~ "unknown backend label"
    end
  end
end
