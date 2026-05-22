defmodule TrinityCoordinator.Sakana.ExporterSyncTest do
  @moduledoc """
  Phase 1 (lazy-backend timing sync) — unit coverage for
  `TrinityCoordinator.Sakana.Exporter.sync_tensor!/1`.

  The helper is the backend-agnostic force point that pulls a tensor
  through `Nx.sum/1 |> Nx.to_number/1` so the exporter's
  `decompose_elapsed_ms` / `reconstruct_elapsed_ms` capture real wall
  time on lazy backends (EMLX, Emily). On already-eager backends (EXLA,
  Nx.BinaryBackend) the call is a near-no-op; this test pins the
  identity contract on `Nx.BinaryBackend` (the only backend we are
  guaranteed to have available).
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.Exporter

  describe "sync_tensor!/1" do
    test "returns the same tensor reference, unchanged" do
      tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Nx.BinaryBackend)
      result = Exporter.sync_tensor!(tensor)

      assert result == tensor
      assert Nx.shape(result) == Nx.shape(tensor)
      assert Nx.type(result) == Nx.type(tensor)
      assert Nx.to_flat_list(result) == Nx.to_flat_list(tensor)
    end

    test "forces materialization (round-trip through to_number/1 succeeds)" do
      # If sync_tensor!/1 did not actually traverse the tensor, this would
      # not raise on a lazy backend. On the BinaryBackend the call is
      # essentially free but still exercises the same code path the
      # exporter uses to flush futures on EMLX/Emily.
      tensor = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.BinaryBackend)

      Exporter.sync_tensor!(tensor)

      # Recompute the same sum via the user-facing API and assert it agrees
      # with what `sync_tensor!/1` had to compute internally.
      assert Nx.to_number(Nx.sum(tensor)) == 10.0
    end

    test "works on differently-shaped tensors (0-d, 1-d, 2-d, 3-d)" do
      scalars = [
        Nx.tensor(42.0, backend: Nx.BinaryBackend),
        Nx.tensor([1, 2, 3], type: :s64, backend: Nx.BinaryBackend),
        Nx.iota({4, 4}, type: :f32, backend: Nx.BinaryBackend),
        Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend)
      ]

      for t <- scalars do
        assert Exporter.sync_tensor!(t) == t
      end
    end

    test "is safe to chain inside a pipeline" do
      result =
        Nx.tensor([1.0, 2.0, 3.0], backend: Nx.BinaryBackend)
        |> Exporter.sync_tensor!()
        |> Nx.add(1.0)
        |> Exporter.sync_tensor!()

      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end
  end
end
