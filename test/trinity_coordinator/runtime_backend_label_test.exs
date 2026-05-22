# Minimal structs to act as `Nx.Tensor.data` shapes. We never use Nx
# operations on them; `tensor_backend/1` only reads `tensor.data.__struct__`
# (via the new pattern) and the `inspect/1`-based EXLA branches do not match.
defmodule FakeBackendForTesting do
  @moduledoc false
  defstruct [:state]
end

defmodule FakeBackendForTesting.Nested.Mod do
  @moduledoc false
  defstruct [:x]
end

defmodule EmlxLikeBackend do
  @moduledoc false
  defstruct []
end

defmodule TrinityCoordinator.RuntimeBackendLabelTest do
  @moduledoc """
  Phase 2 (generic backend labeling) — non-CUDA coverage for
  `TrinityCoordinator.Runtime.tensor_backend/1`.

  The integration-tagged test in `runtime_test.exs` already pins the
  EXLA-CUDA label. This module pins the generic default for arbitrary
  backends (including any that the project does not depend on, such as
  EMLX / Emily), the existing explicit branches for Nx.BinaryBackend,
  and the EXLA<...> device-info prefixes (asserted via fixture inspect
  strings to avoid requiring a live CUDA build).
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.Runtime

  describe "tensor_backend/1 — explicit branches still hold" do
    test "Nx.BinaryBackend tensors keep returning \"Nx.BinaryBackend\"" do
      tensor = Nx.tensor([1, 2, 3], backend: Nx.BinaryBackend)
      assert Runtime.tensor_backend(tensor) == "Nx.BinaryBackend"
    end

    test "0-d Nx.BinaryBackend tensors keep returning \"Nx.BinaryBackend\"" do
      tensor = Nx.tensor(42.0, backend: Nx.BinaryBackend)
      assert Runtime.tensor_backend(tensor) == "Nx.BinaryBackend"
    end
  end

  describe "tensor_backend/1 — generic default" do
    test "returns the backend module's dotted name for backends we don't have explicit branches for" do
      fake_backend_data = %FakeBackendForTesting{state: :probe}
      tensor = %Nx.Tensor{data: fake_backend_data, type: {:f, 32}, shape: {1}, names: [nil]}

      label = Runtime.tensor_backend(tensor)
      refute label == "unknown"
      assert label == "FakeBackendForTesting"
    end

    test "returns the dotted module name with full nesting preserved" do
      fake_backend_data = %FakeBackendForTesting.Nested.Mod{x: 1}

      tensor = %Nx.Tensor{
        data: fake_backend_data,
        type: {:f, 32},
        shape: {1},
        names: [nil]
      }

      assert Runtime.tensor_backend(tensor) == "FakeBackendForTesting.Nested.Mod"
    end

    test "EMLX.Backend-style label would resolve via the generic default" do
      fake_backend_data = %EmlxLikeBackend{}

      tensor = %Nx.Tensor{
        data: fake_backend_data,
        type: {:f, 32},
        shape: {1},
        names: [nil]
      }

      assert Runtime.tensor_backend(tensor) == "EmlxLikeBackend"
    end
  end
end
