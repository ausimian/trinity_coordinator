defmodule TrinityCoordinator.RuntimeTest do
  use ExUnit.Case

  alias TrinityCoordinator.Runtime

  @tag :integration
  test "confirms CUDA is available and tensors can be allocated on the CUDA backend" do
    platforms = Runtime.require_cuda!()
    assert Map.fetch!(platforms, :cuda) >= 1

    Runtime.put_cuda_backend!()

    tensor = Nx.iota({8, 8}, type: :f32) |> Nx.dot(Nx.iota({8, 8}, type: :f32))

    assert Runtime.tensor_backend(tensor) =~ "EXLA.Backend<cuda:"
  end
end
