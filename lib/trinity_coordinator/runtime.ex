defmodule TrinityCoordinator.Runtime do
  @moduledoc """
  Runtime checks for GPU-backed Nx/EXLA execution.
  """

  @doc """
  Returns the EXLA platforms visible to this BEAM instance.
  """
  def supported_platforms do
    EXLA.Client.get_supported_platforms()
  end

  @doc """
  Raises unless EXLA reports at least one CUDA device.
  """
  def require_cuda! do
    platforms = supported_platforms()

    unless Map.get(platforms, :cuda, 0) > 0 do
      raise "EXLA CUDA platform is not available; run with XLA_TARGET=cuda12 for the current Bumblebee stack"
    end

    platforms
  end

  @doc """
  Sets the process default Nx backend to the configured EXLA CUDA client.
  """
  def put_cuda_backend! do
    require_cuda!()
    Nx.global_default_backend({EXLA.Backend, client: :cuda})
  end

  @doc """
  Returns a compact backend label for a tensor.
  """
  def tensor_backend(%Nx.Tensor{} = tensor) do
    case Regex.run(~r/(EXLA\.Backend<[^>]+>|Nx\.BinaryBackend)/, inspect(tensor)) do
      [backend | _] -> backend
      nil -> "unknown"
    end
  end
end
