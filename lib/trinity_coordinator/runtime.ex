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
  Sets the current process default Nx backend to the configured EXLA CUDA client.
  """
  def put_cuda_backend! do
    require_cuda!()
    Nx.default_backend({EXLA.Backend, client: :cuda})
  end

  @doc """
  Runs a function with the current process default backend set to EXLA CUDA.
  """
  def with_cuda_backend!(fun) when is_function(fun, 0) do
    require_cuda!()
    Nx.with_default_backend({EXLA.Backend, client: :cuda}, fun)
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
