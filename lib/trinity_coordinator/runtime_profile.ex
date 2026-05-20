defmodule TrinityCoordinator.RuntimeProfile do
  @moduledoc """
  Declarative description of a coordinator runtime / backend lane.

  A `%RuntimeProfile{}` captures the Nx backend tuple, whether CUDA is
  required, whether the Qwen3-0.6B base must be loadable, and other facts that
  used to be scattered through Mix-task defaults and `Coordinator.load/1`'s
  CUDA-shaped option list.

  ## Built-in profiles

    * `:cuda_exla` — canonical production lane. Requires CUDA platform.
    * `:host_exla` — CPU EXLA lane for compatibility / non-CUDA hosts.
    * `:binary` — pure Nx binary backend; tiny tests only.
    * `:mock_tiny` — synthetic tiny coordinator with no Qwen load (Phase 5).
    * `:emlx` — Apple Silicon path. Descriptive only unless the EMLX backend
      module is actually loaded by the host application.

  Custom profiles can be passed through directly:

      %RuntimeProfile{name: :custom_x, nx_backend: {EXLA.Backend, client: :host}, ...}

  No `lib/**` code reads from `System.get_env/1` to pick the profile — the
  caller decides, either by a Mix task flag, a `Config.Provider`, or an
  explicit struct.
  """

  @enforce_keys [:name, :nx_backend]
  defstruct [
    :name,
    :nx_backend,
    require_cuda?: false,
    qwen_runtime?: true,
    export_svd?: false,
    large_svd?: false,
    artifact_runtime?: true,
    default_slm_profile: :qwen_coordinator,
    notes: [],
    warnings: []
  ]

  @type backend_tuple :: {module(), keyword()} | module()

  @type t :: %__MODULE__{
          name: atom() | binary(),
          nx_backend: backend_tuple(),
          require_cuda?: boolean(),
          qwen_runtime?: boolean(),
          export_svd?: boolean(),
          large_svd?: boolean(),
          artifact_runtime?: boolean(),
          default_slm_profile: atom(),
          notes: [String.t()],
          warnings: [String.t()]
        }

  @doc """
  Resolves a profile name (or passthrough struct) into a `%RuntimeProfile{}`.

  Raises `ArgumentError` for unknown names so misconfiguration fails loud
  rather than silently falling back to a CUDA default.
  """
  @spec resolve(atom() | t() | {atom(), keyword()}) :: t()
  def resolve(%__MODULE__{} = profile), do: profile

  def resolve(:cuda_exla) do
    %__MODULE__{
      name: :cuda_exla,
      nx_backend: {EXLA.Backend, client: :cuda},
      require_cuda?: true,
      qwen_runtime?: true,
      export_svd?: true,
      large_svd?: true,
      artifact_runtime?: true,
      default_slm_profile: :qwen_coordinator,
      notes: ["Canonical production lane; requires a CUDA-capable GPU."]
    }
  end

  def resolve(:host_exla) do
    %__MODULE__{
      name: :host_exla,
      nx_backend: {EXLA.Backend, client: :host},
      require_cuda?: false,
      qwen_runtime?: true,
      export_svd?: true,
      large_svd?: false,
      artifact_runtime?: true,
      default_slm_profile: :qwen_coordinator,
      notes: ["CPU EXLA; useful for compatibility checks but slow for full Qwen."]
    }
  end

  def resolve(:binary) do
    %__MODULE__{
      name: :binary,
      nx_backend: Nx.BinaryBackend,
      require_cuda?: false,
      qwen_runtime?: false,
      export_svd?: false,
      large_svd?: false,
      artifact_runtime?: false,
      default_slm_profile: :tiny_synthetic,
      notes: ["Pure Nx binary backend. Tests / tiny fixtures only."]
    }
  end

  def resolve(:mock_tiny) do
    %__MODULE__{
      name: :mock_tiny,
      nx_backend: Nx.BinaryBackend,
      require_cuda?: false,
      qwen_runtime?: false,
      export_svd?: false,
      large_svd?: false,
      artifact_runtime?: true,
      default_slm_profile: :tiny_synthetic,
      notes: ["Synthetic tiny coordinator; no real Qwen load. See Phase 5."]
    }
  end

  def resolve(:emlx) do
    %__MODULE__{
      name: :emlx,
      nx_backend: Nx.BinaryBackend,
      require_cuda?: false,
      qwen_runtime?: true,
      export_svd?: true,
      large_svd?: false,
      artifact_runtime?: true,
      default_slm_profile: :qwen_coordinator,
      notes: [
        "Apple Silicon profile. Native MLX SVD may materialize full matrices; ",
        "see docs 18/19 for the upstream Nx/EMLX discussion."
      ],
      warnings: [
        "EMLX backend is descriptive here; the host application must actually load it."
      ]
    }
  end

  def resolve({:custom, backend, opts}) when is_atom(backend) and is_list(opts) do
    %__MODULE__{
      name: :custom,
      nx_backend: {backend, opts},
      notes: ["Custom backend tuple supplied by caller."]
    }
  end

  def resolve(other) do
    raise ArgumentError,
          "unknown runtime profile #{inspect(other)}; " <>
            "valid built-ins: :cuda_exla, :host_exla, :binary, :mock_tiny, :emlx; " <>
            "or pass a %TrinityCoordinator.RuntimeProfile{} struct or {:custom, backend, opts}"
  end

  @doc """
  Returns the list of built-in profile names.
  """
  @spec builtin_names() :: [atom()]
  def builtin_names, do: [:cuda_exla, :host_exla, :binary, :mock_tiny, :emlx]
end
