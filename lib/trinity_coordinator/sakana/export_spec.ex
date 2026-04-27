defmodule TrinityCoordinator.Sakana.ExportSpec do
  @moduledoc """
  Validated export contract for Sakana/TRINITY runtime artifacts.

  The default spec preserves the inspected Qwen3-0.6B layer-26 artifact layout:
  `9216` SVF scale offsets plus a `{10, 1024}` router head.
  """

  @enforce_keys [
    :name,
    :base_model_repo,
    :bumblebee_module,
    :architecture,
    :hidden_size,
    :num_agents,
    :num_roles,
    :selected_layer_indices,
    :scale_offset_count,
    :source_vector_tensor,
    :router_head_tensor_key
  ]

  defstruct [
    :name,
    :base_model_repo,
    :bumblebee_module,
    :architecture,
    :hidden_size,
    :num_agents,
    :num_roles,
    :selected_layer_indices,
    :scale_offset_count,
    :source_vector_tensor,
    :router_head_tensor_key,
    source_vector_path: "priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors",
    out_dir: "priv/sakana_trinity/adapted_qwen3_0_6b_layer26",
    xla_target: "cuda12",
    export_backend: "elixir_nx_exla_cuda"
  ]

  @type t :: %__MODULE__{
          name: atom(),
          base_model_repo: String.t(),
          bumblebee_module: module(),
          architecture: atom(),
          hidden_size: pos_integer(),
          num_agents: pos_integer(),
          num_roles: pos_integer(),
          selected_layer_indices: [non_neg_integer()],
          scale_offset_count: non_neg_integer(),
          source_vector_tensor: String.t(),
          router_head_tensor_key: String.t(),
          source_vector_path: String.t(),
          out_dir: String.t(),
          xla_target: String.t(),
          export_backend: String.t()
        }

  @doc "Default Qwen/Sakana layer-26 export spec."
  @spec qwen3_0_6b_layer26() :: t()
  def qwen3_0_6b_layer26 do
    %__MODULE__{
      name: :qwen3_0_6b_layer26,
      base_model_repo: "Qwen/Qwen3-0.6B",
      bumblebee_module: Bumblebee.Text.Qwen3,
      architecture: :for_causal_language_modeling,
      hidden_size: 1_024,
      num_agents: 7,
      num_roles: 3,
      selected_layer_indices: [26],
      scale_offset_count: 9_216,
      source_vector_tensor: "trinity_router_es_vector",
      router_head_tensor_key: "trinity_router_head"
    }
  end

  @doc "Resolves a named or struct spec."
  @spec resolve(atom() | String.t() | t() | nil) :: {:ok, t()} | {:error, term()}
  def resolve(nil), do: {:ok, qwen3_0_6b_layer26()}
  def resolve(%__MODULE__{} = spec), do: validate(spec)
  def resolve(:qwen3_0_6b_layer26), do: {:ok, qwen3_0_6b_layer26()}
  def resolve("qwen3_0_6b_layer26"), do: {:ok, qwen3_0_6b_layer26()}
  def resolve("qwen3-0.6b-layer26"), do: {:ok, qwen3_0_6b_layer26()}
  def resolve("qwen"), do: {:ok, qwen3_0_6b_layer26()}
  def resolve(other), do: {:error, {:unsupported_export_spec, other}}

  @doc "Raises on unsupported spec names."
  @spec resolve!(atom() | String.t() | t() | nil) :: t()
  def resolve!(value) do
    case resolve(value) do
      {:ok, spec} -> spec
      {:error, reason} -> raise ArgumentError, inspect(reason)
    end
  end

  @doc "Returns router output count."
  @spec output_count(t()) :: pos_integer()
  def output_count(%__MODULE__{} = spec), do: spec.num_agents + spec.num_roles

  @doc "Returns number of linear-head parameters stored in the vector."
  @spec head_param_count(t()) :: pos_integer()
  def head_param_count(%__MODULE__{} = spec), do: spec.hidden_size * output_count(spec)

  @doc "Returns expected source vector size."
  @spec source_vector_size(t()) :: pos_integer()
  def source_vector_size(%__MODULE__{} = spec),
    do: spec.scale_offset_count + head_param_count(spec)

  @doc "Returns a serializable compact representation for manifests/logs."
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = spec) do
    %{
      name: spec.name,
      base_model_repo: spec.base_model_repo,
      bumblebee_module: inspect(spec.bumblebee_module),
      architecture: spec.architecture,
      hidden_size: spec.hidden_size,
      num_agents: spec.num_agents,
      num_roles: spec.num_roles,
      output_count: output_count(spec),
      selected_layer_indices: spec.selected_layer_indices,
      scale_offset_count: spec.scale_offset_count,
      head_param_count: head_param_count(spec),
      source_vector_size: source_vector_size(spec),
      source_vector_path: spec.source_vector_path,
      source_vector_tensor: spec.source_vector_tensor,
      router_head_tensor_key: spec.router_head_tensor_key,
      out_dir: spec.out_dir,
      xla_target: spec.xla_target,
      export_backend: spec.export_backend
    }
  end

  @doc "Validates an export spec."
  @spec validate(t()) :: {:ok, t()} | {:error, term()}
  def validate(%__MODULE__{} = spec) do
    cond do
      not non_empty_atom?(spec.name) ->
        {:error, {:invalid_export_spec, :name}}

      not non_empty_string?(spec.base_model_repo) ->
        {:error, {:invalid_export_spec, :base_model_repo}}

      not is_atom(spec.bumblebee_module) ->
        {:error, {:invalid_export_spec, :bumblebee_module}}

      not is_atom(spec.architecture) ->
        {:error, {:invalid_export_spec, :architecture}}

      not pos_int?(spec.hidden_size) ->
        {:error, {:invalid_export_spec, :hidden_size}}

      not pos_int?(spec.num_agents) ->
        {:error, {:invalid_export_spec, :num_agents}}

      not pos_int?(spec.num_roles) ->
        {:error, {:invalid_export_spec, :num_roles}}

      output_count(spec) <= 0 ->
        {:error, {:invalid_export_spec, :output_count}}

      not non_neg_int?(spec.scale_offset_count) ->
        {:error, {:invalid_export_spec, :scale_offset_count}}

      not layer_indices?(spec.selected_layer_indices) ->
        {:error, {:invalid_export_spec, :selected_layer_indices}}

      not non_empty_string?(spec.source_vector_tensor) ->
        {:error, {:invalid_export_spec, :source_vector_tensor}}

      not non_empty_string?(spec.router_head_tensor_key) ->
        {:error, {:invalid_export_spec, :router_head_tensor_key}}

      not non_empty_string?(spec.source_vector_path) ->
        {:error, {:invalid_export_spec, :source_vector_path}}

      not non_empty_string?(spec.out_dir) ->
        {:error, {:invalid_export_spec, :out_dir}}

      true ->
        {:ok, spec}
    end
  end

  def validate(other), do: {:error, {:invalid_export_spec, other}}

  defp non_empty_atom?(value), do: is_atom(value) and not is_nil(value)
  defp non_empty_string?(value), do: is_binary(value) and String.trim(value) != ""
  defp pos_int?(value), do: is_integer(value) and value > 0
  defp non_neg_int?(value), do: is_integer(value) and value >= 0

  defp layer_indices?(values) when is_list(values),
    do: Enum.all?(values, &(is_integer(&1) and &1 >= 0))

  defp layer_indices?(_), do: false
end
