defmodule Mix.Tasks.Trinity.Sakana.RouterTrace do
  @moduledoc """
  Emits and optionally compares a fixed-transcript Sakana router trace.

      XLA_TARGET=cuda12 mix trinity.sakana.router_trace \
        --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python \
        --python-report tmp/sakana_parity/python_router_trace.json \
        --out tmp/sakana_parity/elixir_router_trace.json

  The required gate is exact transcript/token/head/argmax parity plus declared
  f32 alignment thresholds for the hidden vector and logits.
  """

  use Mix.Task

  alias TrinityCoordinator.{Extractor, HITL, MixHelpers, Runtime}
  alias TrinityCoordinator.Sakana.Artifact

  @shortdoc "Emit fixed-transcript Sakana router trace"
  @default_message "Select a TRINITY role for this reasoning task."
  @schema "trinity_sakana_router_trace.v1"

  @default_tolerances %{
    hidden_max_abs_error: 5.0e-2,
    hidden_mean_abs_error: 5.0e-3,
    hidden_min_cosine: 0.99,
    hidden_max_relative_l2: 0.12,
    logits_max_abs_error: 1.0,
    logits_mean_abs_error: 2.0e-1,
    logits_min_cosine: 0.99,
    logits_max_relative_l2: 0.10
  }

  @impl true
  def run(args) do
    Mix.Task.run("app.start")
    opts = parse_args!(args)

    HITL.banner("TRINITY SAKANA ROUTER TRACE")
    Runtime.put_cuda_backend!()

    runtime_profile = MixHelpers.runtime_profile_atom!(Map.get(opts, :runtime_profile, nil))

    coordinator =
      MixHelpers.load_coordinator!(
        Keyword.merge(
          [artifact_dir: opts.artifact_dir],
          if(runtime_profile, do: [runtime_profile: runtime_profile], else: [])
        )
      )

    messages = [%{"role" => "user", "content" => opts.message}]
    manifest_path = Artifact.manifest_path(opts.artifact_dir)
    head = Artifact.load_router_head!(opts.artifact_dir, manifest: coordinator.manifest)

    {:ok, extraction} =
      Extractor.extract_penultimate_hidden_state_with_metadata(
        coordinator.model_info,
        coordinator.tokenizer,
        messages
      )

    hidden_snapshot = Nx.backend_transfer(extraction.vector, Nx.BinaryBackend)

    route =
      route_from_head_snapshot(
        hidden_snapshot,
        head,
        coordinator.num_agents,
        coordinator.num_roles
      )

    report =
      build_report(%{
        artifact_dir: opts.artifact_dir,
        manifest_path: manifest_path,
        coordinator: coordinator,
        messages: messages,
        extraction: Map.put(extraction, :vector_snapshot, hidden_snapshot),
        route: route,
        head: head
      })

    report =
      if opts.python_report do
        python_report = read_json!(opts.python_report)
        comparison = compare_reports(python_report, report, opts.tolerances)
        Map.put(report, "comparison", comparison)
      else
        report
      end

    if opts.out do
      File.mkdir_p!(Path.dirname(opts.out))
      File.write!(opts.out, Jason.encode!(report, pretty: true) <> "\n")
    end

    print_summary!(report)

    case report["comparison"] do
      %{"failed_required" => 0} ->
        HITL.pass("TRINITY SAKANA ROUTER TRACE")

      %{"failed_required" => failed} when is_integer(failed) ->
        Mix.raise("router trace comparison failed_required=#{failed}")

      _ ->
        HITL.pass("TRINITY SAKANA ROUTER TRACE")
    end
  end

  @doc false
  def parse_args!(args) do
    {opts, rest, errors} =
      OptionParser.parse(args,
        strict: [
          artifact_dir: :string,
          runtime_profile: :string,
          python_report: :string,
          out: :string,
          message: :string,
          hidden_max_abs: :float,
          hidden_mean_abs: :float,
          hidden_min_cosine: :float,
          hidden_max_relative_l2: :float,
          logits_max_abs: :float,
          logits_mean_abs: :float,
          logits_min_cosine: :float,
          logits_max_relative_l2: :float
        ]
      )

    unless rest == [], do: Mix.raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: Mix.raise("Invalid options: #{inspect(errors)}")

    %{
      artifact_dir: Keyword.get(opts, :artifact_dir, Artifact.default_output_dir()),
      python_report: Keyword.get(opts, :python_report),
      out: Keyword.get(opts, :out),
      message: Keyword.get(opts, :message, @default_message),
      tolerances: %{
        hidden_max_abs_error:
          Keyword.get(opts, :hidden_max_abs, @default_tolerances.hidden_max_abs_error),
        hidden_mean_abs_error:
          Keyword.get(opts, :hidden_mean_abs, @default_tolerances.hidden_mean_abs_error),
        hidden_min_cosine:
          Keyword.get(opts, :hidden_min_cosine, @default_tolerances.hidden_min_cosine),
        hidden_max_relative_l2:
          Keyword.get(
            opts,
            :hidden_max_relative_l2,
            @default_tolerances.hidden_max_relative_l2
          ),
        logits_max_abs_error:
          Keyword.get(opts, :logits_max_abs, @default_tolerances.logits_max_abs_error),
        logits_mean_abs_error:
          Keyword.get(opts, :logits_mean_abs, @default_tolerances.logits_mean_abs_error),
        logits_min_cosine:
          Keyword.get(opts, :logits_min_cosine, @default_tolerances.logits_min_cosine),
        logits_max_relative_l2:
          Keyword.get(
            opts,
            :logits_max_relative_l2,
            @default_tolerances.logits_max_relative_l2
          )
      }
    }
  end

  @doc false
  def compare_reports(python, elixir, tolerances \\ @default_tolerances)
      when is_map(python) and is_map(elixir) and is_map(tolerances) do
    checks = [
      exact_check("schema", python["schema"], elixir["schema"]),
      exact_check("transcript_sha256", python["transcript_sha256"], elixir["transcript_sha256"]),
      exact_check("token_ids_sha256", python["token_ids_sha256"], elixir["token_ids_sha256"]),
      exact_check("input_ids", python["input_ids"], elixir["input_ids"]),
      exact_check(
        "head_weight_sha256_as_f32",
        python["head_weight_sha256_as_f32"],
        elixir["head_weight_sha256_as_f32"]
      ),
      exact_check(
        "hidden_vector_shape",
        python["hidden_vector_shape"],
        elixir["hidden_vector_shape"]
      ),
      exact_check("logits_shape", python["logits_shape"], elixir["logits_shape"]),
      alignment_check(
        "hidden_vector_f32",
        python["hidden_vector_f32"],
        elixir["hidden_vector_f32"],
        tolerances.hidden_max_abs_error,
        tolerances.hidden_mean_abs_error,
        tolerances.hidden_min_cosine,
        tolerances.hidden_max_relative_l2
      ),
      alignment_check(
        "logits",
        python["logits"],
        elixir["logits"],
        tolerances.logits_max_abs_error,
        tolerances.logits_mean_abs_error,
        tolerances.logits_min_cosine,
        tolerances.logits_max_relative_l2
      ),
      exact_check("argmax_agent_id", python["argmax_agent_id"], elixir["argmax_agent_id"]),
      exact_check("argmax_role_id", python["argmax_role_id"], elixir["argmax_role_id"])
    ]

    failed_required = Enum.count(checks, &(&1["required"] and not &1["passed"]))

    %{
      "schema" => "trinity_sakana_router_trace_comparison.v1",
      "required_checks" => Enum.count(checks, & &1["required"]),
      "failed_required" => failed_required,
      "tolerances" => stringify_tolerances(tolerances),
      "checks" => checks
    }
  end

  defp build_report(context) do
    extraction = context.extraction
    route = context.route
    hidden_vector = Map.get(extraction, :vector_snapshot, extraction.vector)
    hidden_f32 = f32_host_snapshot(hidden_vector)
    head_f32 = f32_host_snapshot(context.head)
    logits_f32 = f32_host_snapshot(route.logits)
    agent_logits_f32 = f32_host_snapshot(route.agent_logits)
    role_logits_f32 = f32_host_snapshot(route.role_logits)
    input_ids = tensor_list(extraction.input_ids)

    %{
      "schema" => @schema,
      "runtime" => "elixir.bumblebee",
      "artifact_dir" => context.artifact_dir,
      "artifact_manifest_sha256" => Artifact.file_sha256!(context.manifest_path),
      "messages" => context.messages,
      "transcript" => extraction.transcript,
      "transcript_sha256" => sha256_string(extraction.transcript),
      "input_ids" => input_ids,
      "token_ids_sha256" => sha256_json(input_ids),
      "attention_mask_shape" => shape_list(extraction.attention_mask),
      "hidden_state_shape" => Tuple.to_list(extraction.hidden_state_shape),
      "hidden_position" => extraction.hidden_position,
      "hidden_index" => extraction.hidden_index,
      "hidden_vector_shape" => Tuple.to_list(extraction.vector_shape),
      "hidden_vector_sha256_as_f32" => tensor_sha256_f32_host(hidden_f32),
      "hidden_vector_prefix_f32" => tensor_prefix_host(hidden_f32, 8),
      "hidden_vector_f32" => tensor_f32_list_host(hidden_f32),
      "head_weight_shape" => Tuple.to_list(Nx.shape(context.head)),
      "head_weight_sha256_as_f32" => tensor_sha256_f32_host(head_f32),
      "logits_shape" => Tuple.to_list(Nx.shape(route.logits)),
      "logits_sha256_as_f32" => tensor_sha256_f32_host(logits_f32),
      "logits" => tensor_f32_list_host(logits_f32),
      "agent_logits" => tensor_f32_list_host(agent_logits_f32),
      "role_logits" => tensor_f32_list_host(role_logits_f32),
      "argmax_agent_id" => route.agent_id,
      "argmax_role_id" => route.role_id,
      "role_name" => HITL.role_name(route.role_id),
      "notes" => [
        "Transcript formatting comes from TrinityCoordinator.Extractor.format_messages/1.",
        "Qwen hidden extraction runs on the adapted CUDA coordinator; the imported linear head is applied to a host f32 snapshot for trace inspectability.",
        "The required router trace gate is exact token/head/argmax parity plus declared hidden/logit tolerances."
      ]
    }
  end

  defp route_from_head_snapshot(hidden_vector, head, num_agents, num_roles) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      logits =
        hidden_vector
        |> Nx.as_type(:f32)
        |> Nx.dot(Nx.transpose(Nx.as_type(head, :f32)))

      logits_1d = Nx.squeeze(logits, axes: [0])
      agent_logits = Nx.slice(logits_1d, [0], [num_agents])
      role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

      %{
        logits: logits,
        agent_logits: agent_logits,
        role_logits: role_logits,
        agent_id: Nx.to_number(Nx.argmax(agent_logits)),
        role_id: Nx.to_number(Nx.argmax(role_logits))
      }
    end)
  end

  defp print_summary!(report) do
    if path = report["artifact_dir"], do: HITL.kv("Artifact dir", path)
    HITL.kv("Token ids sha256", report["token_ids_sha256"])
    HITL.kv("Hidden vector sha256 as f32", report["hidden_vector_sha256_as_f32"])
    HITL.kv("Head weight sha256 as f32", report["head_weight_sha256_as_f32"])
    HITL.kv("Logits sha256 as f32", report["logits_sha256_as_f32"])
    HITL.kv("Argmax agent id", report["argmax_agent_id"])
    HITL.kv("Argmax role id", report["argmax_role_id"])
    HITL.kv("Role name", report["role_name"])

    case report["comparison"] do
      %{"failed_required" => failed, "checks" => checks} ->
        HITL.kv("Router trace failed required checks", failed)

        checks
        |> Enum.reject(& &1["passed"])
        |> Enum.each(fn check ->
          HITL.kv("Failed check #{check["name"]}", Map.drop(check, ["name"]))
        end)

      _ ->
        :ok
    end
  end

  defp exact_check(name, expected, observed) do
    %{
      "name" => name,
      "required" => true,
      "kind" => "exact",
      "passed" => expected == observed,
      "expected" => expected,
      "observed" => observed
    }
  end

  defp alignment_check(
         name,
         expected,
         observed,
         max_abs_tolerance,
         mean_abs_tolerance,
         min_cosine,
         max_relative_l2
       )
       when is_list(expected) and is_list(observed) and length(expected) == length(observed) do
    errors =
      expected
      |> Enum.zip(observed)
      |> Enum.map(fn {left, right} -> abs(float(left) - float(right)) end)

    max_abs = Enum.max(errors, fn -> 0.0 end)
    mean_abs = Enum.sum(errors) / max(length(errors), 1)
    cosine = cosine_similarity(expected, observed)
    relative_l2 = relative_l2(expected, observed)

    %{
      "name" => name,
      "required" => true,
      "kind" => "alignment",
      "passed" => cosine >= min_cosine and relative_l2 <= max_relative_l2,
      "max_abs_error" => max_abs,
      "mean_abs_error" => mean_abs,
      "max_abs_tolerance" => max_abs_tolerance,
      "mean_abs_tolerance" => mean_abs_tolerance,
      "cosine_similarity" => cosine,
      "min_cosine" => min_cosine,
      "relative_l2_error" => relative_l2,
      "max_relative_l2" => max_relative_l2,
      "value_count" => length(errors)
    }
  end

  defp alignment_check(
         name,
         expected,
         observed,
         max_abs_tolerance,
         mean_abs_tolerance,
         min_cosine,
         max_relative_l2
       ) do
    %{
      "name" => name,
      "required" => true,
      "kind" => "alignment",
      "passed" => false,
      "expected_shape" => list_shape(expected),
      "observed_shape" => list_shape(observed),
      "max_abs_tolerance" => max_abs_tolerance,
      "mean_abs_tolerance" => mean_abs_tolerance,
      "min_cosine" => min_cosine,
      "max_relative_l2" => max_relative_l2,
      "reason" => "non-list or length mismatch"
    }
  end

  defp read_json!(path) do
    path
    |> File.read!()
    |> Jason.decode!()
  end

  defp f32_host_snapshot(%Nx.Tensor{} = tensor) do
    tensor
    |> Nx.as_type(:f32)
    |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  defp tensor_sha256_f32_host(%Nx.Tensor{} = tensor) do
    :crypto.hash(:sha256, Nx.to_binary(tensor))
    |> Base.encode16(case: :lower)
  end

  defp tensor_f32_list_host(%Nx.Tensor{} = tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&float/1)
  end

  defp tensor_list(nil), do: nil

  defp tensor_list(%Nx.Tensor{} = tensor) do
    tensor
    |> Nx.backend_transfer(Nx.BinaryBackend)
    |> Nx.to_flat_list()
    |> Enum.map(fn
      value when is_integer(value) -> value
      value when is_float(value) -> trunc(value)
    end)
  end

  defp tensor_prefix_host(%Nx.Tensor{} = tensor, count) do
    tensor
    |> tensor_f32_list_host()
    |> Enum.take(count)
  end

  defp shape_list(nil), do: nil
  defp shape_list(%Nx.Tensor{} = tensor), do: Tuple.to_list(Nx.shape(tensor))

  defp sha256_string(value) when is_binary(value) do
    :crypto.hash(:sha256, value)
    |> Base.encode16(case: :lower)
  end

  defp sha256_json(value) do
    value
    |> Jason.encode!()
    |> sha256_string()
  end

  defp stringify_tolerances(tolerances) do
    Map.new(tolerances, fn {key, value} -> {Atom.to_string(key), value} end)
  end

  defp float(value) when is_float(value), do: value
  defp float(value) when is_integer(value), do: value / 1

  defp cosine_similarity(left, right) do
    dot =
      left
      |> Enum.zip(right)
      |> Enum.reduce(0.0, fn {left_value, right_value}, acc ->
        acc + float(left_value) * float(right_value)
      end)

    left_norm = vector_norm(left)
    right_norm = vector_norm(right)

    if left_norm == 0.0 or right_norm == 0.0 do
      0.0
    else
      dot / (left_norm * right_norm)
    end
  end

  defp relative_l2(left, right) do
    diff_norm =
      left
      |> Enum.zip(right)
      |> Enum.reduce(0.0, fn {left_value, right_value}, acc ->
        diff = float(left_value) - float(right_value)
        acc + diff * diff
      end)
      |> :math.sqrt()

    diff_norm / max(vector_norm(left), 1.0e-12)
  end

  defp vector_norm(values) do
    values
    |> Enum.reduce(0.0, fn value, acc ->
      value = float(value)
      acc + value * value
    end)
    |> :math.sqrt()
  end

  defp list_shape(value) when is_list(value), do: [length(value)]
  defp list_shape(_), do: :not_a_list
end
