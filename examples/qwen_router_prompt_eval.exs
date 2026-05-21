defmodule Examples.QwenRouterPromptEval do
  @moduledoc false

  require Logger

  alias TrinityCoordinator.{HITL, RoleInjector, Runtime, Trace}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator}

  @default_artifact_dir "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
  @native_log_path "tmp/examples/qwen_router_prompt_eval.native.log"

  @agent_names %{
    0 => "gpt-5",
    1 => "claude-sonnet-4-20250514",
    2 => "gemini-2.5-pro",
    3 => "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    4 => "google/gemma-3-27b-it",
    5 => "Qwen/Qwen3-32B (reasoning)",
    6 => "Qwen/Qwen3-32B (direct)"
  }

  @cases_fixture_path Path.join(["examples", "fixtures", "qwen_router_prompt_eval_cases.json"])

  # Phase 11 Option D (margin-floor ratchet):
  # 80% of the empirical worst observed in qwen_router_prompt_eval_logits.json
  # on 2026-05-20 (agent worst = 0.301 on `unicode_emoji`, role worst = 1.335
  # on `root_cause`). Pass `--min-agent-margin 0.0` / `--min-role-margin 0.0`
  # to disable for one-off debug. See ~/jb/docs/20260520/sakana/04_margin_floor_ratchet.md.
  @default_min_agent_margin 0.24
  @default_min_role_margin 1.06

  defp load_cases! do
    body = File.read!(@cases_fixture_path)
    doc = Jason.decode!(body)

    cases =
      Enum.map(doc["cases"], fn c ->
        %{
          id: c["id"],
          tags: c["tags"] || [],
          notes: c["notes"],
          expected: %{
            agent_id: c["expected"]["agent_id"],
            role_id: c["expected"]["role_id"]
          },
          messages:
            Enum.map(c["messages"], fn m ->
              %{role: m["role"], content: m["content"]}
            end)
        }
      end)

    {cases, doc["coverage"] || %{}}
  end

  defp all_cases do
    {cases, _coverage} = load_cases!()
    cases
  end

  def main(argv) do
    argv = normalize_argv(argv)
    maybe_reexec_with_suppressed_stderr!(argv)

    unless "--debug-native-logs" in argv do
      Logger.configure(level: :error)
    end

    Application.ensure_all_started(:trinity_coordinator)

    {opts, rest, errors} =
      argv
      |> OptionParser.parse(
        strict: [
          artifact_dir: :string,
          case: :keep,
          debug_native_logs: :boolean,
          determinism_runs: :integer,
          list_cases: :boolean,
          min_agent_margin: :float,
          min_role_margin: :float,
          no_assert: :boolean,
          show_logits: :boolean,
          snapshot: :string,
          snapshot_out: :string,
          suppress_native_logs_child: :boolean,
          verbose: :boolean
        ]
      )

    unless rest == [], do: raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: raise("Invalid options: #{inspect(errors)}")

    if Keyword.get(opts, :list_cases, false) do
      list_cases()
    else
      run_eval!(opts)
    end
  end

  defp normalize_argv(["--" | rest]), do: rest
  defp normalize_argv(argv), do: argv

  defp maybe_reexec_with_suppressed_stderr!(argv) do
    cond do
      "--suppress-native-logs-child" in argv ->
        :ok

      "--list-cases" in argv ->
        :ok

      "--debug-native-logs" in argv ->
        :ok

      true ->
        File.mkdir_p!(Path.dirname(@native_log_path))
        File.rm(@native_log_path)

        shell = """
        stderr_path=$1
        script_path=$2
        shift 2
        mix run "$script_path" -- --suppress-native-logs-child "$@" 2>"$stderr_path"
        """

        {_output, status} =
          System.cmd("sh", ["-c", shell, "sh", @native_log_path, __ENV__.file | argv],
            into: IO.stream(:stdio, :line)
          )

        if status != 0 do
          IO.puts(:stderr, "\nqwen_router_prompt_eval failed.")
          IO.puts(:stderr, "Native/framework logs were captured at #{@native_log_path}.")
          IO.puts(:stderr, "Re-run with --debug-native-logs to see them inline.\n")
          print_stderr_tail(@native_log_path)
        end

        System.halt(status)
    end
  end

  defp print_stderr_tail(path) do
    if File.exists?(path) do
      path
      |> File.read!()
      |> String.split("\n")
      |> Enum.reject(&(&1 == ""))
      |> Enum.take(-40)
      |> case do
        [] ->
          :ok

        lines ->
          IO.puts(:stderr, "Last stderr lines:")
          Enum.each(lines, &IO.puts(:stderr, &1))
      end
    end
  end

  defp list_cases do
    Enum.each(all_cases(), fn case_spec ->
      IO.puts(case_spec.id)
    end)
  end

  defp run_eval!(opts) do
    artifact_dir = Keyword.get(opts, :artifact_dir, @default_artifact_dir)
    manifest_path = Artifact.manifest_path(artifact_dir)
    ensure_manifest!(manifest_path)

    selected_cases = select_cases!(Keyword.get_values(opts, :case))
    assert? = not Keyword.get(opts, :no_assert, false)
    show_logits? = Keyword.get(opts, :show_logits, false)
    verbose? = Keyword.get(opts, :verbose, false) or show_logits?
    determinism_runs = max(1, Keyword.get(opts, :determinism_runs, 1))
    min_agent_margin = Keyword.get(opts, :min_agent_margin, @default_min_agent_margin)
    min_role_margin = Keyword.get(opts, :min_role_margin, @default_min_role_margin)
    snapshot_in = Keyword.get(opts, :snapshot)
    snapshot_out = Keyword.get(opts, :snapshot_out)
    snapshot_expected = if snapshot_in, do: Jason.decode!(File.read!(snapshot_in)), else: nil

    HITL.banner("QWEN ROUTER PROMPT EVAL")
    Runtime.put_cuda_backend!()

    coordinator = TrinityCoordinator.MixHelpers.load_coordinator!(artifact_dir: artifact_dir)

    IO.puts("""

    What this does
      Loads the local adapted Qwen router once, sends fixed transcripts through it,
      and checks whether the selected agent slot and role match expectations.

      No external LLM/provider calls are made.

    Agent labels
      The agent names are labels from the original Sakana checkpoint.
      Example: agent 4 is labeled "google/gemma-3-27b-it".
      This eval does not call Gemma; it only reports that the router selected
      checkpoint slot 4.

    Artifact
      #{artifact_dir}

    Model
      base router model: #{coordinator.manifest["base_model_repo"]}
      router head shape: #{inspect(coordinator.manifest["router_head_shape"])}
      assertion mode: #{if(assert?, do: "strict", else: "report only")}

    Native logs
      hidden in normal mode: #{@native_log_path}
      use --debug-native-logs to print XLA/CUDA compiler logs inline
    """)

    results =
      selected_cases
      |> Enum.with_index(1)
      |> Enum.map(fn {case_spec, index} ->
        route_case!(
          coordinator,
          case_spec,
          index,
          length(selected_cases),
          assert?,
          show_logits?,
          verbose?,
          min_agent_margin: min_agent_margin,
          min_role_margin: min_role_margin,
          snapshot_expected: snapshot_expected
        )
      end)

    determinism_failures =
      if determinism_runs > 1 do
        verify_determinism!(coordinator, selected_cases, results, determinism_runs)
      else
        []
      end

    if snapshot_out do
      write_snapshot!(results, snapshot_out)
    end

    failures =
      Enum.filter(results, &(&1.status == :fail)) ++
        Enum.map(determinism_failures, fn id -> %{id: id, status: :fail} end)

    if failures == [] do
      print_summary(results)
    else
      ids = failures |> Enum.map(& &1.id) |> Enum.join(", ")
      raise "qwen_router_prompt_eval failed cases=#{ids}"
    end
  end

  defp select_cases!([]), do: all_cases()

  defp select_cases!(ids) do
    cases_by_id = Map.new(all_cases(), &{&1.id, &1})

    Enum.map(ids, fn id ->
      Map.get(cases_by_id, id) || raise("Unknown case #{inspect(id)}. Run with --list-cases.")
    end)
  end

  defp ensure_manifest!(manifest_path) do
    unless File.exists?(manifest_path) do
      raise """
      Missing adapted artifact manifest: #{manifest_path}

      Install the canonical artifact bundle at #{@default_artifact_dir}, or pass
      --artifact-dir path/to/adapted_qwen3_0_6b_layer26.
      """
    end
  end

  defp route_case!(coordinator, case_spec, index, total, assert?, show_logits?, verbose?, extras)
       when is_list(extras) do
    {:ok, routed} = Coordinator.route_messages(coordinator, case_spec.messages)

    route = routed.route
    expected = case_spec.expected
    actual = %{agent_id: route.agent_id, role_id: route.role_id}

    agent_logits_list = Nx.to_flat_list(route.agent_logits)
    role_logits_list = Nx.to_flat_list(route.role_logits)
    agent_margin = top_margin(route.agent_logits)
    role_margin = top_margin(route.role_logits)

    snapshot_expected = Keyword.get(extras, :snapshot_expected)
    min_agent_margin = Keyword.get(extras, :min_agent_margin)
    min_role_margin = Keyword.get(extras, :min_role_margin)

    transcript_hash =
      case_spec.messages
      |> :erlang.term_to_binary([:compressed])
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)

    token_count = routed.extraction.input_ids |> Nx.to_flat_list() |> length()

    status =
      cond do
        not assert? ->
          :report

        not expectation_matches?(expected, actual) ->
          :fail

        not margin_ok?(:agent, agent_margin, min_agent_margin) ->
          :fail

        not margin_ok?(:role, role_margin, min_role_margin) ->
          :fail

        not snapshot_ok?(
          case_spec.id,
          route.agent_id,
          route.role_id,
          token_count,
          transcript_hash,
          snapshot_expected
        ) ->
          :fail

        true ->
          :ok
      end

    print_case(case_spec, routed, index, total, status, show_logits?, verbose?)

    %{
      id: case_spec.id,
      status: status,
      role_id: route.role_id,
      agent_id: route.agent_id,
      agent_margin: agent_margin,
      role_margin: role_margin,
      agent_logits_rounded: Enum.map(agent_logits_list, &Float.round(&1, 6)),
      role_logits_rounded: Enum.map(role_logits_list, &Float.round(&1, 6)),
      token_count: token_count,
      transcript_hash: transcript_hash,
      route_hash: route_hash(route)
    }
  end

  defp margin_ok?(_kind, _margin, nil), do: true
  defp margin_ok?(_kind, :infinity, _min), do: true
  defp margin_ok?(_kind, margin, min) when is_number(margin) and is_number(min), do: margin >= min
  defp margin_ok?(_, _, _), do: false

  # Snapshot assertion semantics. Asserts decision-stable invariants only:
  # agent_id, role_id, token_count, transcript_hash. Raw logits are recorded
  # in the snapshot for diagnostic use but not asserted, because CUDA kernel
  # selection and JIT compilation cause O(1) drift in raw logits across
  # process launches even when argmax is bytewise stable within a single
  # process. In-process logit stability is covered by `--determinism-runs N`.
  defp snapshot_ok?(_id, _agent_id, _role_id, _token_count, _transcript_hash, nil), do: true

  defp snapshot_ok?(id, agent_id, role_id, token_count, transcript_hash, snapshot)
       when is_map(snapshot) do
    expected_case = Enum.find(snapshot["cases"] || [], &(&1["id"] == id))

    case expected_case do
      nil ->
        true

      e ->
        e["agent_id"] == agent_id and
          e["role_id"] == role_id and
          e["token_count"] == token_count and
          e["transcript_hash"] == transcript_hash
    end
  end

  defp expectation_matches?(expected, actual) do
    expected.agent_id == actual.agent_id and expected.role_id == actual.role_id
  end

  defp print_case(case_spec, routed, index, total, status, show_logits?, verbose?) do
    extraction = routed.extraction
    route = routed.route
    expected = case_spec.expected
    token_count = extraction.input_ids |> Nx.to_flat_list() |> length()
    prompt = format_prompt(case_spec.messages)

    IO.puts("""

    [#{index}/#{total}] #{case_spec.id} - #{status_label(status)}

    Prompt sent to router:
    #{indent(prompt, 2)}

    Expected route:
      agent #{expected.agent_id}: #{Map.fetch!(@agent_names, expected.agent_id)}
      role  #{expected.role_id}: #{RoleInjector.role_name(expected.role_id)}

    Router returned:
      agent #{route.agent_id}: #{Map.fetch!(@agent_names, route.agent_id)}
      role  #{route.role_id}: #{RoleInjector.role_name(route.role_id)}

    Router input tokens: #{token_count}
    """)

    if verbose? do
      IO.puts("""
      Debug:
        hidden_index: #{extraction.hidden_index}
        transcript_hash: #{Trace.Hash.messages(case_spec.messages)}
        route_vector_hash: #{Trace.Hash.tensor(extraction.vector_snapshot)}
        agent_margin: #{format_float(top_margin(route.agent_logits))}
        role_margin: #{format_float(top_margin(route.role_logits))}
      """)
    end

    if show_logits? do
      IO.puts("      agent_logits: #{inspect(round_list(Nx.to_flat_list(route.agent_logits)))}")
      IO.puts("      role_logits: #{inspect(round_list(Nx.to_flat_list(route.role_logits)))}")
    end
  end

  defp status_label(:ok), do: "PASS"
  defp status_label(:fail), do: "FAIL"
  defp status_label(:report), do: "REPORT ONLY"

  defp format_prompt(messages) do
    messages
    |> Enum.map(fn message ->
      role = Map.get(message, :role, Map.get(message, "role"))
      content = Map.get(message, :content, Map.get(message, "content"))
      "#{role}: #{content}"
    end)
    |> Enum.join("\n")
  end

  defp indent(text, spaces) do
    padding = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map_join("\n", &(padding <> &1))
  end

  defp print_summary(results) do
    passed = Enum.count(results, &(&1.status in [:ok, :report]))
    failed = Enum.count(results, &(&1.status == :fail))

    role_counts =
      results
      |> Enum.frequencies_by(& &1.role_id)
      |> Enum.sort()
      |> Enum.map_join(", ", fn {role_id, count} ->
        "#{RoleInjector.role_name(role_id)}=#{count}"
      end)

    agent_counts =
      results
      |> Enum.frequencies_by(& &1.agent_id)
      |> Enum.sort()
      |> Enum.map_join(", ", fn {agent_id, count} ->
        "#{agent_id}=#{count}"
      end)

    IO.puts("""

    Summary
      passed: #{passed}
      failed: #{failed}
      roles selected: #{role_counts}
      agent slots selected: #{agent_counts}

    PASS qwen_router_prompt_eval
    """)
  end

  defp top_margin(%Nx.Tensor{} = logits) do
    logits
    |> Nx.to_flat_list()
    |> Enum.sort(:desc)
    |> case do
      [first, second | _] -> first - second
      [_only] -> 0.0
      [] -> 0.0
    end
  end

  defp round_list(values) do
    Enum.map(values, fn
      value when is_float(value) -> Float.round(value, 5)
      value -> value
    end)
  end

  defp format_float(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 5)
  defp format_float(value), do: inspect(value)

  defp write_snapshot!(results, path) do
    File.mkdir_p!(Path.dirname(path))

    payload = %{
      "schema_version" => 1,
      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "cases" =>
        Enum.map(results, fn r ->
          %{
            "id" => r.id,
            "agent_id" => r.agent_id,
            "role_id" => r.role_id,
            "token_count" => Map.get(r, :token_count),
            "agent_margin" => Map.get(r, :agent_margin),
            "role_margin" => Map.get(r, :role_margin),
            "agent_logits_rounded" => Map.get(r, :agent_logits_rounded),
            "role_logits_rounded" => Map.get(r, :role_logits_rounded),
            "transcript_hash" => Map.get(r, :transcript_hash),
            "route_hash" => Map.get(r, :route_hash)
          }
        end)
    }

    File.write!(path, Jason.encode!(payload, pretty: true))
    IO.puts("\nSnapshot written: #{path} (#{length(results)} cases)")
  end

  defp verify_determinism!(coordinator, cases, baseline_results, runs) when runs > 1 do
    baseline = Map.new(baseline_results, fn r -> {r.id, r.route_hash} end)

    Enum.reduce(2..runs, [], fn run_index, mismatches ->
      Enum.reduce(cases, mismatches, fn case_spec, acc ->
        {:ok, %{route: route}} = Coordinator.route_messages(coordinator, case_spec.messages)
        hash = route_hash(route)

        if baseline[case_spec.id] == hash do
          acc
        else
          IO.puts("  determinism mismatch case=#{case_spec.id} run=#{run_index}")
          [case_spec.id | acc]
        end
      end)
    end)
  end

  defp route_hash(route) do
    payload = [
      route.agent_id,
      route.role_id,
      route.logits |> Nx.to_flat_list() |> Enum.map(&Float.round(&1, 6))
    ]

    payload
    |> :erlang.term_to_binary([:compressed])
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end
end

Examples.QwenRouterPromptEval.main(System.argv())
