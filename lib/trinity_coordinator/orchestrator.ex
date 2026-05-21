defmodule TrinityCoordinator.Orchestrator do
  @moduledoc """
  Orchestrates a real TRINITY multi-turn routing loop.

  The core path remains real SLM hidden-state extraction plus Axon routing.  The
  provider call boundary can be replaced with `:mock_agent_fn` for foundational
  bring-up and tests where live LLM calls are intentionally out of scope.
  """

  require Logger

  alias TrinityCoordinator.{
    AgentPool,
    CoordinationHead,
    Extractor,
    GovernedAuthority,
    RoleInjector,
    Runtime,
    StateManager,
    Thinker,
    Trace,
    Verifier
  }

  alias TrinityCoordinator.Sakana.Artifact

  # Imported Sakana/TRINITY checkpoints emit role logits in the supplemental
  # Python order: solver, thinker, verifier. Public docs call solver "Worker".
  @roles %{0 => "Worker", 1 => "Thinker", 2 => "Verifier"}
  @default_max_turns 5

  @doc """
  Run loop with keyword options:

  - `:max_turns` – stop after this many turns if no termination.
  - `:slm_context` – `%{model_info: ...}` or `{model_info, tokenizer}` for real extraction.
  - `:extractor_fn` – optional test hook. Called as `(messages, slm_context)` or `(messages)`.
  - `:mock_agent_fn` – optional provider hook. Called as `(role_atom, messages)` or
    `(role_atom, messages, metadata)`.
  - `:stop_token` – verifier termination token (default `"ACCEPT"`).
  - `:agent_pool_opts` – custom options passed through to `AgentPool`.
  - `:provider_pool` – pool name or explicit pool spec list.
  - `:roles` – optional role-map for index->name decoding.
  - `:num_agents` – number of agent logits in the coordination head.
  - `:num_roles` – number of role logits in the coordination head.
  - `:route_opts` – optional `CoordinationHead.route/6` selection options.
  - `:trace` – trace options (enabled, sink, run_id, content).

  Cost / time budgets (all default `nil` = unbounded). When a budget is
  exceeded the loop returns `{:error, {:budget_exceeded, kind, details}}`
  and emits a `:run_failed` trace event with the same kind and details.

  - `:max_wall_time_ms` – wall-clock cap, checked at `:turn_start`.
  - `:max_provider_calls` – at most N total dispatches; the (N+1)th
    attempt aborts at `:before_dispatch`.
  - `:max_provider_latency_ms` – aborts immediately after a single
    dispatch whose `:provider_latency_ms` exceeds the limit
    (`checkpoint: :after_dispatch`).
  - `:max_verifier_revisions` – counts Verifier dispatches that did
    NOT accept. The (N+1)th rejection aborts at
    `:after_verifier_revision`.
  - `:max_estimated_cost_usd` – aborts when cumulative cost exceeds
    the limit. Requires a `:cost_estimator_fn` to fire; without one,
    a single `Logger.warning/1` is emitted per run_loop call.
  - `:cost_estimator_fn` – `(dispatch_map) :: float()` returning the
    USD cost for the just-completed dispatch. Called only when
    `:max_estimated_cost_usd` is set. `dispatch_map` carries
    `:provider`, `:provider_model`, `:response_text`, `:mode`,
    `:provider_latency_ms`.
  """
  def run_loop(pid, model, params, opts \\ [])

  def run_loop(pid, model, params, opts) when is_list(opts) do
    with {:ok, opts} <- GovernedAuthority.materialize_orchestrator_opts(opts) do
      do_run_loop(pid, model, params, opts)
    end
  end

  def run_loop(pid, model, params, max_turns) when is_integer(max_turns),
    do: run_loop(pid, model, params, max_turns: max_turns)

  def run_loop(pid, model, params, max_turns, slm_context) when is_integer(max_turns),
    do:
      run_loop(pid, model, params,
        max_turns: max_turns,
        slm_context: slm_context
      )

  defp do_run_loop(pid, model, params, opts) do
    max_turns = Keyword.get(opts, :max_turns, @default_max_turns)
    slm_context = Keyword.get(opts, :slm_context)
    stop_token = Keyword.get(opts, :stop_token, "ACCEPT")
    roles = Keyword.get(opts, :roles, @roles)
    agent_pool_opts = Keyword.get(opts, :agent_pool_opts, [])

    num_agents =
      Keyword.get(opts, :num_agents) ||
        AgentPool.agent_count(Keyword.get(opts, :provider_pool, :default))

    num_roles = Keyword.get(opts, :num_roles, 3)
    trace = Trace.Context.new(Keyword.get(opts, :trace, []))

    budgets =
      Keyword.get(opts, :budgets, %{})
      |> Map.put_new(:max_wall_time_ms, Keyword.get(opts, :max_wall_time_ms))
      |> Map.put_new(:max_provider_calls, Keyword.get(opts, :max_provider_calls))
      |> Map.put_new(:max_provider_latency_ms, Keyword.get(opts, :max_provider_latency_ms))
      |> Map.put_new(:max_verifier_revisions, Keyword.get(opts, :max_verifier_revisions))
      |> Map.put_new(:max_estimated_cost_usd, Keyword.get(opts, :max_estimated_cost_usd))

    counters = %{
      started_monotonic_ms: System.monotonic_time(:millisecond),
      provider_calls_ref: :counters.new(1, [:atomics]),
      verifier_revisions_ref: :counters.new(1, [:atomics]),
      estimated_cost_micro_usd_ref: :counters.new(1, [:atomics]),
      cost_warning_emitted_ref: :counters.new(1, [:atomics])
    }

    run_ctx = %{
      roles: roles,
      stop_token: stop_token,
      agent_pool_opts: agent_pool_opts,
      provider_pool: Keyword.get(opts, :provider_pool),
      num_agents: num_agents,
      num_roles: num_roles,
      mock_agent_fn: Keyword.get(opts, :mock_agent_fn),
      extractor_fn: Keyword.get(opts, :extractor_fn),
      route_opts: Keyword.get(opts, :route_opts, []),
      respect_thinker_suggestions: Keyword.get(opts, :respect_thinker_suggestions, true),
      budgets: budgets,
      counters: counters,
      cost_estimator_fn: Keyword.get(opts, :cost_estimator_fn)
    }

    runtime_metadata = build_runtime_metadata(opts[:slm_context])

    case validate_loop_input(pid, model, params) do
      {:ok, _} ->
        emit_trace(
          trace,
          :run_started,
          %{
            runtime_metadata: runtime_metadata,
            max_turns: max_turns,
            num_agents: num_agents,
            num_roles: num_roles,
            provider_mode: if(run_ctx.mock_agent_fn, do: :mock, else: :live)
          }
        )

        do_run_loop(
          pid,
          {model, params},
          0,
          max_turns,
          slm_context,
          run_ctx,
          trace,
          initial_loop_state(pid)
        )

      error ->
        error
    end
  end

  defp build_runtime_metadata(slm_context) do
    model_info =
      case slm_context do
        {model_info, _tokenizer} when is_map(model_info) -> model_info
        %{model_info: model_info} when is_map(model_info) -> model_info
        _ -> nil
      end

    if model_info do
      Artifact.trace_metadata(model_info)
    else
      %{}
    end
  end

  defp validate_loop_input(pid, model, params) do
    cond do
      not is_pid(pid) -> {:error, :invalid_state_pid}
      model == nil -> {:error, :invalid_model}
      params == nil -> {:error, :invalid_params}
      true -> {:ok, :ok}
    end
  end

  defp initial_loop_state(pid) do
    %{
      latest_worker_response: latest_assistant_response(pid),
      suggested_role: nil,
      suggested_role_id: nil,
      suggestion: nil
    }
  end

  defp latest_assistant_response(pid) do
    pid
    |> StateManager.get_messages()
    |> Enum.reverse()
    |> Enum.find_value(fn
      %{role: "assistant", content: content} when is_binary(content) -> content
      _ -> nil
    end)
  end

  defp do_run_loop(_pid, _routing, turn, max_turns, _slm_context, _run_ctx, trace, state)
       when turn >= max_turns do
    case state.latest_worker_response do
      response_text when is_binary(response_text) ->
        emit_trace(trace, :run_completed, %{
          turn: turn,
          final_status: :max_turns_latest_worker_response,
          response_hash: Trace.Hash.text(response_text)
        })

        {:ok, response_text}

      _ ->
        emit_trace(trace, :run_failed, %{reason: :max_turns_reached})
        {:error, :max_turns_reached}
    end
  end

  defp do_run_loop(
         pid,
         routing,
         turn,
         max_turns,
         slm_context,
         run_ctx,
         trace,
         state
       ) do
    case check_budgets(run_ctx, :turn_start, %{turn: turn}) do
      {:budget_exceeded, kind, details} ->
        emit_trace(trace, :run_failed, %{reason: {:budget_exceeded, kind}, details: details})
        {:error, {:budget_exceeded, kind, details}}

      :ok ->
        do_turn(pid, routing, turn, max_turns, slm_context, run_ctx, trace, state)
    end
  end

  defp do_turn(pid, {model, params}, turn, max_turns, slm_context, run_ctx, trace, state) do
    messages = StateManager.get_messages(pid)
    transcript_hash = Trace.Hash.messages(messages)

    with :ok <-
           emit_trace(trace, :turn_started, %{
             turn: turn,
             max_turns: max_turns,
             transcript_hash: transcript_hash,
             message_count: length(messages)
           }),
         {:ok, extraction} <- extract_router_tensor(messages, slm_context, run_ctx),
         :ok <- emit_extraction_trace(trace, extraction, messages, turn),
         route <-
           CoordinationHead.route(
             model,
             params,
             extraction.vector,
             run_ctx.num_agents,
             run_ctx.num_roles,
             run_ctx.route_opts
           )
           |> apply_thinker_suggestion(run_ctx, state),
         :ok <- emit_route_trace(trace, route, extraction.vector, run_ctx.roles, turn),
         role_name = role_name_for(run_ctx.roles, route.role_id),
         role_atom = RoleInjector.role_atom(role_name),
         :ok <- ensure_role_dispatch_allowed(role_atom, state),
         state_for_turn <- clear_consumed_suggestion(state, route),
         injected_messages <- RoleInjector.inject_role(messages, role_name),
         {:ok, dispatch_started} <-
           emit_dispatch_started(trace, turn, run_ctx, route, role_name),
         :ok <-
           bump_and_check_provider_budget(run_ctx, trace, turn),
         {:ok, dispatch} <-
           dispatch_agent_timed(
             run_ctx.mock_agent_fn,
             role_atom,
             role_name,
             injected_messages,
             route.agent_id,
             run_ctx.agent_pool_opts,
             run_ctx.provider_pool
           ),
         :ok <-
           check_latency_budget(run_ctx, trace, turn, dispatch.provider_latency_ms),
         :ok <-
           bump_and_check_cost_budget(run_ctx, trace, turn, dispatch) do
      response_text = dispatch.response_text
      StateManager.append_assistant(pid, response_text)

      verifier_result =
        if Verifier.verifier_role?(role_name) do
          Verifier.parse(response_text, stop_token: run_ctx.stop_token)
        else
          %Verifier{status: :revised, raw: response_text, token: nil, diagnosis: nil}
        end

      verifier_status = Verifier.safe_status(verifier_result)
      accepted? = verifier_status == :accepted
      thinker_result = maybe_parse_thinker(role_atom, response_text)
      next_state = update_loop_state(state_for_turn, role_atom, response_text, thinker_result)

      route_decision =
        TrinityCoordinator.RouteDecision.from_route(route, transcript_hash,
          artifact_identity: build_artifact_identity(slm_context)
        )

      emit_trace(
        trace,
        :provider_called,
        %{
          turn: turn,
          provider: dispatch.provider,
          provider_model: dispatch.provider_model,
          provider_mode: dispatch.mode,
          provider_latency_ms: dispatch.provider_latency_ms,
          mock: dispatch.mode == :mock,
          selected_agent: route.agent_id,
          selected_role: route.role_id,
          selected_role_name: role_name,
          response_hash: Trace.Hash.text(response_text),
          status: :ok,
          dispatch_started: dispatch_started
        }
      )

      emit_turn_completed(trace, %{
        turn: turn,
        transcript_hash: transcript_hash,
        selected_agent: route.agent_id,
        selected_role: role_name,
        selected_role_id: route.role_id,
        provider: dispatch.provider,
        provider_model: dispatch.provider_model,
        provider_mode: dispatch.mode,
        provider_latency_ms: dispatch.provider_latency_ms,
        mock: dispatch.mode == :mock,
        response_hash: Trace.Hash.text(response_text),
        selected_agent_logits: Nx.to_flat_list(route.agent_logits),
        selected_role_logits: Nx.to_flat_list(route.role_logits),
        logits: Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0])),
        vector_shape: extraction.vector_shape,
        hidden_state_shape: extraction.hidden_state_shape,
        vector_backend: Runtime.tensor_backend(extraction.vector),
        verifier_parse_status: verifier_result.status,
        verifier_status: verifier_status,
        verifier_diagnosis_hash: diagnosis_hash(verifier_result),
        thinker_suggested_role: suggested_role_for_trace(thinker_result),
        thinker_suggestion_hash: suggestion_hash(thinker_result),
        raw_selected_role_id: Map.get(route, :raw_role_id, route.role_id),
        raw_selected_role: Map.get(route, :raw_role_name, role_name),
        role_override_from_thinker: Map.get(route, :role_override_from_thinker, false),
        final: accepted?,
        route_decision: TrinityCoordinator.RouteDecision.to_trace_map(route_decision)
      })

      if accepted? do
        emit_trace(trace, :run_completed, %{
          turn: turn,
          final_status: :accepted,
          response_hash: Trace.Hash.text(response_text)
        })

        {:ok, response_text}
      else
        continue_loop_after_turn(
          pid,
          {model, params},
          turn,
          max_turns,
          slm_context,
          run_ctx,
          trace,
          {next_state, role_name, verifier_status}
        )
      end
    else
      {:error, :verifier_before_worker_response} ->
        emit_trace(trace, :run_failed, %{turn: turn, reason: :verifier_before_worker_response})
        {:error, :verifier_before_worker_response}

      {:error, {:provider_dispatch_failed, reason, provider_latency_ms}} ->
        emit_trace(
          trace,
          :provider_called,
          %{
            turn: turn,
            provider_mode: if(run_ctx.mock_agent_fn, do: :mock, else: :live),
            provider_latency_ms: provider_latency_ms,
            mock: not is_nil(run_ctx.mock_agent_fn),
            status: :error,
            error: inspect(reason)
          }
        )

        emit_trace(trace, :run_failed, %{turn: turn, reason: reason})
        {:error, reason}

      {:error, reason} ->
        emit_trace(
          trace,
          :provider_called,
          %{
            turn: turn,
            provider_mode: if(run_ctx.mock_agent_fn, do: :mock, else: :live),
            mock: not is_nil(run_ctx.mock_agent_fn),
            status: :error,
            error: inspect(reason)
          }
        )

        emit_trace(trace, :run_failed, %{turn: turn, reason: reason})
        {:error, reason}

      _ ->
        emit_trace(trace, :run_failed, %{turn: turn, reason: :unexpected_orchestrator_state})
        {:error, :unexpected_orchestrator_state}
    end
  end

  defp continue_loop_after_turn(
         pid,
         routing,
         turn,
         max_turns,
         slm_context,
         run_ctx,
         trace,
         {next_state, role_name, verifier_status}
       ) do
    case bump_and_check_verifier_revisions(run_ctx, trace, turn, role_name, verifier_status) do
      {:error, _} = err ->
        err

      :ok ->
        do_run_loop(pid, routing, turn + 1, max_turns, slm_context, run_ctx, trace, next_state)
    end
  end

  defp dispatch_agent_timed(mock_fn, role_atom, role_name, messages, agent_id, opts, pool) do
    start_time = System.monotonic_time(:millisecond)

    case dispatch_agent(mock_fn, role_atom, role_name, messages, agent_id, opts, pool) do
      {:ok, dispatch} ->
        {:ok, Map.put(dispatch, :provider_latency_ms, elapsed_ms(start_time))}

      {:error, reason} ->
        {:error, {:provider_dispatch_failed, reason, elapsed_ms(start_time)}}
    end
  end

  defp elapsed_ms(start_time), do: max(System.monotonic_time(:millisecond) - start_time, 0)

  defp role_name_for(roles, role_id) when is_map(roles) do
    roles
    |> Map.get(role_id, "Worker")
    |> RoleInjector.role_name()
  end

  defp apply_thinker_suggestion(route, %{respect_thinker_suggestions: true, roles: roles}, %{
         suggested_role: role_name,
         suggested_role_id: role_id,
         suggestion: suggestion
       })
       when is_binary(role_name) and is_integer(role_id) do
    route
    |> annotate_raw_role(roles)
    |> Map.put(:role_id, role_id)
    |> Map.put(:role_override_from_thinker, true)
    |> Map.put(:role_override_suggestion_hash, Trace.Hash.text(suggestion || ""))
  end

  defp apply_thinker_suggestion(route, %{roles: roles}, _state) do
    route
    |> annotate_raw_role(roles)
    |> Map.put(:role_override_from_thinker, false)
  end

  defp annotate_raw_role(route, roles) do
    raw_role_id = Map.get(route, :raw_role_id, route.role_id)

    route
    |> Map.put_new(:raw_role_id, raw_role_id)
    |> Map.put_new(:raw_role_name, role_name_for(roles, raw_role_id))
  end

  defp ensure_role_dispatch_allowed(:verifier, %{latest_worker_response: nil}) do
    {:error, :verifier_before_worker_response}
  end

  defp ensure_role_dispatch_allowed(_role, _state), do: :ok

  defp clear_consumed_suggestion(state, %{role_override_from_thinker: true}) do
    %{state | suggested_role: nil, suggested_role_id: nil, suggestion: nil}
  end

  defp clear_consumed_suggestion(state, _route), do: state

  defp maybe_parse_thinker(:thinker, response_text), do: Thinker.parse(response_text)
  defp maybe_parse_thinker(_role, _response_text), do: nil

  defp update_loop_state(state, :worker, response_text, _thinker_result) do
    %{
      state
      | latest_worker_response: response_text,
        suggested_role: nil,
        suggested_role_id: nil,
        suggestion: nil
    }
  end

  defp update_loop_state(state, :thinker, _response_text, %Thinker{} = thinker_result) do
    %{
      state
      | suggested_role: thinker_result.suggested_role,
        suggested_role_id: thinker_result.suggested_role_id,
        suggestion: thinker_result.suggestion
    }
  end

  defp update_loop_state(state, _role, _response_text, _thinker_result) do
    %{state | suggested_role: nil, suggested_role_id: nil, suggestion: nil}
  end

  defp suggested_role_for_trace(%Thinker{suggested_role: role}), do: role
  defp suggested_role_for_trace(_), do: nil

  defp suggestion_hash(%Thinker{suggestion: suggestion}) when is_binary(suggestion),
    do: Trace.Hash.text(suggestion)

  defp suggestion_hash(_), do: nil

  defp emit_dispatch_started(trace, turn, run_ctx, route, role_name) do
    case dispatch_preview(
           run_ctx.mock_agent_fn,
           route.agent_id,
           run_ctx.agent_pool_opts,
           run_ctx.provider_pool
         ) do
      {:ok, preview} ->
        emit_trace(trace, :provider_called, %{
          turn: turn,
          provider: preview.provider,
          provider_model: preview.provider_model,
          provider_base_url: preview.provider_base_url,
          provider_timeout_ms: preview.provider_timeout_ms,
          provider_max_tokens: preview.provider_max_tokens,
          provider_temperature: preview.provider_temperature,
          provider_mode: preview.mode,
          mock: preview.mode == :mock,
          selected_agent: route.agent_id,
          selected_role: route.role_id,
          selected_role_name: role_name,
          status: :started
        })

        {:ok, true}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp dispatch_preview(mock_fn, _agent_id, _opts, _pool) when is_function(mock_fn) do
    {:ok,
     %{
       provider: :mock,
       provider_model: "mock_agent_fn",
       provider_base_url: nil,
       provider_timeout_ms: nil,
       provider_max_tokens: nil,
       provider_temperature: nil,
       mode: :mock
     }}
  end

  defp dispatch_preview(nil, agent_id, opts, provider_pool) do
    opts = put_provider_pool(opts, provider_pool)

    with {:ok, spec} <- AgentPool.fetch_agent_spec(agent_id, opts) do
      {:ok,
       %{
         provider: spec.provider,
         provider_model: spec.model,
         provider_base_url: Map.get(spec, :base_url),
         provider_timeout_ms: Map.get(spec, :timeout_ms),
         provider_max_tokens: Map.get(spec, :max_tokens),
         provider_temperature: Map.get(spec, :temperature),
         mode: :live
       }}
    end
  end

  defp dispatch_agent(mock_fn, role_atom, role_name, messages, agent_id, _opts, _pool)
       when is_function(mock_fn, 2) do
    mock_fn.(role_atom, messages)
    |> normalize_dispatch_response(%{
      provider: :mock,
      provider_model: "mock_agent_fn",
      provider_role: role_name,
      selected_agent: agent_id,
      mode: :mock
    })
  end

  defp dispatch_agent(mock_fn, role_atom, role_name, messages, agent_id, _opts, _pool)
       when is_function(mock_fn, 3) do
    metadata = %{role_name: role_name, role: role_atom, agent_id: agent_id}

    mock_fn.(role_atom, messages, metadata)
    |> normalize_dispatch_response(%{
      provider: :mock,
      provider_model: "mock_agent_fn",
      provider_role: role_name,
      selected_agent: agent_id,
      mode: :mock
    })
  end

  defp dispatch_agent(nil, _role_atom, _role_name, messages, agent_id, opts, provider_pool) do
    opts = put_provider_pool(opts, provider_pool)

    with {:ok, spec} <- AgentPool.fetch_agent_spec(agent_id, opts),
         {:ok, response_text} <- AgentPool.call_agent_with_spec(spec, messages, opts) do
      {:ok,
       %{
         response_text: response_text,
         provider: spec.provider,
         provider_model: spec.model,
         mode: :live
       }}
    end
  end

  defp dispatch_agent(_mock_fn, _role_atom, _role_name, _messages, _agent_id, _opts, _pool) do
    {:error, :invalid_mock_agent_fn}
  end

  defp normalize_dispatch_response({:ok, response_text}, metadata)
       when is_binary(response_text) do
    {:ok, Map.put(metadata, :response_text, response_text)}
  end

  defp normalize_dispatch_response({:error, reason}, _metadata), do: {:error, reason}

  defp normalize_dispatch_response(response_text, metadata) when is_binary(response_text) do
    {:ok, Map.put(metadata, :response_text, response_text)}
  end

  defp normalize_dispatch_response(other, _metadata) do
    {:error, {:invalid_mock_agent_response, other}}
  end

  defp diagnosis_hash(%Verifier{diagnosis: nil}), do: nil
  defp diagnosis_hash(%Verifier{diagnosis: diagnosis}), do: Trace.Hash.text(diagnosis)

  defp emit_extraction_trace(trace, extraction, messages, turn) do
    emit_trace(trace, :slm_extracted, %{
      turn: turn,
      input_shapes: extraction.input_shapes,
      hidden_state_shape: extraction.hidden_state_shape,
      vector_shape: extraction.vector_shape,
      vector_backend: Runtime.tensor_backend(extraction.vector),
      transcript_hash: Trace.Hash.messages(messages)
    })
  end

  defp emit_route_trace(trace, route, vector, roles, turn) do
    raw_role_id = Map.get(route, :raw_role_id, route.role_id)

    emit_trace(trace, :route_selected, %{
      turn: turn,
      logits: Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0])),
      route_logit_shape: Nx.shape(route.logits),
      route_logit_backend: Runtime.tensor_backend(route.logits),
      vector_shape: Nx.shape(vector),
      vector_backend: Runtime.tensor_backend(vector),
      agent_logits: Nx.to_flat_list(route.agent_logits),
      role_logits: Nx.to_flat_list(route.role_logits),
      selected_agent: route.agent_id,
      selected_role: role_name_for(roles, route.role_id),
      selected_role_id: route.role_id,
      raw_selected_role: Map.get(route, :raw_role_name, role_name_for(roles, raw_role_id)),
      raw_selected_role_id: raw_role_id,
      role_override_from_thinker: Map.get(route, :role_override_from_thinker, false),
      role_override_suggestion_hash: Map.get(route, :role_override_suggestion_hash),
      agent_selection_mode: Map.get(route, :agent_selection_mode, :argmax),
      role_selection_mode: Map.get(route, :role_selection_mode, :argmax),
      selection_temperature: Map.get(route, :selection_temperature),
      selection_seed: Map.get(route, :selection_seed),
      agent_probabilities: tensor_to_list_or_nil(Map.get(route, :agent_probabilities)),
      role_probabilities: tensor_to_list_or_nil(Map.get(route, :role_probabilities))
    })
  end

  defp tensor_to_list_or_nil(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp tensor_to_list_or_nil(nil), do: nil

  defp emit_turn_completed(trace, fields), do: emit_trace(trace, :turn_completed, fields)

  defp emit_trace(%Trace.Context{} = trace, event, fields) do
    Trace.Context.write(trace, Trace.Event.new(event, trace.run_id, fields))
  end

  defp put_provider_pool(opts, nil), do: opts

  defp put_provider_pool(opts, provider_pool),
    do: Keyword.put(opts, :provider_pool, provider_pool)

  defp extract_router_tensor(messages, slm_context, %{extractor_fn: extractor_fn})
       when is_function(extractor_fn, 2) do
    extractor_fn.(messages, slm_context)
    |> normalize_extraction()
  end

  defp extract_router_tensor(messages, _slm_context, %{extractor_fn: extractor_fn})
       when is_function(extractor_fn, 1) do
    extractor_fn.(messages)
    |> normalize_extraction()
  end

  defp extract_router_tensor(_messages, nil, _run_ctx), do: {:error, :missing_slm_context}

  defp extract_router_tensor(messages, {model_info, tokenizer}, _run_ctx) do
    Extractor.extract_penultimate_hidden_state_with_metadata(model_info, tokenizer, messages)
    |> normalize_extraction()
  end

  defp extract_router_tensor(messages, %{model_info: model_info, tokenizer: tokenizer}, run_ctx) do
    extract_router_tensor(messages, {model_info, tokenizer}, run_ctx)
  end

  defp extract_router_tensor(_messages, _context, _run_ctx), do: {:error, :invalid_slm_context}

  defp normalize_extraction({:ok, %{vector: %Nx.Tensor{} = vector} = extraction}) do
    {:ok,
     extraction
     |> Map.put_new(:vector_shape, Nx.shape(vector))
     |> Map.put_new(:hidden_state_shape, Nx.shape(vector))
     |> Map.put_new(:input_shapes, %{})
     |> Map.put_new(:transcript, nil)}
  end

  defp normalize_extraction(%{vector: %Nx.Tensor{} = vector} = extraction) do
    normalize_extraction({:ok, extraction |> Map.put_new(:vector_shape, Nx.shape(vector))})
  end

  defp normalize_extraction({:error, reason}), do: {:error, reason}
  defp normalize_extraction(_), do: {:error, :invalid_extractor_result}

  # --- Budget enforcement (Phase 10) ---

  defp bump_and_check_provider_budget(run_ctx, trace, turn) do
    case run_ctx[:counters][:provider_calls_ref] do
      nil -> :ok
      ref -> :counters.add(ref, 1, 1)
    end

    case check_budgets(run_ctx, :before_dispatch, %{turn: turn}) do
      {:budget_exceeded, kind, details} ->
        emit_trace(trace, :run_failed, %{reason: {:budget_exceeded, kind}, details: details})
        {:error, {:budget_exceeded, kind, details}}

      :ok ->
        :ok
    end
  end

  @doc false
  def check_budgets(run_ctx, kind, extras \\ %{}) when is_atom(kind) and is_map(extras) do
    budgets = Map.get(run_ctx, :budgets, %{})
    counters = Map.get(run_ctx, :counters, %{})

    cond do
      exceeded_wall_time?(budgets, counters) ->
        {:budget_exceeded, :wall_time,
         %{
           limit_ms: budgets[:max_wall_time_ms],
           elapsed_ms: elapsed_ms(counters[:started_monotonic_ms] || 0),
           checkpoint: kind
         }
         |> Map.merge(extras)}

      exceeded_provider_calls?(budgets, counters) ->
        {:budget_exceeded, :provider_calls,
         %{
           limit: budgets[:max_provider_calls],
           observed: provider_call_count(counters),
           checkpoint: kind
         }
         |> Map.merge(extras)}

      exceeded_verifier_revisions?(budgets, counters) ->
        {:budget_exceeded, :verifier_revisions,
         %{
           limit: budgets[:max_verifier_revisions],
           observed: verifier_revision_count(counters),
           checkpoint: kind
         }
         |> Map.merge(extras)}

      exceeded_cost?(budgets, counters) ->
        {:budget_exceeded, :estimated_cost_usd,
         %{
           limit_usd: budgets[:max_estimated_cost_usd],
           observed_usd: estimated_cost_usd(counters),
           checkpoint: kind
         }
         |> Map.merge(extras)}

      true ->
        :ok
    end
  end

  defp exceeded_wall_time?(%{max_wall_time_ms: nil}, _), do: false

  defp exceeded_wall_time?(%{max_wall_time_ms: limit}, counters) when is_integer(limit) do
    elapsed_ms(counters[:started_monotonic_ms] || 0) >= limit
  end

  defp exceeded_wall_time?(_, _), do: false

  defp exceeded_provider_calls?(%{max_provider_calls: nil}, _), do: false

  defp exceeded_provider_calls?(%{max_provider_calls: limit}, counters)
       when is_integer(limit) do
    provider_call_count(counters) > limit
  end

  defp exceeded_provider_calls?(_, _), do: false

  defp exceeded_verifier_revisions?(%{max_verifier_revisions: nil}, _), do: false

  defp exceeded_verifier_revisions?(%{max_verifier_revisions: limit}, counters)
       when is_integer(limit) do
    verifier_revision_count(counters) > limit
  end

  defp exceeded_verifier_revisions?(_, _), do: false

  defp exceeded_cost?(%{max_estimated_cost_usd: nil}, _), do: false

  defp exceeded_cost?(%{max_estimated_cost_usd: limit}, counters)
       when is_number(limit) do
    estimated_cost_usd(counters) >= limit
  end

  defp exceeded_cost?(_, _), do: false

  defp provider_call_count(%{provider_calls_ref: ref}), do: :counters.get(ref, 1)
  defp provider_call_count(_), do: 0

  defp verifier_revision_count(%{verifier_revisions_ref: ref}), do: :counters.get(ref, 1)
  defp verifier_revision_count(_), do: 0

  defp estimated_cost_usd(%{estimated_cost_micro_usd_ref: ref}),
    do: :counters.get(ref, 1) / 1_000_000

  defp estimated_cost_usd(_), do: 0.0

  # --- Phase 11.2: provider latency budget ---

  defp check_latency_budget(run_ctx, trace, turn, latency_ms) when is_integer(latency_ms) do
    case run_ctx[:budgets][:max_provider_latency_ms] do
      nil ->
        :ok

      limit when is_integer(limit) and latency_ms > limit ->
        details = %{
          limit_ms: limit,
          observed_ms: latency_ms,
          checkpoint: :after_dispatch,
          turn: turn
        }

        emit_trace(trace, :run_failed, %{
          reason: {:budget_exceeded, :provider_latency_ms},
          details: details
        })

        {:error, {:budget_exceeded, :provider_latency_ms, details}}

      _ ->
        :ok
    end
  end

  # --- Phase 11.4: estimated cost budget ---

  defp bump_and_check_cost_budget(run_ctx, trace, turn, dispatch) do
    case {run_ctx[:budgets][:max_estimated_cost_usd], run_ctx[:cost_estimator_fn]} do
      {nil, _} ->
        :ok

      {_limit, fun} when is_function(fun, 1) ->
        cost_usd = fun.(dispatch)
        bump_cost(run_ctx, cost_usd)
        check_cost_budget(run_ctx, trace, turn)

      {_limit, _no_fn} ->
        maybe_warn_missing_cost_estimator(run_ctx)
        :ok
    end
  end

  defp bump_cost(run_ctx, cost_usd) when is_number(cost_usd) and cost_usd >= 0 do
    case run_ctx[:counters][:estimated_cost_micro_usd_ref] do
      nil ->
        :ok

      ref ->
        # Store as integer micro-USD to keep :counters as integer-only.
        :counters.add(ref, 1, round(cost_usd * 1_000_000))
    end
  end

  defp bump_cost(_run_ctx, _other), do: :ok

  defp check_cost_budget(run_ctx, trace, turn) do
    counters = run_ctx[:counters] || %{}
    budgets = run_ctx[:budgets] || %{}

    if exceeded_cost?(budgets, counters) do
      details = %{
        limit_usd: budgets[:max_estimated_cost_usd],
        observed_usd: estimated_cost_usd(counters),
        checkpoint: :after_dispatch,
        turn: turn
      }

      emit_trace(trace, :run_failed, %{
        reason: {:budget_exceeded, :estimated_cost_usd},
        details: details
      })

      {:error, {:budget_exceeded, :estimated_cost_usd, details}}
    else
      :ok
    end
  end

  defp maybe_warn_missing_cost_estimator(run_ctx) do
    case run_ctx[:counters][:cost_warning_emitted_ref] do
      nil ->
        :ok

      ref ->
        if :counters.get(ref, 1) == 0 do
          :counters.add(ref, 1, 1)

          Logger.warning(
            "max_estimated_cost_usd was set but no cost_estimator_fn was provided; " <>
              "cost budget will not fire. See docs/agent_slot_provider_mapping.md."
          )
        end

        :ok
    end
  end

  # --- Phase 11.3: verifier revisions budget ---

  defp bump_and_check_verifier_revisions(run_ctx, trace, turn, role_name, verifier_status) do
    if Verifier.verifier_role?(role_name) and verifier_status != :accepted do
      case run_ctx[:counters][:verifier_revisions_ref] do
        nil -> :ok
        ref -> :counters.add(ref, 1, 1)
      end

      counters = run_ctx[:counters] || %{}
      budgets = run_ctx[:budgets] || %{}

      if exceeded_verifier_revisions?(budgets, counters) do
        details = %{
          limit: budgets[:max_verifier_revisions],
          observed: verifier_revision_count(counters),
          checkpoint: :after_verifier_revision,
          turn: turn
        }

        emit_trace(trace, :run_failed, %{
          reason: {:budget_exceeded, :verifier_revisions},
          details: details
        })

        {:error, {:budget_exceeded, :verifier_revisions, details}}
      else
        :ok
      end
    else
      :ok
    end
  end

  # --- Phase 11.5: artifact identity for RouteDecision ---

  defp build_artifact_identity(slm_context) do
    case build_runtime_metadata(slm_context) do
      meta when is_map(meta) and map_size(meta) > 0 -> meta
      _ -> nil
    end
  end
end
