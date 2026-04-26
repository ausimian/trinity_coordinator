defmodule TrinityCoordinator.AgentPool.OpenAI do
  @moduledoc """
  OpenAI-compatible provider adapter used by the agent pool.
  """

  @behaviour TrinityCoordinator.AgentPool.Adapter

  @default_base_url "https://api.openai.com/v1"

  @impl true
  def call(agent_spec, messages, opts) do
    api_key = Keyword.get(opts, :openai_api_key, System.get_env("OPENAI_API_KEY"))

    base_url =
      Keyword.get(
        opts,
        :openai_base_url,
        System.get_env("TRINITY_OPENAI_BASE_URL", @default_base_url)
      )

    timeout = Keyword.get(opts, :openai_timeout_ms, 30_000)

    with :ok <- validate_api_key(api_key),
         {:ok, payload} <- build_payload(agent_spec[:model], messages),
         {:ok, response} <- request(base_url, payload, api_key, timeout) do
      parse_response(response)
    end
  end

  defp validate_api_key(api_key) when is_binary(api_key) and byte_size(api_key) > 0, do: :ok
  defp validate_api_key(_), do: {:error, :missing_openai_api_key}

  defp build_payload(model, messages) when is_binary(model) do
    {:ok, %{model: model, messages: messages, max_tokens: 200, temperature: 0.2}}
  end

  defp build_payload(_, _), do: {:error, :invalid_model}

  defp request(base_url, payload, api_key, timeout) do
    url = Path.join(base_url, "chat/completions")

    case Req.post(url,
           json: payload,
           headers: [{"authorization", "Bearer #{api_key}"}, {"content-type", "application/json"}],
           receive_timeout: timeout,
           connect_options: [timeout: timeout]
         ) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_response(%{"choices" => [%{"message" => %{"content" => response}} | _]})
       when is_binary(response) do
    {:ok, response}
  end

  defp parse_response(%{"choices" => [%{"text" => response} | _]}) when is_binary(response) do
    {:ok, response}
  end

  defp parse_response(_), do: {:error, :invalid_provider_response}
end
