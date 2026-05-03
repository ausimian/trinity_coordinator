{:ok, _} = Application.ensure_all_started(:emily)
Nx.global_default_backend(Emily.Backend)

ExUnit.start()
ExUnit.configure(exclude: [integration: true, expensive_qwen_svd: true, slow_qwen_svd: true])
