defmodule TrinityCoordinator.RuntimeProfileMarginsTest do
  @moduledoc """
  Phase 5 — pins per-profile default margin floors for the prompt-eval suite.

  All built-in profiles inherit the existing global defaults
  (`0.24` agent, `1.06` role). New profiles (or a future first-class
  `:emily`) override only their own margins; CUDA stays bytewise unchanged.
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.RuntimeProfile

  describe "default_margins/1 — every built-in returns the canonical CUDA defaults" do
    for name <- [:cuda_exla, :host_exla, :binary, :mock_tiny, :emlx] do
      @tag name: name
      test "#{name} → %{agent: 0.24, role: 1.06}", %{name: name} do
        profile = RuntimeProfile.resolve(name)
        assert RuntimeProfile.default_margins(profile) == %{agent: 0.24, role: 1.06}
      end
    end

    test ":emily ships Emily-empirical margin floors NOT the canonical CUDA defaults" do
      profile = RuntimeProfile.resolve(:emily)
      assert RuntimeProfile.default_margins(profile) == %{agent: 0.33, role: 0.82}

      # Sanity: the canonical defaults are still 0.24 / 1.06 — :emily's
      # override does not bleed into the other built-ins.
      assert RuntimeProfile.default_margins(RuntimeProfile.resolve(:cuda_exla)) ==
               %{agent: 0.24, role: 1.06}
    end

    test ":custom inherits the CUDA defaults unless overridden via override_default_margins/2" do
      profile = RuntimeProfile.resolve({:custom, Nx.BinaryBackend, []})
      assert RuntimeProfile.default_margins(profile) == %{agent: 0.24, role: 1.06}
    end
  end

  describe "override_default_margins/2 — opt-in per-profile overrides" do
    test "lets a caller (or a future :emily clause) override only one axis" do
      profile = RuntimeProfile.resolve(:emlx)

      profile = RuntimeProfile.override_default_margins(profile, agent: 0.33)

      assert RuntimeProfile.default_margins(profile) == %{agent: 0.33, role: 1.06}
    end

    test "lets a caller override both axes at once" do
      profile = RuntimeProfile.resolve(:emlx)

      profile = RuntimeProfile.override_default_margins(profile, agent: 0.33, role: 0.82)

      assert RuntimeProfile.default_margins(profile) == %{agent: 0.33, role: 0.82}
    end

    test "does not mutate the original profile struct" do
      original = RuntimeProfile.resolve(:cuda_exla)
      overridden = RuntimeProfile.override_default_margins(original, agent: 0.99)

      assert RuntimeProfile.default_margins(original) == %{agent: 0.24, role: 1.06}
      assert RuntimeProfile.default_margins(overridden) == %{agent: 0.99, role: 1.06}
    end
  end
end
