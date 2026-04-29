defmodule TrinityCoordinator.ThinkerTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Thinker

  test "parses solver suggestion and maps it to Worker" do
    parsed =
      Thinker.parse("""
      <suggestion>Compute the concrete answer.</suggestion>
      <suggested_role>solver</suggested_role>
      """)

    assert parsed.suggestion == "Compute the concrete answer."
    assert parsed.suggested_role == "Worker"
    assert parsed.suggested_role_id == 0
  end

  test "parses verifier suggestion" do
    parsed =
      Thinker.parse("""
      <suggestion>Check the final answer.</suggestion>
      <suggested_role>verifier</suggested_role>
      """)

    assert parsed.suggestion == "Check the final answer."
    assert parsed.suggested_role == "Verifier"
    assert parsed.suggested_role_id == 2
  end

  test "requires both suggestion and valid suggested role" do
    invalid_role =
      Thinker.parse("""
      <suggestion>Think more.</suggestion>
      <suggested_role>thinker</suggested_role>
      """)

    missing_suggestion = Thinker.parse("<suggested_role>solver</suggested_role>")
    missing_role = Thinker.parse("<suggestion>Use a solver.</suggestion>")

    assert invalid_role.suggestion == nil
    assert invalid_role.suggested_role == nil
    assert missing_suggestion.suggested_role == nil
    assert missing_role.suggestion == nil
  end
end
