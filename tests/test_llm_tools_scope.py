"""Tests for parsing LLM_TOOLS_SCOPE metadata."""

import pytest

from statek.llm_tools_scope import LLM_ToolsScope, parse_llm_tools_scope


def test_parse_empty_scope_returns_unset_dataclass():
    """Empty input returns an unset scope object, not None."""
    result = parse_llm_tools_scope("   ")

    assert result == LLM_ToolsScope()
    assert result.category is None
    assert result.additional_tools is None
    assert result.removed_tools is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SYSTEM", "SYSTEM"),
        ("application", "APPLICATION"),
        (" All ", "ALL"),
    ],
)
def test_parse_category_only(raw, expected):
    """Category-only scopes are normalized and leave explicit lists unset."""
    result = parse_llm_tools_scope(raw)

    assert result.category == expected
    assert result.additional_tools is None
    assert result.removed_tools is None


def test_parse_additions_only():
    """Explicit additions can be provided without a category."""
    result = parse_llm_tools_scope("+docstr, list_of_examples")

    assert result.category is None
    assert result.additional_tools == ["docstr", "list_of_examples"]
    assert result.removed_tools is None


def test_parse_category_with_additions():
    """A category can be followed by an addition list."""
    result = parse_llm_tools_scope("SYSTEM+send_message,dispatch_to")

    assert result.category == "SYSTEM"
    assert result.additional_tools == ["send_message", "dispatch_to"]
    assert result.removed_tools is None


def test_parse_category_with_removals():
    """A category can be followed by a removal list."""
    result = parse_llm_tools_scope("ALL-send_message")

    assert result.category == "ALL"
    assert result.additional_tools is None
    assert result.removed_tools == ["send_message"]


def test_parse_removals_only():
    """Explicit removals can be provided without a category."""
    result = parse_llm_tools_scope("-docstr")

    assert result.category is None
    assert result.additional_tools is None
    assert result.removed_tools == ["docstr"]


def test_parse_multiple_lists_and_whitespace():
    """Multiple add/remove lists are aggregated in source order."""
    result = parse_llm_tools_scope(
        " system + send_message , dispatch_to - docstr + brief "
    )

    assert result.category == "SYSTEM"
    assert result.additional_tools == ["send_message", "dispatch_to", "brief"]
    assert result.removed_tools == ["docstr"]


def test_parse_adjacent_list_markers_after_names():
    """A new list can start immediately after the previous list's final name."""
    result = parse_llm_tools_scope("SYSTEM+docstr-brief")

    assert result.category == "SYSTEM"
    assert result.additional_tools == ["docstr"]
    assert result.removed_tools == ["brief"]


def test_parse_preserves_duplicates_for_integration_deduplication():
    """The parser preserves duplicate names for integration-layer resolution."""
    result = parse_llm_tools_scope("SYSTEM+a+a")

    assert result.category == "SYSTEM"
    assert result.additional_tools == ["a", "a"]
    assert result.removed_tools is None


def test_parse_preserves_same_name_in_added_and_removed_lists():
    """The parser records conflicts and leaves removal-wins semantics downstream."""
    result = parse_llm_tools_scope("SYSTEM+a+b-a")

    assert result.category == "SYSTEM"
    assert result.additional_tools == ["a", "b"]
    assert result.removed_tools == ["a"]


@pytest.mark.parametrize(
    "raw",
    [
        "TOOLS",
        "docstr",
        "SYSTEM docstr",
        "SYSTEM+",
        "SYSTEM++docstr",
        "SYSTEM+,docstr",
        "SYSTEM+docstr,",
        "SYSTEM+docstr,,brief",
        "+docstr-",
        "+ -",
    ],
)
def test_parse_rejects_malformed_scope(raw):
    """Malformed scope definitions fail fast."""
    with pytest.raises(ValueError):
        parse_llm_tools_scope(raw)
