# pylint: disable=no-member,redefined-outer-name
"""Tests for parsing LLM_TOOLS_SCOPE metadata."""

import pytest

from statek.chat_style import ChatStyle
from statek.llm_api import select_request_tools
from statek.llm_tools_scope import LLM_ToolsScope, parse_llm_tools_scope
from statek.system import tool


@tool
def llm_scope_app_tool(value: str, **kwargs) -> str:
    """Return an application-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


@tool
def llm_scope_other_app_tool(value: str, **kwargs) -> str:
    """Return another application-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


@tool(system=True)
def llm_scope_system_tool(value: str, **kwargs) -> str:
    """Return a system-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


@tool(hidden=True)
def llm_scope_hidden_app_tool(value: str, **kwargs) -> str:
    """Return a hidden application-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


@tool(system=True, hidden=True)
def llm_scope_hidden_system_tool(value: str, **kwargs) -> str:
    """Return a hidden system-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


@tool(target={ChatStyle.DIRECT})  # pylint: disable=no-member
def llm_scope_direct_app_tool(value: str, **kwargs) -> str:
    """Return a direct-only application-scoped value.

    Args:
        value: Value to return.
    """
    del kwargs
    return value


def _tool_names(tools):
    """Return tool names for assertions."""
    return [tool_func.__name__ for tool_func in tools]


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


def test_select_system_scope_with_explicit_application_addition():
    """SYSTEM+app_tool adds the named application tool to system tools."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "SYSTEM+llm_scope_app_tool"},
        available_tools=[llm_scope_app_tool],
        chat_style=ChatStyle.DIRECT,
    )

    names = _tool_names(selected)
    assert "llm_scope_app_tool" in names
    assert "llm_scope_system_tool" in names


def test_select_application_scope_with_explicit_system_addition():
    """APPLICATION+system_tool adds a named registered system tool."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION+llm_scope_system_tool"},
        available_tools=[llm_scope_app_tool, llm_scope_system_tool],
    )

    names = _tool_names(selected)
    assert names.count("llm_scope_app_tool") == 1
    assert names.count("llm_scope_system_tool") == 1


def test_select_all_scope_with_explicit_removal():
    """ALL-app_tool removes the named tool from the selected set."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "ALL-llm_scope_app_tool"},
        available_tools=[llm_scope_app_tool, llm_scope_system_tool],
    )

    names = _tool_names(selected)
    assert "llm_scope_app_tool" not in names
    assert "llm_scope_system_tool" in names


def test_select_all_scope_excludes_hidden_tools():
    """ALL scope excludes hidden tools from LLM-facing selection."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "ALL"},
        available_tools=[llm_scope_app_tool, llm_scope_hidden_app_tool],
    )

    names = _tool_names(selected)
    assert "llm_scope_app_tool" in names
    assert "llm_scope_hidden_app_tool" not in names
    assert "llm_scope_hidden_system_tool" not in names


def test_select_explicit_hidden_app_addition_raises():
    """Explicit additions cannot surface hidden application tools by name."""
    with pytest.raises(ValueError, match="unknown tool"):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "+llm_scope_hidden_app_tool"},
            available_tools=[llm_scope_hidden_app_tool],
        )


def test_select_explicit_hidden_system_addition_raises():
    """Explicit additions cannot surface hidden registered system tools by name."""
    with pytest.raises(ValueError, match="unknown tool"):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "+llm_scope_hidden_system_tool"},
            available_tools=[],
        )


def test_select_additions_only_reports_only_added_tool():
    """No-category explicit additions start from an empty base."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "+llm_scope_system_tool"},
        available_tools=[llm_scope_app_tool],
    )

    assert _tool_names(selected) == ["llm_scope_system_tool"]


def test_select_removal_wins_over_explicit_addition():
    """A removed tool is absent even when explicitly added earlier."""
    selected = select_request_tools(
        metadata={
            "LLM_TOOLS_SCOPE": "SYSTEM+llm_scope_app_tool-llm_scope_app_tool"
        },
        available_tools=[llm_scope_app_tool],
    )

    names = _tool_names(selected)
    assert "llm_scope_app_tool" not in names
    assert "llm_scope_system_tool" in names


def test_select_deduplicates_category_and_additions_by_name():
    """Category-selected tools win over duplicate explicit additions."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION+llm_scope_app_tool"},
        available_tools=[llm_scope_app_tool],
    )

    assert _tool_names(selected).count("llm_scope_app_tool") == 1


def test_select_does_not_resolve_global_application_tools():
    """Explicit additions cannot leak application tools from the global registry."""
    with pytest.raises(ValueError):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "+llm_scope_other_app_tool"},
            available_tools=[llm_scope_app_tool],
        )


def test_select_unresolved_explicit_removal_raises():
    """Explicit removals fail fast when the named tool cannot be resolved."""
    with pytest.raises(ValueError):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "ALL-missing_tool"},
            available_tools=[llm_scope_app_tool],
        )


def test_select_resolved_removal_outside_candidate_set_raises():
    """Resolved removals fail fast when they do not affect selected candidates."""
    with pytest.raises(ValueError):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "APPLICATION-llm_scope_system_tool"},
            available_tools=[llm_scope_app_tool, llm_scope_system_tool],
        )


def test_select_category_chat_style_filtering_is_silent():
    """Broad category selection silently excludes incompatible target tools."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION"},
        available_tools=[llm_scope_app_tool, llm_scope_direct_app_tool],
        chat_style=ChatStyle.MARKDOWN,
    )

    names = _tool_names(selected)
    assert "llm_scope_app_tool" in names
    assert "llm_scope_direct_app_tool" not in names


def test_select_explicit_chat_style_mismatch_raises():
    """Explicit additions fail fast when the named tool cannot target chat_style."""
    with pytest.raises(ValueError):
        select_request_tools(
            metadata={"LLM_TOOLS_SCOPE": "+llm_scope_direct_app_tool"},
            available_tools=[llm_scope_direct_app_tool],
            chat_style=ChatStyle.MARKDOWN,
        )


def test_select_available_tools_none_skips_explicit_resolution():
    """None available_tools keeps current no-tools behavior."""
    selected = select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "+missing_tool"},
        available_tools=None,
    )

    assert selected is None


def test_preview_request_raises_for_unresolved_scope_addition(openrouter_api):
    """preview_request fails fast on unresolved explicit additions."""
    with pytest.raises(ValueError):
        openrouter_api.preview_request(
            model="gpt-4o",
            metadata={"LLM_TOOLS_SCOPE": "+missing_tool"},
            available_tools=[llm_scope_app_tool],
        )


@pytest.mark.asyncio
async def test_process_request_raises_for_unresolved_scope_removal(openrouter_api):
    """process_request fails fast on unresolved explicit removals."""
    with pytest.raises(ValueError):
        await openrouter_api.process_request(
            model="gpt-4o",
            metadata={"LLM_TOOLS_SCOPE": "ALL-missing_tool"},
            available_tools=[llm_scope_app_tool],
        )


def test_openrouter_preview_uses_explicit_scope_tools(openrouter_api):
    """OpenRouter preview uses parsed explicit scope for OpenAI tools payload."""
    payload = openrouter_api.preview_request(
        model="gpt-4o",
        metadata={"LLM_TOOLS_SCOPE": "+llm_scope_system_tool"},
        available_tools=[llm_scope_app_tool],
    )

    names = [entry["function"]["name"] for entry in payload["tools"]]
    assert names == ["llm_scope_system_tool"]


def test_openrouter_preview_excludes_hidden_tools(openrouter_api):
    """OpenRouter preview does not include hidden tools in the provider payload."""
    payload = openrouter_api.preview_request(
        model="gpt-4o",
        metadata={"LLM_TOOLS_SCOPE": "ALL"},
        available_tools=[llm_scope_app_tool, llm_scope_hidden_app_tool],
    )

    names = [entry["function"]["name"] for entry in payload["tools"]]
    assert "llm_scope_app_tool" in names
    assert "llm_scope_hidden_app_tool" not in names
    assert "llm_scope_hidden_system_tool" not in names


def test_vertexai_preview_uses_explicit_scope_tools(vertexai_api):
    """VertexAI preview formats parsed scope tools as functionDeclarations."""
    payload = vertexai_api.preview_request(
        model="gemini-2.5-flash",
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION+llm_scope_system_tool"},
        available_tools=[llm_scope_app_tool],
    )

    declarations = payload["tools"][0]["functionDeclarations"]
    names = [entry["name"] for entry in declarations]
    assert names == ["llm_scope_app_tool", "llm_scope_system_tool"]


def test_claude_preview_uses_explicit_scope_removal(claude_api):
    """Claude preview formats parsed scope tools after removals are applied."""
    payload = claude_api.preview_request(
        model="claude-3",
        metadata={"LLM_TOOLS_SCOPE": "ALL-llm_scope_app_tool"},
        available_tools=[llm_scope_app_tool, llm_scope_system_tool],
    )

    names = [entry["name"] for entry in payload["tools"]]
    assert "llm_scope_app_tool" not in names
    assert "llm_scope_system_tool" in names


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
