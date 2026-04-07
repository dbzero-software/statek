"""Tests for ``format_chat_history_item``.

Each test exercises a single ``ChatHistoryItem`` shape across the relevant
chat styles, with and without xml-box console tags.  No integration with
the rest of Statek is performed — a stub settings object provides the
``get_xml_box_tags`` method that ``format_chat_history_item`` consults.
"""

# pylint: disable=unused-argument,no-member

import json
from dataclasses import dataclass
from typing import Dict

import pytest

from statek.chat_history import (
    ChatHistoryItem,
    ChatRole,
    ContentSource,
    format_chat_history_item,
)
from statek.chat_style import ChatStyle
from statek.utils import CallSpec


@dataclass
class _StubSettings:
    """Minimal settings stub providing only ``get_xml_box_tags``."""
    xml_tags: Dict[str, str] = None

    def get_xml_box_tags(self):
        return self.xml_tags or {}


def _settings(**tags) -> _StubSettings:
    return _StubSettings(xml_tags=tags or None)


# ----------------------- system prompt ----------------------------------

def test_system_prompt(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.SYSTEM,
        content="You are a helpful assistant.",
        content_src=ContentSource.SYSTEM,
    )
    assert format_chat_history_item(item, ChatStyle.MARKDOWN, _settings()) == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }


# ----------------------- user message -----------------------------------

def test_user_message_plain(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="Hello",
        content_src=ContentSource.USER,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {"role": "user", "content": "Hello"}


def test_user_message_not_xml_boxed_when_user_source(db0_fixture):
    """Plain user messages are NEVER xml-boxed (only CONSOLE-source content is)."""
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="Hello",
        content_src=ContentSource.USER,
    )
    result = format_chat_history_item(
        item, ChatStyle.MARKDOWN, _settings(console="OUT"))
    assert result == {"role": "user", "content": "Hello"}


# ----------------------- code-result / console output -------------------

def test_code_result_console_style(db0_fixture):
    """Code result (USER role + CONSOLE source) gets ``> ``-prefixed in CONSOLE style."""
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1\nline2",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(item, ChatStyle.CONSOLE, _settings())
    assert result == {"role": "user", "content": "> line1\n> line2"}


def test_code_result_markdown_style(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1\nline2",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {"role": "user", "content": "line1\nline2"}


def test_code_result_md_dialog_style_wraps_in_console_tag(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1\nline2",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(item, ChatStyle.MD_DIALOG, _settings())
    assert result == {
        "role": "user",
        "content": "<CONSOLE>\nline1\nline2\n</CONSOLE>",
    }


def test_code_result_xml_box_console_stacks_with_md_dialog(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(
        item, ChatStyle.MD_DIALOG, _settings(console="OUT"))
    assert result == {
        "role": "user",
        "content": "<OUT>\n<CONSOLE>\nline1\n</CONSOLE>\n</OUT>",
    }


def test_code_result_direct_style_still_xml_boxes(db0_fixture):
    """USER + CONSOLE content is xml-boxed in every style when configured."""
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1\nline2",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(
        item, ChatStyle.DIRECT, _settings(console="OUT"))
    assert result == {
        "role": "user",
        "content": "<OUT>\nline1\nline2\n</OUT>",
    }


def test_code_result_markdown_style_xml_boxes(db0_fixture):
    """xml-box console wrapping also applies in MARKDOWN style."""
    item = ChatHistoryItem(
        role=ChatRole.USER,
        content="line1",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(
        item, ChatStyle.MARKDOWN, _settings(console="OUT"))
    assert result == {
        "role": "user",
        "content": "<OUT>\nline1\n</OUT>",
    }


# ----------------------- tool result ------------------------------------

def test_tool_result_basic(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.TOOL,
        content="42",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {"role": "tool", "content": "42"}


def test_tool_result_with_call_spec_uses_id(db0_fixture):
    """When tool_calls is provided on a TOOL item it supplies the tool_call_id."""
    cs = CallSpec(id="call_123", func_name="get_x", args=[], kwargs={})
    item = ChatHistoryItem(
        role=ChatRole.TOOL,
        content="42",
        content_src=ContentSource.CONSOLE,
        tool_calls=cs,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {"role": "tool", "content": "42", "tool_call_id": "call_123"}


def test_tool_result_xml_box_applies(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.TOOL,
        content="line1\nline2",
        content_src=ContentSource.CONSOLE,
    )
    result = format_chat_history_item(
        item, ChatStyle.MARKDOWN, _settings(console="OUT"))
    assert result == {
        "role": "tool",
        "content": "<OUT>\nline1\nline2\n</OUT>",
    }


# ----------------------- tool calls (assistant) -------------------------

def test_assistant_single_tool_call(db0_fixture):
    cs = CallSpec(
        id="call_1", func_name="get_weather", args=[], kwargs={"city": "Boston"}
    )
    item = ChatHistoryItem(role=ChatRole.ASSISTANT, tool_calls=cs)
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Boston"}),
                },
            }
        ],
    }


def test_assistant_multiple_tool_calls(db0_fixture):
    cs1 = CallSpec(id="c1", func_name="a", args=[], kwargs={"x": 1})
    cs2 = CallSpec(id="c2", func_name="b", args=[], kwargs={})
    item = ChatHistoryItem(role=ChatRole.ASSISTANT, tool_calls=[cs1, cs2])
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result["role"] == "assistant"
    assert result["content"] is None
    assert [tc["id"] for tc in result["tool_calls"]] == ["c1", "c2"]
    assert result["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert result["tool_calls"][1]["function"]["arguments"] == "{}"


def test_assistant_python_cli_tool_call_strips_markdown(db0_fixture):
    """python_cli tool: code parameter must NOT contain markdown fences."""
    cs = CallSpec(
        id="call_cli",
        func_name="python_cli",
        args=[],
        kwargs={"code": "```python\nprint('hi')\n```"},
    )
    item = ChatHistoryItem(role=ChatRole.ASSISTANT, tool_calls=cs)
    result = format_chat_history_item(item, ChatStyle.DIRECT, _settings())
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args == {"code": "print('hi')"}


def test_assistant_python_cli_plain_code_unchanged(db0_fixture):
    """Plain python_cli code (no fences) passes through untouched."""
    cs = CallSpec(
        id="call_cli",
        func_name="python_cli",
        args=[],
        kwargs={"code": "print('hi')"},
    )
    item = ChatHistoryItem(role=ChatRole.ASSISTANT, tool_calls=cs)
    result = format_chat_history_item(item, ChatStyle.DIRECT, _settings())
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args == {"code": "print('hi')"}


# ----------------------- warmup block (assistant) -----------------------

def test_warmup_block_markdown_style_wraps_code(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="x = 1\ny = 2",
        content_src=ContentSource.SYSTEM,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {
        "role": "assistant",
        "content": "```python\nx = 1\ny = 2\n```",
    }


def test_warmup_block_md_dialog_style_wraps_code(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="x = 1",
        content_src=ContentSource.SYSTEM,
    )
    result = format_chat_history_item(item, ChatStyle.MD_DIALOG, _settings())
    assert result == {
        "role": "assistant",
        "content": "```python\nx = 1\n```",
    }


def test_warmup_block_direct_style_no_wrapping(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="x = 1",
        content_src=ContentSource.SYSTEM,
    )
    result = format_chat_history_item(item, ChatStyle.DIRECT, _settings())
    assert result == {"role": "assistant", "content": "x = 1"}


# ----------------------- LLM response (assistant) -----------------------

def test_llm_response_markdown_wraps_code(db0_fixture):
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="print(42)",
        content_src=ContentSource.ASSISTANT,
    )
    result = format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
    assert result == {
        "role": "assistant",
        "content": "```python\nprint(42)\n```",
    }


def test_llm_response_direct_passes_through(db0_fixture):
    """In DIRECT mode the LLM response is plain text — no wrapping."""
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="I will use a tool now.",
        content_src=ContentSource.ASSISTANT,
    )
    result = format_chat_history_item(item, ChatStyle.DIRECT, _settings())
    assert result == {"role": "assistant", "content": "I will use a tool now."}


def test_llm_response_console_style_no_wrapping(db0_fixture):
    """CONSOLE style does not wrap python code in markdown fences."""
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="print(1)",
        content_src=ContentSource.ASSISTANT,
    )
    result = format_chat_history_item(item, ChatStyle.CONSOLE, _settings())
    # CONSOLE style historically wraps assistant code too — keep parity with
    # prompt_append_console which does wrap for MARKDOWN/MD_DIALOG/DIRECT
    # but NOT for CONSOLE.
    assert result == {
        "role": "assistant",
        "content": "```python\nprint(1)\n```",
    }


# ----------------------- error / validation -----------------------------

def test_assistant_without_content_or_tool_calls_raises(db0_fixture):
    item = ChatHistoryItem(role=ChatRole.ASSISTANT)
    with pytest.raises(ValueError):
        format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())


def test_content_without_source_raises(db0_fixture):
    item = ChatHistoryItem(role=ChatRole.USER, content="hi")
    with pytest.raises(ValueError):
        format_chat_history_item(item, ChatStyle.MARKDOWN, _settings())
