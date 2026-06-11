"""Tests for the StatekWebUI agents page helper functions."""

import json
from typing import List

from web_ui.pages.agents import (
    _format_warmup_code,
    _get_agent_system_prompt,
    _get_tool_info,
    _get_tool_signature,
)
from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt


class _FakeCodeBlock:  # pylint: disable=too-few-public-methods
    """Duck-typed stand-in for CodeBlock (avoids db0 initialization)."""
    def __init__(self, code=None, tool_calls=None):
        self.code = code
        self.tool_calls = tool_calls


class _FakeCallSpec:  # pylint: disable=too-few-public-methods
    """Duck-typed stand-in for CallSpec."""
    def __init__(self, func_name, args=None, kwargs=None):
        self.func_name = func_name
        self.args = args or []
        self.kwargs = kwargs or {}

    def format(self) -> str:
        """Format as a human-friendly function call string."""
        parts = [repr(a) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.func_name}({', '.join(parts)})"


class _FakeAgentWithBadFormatter:  # pylint: disable=too-few-public-methods
    def __init__(self, raw_prompt):
        self._system_prompt = raw_prompt

    def system_prompt(self, *_args, **_kwargs):
        raise KeyError('missing template value')


class _FakePromptWithTypingList:  # pylint: disable=too-few-public-methods
    intro = 'Intro'
    sections = List


def _tool_with_full_docs(value: str) -> str:
    """Return the value uppercased.

    Args:
        value (str): The string to uppercase.

    Returns:
        str: The uppercased string.
    """
    return value.upper()


def _tool_no_args() -> None:
    """Do nothing."""


def _tool_no_docstring(x):  # pylint: disable=unused-argument
    pass


def _tool_with_default(name: str, limit: int = 10) -> str:  # pylint: disable=unused-argument
    """Fetch items by name.

    Args:
        name (str): The name to look up.
        limit (int): Maximum results.

    Returns:
        str: Result string.
    """
    return name


class TestGetToolInfo:
    def test_brief_contains_signature_and_description(self):
        brief, _, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'def _tool_with_full_docs' in brief
        assert 'Return the value uppercased.' in brief

    def test_brief_does_not_contain_args_section(self):
        brief, _, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'Args:' not in brief

    def test_full_docs_contains_args_section(self):
        _, full_docs, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'def _tool_with_full_docs' in full_docs
        assert 'Args:' in full_docs
        assert 'value' in full_docs
        assert 'Returns:' in full_docs

    def test_no_docstring_returns_none_none(self):
        brief, full_docs, error = _get_tool_info(_tool_no_docstring)
        assert brief is None
        assert full_docs is None
        assert error is not None

    def test_non_callable_returns_none_none(self):
        brief, full_docs, error = _get_tool_info("not_a_function")
        assert brief is None
        assert full_docs is None
        assert error is None


class TestGetToolSignature:
    def test_simple_param(self):
        assert _get_tool_signature(_tool_with_full_docs) == '_tool_with_full_docs(value)'

    def test_no_params(self):
        assert _get_tool_signature(_tool_no_args) == '_tool_no_args()'

    def test_default_param_included(self):
        sig = _get_tool_signature(_tool_with_default)
        assert sig == '_tool_with_default(name, limit)'

    def test_non_callable_returns_str(self):
        result = _get_tool_signature("mytool")  # type: ignore[arg-type]
        assert result == 'mytool'


class TestGetAgentSystemPrompt:
    def test_formats_persistent_prompt_to_json_safe_text(self, db0_fixture):  # pylint: disable=unused-argument
        agent = Agent(
            role='test',
            _system_prompt=make_system_prompt('Intro\n\n--- Details ---\nUse tools.'),
            _tools=[],
        )

        result = _get_agent_system_prompt(agent)

        assert result == 'Intro\n\n--- Details ---\nUse tools.'
        json.dumps({'system_prompt': result})

    def test_does_not_return_persistent_prompt_with_typing_list(self, db0_fixture):  # pylint: disable=unused-argument
        result = _get_agent_system_prompt(_FakeAgentWithBadFormatter(_FakePromptWithTypingList()))

        assert isinstance(result, str)
        json.dumps({'system_prompt': result})


class TestFormatWarmupCode:
    def test_none_returns_none(self):
        assert _format_warmup_code(None) is None

    def test_plain_string(self):
        assert _format_warmup_code("x = 1") == "x = 1"

    def test_code_block_code_only(self):
        cb = _FakeCodeBlock(code="x = 1")
        result = _format_warmup_code(cb)
        assert "x = 1" in result

    def test_code_block_with_tool_calls(self):
        cs = _FakeCallSpec(func_name="my_tool", args=["a"], kwargs={"k": "v"})
        cb = _FakeCodeBlock(code="x = 1", tool_calls=[cs])
        result = _format_warmup_code(cb)
        assert "x = 1" in result
        assert "my_tool('a', k='v')  #STATEK: as tool" in result

    def test_code_block_tool_call_no_args(self):
        cs = _FakeCallSpec(func_name="do_stuff")
        cb = _FakeCodeBlock(code="", tool_calls=[cs])
        result = _format_warmup_code(cb)
        assert "do_stuff()  #STATEK: as tool" in result

    def test_list_of_strings(self):
        result = _format_warmup_code(["block1", "block2"])
        assert "Block 1" in result
        assert "block1" in result
        assert "Block 2" in result
        assert "block2" in result

    def test_list_with_mixed_types(self):
        cb = _FakeCodeBlock(code="y = 2")
        result = _format_warmup_code(["x = 1", cb])
        assert "Block 1" in result
        assert "x = 1" in result
        assert "Block 2" in result
        assert "y = 2" in result

    def test_empty_string_returns_none(self):
        assert _format_warmup_code("") is None

    def test_empty_list_returns_none(self):
        assert _format_warmup_code([]) is None
