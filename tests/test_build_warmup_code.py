"""Tests for build_warmup_code function."""

import pytest
from statek.utils import (
    ParsedFuncCall, ParsedWarmupBlock,
    build_warmup_code, CodeBlock, CallSpec,
)

pytestmark = pytest.mark.usefixtures("db0_fixture")


def _parsed(code, tool_calls=None):
    """Helper to create a ParsedWarmupBlock directly."""
    return ParsedWarmupBlock(code=code, tool_calls=tool_calls or [])


def test_build_warmup_code_single_no_tool_calls_returns_str():
    """A single block without tool calls returns a plain string."""
    result = build_warmup_code([_parsed("x = 1")])
    assert result == "x = 1"
    assert isinstance(result, str)


def test_build_warmup_code_single_with_tool_calls_returns_code_block():
    """A single block with tool calls returns a CodeBlock."""
    tc = ParsedFuncCall(name='show_example', args=[1], kwargs=None)
    result = build_warmup_code([_parsed("show_example(1)", [tc])])
    assert isinstance(result, CodeBlock)


def test_build_warmup_code_code_block_has_code_and_tool_calls():
    """CodeBlock has .code and .tool_calls with CallSpec items."""
    tc = ParsedFuncCall(name='ping', args=[], kwargs=None)
    result = build_warmup_code([_parsed("ping()", [tc])])
    assert result.code == "ping()"
    assert len(result.tool_calls) == 1
    assert isinstance(result.tool_calls[0], CallSpec)
    assert result.tool_calls[0].func_name == 'ping'


def test_build_warmup_code_call_id_format():
    """Call IDs follow the STATEK-NNN format."""
    tc = ParsedFuncCall(name='list_examples', args=[], kwargs=None)
    result = build_warmup_code([_parsed("list_examples()", [tc])])
    assert result.tool_calls[0].id == "STATEK-001"


def test_build_warmup_code_call_ids_sequential():
    """Multiple tool calls in the same block get sequential IDs."""
    tc1 = ParsedFuncCall(name='alpha', args=[], kwargs=None)
    tc2 = ParsedFuncCall(name='beta', args=[], kwargs=None)
    result = build_warmup_code([_parsed("alpha()\nbeta()", [tc1, tc2])])
    assert [cp.id for cp in result.tool_calls] == ["STATEK-001", "STATEK-002"]


def test_build_warmup_code_call_ids_unique_across_blocks():
    """Call IDs are unique across multiple blocks."""
    tc1 = ParsedFuncCall(name='alpha', args=[], kwargs=None)
    tc2 = ParsedFuncCall(name='beta', args=[], kwargs=None)
    result = build_warmup_code([_parsed("alpha()", [tc1]), _parsed("beta()", [tc2])])
    ids = [cp.id for cb in result if isinstance(cb, CodeBlock) for cp in cb.tool_calls]
    assert ids == ["STATEK-001", "STATEK-002"]


def test_build_warmup_code_multiple_returns_list():
    """Multiple blocks are returned as a list."""
    result = build_warmup_code([_parsed("x = 1"), _parsed("y = 2")])
    assert isinstance(result, list)
    assert len(result) == 2


def test_build_warmup_code_multiple_mixed_types():
    """Mixed blocks: strings and CodeBlocks in the returned list."""
    tc = ParsedFuncCall(name='tool_fn', args=[], kwargs=None)
    result = build_warmup_code([_parsed("x = 1"), _parsed("tool_fn()", [tc])])
    assert isinstance(result, list)
    assert isinstance(result[0], str)
    assert isinstance(result[1], CodeBlock)


def test_build_warmup_code_call_params_args_and_kwargs():
    """Tool call arguments and kwargs are preserved in CallSpec."""
    tc = ParsedFuncCall(name='search', args=['query'], kwargs={'max_results': 5})
    result = build_warmup_code([_parsed("search('query', max_results=5)", [tc])])
    cp = result.tool_calls[0]
    assert cp.args == ['query']
    assert cp.kwargs == {'max_results': 5}


def test_build_warmup_code_none_kwargs_becomes_empty_dict():
    """ParsedFuncCall with kwargs=None maps to CallSpec with kwargs={}."""
    tc = ParsedFuncCall(name='ping', args=[], kwargs=None)
    result = build_warmup_code([_parsed("ping()", [tc])])
    assert result.tool_calls[0].kwargs == {}
