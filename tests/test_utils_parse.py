# pylint: disable=unused-argument, E1101
"""Tests for statek.utils parse_func_call, parse_warmup_block, and CodeBlock."""

import pytest
from statek.llm_api import CallParams
from statek.utils import (parse_func_call, ParsedFuncCall,
                          parse_warmup_block, ParsedWarmupBlock,
                          CodeBlock, CallSpec, parse_tool_log,
                          print_tool_log)


# ---------------------------------------------------------------------------
# parse_func_call tests
# ---------------------------------------------------------------------------

def test_parse_func_call_single_int_arg():
    """Parse a function call with a single integer argument."""
    result = parse_func_call('show_example(71)')
    assert result == ParsedFuncCall(name='show_example', args=[71], kwargs=None)


def test_parse_func_call_no_args():
    """Parse a function call with no arguments."""
    result = parse_func_call('ping()')
    assert result.name == 'ping'
    assert result.args == []
    assert result.kwargs is None


def test_parse_func_call_string_arg():
    """Parse a function call with a string argument."""
    result = parse_func_call('greet("hello")')
    assert result == ParsedFuncCall(name='greet', args=['hello'], kwargs=None)


def test_parse_func_call_single_quoted_string_arg():
    """Parse a function call with single-quoted string arguments."""
    result = parse_func_call("find_user('Alice', max_results=5)")
    assert result == ParsedFuncCall(name='find_user', args=['Alice'], kwargs={'max_results': 5})


def test_parse_func_call_multiple_positional_args():
    """Parse a function call with multiple positional arguments."""
    result = parse_func_call('find_user("Alice", 10)')
    assert result == ParsedFuncCall(name='find_user', args=['Alice', 10], kwargs=None)


def test_parse_func_call_kwargs_only():
    """Parse a function call with keyword arguments only."""
    result = parse_func_call('connect(host="localhost", port=5432)')
    assert result.name == 'connect'
    assert result.args == []
    assert result.kwargs == {'host': 'localhost', 'port': 5432}


def test_parse_func_call_mixed_args_and_kwargs():
    """Parse a function call with both positional and keyword arguments."""
    result = parse_func_call('search("query", max_results=10)')
    assert result.name == 'search'
    assert result.args == ['query']
    assert result.kwargs == {'max_results': 10}


def test_parse_func_call_float_arg():
    """Parse a function call with a float argument."""
    result = parse_func_call('set_threshold(0.5)')
    assert result == ParsedFuncCall(name='set_threshold', args=[0.5], kwargs=None)


def test_parse_func_call_bool_arg():
    """Parse a function call with a boolean argument."""
    result = parse_func_call('toggle(True)')
    assert result == ParsedFuncCall(name='toggle', args=[True], kwargs=None)


def test_parse_func_call_variable_name_arg():
    """Variable names in args/kwargs are preserved as strings."""
    result = parse_func_call('docstr(OnCallCalendar)')
    assert result == ParsedFuncCall(name='docstr', args=['OnCallCalendar'], kwargs=None)


def test_parse_func_call_variable_name_kwarg():
    """Variable names in kwargs are preserved as strings."""
    result = parse_func_call('run(target=MyModel)')
    assert result == ParsedFuncCall(name='run', args=[], kwargs={'target': 'MyModel'})


def test_parse_func_call_mixed_literal_and_variable_args():
    """Literal and variable arguments can be mixed."""
    result = parse_func_call('show(MyClass, 5)')
    assert result == ParsedFuncCall(name='show', args=['MyClass', 5], kwargs=None)


def test_parse_func_call_bare_name_raises():
    """A bare name (not a call) raises an exception."""
    with pytest.raises(Exception):
        parse_func_call('my_function')


def test_parse_func_call_invalid_syntax_raises():
    """Invalid Python syntax raises an exception."""
    with pytest.raises(Exception):
        parse_func_call('not a function call!!!')


# ---------------------------------------------------------------------------
# parse_tool_log / print_tool_log tests
# ---------------------------------------------------------------------------

def test_parse_tool_log_valid_call_params():
    """A log line with one function call is parsed into CallParams."""
    line = (
        "log: answer(body='Brak dyzurow w przyszlym tygodniu "
        "(2026-04-27-2026-05-03).', media=None)"
    )
    result = parse_tool_log(line)
    assert isinstance(result, CallParams)
    assert result.id == ""
    assert result.name == "answer"
    assert result.args == []
    assert result.kwargs == {
        "body": "Brak dyzurow w przyszlym tygodniu (2026-04-27-2026-05-03).",
        "media": None,
    }


def test_parse_tool_log_valid_positional_and_keyword_args():
    """Positional and keyword arguments are preserved."""
    result = parse_tool_log("log: search('alice', limit=5)")
    assert result is not None
    assert result.name == "search"
    assert result.args == ["alice"]
    assert result.kwargs == {"limit": 5}


@pytest.mark.parametrize(
    "line",
    [
        "answer(body='missing prefix')",
        " log: answer(body='leading space')",
        "log:",
        "log: answer(body='unterminated'",
        "log: answer(body='ok'); other()",
        "log: answer(**{'body': 'unsupported expansion'})",
    ],
)
def test_parse_tool_log_invalid_inputs_return_none(line):
    """Non-log and malformed inputs are ignored."""
    assert parse_tool_log(line) is None


def test_print_tool_log_outputs_log_line(capsys):
    """print_tool_log writes the canonical log format."""
    call_params = CallParams(
        call_id="call_1",
        name="answer",
        args=[],
        kwargs={"body": "done", "media": None},
    )
    print_tool_log(call_params)
    assert capsys.readouterr().out == "log: answer(body='done', media=None)\n"


# ---------------------------------------------------------------------------
# parse_warmup_block tests
# ---------------------------------------------------------------------------

def test_parse_warmup_block_no_tool_calls():
    """Code without annotations produces empty tool_calls and unchanged code."""
    code = "user, message = fetch_next_message()\nprint(message)"
    result = parse_warmup_block(code)
    assert result.code == code
    assert result.tool_calls == []


def test_parse_warmup_block_single_tool_call():
    """A line annotated with #STATEK: as tool is extracted as a tool call."""
    code = "list_of_examples() #STATEK: as tool\nuser, message = fetch_next_message()"
    result = parse_warmup_block(code)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0] == ParsedFuncCall(name='list_of_examples', args=[], kwargs=None)


def test_parse_warmup_block_removes_tool_call_lines_from_code():
    """Lines annotated with #STATEK: as tool are excluded from the returned code field."""
    code = "list_of_examples() #STATEK: as tool\nuser, message = fetch_next_message()"
    result = parse_warmup_block(code)
    assert "list_of_examples()" not in result.code
    assert "user, message = fetch_next_message()" in result.code


def test_parse_warmup_block_tool_call_with_args():
    """Tool call lines with arguments are parsed correctly."""
    code = "show_example('myagent', 3) #STATEK: as tool"
    result = parse_warmup_block(code)
    assert result.tool_calls[0] == ParsedFuncCall(
        name='show_example', args=['myagent', 3], kwargs=None
    )


def test_parse_warmup_block_tool_call_with_kwargs():
    """Tool call lines with keyword arguments are parsed correctly."""
    code = "search(query='hello', max_results=5) #STATEK: as tool"
    result = parse_warmup_block(code)
    assert result.tool_calls[0] == ParsedFuncCall(
        name='search', args=[], kwargs={'query': 'hello', 'max_results': 5}
    )


def test_parse_warmup_block_multiple_tool_calls():
    """Multiple annotated lines produce multiple tool calls in order."""
    code = (
        "list_of_examples() #STATEK: as tool\n"
        "show_example('agent', 1) #STATEK: as tool\n"
        "user, message = fetch_next_message()"
    )
    result = parse_warmup_block(code)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == 'list_of_examples'
    assert result.tool_calls[1].name == 'show_example'
    assert isinstance(result, ParsedWarmupBlock)


# --- CodeBlock.get_tool_call_id tests ---


def test_get_tool_call_id_found(db0_fixture):
    """get_tool_call_id returns the index when the CallSpec is present."""
    cs0 = CallSpec(id="STATEK-001", func_name="foo")
    cs1 = CallSpec(id="STATEK-002", func_name="bar", args=[1])
    block = CodeBlock(code="x = 1", tool_calls=[cs0, cs1])
    assert block.get_tool_call_id(cs0) == 0
    assert block.get_tool_call_id(cs1) == 1


def test_get_tool_call_id_not_found(db0_fixture):
    """get_tool_call_id returns None when the CallSpec is not in tool_calls."""
    cs0 = CallSpec(id="STATEK-001", func_name="foo")
    cs_other = CallSpec(id="STATEK-999", func_name="baz")
    block = CodeBlock(code="x = 1", tool_calls=[cs0])
    assert block.get_tool_call_id(cs_other) is None


def test_get_tool_call_id_no_tool_calls(db0_fixture):
    """get_tool_call_id returns None when tool_calls is None."""
    cs = CallSpec(id="STATEK-001", func_name="foo")
    block = CodeBlock(code="x = 1")
    assert block.get_tool_call_id(cs) is None


def test_get_tool_call_id_empty_tool_calls(db0_fixture):
    """get_tool_call_id returns None when tool_calls is empty."""
    cs = CallSpec(id="STATEK-001", func_name="foo")
    block = CodeBlock(code="x = 1", tool_calls=[])
    assert block.get_tool_call_id(cs) is None
