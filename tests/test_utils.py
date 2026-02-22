# pylint: disable=unused-argument, E1101
"""Tests for statek.utils module."""

from typing import Iterable, Union, List, Dict, Optional, ForwardRef
import dbzero as db0
from statek.utils import format_callable_decl, prompt_append_console, block_comment, strip_markup
from statek.future import temporal, FutureResult
from statek.settings import ChatStyle


def test_simple_function():
    """Test formatting a simple function with type hints."""

    def send_message(message: str) -> str:
        return message

    result = format_callable_decl(send_message)
    assert result == "def send_message(message: str) -> str"


def test_function_with_default_none():
    """Test formatting a function with default None value."""

    def find_user(pattern: str, max_result: int = None) -> str:  # pylint: disable=unused-argument
        return pattern

    result = format_callable_decl(find_user)
    assert result == "def find_user(pattern: str, max_result: int = None) -> str"


def test_function_with_union_return():
    """Test formatting a function with Union return type."""

    def find_items(pattern: str) -> Union[str, List[str]]:
        return pattern

    result = format_callable_decl(find_items)
    assert "str | list" in result.lower() or "Union[str" in result


def test_function_with_multiple_params():
    """Test formatting a function with multiple parameters."""

    def process_data(
        name: str, value: int, active: bool = True
    ) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {name: value}

    result = format_callable_decl(process_data)
    assert "name: str" in result
    assert "value: int" in result
    assert "active: bool = True" in result
    assert "Dict[str, int]" in result or "dict[str, int]" in result


def test_function_without_annotations():
    """Test formatting a function without type annotations."""

    def plain_function(x, y=10):
        return x + y

    result = format_callable_decl(plain_function)
    assert result == "def plain_function(x, y = 10)"


def test_function_with_optional():
    """Test formatting a function with Optional type."""

    def get_value(
        key: str, default: Optional[str] = None
    ) -> Optional[str]:  # pylint: disable=unused-argument
        return default

    result = format_callable_decl(get_value)
    assert "key: str" in result
    assert "default:" in result
    assert "None" in result


def test_function_with_string_default():
    """Test formatting a function with string default value."""

    def greet(name: str = "World") -> str:
        return f"Hello, {name}"

    result = format_callable_decl(greet)
    assert 'name: str = "World"' in result


def test_function_with_numeric_defaults():
    """Test formatting a function with numeric default values."""

    def calculate(
        x: int = 0, y: float = 1.5, enabled: bool = False
    ) -> float:  # pylint: disable=unused-argument
        return x + y

    result = format_callable_decl(calculate)
    assert "x: int = 0" in result
    assert "y: float = 1.5" in result
    assert "enabled: bool = False" in result


def test_function_with_no_return_annotation():
    """Test formatting a function without return type annotation."""

    def do_something(value: str):
        print(value)

    result = format_callable_decl(do_something)
    assert result == "def do_something(value: str)"
    assert "->" not in result


def test_function_with_iterable():
    """Test formatting a function with Iterable type."""

    def process_items(items: Iterable[str]) -> List[str]:
        return list(items)

    result = format_callable_decl(process_items)
    assert "items: Iterable[str]" in result or "items: Iterable" in result
    assert "List[str]" in result or "list[str]" in result


def test_function_with_forward_ref():
    """Test formatting a function with ForwardRef type hints."""

    def send_message(msg: ForwardRef('Message')) -> ForwardRef('Response'):
        pass

    result = format_callable_decl(send_message)
    assert result == "def send_message(msg: Message) -> Response"
    # Ensure ForwardRef(...) wrapper is not present
    assert "ForwardRef" not in result


def test_function_with_forward_ref_in_generic():
    """Test formatting a function with ForwardRef inside generic types."""

    def get_messages(count: int = 10) -> List[ForwardRef('Message')]:
        pass

    result = format_callable_decl(get_messages)
    # Should format as list[Message] not list[ForwardRef('Message')]
    assert "Message" in result
    assert "ForwardRef" not in result
    assert "count: int = 10" in result


def test_prompt_append_console_basic():
    """Test basic usage with prompt and console."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    prompt = 'print(user)\nprint(clock.now())'
    result = prompt_append_console(console, ChatStyle.CONSOLE, prompt)

    expected = (
        'print(user)\nprint(clock.now())\n'
        '> User(name = "Kowalski Adam")\n> 2026-01-03 12:13:32'
    )
    assert result == expected


def test_prompt_append_console_no_prompt():
    """Test console output without initial prompt."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    result = prompt_append_console(console, ChatStyle.CONSOLE)

    expected = '> User(name = "Kowalski Adam")\n> 2026-01-03 12:13:32'
    assert result == expected


def test_prompt_append_console_empty_console():
    """Test with empty console list."""
    result = prompt_append_console([], ChatStyle.CONSOLE, 'prompt only')
    assert result == 'prompt only'


def test_prompt_append_console_with_from_pos():
    """Test with from_pos parameter to skip initial elements."""
    console = ['line1', 'line2', 'line3', 'line4']
    result = prompt_append_console(console, ChatStyle.CONSOLE, 'test', from_pos=2)

    expected = 'test\n> line3\n> line4'
    assert result == expected


def test_prompt_append_console_with_limit():
    """Test with limit parameter to restrict number of elements."""
    console = ['line1', 'line2', 'line3', 'line4', 'line5']
    result = prompt_append_console(console, ChatStyle.CONSOLE, 'test', limit=3)

    expected = 'test\n> line1\n> line2\n> line3'
    assert result == expected


def test_prompt_append_console_limit_exceeds_length():
    """Test when limit exceeds available elements."""
    console = ['line1', 'line2']
    result = prompt_append_console(console, ChatStyle.CONSOLE, from_pos=0, limit=10)

    expected = '> line1\n> line2'
    assert result == expected


def test_prompt_append_console_style_console():
    """CONSOLE style: prompt as-is, console lines prefixed with '> '."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    prompt = 'print(user)\nprint(clock.now())'
    result = prompt_append_console(console, ChatStyle.CONSOLE, prompt)

    assert result == (
        'print(user)\nprint(clock.now())\n'
        '> User(name = "Kowalski Adam")\n> 2026-01-03 12:13:32'
    )


def test_prompt_append_console_style_markdown():
    """MARKDOWN style: prompt wrapped in ```python fences, console lines as-is."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    prompt = 'print(user)\nprint(clock.now())'
    result = prompt_append_console(console, ChatStyle.MARKDOWN, prompt)

    assert result == (
        '```python\nprint(user)\nprint(clock.now())\n```\n'
        'User(name = "Kowalski Adam")\n2026-01-03 12:13:32'
    )


def test_prompt_append_console_xml_tags_none_no_boxing():
    """xml_tags=None leaves output unchanged."""
    console = ['line1', 'line2']
    result = prompt_append_console(console, ChatStyle.CONSOLE, 'code', xml_tags=None)
    assert result == 'code\n> line1\n> line2'


def test_prompt_append_console_xml_tags_no_console_key_no_boxing():
    """xml_tags without 'console' key leaves output unchanged."""
    console = ['line1', 'line2']
    result = prompt_append_console(console, ChatStyle.CONSOLE, 'code', xml_tags={"example": "ex"})
    assert result == 'code\n> line1\n> line2'


def test_prompt_append_console_xml_tags_console_key_boxes_console_lines_console_style():
    """xml_tags with 'console' key wraps the console portion in XML tags (CONSOLE style)."""
    console = ['line1', 'line2']
    result = prompt_append_console(
        console, ChatStyle.CONSOLE, 'code', xml_tags={"console": "output"})
    assert result == 'code\n<output>\n> line1\n> line2\n</output>'


def test_prompt_append_console_xml_tags_console_key_boxes_console_lines_markdown_style():
    """xml_tags with 'console' key wraps the console portion in XML tags (MARKDOWN style)."""
    console = ['line1', 'line2']
    result = prompt_append_console(
        console, ChatStyle.MARKDOWN, 'code', xml_tags={"console": "output"})
    assert result == '```python\ncode\n```\n<output>\nline1\nline2\n</output>'


def test_prompt_append_console_xml_tags_boxing_no_prompt():
    """Boxing works when no prompt is provided."""
    console = ['line1', 'line2']
    result = prompt_append_console(console, ChatStyle.CONSOLE, xml_tags={"console": "out"})
    assert result == '<out>\n> line1\n> line2\n</out>'


def test_prompt_append_console_xml_tags_empty_console_no_boxing():
    """Empty console produces no XML box even when tag is configured."""
    result = prompt_append_console([], ChatStyle.CONSOLE, 'code', xml_tags={"console": "output"})
    assert result == 'code'


def test_format_callable_decl_temporal_function():
    """Test formatting a temporal function shows non-FutureResult return type."""

    def complement_func(fut: FutureResult) -> str:
        return "result"

    def condition_func(fut: FutureResult) -> bool:
        return True

    @temporal(complement_func, condition_func)
    def temporal_func(x: int) -> FutureResult | int:
        return x

    result = format_callable_decl(temporal_func)
    # Should show the complement function's return type
    assert result == "def temporal_func(x: int) -> str"


def test_format_callable_decl_multiple_temporal_function():
    """Test formatting a temporal function shows non-FutureResult return type."""

    def complement_func(fut: FutureResult) -> str:
        return "result"

    def complement_func_2(fut: FutureResult) -> int:
        return 5

    def condition_func(fut: FutureResult) -> bool:
        return True

    @temporal(complement_func, condition_func)
    def temporal_func(x: int) -> FutureResult | int:
        return x

    @temporal(complement_func_2, condition_func)
    def temporal_func_2(x: int) -> FutureResult | int:
        return x

    result = format_callable_decl(temporal_func)
    # Should show the complement function's return type
    assert result == "def temporal_func(x: int) -> str"

    result = format_callable_decl(temporal_func_2)
    # Should show the complement function's return type
    assert result == "def temporal_func_2(x: int) -> int"

    result = format_callable_decl(temporal_func)
    # Should show the complement function's return type
    assert result == "def temporal_func(x: int) -> str"


def test_format_callable_decl_enum_param_shown_as_str(db0_fixture):
    """Test that a db0 enum parameter type is reported as str."""
    SeverityLevel = db0.enum("SeverityLevel", ["INFO", "WARNING", "ERROR"])

    def alert(severity: SeverityLevel):
        pass

    result = format_callable_decl(alert)
    assert result == "def alert(severity: str)"


def test_format_callable_decl_enum_mixed_with_regular_types(db0_fixture):
    """Test mixed enum and regular types — only enum is converted."""
    Status = db0.enum("Status", ["ACTIVE", "INACTIVE"])

    def update(name: str, status: Status, count: int = 0):
        pass

    result = format_callable_decl(update)
    assert result == "def update(name: str, status: str, count: int = 0)"


def test_format_callable_decl_enum_does_not_affect_return_type(db0_fixture):
    """Test that a db0 enum return type is also reported as str."""
    Priority = db0.enum("Priority", ["LOW", "HIGH"])

    def get_priority(task: str) -> Priority:
        pass

    result = format_callable_decl(get_priority)
    assert result == "def get_priority(task: str) -> str"


def test_block_comment_single_line():
    """Test block_comment with a single line."""
    result = block_comment("print('hello')")
    assert result == "# print('hello')"


def test_block_comment_multiple_lines():
    """Test block_comment with multiple lines."""
    code = "x = 1\ny = 2\nprint(x + y)"
    result = block_comment(code)
    expected = "# x = 1\n# y = 2\n# print(x + y)"
    assert result == expected


def test_block_comment_empty_string():
    """Test block_comment with an empty string."""
    result = block_comment("")
    assert result == "# "


def test_block_comment_with_empty_lines():
    """Test block_comment preserves empty lines with comment prefix."""
    code = "line1\n\nline3"
    result = block_comment(code)
    expected = "# line1\n# \n# line3"
    assert result == expected


def test_strip_markup_code_block_with_surrounding_text():
    """Test strip_markup extracts code and comments surrounding text."""
    input_text = 'Let me think about the first instruction.\n```python\nprint("Hello")\n```'
    result = strip_markup(input_text, strict=False)
    expected = '# Let me think about the first instruction.\nprint("Hello")'
    assert result == expected


def test_strip_markup_no_fences():
    """Test strip_markup returns input unchanged when no code fences present."""
    code = 'x = 1\nprint(x)'
    assert strip_markup(code, strict=False) == code


def test_strip_markup_code_only():
    """Test strip_markup with a code block and no surrounding text."""
    input_text = '```python\nprint("Hello")\n```'
    result = strip_markup(input_text, strict=False)
    assert result == 'print("Hello")'


def test_strip_markup_multiple_code_blocks():
    """Test strip_markup with multiple code blocks separated by text."""
    input_text = (
        'First, let me define a variable:\n'
        '```python\nx = 42\n```\n'
        'Now let me print it:\n'
        '```python\nprint(x)\n```'
    )
    result = strip_markup(input_text, strict=False)
    expected = (
        '# First, let me define a variable:\n'
        'x = 42\n'
        '# Now let me print it:\n'
        'print(x)'
    )
    assert result == expected


def test_strip_markup_code_block_without_language():
    """Test strip_markup handles code fences without a language specifier."""
    input_text = 'Here is the code:\n```\nprint("hello")\n```'
    result = strip_markup(input_text, strict=False)
    expected = '# Here is the code:\nprint("hello")'
    assert result == expected


def test_strip_markup_multiline_text_becomes_block_comment():
    """Test that multi-line text outside code blocks is fully commented."""
    input_text = (
        'I will solve this step by step.\n'
        'First, create a variable.\n'
        '```python\nx = 1\n```'
    )
    result = strip_markup(input_text, strict=False)
    expected = (
        '# I will solve this step by step.\n'
        '# First, create a variable.\n'
        'x = 1'
    )
    assert result == expected


def test_strip_markup_strict_python_only():
    """Strict mode: ```python block content is returned as plain code."""
    input_text = (
        'Let me think about the first instruction.\n'
        '```python\nprint("Hello")\n```'
    )
    result = strip_markup(input_text, strict=True)
    assert result == '# Let me think about the first instruction.\nprint("Hello")'


def test_strip_markup_strict_non_python_fence_commented():
    """Strict mode: input with no ```python fences is fully commented out."""
    input_text = 'Here is the code:\n```\nprint("hello")\n```'
    result = strip_markup(input_text, strict=True)
    assert result == '# Here is the code:\n# ```\n# print("hello")\n# ```'
