# pylint: disable=unused-argument, E1101
"""Tests for statek.utils module."""

from typing import Union, List, Dict, Optional, Iterable, ForwardRef
import dbzero as db0
from statek.utils import (format_callable_decl,
                          prompt_append_console, block_comment, strip_markup,
                          extract_media, extract_dialog, parse_dialog)
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


def test_prompt_append_console_style_md_dialog():
    """MD_DIALOG style: prompt in fences, console wrapped in <CONSOLE> tags."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    prompt = 'print(user)\nprint(clock.now())'
    result = prompt_append_console(console, ChatStyle.MD_DIALOG, prompt)

    assert result == (
        '```python\nprint(user)\nprint(clock.now())\n```\n'
        '<CONSOLE>\nUser(name = "Kowalski Adam")\n2026-01-03 12:13:32\n</CONSOLE>'
    )


def test_prompt_append_console_md_dialog_no_prompt():
    """MD_DIALOG without prompt: only <CONSOLE>-wrapped console output."""
    console = ['2026-03-18 14:31']
    result = prompt_append_console(console, ChatStyle.MD_DIALOG)
    assert result == '<CONSOLE>\n2026-03-18 14:31\n</CONSOLE>'


def test_prompt_append_console_md_dialog_empty_console():
    """MD_DIALOG with empty console returns prompt only."""
    result = prompt_append_console([], ChatStyle.MD_DIALOG, 'print(x)')
    assert result == '```python\nprint(x)\n```'


def test_prompt_append_console_md_dialog_xml_tags_stack():
    """MD_DIALOG <CONSOLE> wrapping stacks with xml_tags boxing."""
    console = ['output']
    result = prompt_append_console(
        console, ChatStyle.MD_DIALOG, xml_tags={"console": "outer"})
    assert result == '<outer>\n<CONSOLE>\noutput\n</CONSOLE>\n</outer>'


def test_prompt_append_console_direct_style():
    """DIRECT style: prompt in fences, console as plain text (no CONSOLE XML)."""
    console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
    prompt = 'print(user)\nprint(clock.now())'
    result = prompt_append_console(console, ChatStyle.DIRECT, prompt)

    assert result == (
        '```python\nprint(user)\nprint(clock.now())\n```\n'
        'User(name = "Kowalski Adam")\n2026-01-03 12:13:32'
    )


def test_prompt_append_console_direct_no_prompt():
    """DIRECT without prompt: plain console output (no CONSOLE XML)."""
    console = ['2026-03-18 14:31']
    result = prompt_append_console(console, ChatStyle.DIRECT)
    assert result == '2026-03-18 14:31'


def test_prompt_append_console_direct_ignores_xml_tags():
    """DIRECT style ignores xml_tags boxing — tool call structure is sufficient."""
    console = ['output']
    result = prompt_append_console(
        console, ChatStyle.DIRECT, xml_tags={"console": "CONSOLE_OUTPUT"})
    assert result == 'output'


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


# ---------------------------------------------------------------------------
# extract_media
# ---------------------------------------------------------------------------

class TestExtractMedia:
    """Tests for extract_media utility."""

    def test_body_and_media_interleaved(self):
        """Mixed body text and media paths are split into (body, media) tuples."""
        text = (
            "Here's your calendar for march: gen/1234abc.svg \n"
            "and this is for april : gen/1235abc.svg gen/1236abc.svg Bye !"
        )
        result = list(extract_media(text))
        assert result == [
            ("Here's your calendar for march:", "gen/1234abc.svg"),
            ("and this is for april :", "gen/1235abc.svg"),
            ("", "gen/1236abc.svg"),
            ("Bye !", None),
        ]

    def test_no_media(self):
        """Input without media paths yields a single (body, None) tuple."""
        result = list(extract_media("Hello world"))
        assert result == [("Hello world", None)]

    def test_only_media(self):
        """Input that is just a media path yields ('', media)."""
        result = list(extract_media("images/photo.png"))
        assert result == [("", "images/photo.png")]

    def test_multiple_media_no_body(self):
        """Consecutive media paths yield ('', media) for each."""
        result = list(extract_media("a/b.svg c/d.png"))
        assert result == [("", "a/b.svg"), ("", "c/d.png")]

    def test_extra_whitespace_eliminated(self):
        """Extra spaces and newlines between tokens are normalized."""
        result = list(extract_media("Hello   world   dir/img.jpg   done"))
        assert result == [("Hello world", "dir/img.jpg"), ("done", None)]

    def test_empty_input(self):
        """Empty string yields no tuples."""
        assert not list(extract_media(""))

    def test_absolute_path(self):
        """Absolute file paths are recognized as media."""
        result = list(extract_media("See /tmp/output/chart.svg here"))
        assert result == [("See", "/tmp/output/chart.svg"), ("here", None)]

    def test_media_delimited(self):
        result = list(extract_media("[Media](private/chart.svg) is here"))
        assert result == [("Media", "private/chart.svg"), ("is here", None)]

    def test_markdown_image_syntax(self):
        """Markdown image ![alt](path) is preserved as full markdown media."""
        text = " ![April On-Call Schedule](private/calendar_zeo144sc.svg)"
        result = list(extract_media(text))
        assert result == [
            ("", "![April On-Call Schedule](private/calendar_zeo144sc.svg)"),
        ]

    def test_markdown_image_with_title(self):
        """Markdown image ![alt](path "title") is preserved as full markdown media."""
        text = "Here's your calendar for march: " \
               '![Calendar](gen/1234abc.svg "March Calendar")'
        result = list(extract_media(text))
        assert result == [
            ("Here's your calendar for march:",
             '![Calendar](gen/1234abc.svg "March Calendar")'),
        ]

    def test_markdown_image_mixed_with_plain(self):
        """Markdown images and plain paths can coexist."""
        text = "Chart: ![Chart](gen/chart.svg) and gen/data.csv here"
        result = list(extract_media(text))
        assert result == [
            ("Chart:", "![Chart](gen/chart.svg)"),
            ("and", "gen/data.csv"),
            ("here", None),
        ]


# ---------------------------------------------------------------------------
# extract_dialog
# ---------------------------------------------------------------------------

class TestExtractDialog:
    """Tests for extract_dialog utility."""

    def test_no_code_blocks(self):
        """Plain text without markdown blocks is returned as-is."""
        assert extract_dialog("Hello world") == "Hello world"

    def test_code_block_only(self):
        """Input that is only a code block returns None."""
        assert extract_dialog('```python\nprint("Hello")\n```') is None

    def test_text_around_code_block(self):
        """Text before and after a code block is extracted and joined."""
        text = 'Let me think about this.\n```python\nprint("Hello")\n```\nDone!'
        assert extract_dialog(text) == "Let me think about this. Done!"

    def test_text_before_code_block(self):
        """Text before a code block is extracted."""
        text = 'Here is the solution:\n```python\nx = 42\n```'
        assert extract_dialog(text) == "Here is the solution:"

    def test_multiple_code_blocks(self):
        """Text between multiple code blocks is extracted."""
        text = (
            'First, define a variable:\n'
            '```python\nx = 42\n```\n'
            'Now print it:\n'
            '```python\nprint(x)\n```'
        )
        assert extract_dialog(text) == "First, define a variable: Now print it:"


# ---------------------------------------------------------------------------
# parse_dialog
# ---------------------------------------------------------------------------

class TestParseDialog:
    """Tests for parse_dialog utility."""

    def test_text_only(self):
        """Plain text without code blocks or media yields (body, None)."""
        result = list(parse_dialog("Hello world"))
        assert result == [("Hello world", None)]

    def test_code_block_only(self):
        """Input that is only a code block yields nothing."""
        result = list(parse_dialog('```python\nprint("hi")\n```'))
        assert not result

    def test_text_with_media(self):
        """Dialog text containing a media path is split correctly."""
        text = 'Here is your chart: gen/chart.svg\n```python\ncode\n```'
        result = list(parse_dialog(text))
        assert result == [("Here is your chart:", "gen/chart.svg")]

    def test_text_and_media_interleaved(self):
        """Dialog with interleaved body and media paths."""
        text = (
            'First chart: gen/a.svg and second: gen/b.svg Done!\n'
            '```python\ncode\n```'
        )
        result = list(parse_dialog(text))
        assert result == [
            ("First chart:", "gen/a.svg"),
            ("and second:", "gen/b.svg"),
            ("Done!", None),
        ]
