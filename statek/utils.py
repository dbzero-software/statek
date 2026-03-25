"""Utility functions for statek package."""  # pylint: disable=too-many-lines

import ast
import difflib
import logging
import re
import inspect
import sys
from collections import namedtuple
from dataclasses import dataclass, is_dataclass, fields as dataclass_fields
from datetime import datetime
from decimal import Decimal
from typing import (Callable, Iterable, List, Dict, Optional, Sequence, Tuple, Type, Any,
                    get_type_hints, get_origin, get_args, Union, ForwardRef)
import dbzero as db0

logger = logging.getLogger(__name__)

ParsedFuncCall = namedtuple("ParsedFuncCall", ["name", "args", "kwargs"])
ParsedWarmupBlock = namedtuple("ParsedWarmupBlock", ["code", "tool_calls"])


@db0.memo
@dataclass
class CallSpec:
    """LLM or system-assigned tool call specification."""
    id: str
    func_name: str
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    def format(self) -> str:  # pylint: disable=redefined-builtin
        """Format this call spec as a Python-style function call string.

        Returns:
            A string like ``func_name(arg1, key='val')``
        """
        parts = []
        if self.args:
            parts.extend(repr(a) for a in self.args)
        if self.kwargs:
            parts.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.func_name}({', '.join(parts)})"


@db0.memo
@dataclass
class CodeBlock:
    """A block of executable Python code with optional tool-call requests."""
    code: Optional[str] = None
    tool_calls: Optional[Sequence[CallSpec]] = None

    def __eq__(self, other):
        if isinstance(other, CodeBlock):
            if self.code != other.code:
                return False
            self_tcs = list(self.tool_calls) if self.tool_calls else []
            other_tcs = list(other.tool_calls) if other.tool_calls else []
            if len(self_tcs) != len(other_tcs):
                return False
            return all(
                a.id == b.id and a.func_name == b.func_name
                and a.args == b.args and a.kwargs == b.kwargs
                for a, b in zip(self_tcs, other_tcs)
            )
        if isinstance(other, str) and not self.tool_calls:
            return self.code == other
        return NotImplemented

    def get_tool_call_id(self, call_spec: CallSpec) -> Optional[int]:
        """Return the index of *call_spec* in tool_calls, or None if absent."""
        if not self.tool_calls:
            return None
        for idx, tc in enumerate(self.tool_calls):
            if tc is call_spec:
                return idx
        return None

    def __hash__(self):
        tcs = tuple(self.tool_calls) if self.tool_calls else ()
        return hash((self.code, tcs))

_STATEK_TOOL_MARKER = "#STATEK: as tool"


def parse_func_call(input: str) -> ParsedFuncCall:  # pylint: disable=redefined-builtin
    """Parse a string representation of a function call into a structured result.

    Args:
        input: the input function call string

    Returns:
        ParsedFuncCall with name, args (list), and kwargs (dict or None)

    Raises:
        ValueError: if the input is not a valid function call expression
        SyntaxError: if the input is not valid Python syntax
    """
    tree = ast.parse(input, mode='eval')
    if not isinstance(tree.body, ast.Call):
        raise ValueError(f"Not a function call: {input!r}")
    call = tree.body
    if isinstance(call.func, ast.Name):
        name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        name = call.func.attr
    else:
        raise ValueError(f"Unsupported function expression: {input!r}")
    args = [arg.id if isinstance(arg, ast.Name) else ast.literal_eval(arg) for arg in call.args]
    kwargs = {
        kw.arg: kw.value.id if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value)
        for kw in call.keywords
    } or None
    return ParsedFuncCall(name=name, args=args, kwargs=kwargs)


def parse_warmup_block(code: str) -> ParsedWarmupBlock:
    """Parse a single warmup block into code and tool call definitions.

    Lines annotated with ``#STATEK: as tool`` are extracted as tool calls
    and removed from the returned code field.

    Args:
        code: Python code block to be parsed

    Returns:
        ParsedWarmupBlock with clean code and a list of ParsedFuncCall tool calls

    Raises:
        ValueError: if an annotated line is not a valid function call
        SyntaxError: if an annotated line contains invalid Python syntax
    """
    clean_lines = []
    tool_calls = []
    for line in code.splitlines():
        marker_pos = line.find(_STATEK_TOOL_MARKER)
        if marker_pos != -1:
            call_str = line[:marker_pos].rstrip()
            tool_calls.append(parse_func_call(call_str))
        else:
            clean_lines.append(line)
    return ParsedWarmupBlock(code="\n".join(clean_lines), tool_calls=tool_calls)


def build_warmup_code(
    parsed_warmup_code: List["ParsedWarmupBlock"],
) -> "str | CodeBlock | List[str | CodeBlock]":
    """Build warmup code items from parsed warmup blocks.

    Converts each ParsedWarmupBlock into either a plain string (when the block
    has no tool calls) or a CodeBlock (when tool calls are present). Tool calls
    receive unique call IDs in the ``STATEK-NNN`` format, numbered sequentially
    starting from 1 across all blocks in the list.

    Args:
        parsed_warmup_code: list of parsed warmup blocks to convert

    Returns:
        A single str or CodeBlock when the list contains exactly one item,
        otherwise a list of str and/or CodeBlock values.
    """
    counter = 0
    blocks = []
    for parsed_block in parsed_warmup_code:
        if not parsed_block.tool_calls:
            blocks.append(parsed_block.code)
        else:
            tool_calls = []
            for tc in parsed_block.tool_calls:
                counter += 1
                tool_calls.append(CallSpec(
                    id=f"STATEK-{counter:03d}",
                    func_name=tc.name,
                    args=tc.args if tc.args else [],
                    kwargs=tc.kwargs if tc.kwargs is not None else {},
                ))
            blocks.append(CodeBlock(code=parsed_block.code, tool_calls=tool_calls))

    if len(blocks) == 1:
        return blocks[0]
    return blocks


_STATEK_PRINT_SCALAR_TYPES = (
    bool, int, float, Decimal, str, datetime, list, tuple, set, frozenset, dict
)

_NONE_TYPE = type(None)

_JSON_SCHEMA_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
}


def strip_markup(input: str, strict: bool) -> str:  # pylint: disable=redefined-builtin
    """Strip markdown code fences from LLM output, returning clean executable code.

    Splits the input on code fence markers. Even-indexed parts are plain text
    and become block comments; odd-indexed parts are code and are returned as-is.

    Args:
        input: The raw LLM response string, potentially containing markdown
        strict: If True, only content inside ```python fences is returned as
                plain code; everything else (plain text and other fenced blocks)
                is converted to block comments.
                If False, any fenced block is treated as code and the input is
                returned unchanged when no fences are present.

    Returns:
        Clean Python code with non-code text converted to block comments
    """
    if strict:
        pattern = r'```python\n(.*?)```'
    else:
        if '```' not in input:
            return input
        pattern = r'```\w*\n?'

    # Parts alternate: text (even indices), code (odd indices)
    parts = re.split(pattern, input, flags=re.DOTALL)
    result_parts = []
    for i, part in enumerate(parts):
        stripped = part.strip()
        if not stripped:
            continue
        if i % 2 == 0:
            result_parts.append(block_comment(stripped))
        else:
            result_parts.append(stripped)
    return '\n'.join(result_parts)


def extract_dialog(input: str) -> Optional[str]:  # pylint: disable=redefined-builtin
    """Extract the free-text / dialog part from LLM output, stripping markdown blocks.

    This is the opposite of strip_markup: it returns only the text *outside*
    any markdown code fences, cleaned up from extra whitespace.

    Args:
        input: The input text (possibly containing markdown blocks).

    Returns:
        The free-text portion of the input, or None if there is no dialog text.
    """
    parts = re.split(r'```\w*\n?', input, flags=re.DOTALL)
    text_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            cleaned = ' '.join(part.split())
            if cleaned:
                text_parts.append(cleaned)
    result = ' '.join(text_parts).strip()
    return result or None


def parse_md_dialog(input: str) -> Iterable[Tuple[str, Optional[str]]]:  # pylint: disable=redefined-builtin
    """Parse LLM-generated output into (body, media) tuples for message delivery.

    Chains extract_dialog (to strip markdown blocks) with extract_media
    (to split body text from media paths).

    Args:
        input: The LLM generated text to be processed.

    Yields:
        (body, media) tuples where either or both may be None.
    """
    dialog = extract_dialog(input)
    if dialog is None:
        return
    yield from extract_media(dialog)


def block_comment(text: str) -> str:
    """Put input string in a Python block comment.

    Args:
        text: The input string/code to comment out

    Returns:
        str: The commented-out block with each line prefixed by '# '
    """
    lines = text.split('\n')
    commented_lines = ['# ' + line for line in lines]
    return '\n'.join(commented_lines)


def format_callable_decl(func: Callable) -> str:
    """
    Format a callable's declaration to be presented to LLMs.

    Reproduces the original Python syntax used in the callable's declaration,
    without accessing the source code. Includes the function's docstring if available.

    For temporal functions (decorated with @temporal), reports the return type
    of the complement function instead of the original return type.

    Args:
        func: A callable (e.g., function) to format

    Returns:
        The Python declaration string including callable name, argument names,
        argument types, default values, return type(s), and docstring.

    Examples:
        >>> def send_message(message: str) -> str: pass
        >>> format_callable_decl(send_message)
        'def send_message(message: str) -> str'

        >>> def find_user(pattern: str, max_result: int = None) -> User | Iterable[User]: pass
        >>> format_callable_decl(find_user)
        'def find_user(pattern: str, max_result: int = None) -> User | Iterable[User]'
    """
    func_name = func.__name__
    hints = _get_type_hints(func)
    params_str = _format_parameters(func, hints)
    return_annotation = _format_return_annotation(func, hints)

    decl = f"def {func_name}({params_str}){return_annotation}"

    # Add docstring if available
    docstring = inspect.getdoc(func)
    if docstring:
        decl += f"\n{docstring}"

    return decl


def format_tool_spec(func: Callable) -> Dict:
    """
    Format a callable as a JSON tool specification for OpenAI/Anthropic APIs.

    Inspects the callable's name, type hints, and docstring to produce a formal
    tool specification in the OpenAI function-calling format, which is also
    compatible with Anthropic's Claude API and services like OpenRouter.

    Args:
        func: A callable (e.g., function) to format

    Returns:
        The tool specification as a Python dict following the OpenAI function-calling
        schema, with ``"type": "function"`` wrapper.

    Examples:
        >>> def execute_python(code: str) -> str: ...
        >>> spec = format_tool_spec(execute_python)
        >>> spec["type"]
        'function'
        >>> spec["function"]["name"]
        'execute_python'
    """
    func_name = func.__name__
    description, param_descs = _extract_docstring_info(func)
    hints = _get_type_hints(func)
    sig = inspect.signature(func)

    properties: Dict = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name.startswith('_') or param.kind in (
            inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL
        ):
            continue

        prop: Dict = {"type": _type_to_json_schema(hints[name]) if name in hints else "string"}
        if name in param_descs:
            prop["description"] = param_descs[name]
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    parameters: Dict = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": parameters,
        },
    }


def _extract_docstring_info(func: Callable) -> tuple:
    """Extract brief description and parameter descriptions from a callable's docstring.

    Returns a (description, param_descs) tuple. Falls back gracefully when the
    docstring is absent or incompletely documents the function's parameters.
    """
    from statek.docstring import parse_tool_docstring, DocstringParseError  # pylint: disable=import-outside-toplevel
    try:
        doc = parse_tool_docstring(func)
        description = doc.brief_desc or ""
        param_descs = {arg.name: arg.desc for arg in doc.args} if doc.args else {}
        return description, param_descs
    except DocstringParseError:
        docstring = inspect.getdoc(func)
        description = docstring.split('\n', maxsplit=1)[0].strip() if docstring else ""
        return description, {}


def _type_to_json_schema(type_hint) -> str:
    """Map a Python type hint to a JSON Schema type string.

    Handles primitives, generics (List, Dict, Optional, Union), and db0 enum types.
    Unknown or complex types default to ``"string"``.
    """
    # db0 enum types are presented as string values
    if db0.is_enum(type_hint):  # pylint: disable=no-member
        return "string"

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Optional[X] = Union[X, None] and other Union types
    if origin is Union:
        non_none = [a for a in args if a is not _NONE_TYPE]
        return _type_to_json_schema(non_none[0]) if non_none else "null"

    # Generic collection types: infer from origin
    if origin is not None:
        if origin is dict or (isinstance(origin, type) and issubclass(origin, dict)):
            return "object"
        return "array"

    # NoneType and primitive types
    return _JSON_SCHEMA_TYPE_MAP.get(type_hint, "null" if type_hint is _NONE_TYPE else "string")


def _get_type_hints(func: Callable) -> dict:
    """Get type hints for a callable, with fallback to annotations."""
    try:
        return get_type_hints(func)
    except Exception:  # pylint: disable=broad-exception-caught
        return getattr(func, "__annotations__", {})


def _format_parameters(func: Callable, hints: dict) -> str:
    """Format all parameters of a callable."""
    sig = inspect.signature(func)
    params = [
        _format_parameter(name, param, hints)
        for name, param in sig.parameters.items()
        if not name.startswith('_') and param.kind != inspect.Parameter.VAR_KEYWORD
    ]  # Skip internal parameters and **kwargs
    return ", ".join(params)


def _format_parameter(param_name: str, param: inspect.Parameter, hints: dict) -> str:
    """Format a single parameter with its type and default value."""
    parts = [param_name]

    if param_name in hints:
        parts.append(f": {_format_type(hints[param_name])}")

    if param.default is not inspect.Parameter.empty:
        parts.append(_format_default_value(param.default))

    return "".join(parts)


def _format_default_value(default) -> str:
    """Format a default value for a parameter."""
    match default:
        case None:
            return " = None"
        case str():
            return f' = "{default}"'
        case int() | float() | bool():
            return f" = {default}"
        case _:
            return f" = {repr(default)}"


def _format_return_annotation(func: Callable, hints: dict) -> str:
    """Format the return type annotation, handling temporal functions."""
    if "return" not in hints:
        return ""

    return_type_str = _format_type(hints["return"])

    if getattr(func, "__is_temporal__", False):
        complement_type = _extract_complement_return_type(func)
        if complement_type:
            return_type_str = complement_type

    return f" -> {return_type_str}"


def _extract_complement_return_type(func: Callable) -> str:
    """Extract return type from a temporal function's complement function."""
    # First, try to get complement from the __temporal_complement__ attribute
    # (set by the @temporal decorator for extended functions)
    if hasattr(func, "__temporal_complement__"):
        complement_func = func.__temporal_complement__
        try:
            complement_hints = get_type_hints(complement_func)
            if "return" in complement_hints:
                return _format_type(complement_hints["return"])
        except (AttributeError, ValueError, TypeError):
            pass

    # Fallback to closure extraction for older temporal functions
    if not (hasattr(func, "__closure__") and func.__closure__):
        return None

    for cell in func.__closure__:
        try:
            complement_func = cell.cell_contents
            if not callable(complement_func):
                continue
            complement_hints = get_type_hints(complement_func)
            if "return" in complement_hints:
                return _format_type(complement_hints["return"])
        except (AttributeError, ValueError, TypeError):
            continue

    return None


def _format_type(type_hint) -> str:  # pylint: disable=too-many-return-statements
    """
    Format a type hint into a readable string.

    Args:
        type_hint: A type hint to format

    Returns:
        A string representation of the type hint
    """
    # Handle db0 enum types — report as str for LLM consumption
    if db0.is_enum(type_hint):  # pylint: disable=no-member
        return "str"

    # Handle ForwardRef objects
    if isinstance(type_hint, ForwardRef):
        return type_hint.__forward_arg__

    # Handle None type
    if isinstance(type_hint, type(None)):
        return "None"

    # Get the origin and args for generic types (check before __name__)
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Union types (including Optional)
    if origin is Union:
        # Format as Type1 | Type2 | Type3 (modern syntax)
        formatted_args = [_format_type(arg) for arg in args]
        return " | ".join(formatted_args)

    # Handle generic types like List[str], Dict[str, int], etc.
    if origin is not None:
        if args:
            formatted_args = ", ".join(_format_type(arg) for arg in args)
            origin_name = getattr(origin, "__name__", str(origin))
            return f"{origin_name}[{formatted_args}]"
        return getattr(origin, "__name__", str(origin))

    # Handle basic types with __name__
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__

    # Fallback: clean up string representation
    type_str = str(type_hint)
    if type_str.startswith("typing."):
        type_str = type_str.replace("typing.", "")
    return type_str


def _fmt_console_lines(lines: List[str], chat_style) -> str:
    """Format a slice of console lines according to chat_style."""
    from statek.settings import ChatStyle  # pylint: disable=import-outside-toplevel
    if chat_style in (ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
        joined = "\n".join(lines)
        return f"<CONSOLE>\n{joined}\n</CONSOLE>"
    if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
        return "\n".join(lines)
    return "\n".join(f"> {line}" for line in lines)


def prompt_append_console(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        console: List[str], chat_style,
        prompt: str = None, from_pos: int = 0,
        limit: int = None,
        xml_tags: dict = None) -> str:
    """
    Extend a prompt with the console outputs.

    This is a helper function to format console output for LLM consumption.
    Formatting depends on chat_style:
      CONSOLE  - prompt is presented as-is; console lines are prefixed with "> ".
      MARKDOWN - prompt is wrapped in ```python fences; console lines are as-is.

    Args:
        console: The list representation of the console state
        chat_style: The ChatStyle to apply (CONSOLE or MARKDOWN)
        prompt: Optional leading prompt (code text)
        from_pos: First element to start output from
        limit: Optional limit of consecutive console elements to be included

    Returns:
        The complete formatted text ready for the LLM recipient

    Examples:
        >>> console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
        >>> prompt = 'print(user)\\nprint(clock.now())'
        >>> prompt_append_console(console, ChatStyle.CONSOLE, prompt)
        'print(user)\\nprint(clock.now())\\n> User(name = "Kowalski Adam")\\n> 2026-01-03 12:13:32'
        >>> prompt_append_console(console, ChatStyle.MARKDOWN, prompt)
        '```python\\nprint(user)\\nprint(clock.now())\\n```\\n'
        'User(name = "Kowalski Adam")\\n2026-01-03 12:13:32'
    """
    from statek.settings import ChatStyle  # pylint: disable=import-outside-toplevel
    # In MARKDOWN mode wrap the prompt code in python fences; otherwise use as-is
    if prompt:
        if chat_style in (ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
            result = f"```python\n{prompt}\n```"
        else:
            result = prompt
    else:
        result = ""

    # Handle case when console is None or empty
    if not console:
        return result

    # Determine the range of console elements to include
    end_pos = len(console)
    if limit is not None:
        end_pos = min(from_pos + limit, end_pos)

    # Format console outputs according to chat_style
    console_slice = console[from_pos:end_pos]
    if console_slice:
        console_text = _fmt_console_lines(console_slice, chat_style)
        tag = xml_tags and xml_tags.get("console")
        if tag:
            console_text = f"<{tag}>\n{console_text}\n</{tag}>"
        if result:
            result += "\n" + console_text
        else:
            result = console_text

    return result


def get_current_agent():
    """Retrieve the Agent from the current execution context (_STATEK_CTX).

    Returns:
        The Agent object from _STATEK_CTX, or None if not available.
    """
    ctx = next(iter(find_locals(var_name='_STATEK_CTX')), None)
    if not isinstance(ctx, dict):
        return None
    return ctx.get('agent')


def get_current_job():
    """Retrieve the Job from the current execution context (_STATEK_CTX).

    Returns:
        The Job object from _STATEK_CTX, or None if not available.
    """
    ctx = next(iter(find_locals(var_name='_STATEK_CTX')), None)
    if not isinstance(ctx, dict):
        return None
    return ctx.get('job')


def get_current_agent_name() -> Optional[str]:
    """Get the current agent's type name without module qualifiers.

    Returns:
        The agent's type name (e.g. 'MessageDispatcher'), or None if not available.
    """
    agent = get_current_agent()
    if agent is None:
        return None
    return _get_class_name(agent)


def _collect_aggregated_locals(start_frame, max_frames: int = 40) -> dict:
    """Collect and merge locals from up to max_frames call-stack frames."""
    aggregated: dict = {}
    frames = []
    frame = start_frame
    for _ in range(max_frames):
        if frame is None:
            break
        frames.append(frame)
        frame = frame.f_back

    for f in reversed(frames):
        f_locals = f.f_locals.copy()
        lctx = f_locals.get('_local_context')
        if isinstance(lctx, dict):
            aggregated.update(lctx)
        kwargs = f_locals.get('kwargs')
        if isinstance(kwargs, dict):
            lctx = kwargs.get('_local_context')
            if isinstance(lctx, dict):
                aggregated.update(lctx)
        aggregated.update(f_locals)
    return aggregated


def _yield_future_by_name(value, var_type):  # pylint: disable=import-outside-toplevel
    """Resolve a FutureResult by name; propagate FutureError if not ready."""
    actual = value.value  # raises FutureError when not ready
    if var_type is None or isinstance(actual, var_type):
        yield actual


def _yield_future_by_type(value, var_type):
    """Resolve a FutureResult by type; silently skip if not ready."""
    from statek.exceptions import FutureError  # pylint: disable=import-outside-toplevel
    try:
        actual = value.value
        if isinstance(actual, var_type):
            yield actual
    except FutureError:
        pass


def find_locals(var_type: Optional[Type] = None,
                var_name: Optional[str] = None) -> Iterable[Any]:
    """
    Search through the caller's local context - retrieving variables matching
    a specific type or name. This function is helpful when implementing temporal
    functions which need to be context-aware.

    FutureResult / FutureElement resolution:
      - By name (var_name specified): if the variable is a FutureResult or FutureElement,
        resolve its value (raises FutureError if not ready yet), then check var_type.
      - By type only (var_type specified, var_name=None): also scan all FutureResult /
        FutureElement locals, attempt to resolve each, and yield matching values; if a
        future is not ready (FutureError), it is silently skipped.

    Args:
        var_type: Optional type to identify local variables by (e.g. SMS_Message or User)
        var_name: Optional variable name to match

    Yields:
        Matching variables from the caller's context. If neither var_type nor var_name
        is specified, all variables from the local context will be yielded.
    """
    from statek.future import FutureResult  # pylint: disable=import-outside-toplevel

    aggregated_locals = _collect_aggregated_locals(inspect.currentframe().f_back)

    found = False
    for name, value in aggregated_locals.items():
        if var_type is None and var_name is None:
            yield value
            continue

        if var_name is not None and name != var_name:
            continue

        is_future = isinstance(value, FutureResult)

        if is_future and var_name is not None:
            yielded = list(_yield_future_by_name(value, var_type))
            if yielded:
                found = True
                yield from yielded
        elif is_future:
            yield from _yield_future_by_type(value, var_type)
        elif var_type is None or isinstance(value, var_type):
            found = True
            yield value

    if var_name is not None and not found and logger.isEnabledFor(logging.DEBUG):
        available = list(aggregated_locals.keys())
        close = difflib.get_close_matches(var_name, available, n=3, cutoff=0.5)
        logger.debug(
            "find_locals: variable %r not found. "
            "Available: %s. Closest matches: %s",
            var_name,
            available,
            close if close else "(none)",
        )


def format_value_repr(value: Any) -> str:
    """Format a single value for LLM representation.

    Args:
        value: the value to be formatted

    Returns:
        A string representation suitable for LLM consumption.
    """
    if isinstance(value, (bool, int, float, Decimal)):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, datetime):
        return f'datetime("{value.strftime("%Y-%m-%d %H:%M")}")'
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        type_name = type(value).__name__.capitalize()
        return f"<{type_name} of {len(value)} items>"
    return "<Object>"


def format_llm_repr(value: Any, hide: List[str] = None,
                    expand: List[str] = None, show_only: List[str] = None,
                    max_len: int = 25, **kwargs) -> str:
    """Format the LLM-friendly representation of a Python object.

    For simple scalar values, forwards to format_value_repr.
    For collections, shows up to max_len items with a truncation notice.
    For objects (including db0.memo objects), shows ClassName(member=value, ...).

    Args:
        value: the value to be formatted
        hide: list of members to be hidden in the result
        expand: list of members to be shown expanded (using format_llm_repr recursively)
        show_only: if defined, only these members are shown (hide still takes precedence)
        max_len: maximum number of collection items to display

    Returns:
        LLM-friendly string representation.
    """
    if isinstance(value, (bool, int, float, Decimal, str, datetime)):
        return format_value_repr(value)

    if isinstance(value, (list, tuple)):
        brackets = ('[', ']') if isinstance(value, list) else ('(', ')')
        return _format_sequence(list(value), *brackets, max_len)
    if isinstance(value, (set, frozenset)):
        return _format_sequence(sorted(value, key=repr), '{', '}', max_len)
    if isinstance(value, dict):
        return _format_dict_llm(value, max_len)

    members = _get_object_members(value)
    if members is not None:
        return _format_object_llm(
            _get_class_name(value), members, hide, expand, show_only, **kwargs)

    return format_value_repr(value)


def _format_element(item: Any, max_len: int, repeated: bool) -> str:
    """Format a single element within a collection.

    Applies the same dispatch as format_default_llm_repr, threading max_len and
    repeated through to the fallback format_llm_repr call for plain objects.
    """
    if isinstance(item, _STATEK_PRINT_SCALAR_TYPES):
        return format_llm_repr(item, max_len=max_len)
    if hasattr(item, '__llm_repr__'):
        method = item.__llm_repr__
        sig = inspect.signature(method)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        return method(repeated=repeated) if accepts_kwargs else method()
    if type(item).__str__ is not object.__str__:
        return str(item)
    return format_llm_repr(item, max_len=max_len, repeated=repeated)


def _format_sequence(items, open_br: str, close_br: str, max_len: int) -> str:
    """Format a sequence with optional truncation."""
    all_items = list(items)
    total = len(all_items)
    shown = all_items[:max_len]
    seen_types: set = set()
    parts = []
    for item in shown:
        item_type = type(item)
        is_repeated = item_type in seen_types
        seen_types.add(item_type)
        parts.append(_format_element(item, max_len=max_len, repeated=is_repeated))
    formatted = ",".join(parts)
    if total > max_len:
        return f"{open_br}{formatted}, ...{close_br} ({total} items total)"
    return f"{open_br}{formatted}{close_br}"


def _format_dict_llm(value: dict, max_len: int) -> str:
    """Format a dict with optional truncation."""
    items = list(value.items())
    total = len(items)
    shown = items[:max_len]
    seen_types: set = set()
    parts = []
    for k, v in shown:
        v_type = type(v)
        is_repeated = v_type in seen_types
        seen_types.add(v_type)
        parts.append(
            f"{format_value_repr(k)}:{_format_element(v, max_len=max_len, repeated=is_repeated)}")
    formatted = ",".join(parts)
    if total > max_len:
        return "{" + formatted + ", ...} (" + str(total) + " items total)"
    return "{" + formatted + "}"


def _get_class_name(value: Any) -> str:
    """Return the display class name, stripping db0.memo's 'Memo_' prefix if present."""
    cls = type(value)
    name = cls.__name__
    if not name.startswith('Memo_'):
        return name
    # db0.memo wraps classes with a 'Memo_' prefix; find original name via MRO
    for base in cls.__mro__[1:]:
        base_name = base.__name__
        if base_name and base_name != 'object' and not base_name.startswith('Memo_'):
            return base_name
    return name[5:]  # fallback: strip prefix


def _get_object_members(value: Any) -> Optional[Dict[str, Any]]:
    """Return an ordered dict of an object's members, or None if not applicable."""
    if isinstance(value, type):
        return None
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in dataclass_fields(value)}
    if hasattr(value, '__dict__'):
        return dict(vars(value))
    return None


def format_default_llm_repr(obj: Any, **kwargs) -> str:
    """Format the default LLM-friendly representation of a value.

    Applies the following resolution order:
    1. __llm_repr__ if defined on the object
    2. __str__ if explicitly defined (not just inherited from object)
    3. format_llm_repr with default arguments

    Simple values (bool, int, float, Decimal, str, datetime) and collections
    always use format_llm_repr, preserving LLM-friendly formatting (e.g.
    strings are quoted, datetimes use the compact format).

    kwargs (e.g. hide, expand, show_only, max_len) are forwarded to
    format_llm_repr and also to __llm_repr__ if it accepts **kwargs.
    __str__ never receives kwargs.

    If __llm_repr__ calls format_default_llm_repr(self, ...) for the same
    type, the recursion is detected via the call stack and the call falls
    through directly to format_llm_repr, preventing infinite recursion.

    Args:
        obj: The object to format.
        **kwargs: Forwarded to format_llm_repr; also forwarded to __llm_repr__
            if its signature accepts **kwargs.

    Returns:
        LLM-friendly string representation.
    """
    if isinstance(obj, _STATEK_PRINT_SCALAR_TYPES):
        return format_llm_repr(obj, **kwargs)

    if hasattr(obj, '__llm_repr__'):
        obj_type = type(obj)
        frame = sys._getframe(1)  # pylint: disable=protected-access
        while frame is not None:
            if (frame.f_code.co_name == '__llm_repr__' and
                    type(frame.f_locals.get('self')) is obj_type):  # pylint: disable=unidiomatic-typecheck
                break
            frame = frame.f_back
        else:
            method = obj.__llm_repr__
            sig = inspect.signature(method)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            return method(**kwargs) if accepts_kwargs else method()
        return format_llm_repr(obj, **kwargs)

    if type(obj).__str__ is not object.__str__:
        return str(obj)
    return format_llm_repr(obj, **kwargs)


def statek_print(*objects, sep=' ', end='\n', file=None, flush=False):
    """LLM-friendly replacement for Python's built-in print.

    Prints each object using format_default_llm_repr.

    Args:
        *objects: Objects to print.
        sep: String inserted between values. Defaults to ' '.
        end: String appended after the last value. Defaults to '\\n'.
        file: File-like object to write to. Defaults to sys.stdout.
        flush: Whether to forcibly flush the stream. Defaults to False.
    """
    text = sep.join(format_default_llm_repr(obj) for obj in objects)
    print(text, end=end, file=file, flush=flush)


def _format_object_llm(class_name: str, members: Dict[str, Any],
                        hide: List[str], expand: List[str],
                        show_only: List[str], **kwargs) -> str:
    """Format an object representation as ClassName(member=value, ...)."""
    repeated = kwargs.get('repeated', False)

    if show_only is not None:
        shown = {k: v for k, v in members.items() if k in show_only}
    else:
        shown = dict(members)

    if hide:
        shown = {k: v for k, v in shown.items() if k not in hide}

    parts = []
    has_omitted = False
    for name, val in shown.items():
        if expand and name in expand:
            formatted_val = format_llm_repr(val, **kwargs)
        else:
            formatted_val = format_value_repr(val)
            if repeated and formatted_val == '<Object>':
                has_omitted = True
                continue
        parts.append(f"{name}={formatted_val}")

    if has_omitted:
        parts.append('...')

    return f"{class_name}({','.join(parts)})"


# Media path pattern: no whitespace, at least one folder separator, 3-char extension
_MEDIA_PATH_RE = re.compile(r'^/?(?:\S+/)+\S+\.\w{3}$')


def extract_media(text: str) -> Iterable[Tuple[str, Optional[str]]]:
    """Parse text into (body, media) tuples.

    Media paths are tokens that contain at least one folder separator and end
    with a 3-character extension.  Extra whitespace is eliminated during parsing.

    Args:
        text: The input string to parse.

    Yields:
        (body, media) tuples where *media* is a file path or None for the
        trailing body fragment.
    """
    tokens = text.split()
    body_parts: List[str] = []

    for token in tokens:
        if _MEDIA_PATH_RE.match(token):
            yield (" ".join(body_parts), token)
            body_parts = []
        else:
            body_parts.append(token)

    if body_parts:
        yield (" ".join(body_parts), None)
