"""Utility functions for statek package."""

import re
import inspect
from typing import (Callable, Iterable, List, Optional, Type, Any,
                    get_type_hints, get_origin, get_args, Union, ForwardRef)
import dbzero as db0


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


def prompt_append_console(console: List[str], chat_style,
                          prompt: str = None, from_pos: int = 0,
                          limit: int = None) -> str:
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
        if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
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
    console_outputs = []
    for i in range(from_pos, end_pos):
        if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
            console_outputs.append(console[i])
        else:
            console_outputs.append(f"> {console[i]}")

    # Join console outputs and append to result
    if console_outputs:
        console_text = "\n".join(console_outputs)
        if result:
            result += "\n" + console_text
        else:
            result = console_text

    return result


def find_locals(var_type: Optional[Type] = None,
                var_name: Optional[str] = None) -> Iterable[Any]:
    """
    Search through the caller's local context - retrieving variables matching
    a specific type or name. This function is helpful when implementing temporal
    functions which need to be context-aware.

    Args:
        var_type: Optional type to identify local variables by (e.g. SMS_Message or User)
        var_name: Optional variable name to match

    Yields:
        Matching variables from the caller's context. If neither var_type nor var_name
        is specified, all variables from the local context will be yielded.
    """
    # Search through up to 10 frames up the call stack
    caller_frame = inspect.currentframe().f_back
    frames_to_search = []

    # Collect up to 10 frames
    current_frame = caller_frame
    for _ in range(10):
        if current_frame is None:
            break
        frames_to_search.append(current_frame)
        current_frame = current_frame.f_back

    # Aggregate locals from all frames, with priority to closer frames
    aggregated_locals = {}
    for frame in reversed(frames_to_search):
        frame_locals = frame.f_locals.copy()

        # Check if _local_context is set and extend with it
        if '_local_context' in frame_locals:
            local_context = frame_locals['_local_context']
            if local_context is not None and isinstance(local_context, dict):
                aggregated_locals.update(local_context)

        # Also check if kwargs contains _local_context
        if 'kwargs' in frame_locals and isinstance(frame_locals['kwargs'], dict):
            if '_local_context' in frame_locals['kwargs']:
                local_context = frame_locals['kwargs']['_local_context']
                if local_context is not None and isinstance(local_context, dict):
                    aggregated_locals.update(local_context)

        # Update with frame locals (closer frames override)
        aggregated_locals.update(frame_locals)

    # Iterate through aggregated local variables
    for name, value in aggregated_locals.items():
        # If no filters specified, yield all variables
        if var_type is None and var_name is None:
            yield value
        else:
            # Apply filters
            type_match = var_type is None or isinstance(value, var_type)
            name_match = var_name is None or name == var_name

            if type_match and name_match:
                yield value
