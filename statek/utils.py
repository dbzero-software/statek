"""Utility functions for statek package."""

import inspect
from typing import Callable, List, get_type_hints, get_origin, get_args, Union


def format_callable_decl(func: Callable) -> str:
    """
    Format a callable's declaration to be presented to LLMs.

    Reproduces the original Python syntax used in the callable's declaration,
    without accessing the source code.

    For temporal functions (decorated with @temporal), reports the return type
    of the complement function instead of the original return type.

    Args:
        func: A callable (e.g., function) to format

    Returns:
        The Python declaration string including callable name, argument names,
        argument types, default values, and return type(s).

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

    return f"def {func_name}({params_str}){return_annotation}"


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
        if not name.startswith('_') and name != 'kwargs'
    ]  # Skip internal parameters and kwargs
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


def _format_type(type_hint) -> str:
    """
    Format a type hint into a readable string.

    Args:
        type_hint: A type hint to format

    Returns:
        A string representation of the type hint
    """
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

    # Handle basic types
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__

    # Fallback to string representation
    type_str = str(type_hint)

    # Clean up common type string representations
    if type_str.startswith("typing."):
        type_str = type_str.replace("typing.", "")

    return type_str


def prompt_append_console(console: List[str], prompt: str = None,
                          from_pos: int = 0, limit: int = None) -> str:
    """
    Extend a prompt with the console outputs.

    This is a helper function to format console output for LLM consumption by
    appending console items (prefixed with "> ") to an optional initial prompt.

    Args:
        console: The list representation of the console state
        prompt: Optional leading prompt (regular text)
        from_pos: First element to start output from
        limit: Optional limit of consecutive console elements to be included

    Returns:
        The complete formatted text ready for the LLM recipient

    Examples:
        >>> console = ['User(name = "Kowalski Adam")', '2026-01-03 12:13:32']
        >>> prompt = 'print(user)\\nprint(clock.now())'
        >>> prompt_append_console(console, prompt)
        'print(user)\\nprint(clock.now())\\n> User(name = "Kowalski Adam")\\n> 2026-01-03 12:13:32'
    """
    # Start with the initial prompt if provided
    result = prompt if prompt else ""

    # Handle case when console is None or empty
    if not console:
        return result

    # Determine the range of console elements to include
    end_pos = len(console)
    if limit is not None:
        end_pos = min(from_pos + limit, end_pos)

    # Append console outputs with "> " prefix
    console_outputs = []
    for i in range(from_pos, end_pos):
        console_outputs.append(f"> {console[i]}")

    # Join console outputs and append to result
    if console_outputs:
        console_text = "\n".join(console_outputs)
        if result:
            result += "\n" + console_text
        else:
            result = console_text

    return result
