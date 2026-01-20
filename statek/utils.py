"""Utility functions for statek package."""

import inspect
from typing import Callable, List, get_type_hints, get_origin, get_args, Union


def format_callable_decl(func: Callable) -> str:
    """
    Format a callable's declaration to be presented to LLMs.

    Reproduces the original Python syntax used in the callable's declaration,
    without accessing the source code.

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
    # Get function name
    func_name = func.__name__

    # Get signature
    sig = inspect.signature(func)

    # Get type hints (this properly resolves forward references and string annotations)
    try:
        hints = get_type_hints(func)
    except Exception:  # pylint: disable=broad-exception-caught
        # Fallback to annotations if get_type_hints fails
        hints = getattr(func, "__annotations__", {})

    # Format parameters
    params = []
    for param_name, param in sig.parameters.items():
        parts = [param_name]

        # Add type annotation if available
        if param_name in hints:
            type_str = _format_type(hints[param_name])
            parts.append(f": {type_str}")

        # Add default value if present
        if param.default is not inspect.Parameter.empty:
            match param.default:
                case None:
                    parts.append(" = None")
                case str():
                    parts.append(f' = "{param.default}"')
                case int() | float() | bool():
                    parts.append(f" = {param.default}")
                case _:
                    parts.append(f" = {repr(param.default)}")

        params.append("".join(parts))

    # Format return type
    return_annotation = ""
    if "return" in hints:
        return_type_str = _format_type(hints["return"])
        return_annotation = f" -> {return_type_str}"

    # Construct the declaration
    params_str = ", ".join(params)
    declaration = f"def {func_name}({params_str}){return_annotation}"

    return declaration


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
