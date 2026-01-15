"""Utility functions for statek package."""

import inspect
from typing import Callable, get_type_hints, get_origin, get_args, Union


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
