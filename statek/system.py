from typing import Any, Callable, Tuple, Dict
import asyncio
import functools
import inspect
from functools import wraps
from copy import copy
import nest_asyncio
import dbzero as db0
from .future import get_any_future, get_all_future
from .docstring import parse_docstring, format_docstring


def inject_context(func, __local_context):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if "_local_context" in kwargs:
            raise RuntimeError("_local_context is already set")

        # defensive copy per invocation
        kwargs["_local_context"] = copy(__local_context)

        return func(*args, **kwargs)
    return wrapped


_SKIP_KINDS = {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
_POSITIONAL_KINDS = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}


def _convert_enum_args(f, args, kwargs):
    """Convert string arguments to db0 enum values where the parameter is typed as a db0 enum."""
    try:
        hints = f.__annotations__
        sig = inspect.signature(f)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except Exception:  # pylint: disable=broad-except
        return args, kwargs

    if not hints:
        return args, kwargs

    converted = dict(bound.arguments)
    changed = False

    for name, value in bound.arguments.items():
        param = sig.parameters[name]
        annotation = hints.get(name)
        if name.startswith("_") or param.kind in _SKIP_KINDS or annotation is None:
            continue
        if isinstance(value, str) and db0.is_enum(annotation):  # pylint: disable=no-member
            converted[name] = annotation[value]
            changed = True

    if not changed:
        return args, kwargs

    # Rebuild args and kwargs from converted arguments
    new_args = []
    new_kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in _POSITIONAL_KINDS and name in converted:
            new_args.append(converted[name])
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            new_args.extend(converted.get(name, ()))
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and name in converted:
            new_kwargs[name] = converted[name]
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            new_kwargs.update(converted.get(name, {}))

    return tuple(new_args), new_kwargs


def tool(f):
    """Marks a function as a tool for LLM agent."""

    # Check if the function signature includes **kwargs
    sig = inspect.signature(f)
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )

    if not has_var_keyword:
        raise TypeError(
            f"Function '{f.__name__}' must accept **kwargs to be used as a tool. "
            f"Current signature: {sig}"
        )

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args, kwargs = _convert_enum_args(f, args, kwargs)

        # update globals with local context
        result = None
        if inspect.iscoroutinefunction(f):
            # This library patches asyncio to allow nested event loops
            nest_asyncio.apply()
            # If f is async, run it using the event loop
            result = asyncio.get_running_loop().run_until_complete(f(*args, **kwargs))
        else:
            # If f is sync, just call it
            result = f(*args, **kwargs)
        return result
    return wrapper


# pylint: disable=redefined-builtin
def create_tool(tool_name: str, callable: Callable, docstring: str,
                context: Dict, *args, **kwargs) -> Callable:
    """Creates a zero-argument tool function from a callable with bound
    arguments and injects it into context.

    Args:
        tool_name: The name to assign to the created tool function.
        callable: The callable to wrap into a tool.
        docstring: The docstring to assign to the tool.
        context: The context to put the created tool into.
        *args: Positional arguments to bind to the callable.
        **kwargs: Keyword arguments to bind to the callable.

    Returns:
        A zero-argument callable with the specified name and docstring.
    """
    if tool_name in context:
        raise ValueError(f"tool {tool_name} already exists within the context")

    # Capture the bound kwargs
    bound_kwargs = kwargs

    def tool_func(**_tool_kwargs):
        # Call the function with bound arguments
        return callable(*args, **bound_kwargs)

    # Set the function name and docstring
    tool_func.__name__ = tool_name
    tool_func.__doc__ = docstring

    # Apply the @tool decorator
    new_tool = tool(tool_func)
    context[tool_name] = new_tool
    return new_tool


@tool
def docs(tool_or_class: type | Callable, method_name: str = None, **kwargs):  # pylint: disable=unused-argument
    """Prints the docstring associated with either a tool, class or its member name.

    Args:
        tool_or_class: A function, tool, or class to get documentation for.
        method_name: Optional method name if tool_or_class is a class. If provided,
                    returns documentation for the specific method of the class.

    Returns:
        None. Prints the documentation directly to console.

    Examples:
        docs(add)  # Get documentation for the 'add' function
        docs(User, "send_message")  # Get documentation for User.send_message method
    """
    # If method_name is provided, get the method from the class
    if method_name is not None:
        if not isinstance(tool_or_class, type):
            print(f"Error: {tool_or_class} is not a class")
            return

        # Get the method from the class
        if not hasattr(tool_or_class, method_name):
            print(f"Error: {tool_or_class.__name__} has no method '{method_name}'")
            return

        target = getattr(tool_or_class, method_name)
    else:
        target = tool_or_class

    # Use format_docstring for all tools (including temporal)
    parsed = parse_docstring(target)
    formatted = format_docstring(parsed, brief=False, py_syntax=True)
    print(formatted)


@tool
def get_any(*args: Any, **kwargs) -> Any:  # pylint: disable=unused-argument
    """Waits until evaluation of given values completes and returns the first available result.
    
    Args:
        *args: Variable number of values to evaluate.
    
    Returns:
        The first value that becomes available.
    
    Examples:
        result = get_any(value1, value2, value3)
    """
    return get_any_future(*args)


@tool
def get_all(*args: Any, **kwargs) -> Tuple[Any]:  # pylint: disable=unused-argument
    """Waits until evaluation of all given values completes and combines the results.
    
    Args:
        *args: Variable number of values to evaluate.
    
    Returns:
        A tuple containing all the evaluated values, in order.
    
    Examples:
        results = get_all(value1, value2)
    """
    return get_all_future(*args)
