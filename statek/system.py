from typing import Any, Callable, Iterable, Optional, Tuple, Dict
import asyncio
import functools
import inspect
from functools import wraps
from copy import copy
import nest_asyncio
import dbzero as db0
from .future import get_any_future, get_all_future
from .docstring import parse_docstring, format_docstring
from .utils import find_locals


_TOOL_REGISTRY: list[Callable] = []


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


def _rebuild_args(sig, converted):
    """Rebuild args and kwargs tuples from a converted bound-arguments dict."""
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


def _bind_by_name(f, args, kwargs):
    """Bind string arguments to local context variables on type mismatch.

    When a tool receives a string argument but the parameter expects a non-string
    type, looks up the string value as a variable name via find_locals and
    substitutes with the variable's value if found.
    """

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
        if (isinstance(value, str)
                and isinstance(annotation, type)
                and not issubclass(str, annotation)
                and not db0.is_enum(annotation)):  # pylint: disable=no-member
            matches = list(find_locals(var_name=value))
            if matches:
                converted[name] = matches[0]
                changed = True

    if not changed:
        return args, kwargs

    return _rebuild_args(sig, converted)


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

    return _rebuild_args(sig, converted)


def tool(f=None, *, system: bool = False):
    """Marks a function as a tool for LLM agent.

    Can be used as ``@tool`` or ``@tool(system=True)``.

    Args:
        system: When True, the tool is classified as a system-level tool
            (e.g. docs, brief) rather than an application-level tool.
    """

    def _decorate(func):
        # Check if the function signature includes **kwargs
        sig = inspect.signature(func)
        has_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        if not has_var_keyword:
            raise TypeError(
                f"Function '{func.__name__}' must accept **kwargs to be used as a tool. "
                f"Current signature: {sig}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _convert_enum_args(func, args, kwargs)
            args, kwargs = _bind_by_name(func, args, kwargs)

            # update globals with local context
            result = None
            if inspect.iscoroutinefunction(func):
                # This library patches asyncio to allow nested event loops
                nest_asyncio.apply()
                # If func is async, run it using the event loop
                result = asyncio.get_running_loop().run_until_complete(func(*args, **kwargs))
            else:
                # If func is sync, just call it
                result = func(*args, **kwargs)
            return result

        wrapper.tool_system = system
        _TOOL_REGISTRY.append(wrapper)
        return wrapper

    if f is None:
        # Called as @tool() or @tool(system=True)
        return _decorate
    # Called as @tool (without parentheses)
    return _decorate(f)


def find_tools(scope: Optional[str] = None) -> Iterable[Callable]:
    """Returns registered tools, optionally filtered by scope.

    Args:
        scope: Optional scope filter.
            ``"SYSTEM"`` returns only tools decorated with ``system=True``.
            ``"APPLICATION"`` returns only non-system tools.
            ``None`` returns all registered tools.

    Returns:
        An iterable of tool callables matching the requested scope.
    """
    if scope == "SYSTEM":
        return [t for t in _TOOL_REGISTRY if t.tool_system]
    if scope == "APPLICATION":
        return [t for t in _TOOL_REGISTRY if not t.tool_system]
    return list(_TOOL_REGISTRY)


def select_tools(tools: Iterable[Callable], scope: str) -> Iterable[Callable]:
    """Select tools from an iterable by the requested scope.

    Filters the provided tools based on their ``tool_system`` attribute (set by
    the ``@tool`` decorator).  Callables that do not carry the attribute are
    treated as application-level tools (i.e. they are *excluded* from
    ``"SYSTEM"`` and *included* in ``"APPLICATION"`` and ``"ALL"``).

    Args:
        tools: The sequence of tools to select from.
        scope: ``"SYSTEM"`` for system tools only,
               ``"APPLICATION"`` for non-system tools only,
               ``"ALL"`` or ``None`` to return all tools unchanged.

    Returns:
        A list of callables from *tools* that match the requested scope.
    """
    if scope == "SYSTEM":
        return [t for t in tools if getattr(t, "tool_system", False)]
    if scope == "APPLICATION":
        return [t for t in tools if not getattr(t, "tool_system", False)]
    return list(tools)


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


@tool(system=True)
def docs(what: type | Callable | Any, method_name: str = None, **kwargs):  # pylint: disable=unused-argument
    """Prints the docstring associated with a tool, class, object instance or method.

    Args:
        what: A function, type, class or object instance to get documentation for.
        method_name: Optional method name if what is a class/type.

    Returns:
        None. Prints the documentation directly to console.

    Examples:
        docs(add)  # Get documentation for the 'add' function
        docs(user)  # Get documentation for an object's class
        docs(User, "send_message")  # Get documentation for User.send_message method
    """
    # Handle object instances - get their class
    target_type = what
    if not isinstance(what, type) and not callable(what):
        target_type = type(what)

    # If method_name is provided, get the method from the class/type
    if method_name is not None:
        if not isinstance(target_type, type):
            print(f"Error: {target_type} is not a class")
            return

        # Get the method from the class
        if not hasattr(target_type, method_name):
            print(f"Error: {target_type.__name__} has no method '{method_name}'")
            return

        target = getattr(target_type, method_name)
    else:
        target = target_type

    # Use format_docstring for all tools (including temporal)
    parsed = parse_docstring(target)
    formatted = format_docstring(parsed, brief=False, py_syntax=True)
    print(formatted)


@tool(system=True)
def brief(what: type | Callable | Any, method_name: str = None, **kwargs):  # pylint: disable=unused-argument
    """Prints brief documentation for a tool, class, object instance or method.

    Args:
        what: A function, type, class or object instance to get documentation for.
        method_name: Optional method name if what is a class/type.

    Returns:
        None. Prints the brief documentation directly to console.

    Examples:
        brief(add)  # Get brief documentation for the 'add' function
        brief(user)  # Get brief documentation for an object's class
        brief(User, "send_message")  # Get brief documentation for User.send_message
    """
    # Handle object instances - get their class
    target_type = what
    if not isinstance(what, type) and not callable(what):
        target_type = type(what)

    # If method_name is provided, get the method from the class/type
    if method_name is not None:
        if not isinstance(target_type, type):
            print(f"Error: {target_type} is not a class")
            return

        if not hasattr(target_type, method_name):
            print(f"Error: {target_type.__name__} has no method '{method_name}'")
            return

        target = getattr(target_type, method_name)
    else:
        target = target_type

    parsed = parse_docstring(target)
    formatted = format_docstring(parsed, brief=True, py_syntax=False)
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
