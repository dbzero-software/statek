from typing import Any, Callable, Iterable, Optional, Tuple, Dict, Union
import asyncio
import functools
import inspect
import types as _types_module
from copy import copy
from datetime import date, datetime, time as _time
from functools import wraps
import nest_asyncio
import dbzero as db0
from .future import get_any_future, get_all_future, FutureResult
from .docstring import parse_tool_docstring, format_docstring
from .utils import find_locals, get_current_agent_name, get_current_job
from .chat_style import ChatStyle


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

# Python 3.10+ introduces types.UnionType for X | Y syntax; guard for older versions.
_UnionType = getattr(_types_module, 'UnionType', None)


def _expects_non_string_type(annotation) -> bool:
    """Return True if annotation implies a non-string value is expected.

    Handles plain types, Python 3.10+ union syntax (X | Y | Z) and typing.Union.
    """
    if annotation is None:
        return False
    if isinstance(annotation, type):
        return not issubclass(str, annotation)
    # Python 3.10+: X | Y | Z  →  types.UnionType
    if _UnionType is not None and isinstance(annotation, _UnionType):
        return any(isinstance(a, type) and not issubclass(str, a) for a in annotation.__args__)
    # typing.Union[X, Y, Z]
    if getattr(annotation, '__origin__', None) is Union:
        return any(isinstance(a, type) and not issubclass(str, a) for a in annotation.__args__)
    return False


_ISO_PARSEABLE_TYPES = (datetime, date, _time)


def _try_iso_parse(value: str, annotation):
    """Try parsing *value* as an ISO date/time for the given annotation.

    Iterates candidate types extracted from the annotation (plain type or union)
    in declaration order and returns the first successful parse, or ``None`` if
    no candidate type succeeds.
    """
    if isinstance(annotation, type):
        candidates = [annotation]
    elif _UnionType is not None and isinstance(annotation, _UnionType):
        candidates = [a for a in annotation.__args__ if isinstance(a, type)]
    elif getattr(annotation, '__origin__', None) is Union:
        candidates = [a for a in annotation.__args__ if isinstance(a, type)]
    else:
        return None

    for typ in candidates:
        if typ in _ISO_PARSEABLE_TYPES:
            try:
                return typ.fromisoformat(value)
            except (ValueError, TypeError):
                continue
    return None


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
                and _expects_non_string_type(annotation)
                and not db0.is_enum(annotation)):  # pylint: disable=no-member
            parsed = _try_iso_parse(value, annotation)
            if parsed is not None:
                converted[name] = parsed
                changed = True
            else:
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


def docs_style(f=None, *, brief_types: bool = False):
    """Customizes how docstring tools (brief, docs) display a function.

    Args:
        brief_types: When True, the ``brief`` tool includes parameter type
            annotations in the function signature.  Default is False.
    """

    def _decorate(func):
        func._docs_style = {"brief_types": brief_types}  # pylint: disable=protected-access
        return func

    if f is None:
        return _decorate
    return _decorate(f)


def tool(f=None, *, system: bool = False, target=None, error_handler=None): # pylint: disable=W0621
    """Marks a function as a tool for LLM agent.

    Can be used as ``@tool`` or ``@tool(system=True)``.

    Args:
        system: When True, the tool is classified as a system-level tool
            (e.g. docs, brief) rather than an application-level tool.
        target: Optional ChatStyle (or list of ChatStyles) this tool is
            intended for.  When set, ``find_tools`` and ``select_tools``
            can filter by chat style so that only matching tools are
            returned.  Tools without a target are considered universal.
        error_handler: Optional error-handling callable to associate with this
            tool.  Must be decorated with :func:`error_handler` and satisfy the
            error-handler protocol (name starting with ``_``, accepting
            ``(context, error=None)``).  When provided, the handler is
            registered on the current job (via ``job.add_error_handler``) with
            the tool's return value as context, just before the value is
            returned to the caller.
    """

    def _decorate(func):
        if error_handler is not None and not is_valid_error_handler(error_handler):
            raise TypeError(
                "error_handler passed to @tool must be decorated with @error_handler "
                "and satisfy the error-handler protocol: name starting with '_', "
                "accepting (context, error=None)."
            )

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

            if error_handler is not None:
                job = get_current_job()
                if job is not None:
                    if isinstance(result, FutureResult):
                        result.bind_error_handler(error_handler, job)
                    else:
                        job.add_error_handler(error_handler, result)

            return result

        wrapper.tool_system = system
        wrapper.tool_target = target
        _TOOL_REGISTRY.append(wrapper)
        return wrapper

    if f is None:
        # Called as @tool() or @tool(system=True)
        return _decorate
    # Called as @tool (without parentheses)
    return _decorate(f)


def error_handler(func: Callable) -> Callable:
    """Marks a callable as an error handler.

    An error handler is called when a job must be terminated with an error,
    allowing the application to notify users awaiting a response.

    The decorated function must accept ``(context, error=None)`` and its name
    must start with a single underscore ``_``.  Any exception raised by the
    handler at call time is suppressed automatically.

    Args:
        func: The callable to mark as an error handler.

    Returns:
        The wrapped callable with ``is_error_handler = True`` set.
    """
    @functools.wraps(func)
    def wrapper(context, error=None):
        try:
            func(context, error=error)
        except Exception:  # pylint: disable=broad-except
            pass

    wrapper.is_error_handler = True
    return wrapper


def is_valid_error_handler(funct: Callable) -> bool:
    """Return True if *funct* satisfies the error-handler protocol.

    A valid error handler must:

    * Be decorated with :func:`error_handler` (``is_error_handler`` attribute
      is ``True``).
    * Have a name starting with a single underscore ``_``.
    * Accept exactly two parameters where the second (``error``) has a default
      value (i.e. is optional).

    Args:
        funct: The callable to inspect.

    Returns:
        ``True`` if all criteria are met, ``False`` otherwise.
    """
    if not (callable(funct)
            and getattr(funct, 'is_error_handler', False)
            and funct.__name__.startswith('_')):
        return False
    try:
        sig = inspect.signature(funct)
        params = [
            p for p in sig.parameters.values()
            if p.kind not in {inspect.Parameter.VAR_KEYWORD,
                               inspect.Parameter.VAR_POSITIONAL}
        ]
        if len(params) != 2:
            return False
        if params[1].default is inspect.Parameter.empty:
            return False
    except (ValueError, TypeError):
        return False
    return True


def _matches_chat_style(tool_func: Callable, chat_style) -> bool:
    """Return True if *tool_func* is compatible with *chat_style*.

    A tool matches when its ``tool_target`` is ``None`` (universal) or
    when *chat_style* is contained in the target (single value or list).
    """
    target = getattr(tool_func, 'tool_target', None)
    if target is None:
        return True
    if isinstance(target, (list, tuple)):
        return chat_style in target
    return target == chat_style


def find_tools(scope: Optional[str] = None,
               chat_style=None) -> Iterable[Callable]:
    """Returns registered tools, optionally filtered by scope and chat style.

    Args:
        scope: Optional scope filter.
            ``"SYSTEM"`` returns only tools decorated with ``system=True``.
            ``"APPLICATION"`` returns only non-system tools.
            ``None`` returns all registered tools.
        chat_style: Optional target chat style.  When provided, only tools
            whose ``tool_target`` is ``None`` (universal) or includes
            *chat_style* are returned.

    Returns:
        An iterable of tool callables matching the requested scope.
    """
    if scope == "SYSTEM":
        result = [t for t in _TOOL_REGISTRY if t.tool_system]
    elif scope == "APPLICATION":
        result = [t for t in _TOOL_REGISTRY if not t.tool_system]
    else:
        result = list(_TOOL_REGISTRY)
    if chat_style is not None:
        result = [t for t in result if _matches_chat_style(t, chat_style)]
    return result


def select_tools(tools: Iterable[Callable], scope: str,
                  chat_style=None) -> Iterable[Callable]:
    """Select tools from an iterable by the requested scope and chat style.

    Filters the provided tools based on their ``tool_system`` attribute (set by
    the ``@tool`` decorator).  Callables that do not carry the attribute are
    treated as application-level tools (i.e. they are *excluded* from
    ``"SYSTEM"`` and *included* in ``"APPLICATION"`` and ``"ALL"``).

    Args:
        tools: The sequence of tools to select from.
        scope: ``"SYSTEM"`` for system tools only,
               ``"APPLICATION"`` for non-system tools only,
               ``"ALL"`` or ``None`` to return all tools unchanged.
        chat_style: Optional target chat style.  When provided, only tools
            whose ``tool_target`` is ``None`` (universal) or includes
            *chat_style* are returned.

    Returns:
        A list of callables from *tools* that match the requested scope.
    """
    if scope == "SYSTEM":
        result = [t for t in tools if getattr(t, "tool_system", False)]
    elif scope == "APPLICATION":
        result = [t for t in tools if not getattr(t, "tool_system", False)]
    else:
        result = list(tools)
    if chat_style is not None:
        result = [t for t in result if _matches_chat_style(t, chat_style)]
    return result


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

    def tool_func(*_tool_args, **_tool_kwargs):
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
    # A string passed to docs is almost certainly an unresolved name
    if isinstance(what, str):
        print(f"NameError: name '{what}' is not defined")
        return

    try:
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
        parsed = parse_tool_docstring(target)
        formatted = format_docstring(parsed, brief=False, py_syntax=True,
                                     agent=get_current_agent_name())
        print(formatted)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"{type(e).__name__}: {e}")


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
    # A string passed to brief is almost certainly an unresolved name
    if isinstance(what, str):
        print(f"NameError: name '{what}' is not defined")
        return

    try:
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

        parsed = parse_tool_docstring(target)
        formatted = format_docstring(parsed, brief=True, py_syntax=False,
                                     agent=get_current_agent_name())
        print(formatted)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"{type(e).__name__}: {e}")


@tool(system=True, target=ChatStyle.DIRECT)  # pylint: disable=no-member
def python_cli(code: str, **kwargs):  # pylint: disable=unused-argument
    """Execute Python code within the current job context.

    This is the only way to run Python code in the DIRECT chat style — inline
    code blocks in LLM responses are ignored.  The submitted code is executed
    in the job's local context so it can read and modify local variables,
    call temporal functions, and invoke other tools exactly as code submitted
    via chat or markdown blocks would.

    Executions are stateful: each invocation shares and mutates the job's
    local context, so variables assigned in one call are visible in subsequent
    calls within the same job.

    Args:
        code (str): The Python source code to execute.

    Examples:
        python_cli(code="result = find_on_calls(start_date, end_date)")
        python_cli(code="send_message('Hello!')")
    """


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
