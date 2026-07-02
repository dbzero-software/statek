# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Iterable, Optional, Tuple, Dict, Union
import asyncio
import functools
import importlib
import inspect
import types as _types_module
from copy import copy
from datetime import date, datetime, time as _time
import nest_asyncio
import dbzero as db0
from .chat_style import ChatStyle
from .dbzero_restricted import as_unrestricted
from .future import get_any_future, get_all_future, FutureResult
from .docstring import parse_tool_docstring, format_docstring
from .utils import find_locals, get_current_agent_name, get_current_job, _statek_ctx_scope


_TOOL_REGISTRY: list[Callable] = []


def inject_context(func, __local_context):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "_local_context" in kwargs:
            raise RuntimeError("_local_context is already set")

        # defensive copy per invocation
        local_context = copy(__local_context)
        kwargs["_local_context"] = local_context

        statek_ctx = local_context.get("_STATEK_CTX")
        if isinstance(statek_ctx, dict):
            with _statek_ctx_scope(statek_ctx):
                return func(*args, **kwargs)
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
    """Customizes how docstring tools (brief, docstr) display a function.

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


def _validate_callable_accepts_kwargs(
    func: Callable,
    decorator_name: str
) -> inspect.Signature:
    """Validate that a decorated callable accepts framework keyword arguments."""
    sig = inspect.signature(func)
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if not has_var_keyword:
        raise TypeError(
            f"Function '{func.__name__}' must accept **kwargs to be used as "
            f"a {decorator_name}. "
            f"Current signature: {sig}"
        )
    return sig


def _validate_subtask_signature(func: Callable) -> None:
    """Validate that a subtask callable uses framework-managed parameters correctly."""
    sig = _validate_callable_accepts_kwargs(func, "subtask")
    if "id" in sig.parameters:
        raise TypeError(
            f"Function '{func.__name__}' cannot define parameter 'id'; "
            "subtask id is reserved for framework-managed kwargs."
        )


def _prepare_decorated_call(func: Callable, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Apply shared tool/subtask argument conversions."""
    args, kwargs = _convert_enum_args(func, args, kwargs)
    args, kwargs = _bind_by_name(func, args, kwargs)
    return args, kwargs


def _run_coroutine_result(result: Any) -> Any:
    """Run a coroutine result through the current event loop."""
    nest_asyncio.apply()
    return asyncio.get_running_loop().run_until_complete(result)


def _register_wrapper(wrapper: Callable, system: bool, target, hidden: bool = False) -> Callable:
    """Register an LLM-facing callable and annotate its scope metadata."""
    wrapper.tool_system = system
    wrapper.tool_target = target
    wrapper.tool_hidden = hidden
    _TOOL_REGISTRY.append(wrapper)
    return wrapper


def _bind_error_handler_result(result: Any, handler: Optional[Callable]) -> None:
    """Attach a tool error handler to either a future or the current job."""
    if handler is None:
        return
    job = get_current_job()
    if job is None:
        return
    if isinstance(result, FutureResult):
        result.bind_error_handler(handler, job)
    else:
        job.add_error_handler(handler, result)


# pylint: disable=redefined-outer-name
def tool(f=None, *, system: bool = False, target=None, error_handler=None,
         hidden: bool = False):
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
        hidden: When True, the tool remains callable internally but is excluded
            from LLM-facing tool discovery, prompt tool placeholders, and formal
            provider tool payloads.
    """

    def _decorate(func):
        if error_handler is not None and not is_valid_error_handler(error_handler):
            raise TypeError(
                "error_handler passed to @tool must be decorated with @error_handler "
                "and satisfy the error-handler protocol: name starting with '_', "
                "accepting (context, error=None)."
            )

        _validate_callable_accepts_kwargs(func, "tool")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with as_unrestricted():
                args, kwargs = _prepare_decorated_call(func, args, kwargs)
                if inspect.iscoroutinefunction(func):
                    result = _run_coroutine_result(func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)
                _bind_error_handler_result(result, error_handler)
                return result

        return _register_wrapper(wrapper, system, target, hidden)

    if f is None:
        # Called as @tool() or @tool(system=True)
        return _decorate
    # Called as @tool (without parentheses)
    return _decorate(f)


def _subtask_signature(func: Callable, return_type: type) -> inspect.Signature:
    """Return the LLM-facing signature for a subtask callable."""
    sig = inspect.signature(func)
    params = []
    inserted_id = False
    force_keyword_only = False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            if not inserted_id:
                params.append(_subtask_id_param(keyword_only=force_keyword_only))
                inserted_id = True
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY and not inserted_id:
            params.append(_subtask_id_param(keyword_only=True))
            inserted_id = True
        params.append(param)
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            force_keyword_only = True

    if not inserted_id:
        params.append(_subtask_id_param(keyword_only=force_keyword_only))

    return sig.replace(parameters=params, return_annotation=return_type)


def _subtask_id_param(keyword_only: bool) -> inspect.Parameter:
    """Return the synthetic framework-managed subtask id parameter."""
    kind = (
        inspect.Parameter.KEYWORD_ONLY
        if keyword_only
        else inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    return inspect.Parameter("id", kind, default=None, annotation=str)


def _wrap_subtask_result(result: Any, task_id: Optional[Any]) -> Any:
    """Convert a subtask function result to a SubTaskHandler."""
    task_module = importlib.import_module("statek.task")
    job_module = importlib.import_module("statek.executors.job")
    SubTaskHandler = task_module.SubTaskHandler  # pylint: disable=invalid-name
    create_sub_task = task_module.create_sub_task
    Job = job_module.Job  # pylint: disable=invalid-name

    if isinstance(result, SubTaskHandler):
        if result.id is None and task_id is not None:
            result.id = task_id
        if result.job.py_env.local_state.get("sub_task_handler") is not result:
            result.job.add_locals(sub_task_handler=result)
        return result
    if isinstance(result, Job):
        return create_sub_task(result, id=task_id)
    raise TypeError(
        "@subtask functions must return Job or SubTaskHandler, "
        f"got {type(result).__name__}"
    )


def subtask(f=None, *, system: bool = False, target=None):  # pylint: disable=W0621
    """Mark a function as a subtask factory for LLM agents.

    Decorated functions must accept ``**kwargs`` and return either a ``Job``
    or a ``SubTaskHandler``. The wrapper exposes an optional framework-managed
    ``id`` keyword to LLM callers. If the function returns a ``Job``, that id
    is used when creating the handler. If the function returns an existing
    handler, the handler is passed through, missing handler ids adopt the
    framework id, existing handler ids are preserved, and child job local state
    is repaired with ``sub_task_handler``.
    """

    def _decorate(func):
        SubTaskHandler = importlib.import_module("statek.task").SubTaskHandler

        _validate_subtask_signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            task_id = kwargs.pop("id", None)
            args, kwargs = _prepare_decorated_call(func, args, kwargs)
            if inspect.iscoroutinefunction(func):
                result = _run_coroutine_result(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            return _wrap_subtask_result(result, task_id)

        annotations = dict(getattr(func, "__annotations__", {}))
        annotations["id"] = str
        annotations["return"] = SubTaskHandler
        wrapper.__annotations__ = annotations
        wrapper.__signature__ = _subtask_signature(func, SubTaskHandler)
        wrapper.__is_subtask__ = True
        return _register_wrapper(wrapper, system, target)

    if f is None:
        return _decorate
    return _decorate(f)


@tool(system=True)
def find_sub_task_handler(id: Optional[Any] = None, **kwargs) -> Any:  # pylint: disable=redefined-builtin,unused-argument
    """Return a subtask handler from the current job notification history.

    Args:
        id: Optional subtask id to match. When omitted, returns the most recent
            pending or logged subtask handler.

    Returns:
        The matching subtask handler, or None when no matching handler exists.
    """
    job = get_current_job()
    if job is None:
        return None
    return job.find_sub_task_handler(id=id)


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
    if isinstance(target, (list, tuple, set, frozenset)):
        return chat_style in target
    return target == chat_style


def _is_hidden_tool(tool_func: Callable) -> bool:
    """Return whether a tool is hidden from LLM-facing discovery."""
    return bool(getattr(tool_func, "tool_hidden", False))


def find_tools(scope: Optional[str] = None,
               chat_style=None,
               include_hidden: bool = False) -> Iterable[Callable]:
    """Returns registered tools, optionally filtered by scope and chat style.

    Args:
        scope: Optional scope filter.
            ``"SYSTEM"`` returns only tools decorated with ``system=True``.
            ``"APPLICATION"`` returns only non-system tools.
            ``None`` returns all registered tools.
        chat_style: Optional target chat style.  When provided, only tools
            whose ``tool_target`` is ``None`` (universal) or includes
            *chat_style* are returned.
        include_hidden: When True, include tools decorated with
            ``@tool(hidden=True)``. Hidden tools are excluded by default because
            this function is normally used for LLM-facing discovery.

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
    if not include_hidden:
        result = [t for t in result if not _is_hidden_tool(t)]
    # Deduplicate by name, keeping the first occurrence
    seen: set[str] = set()
    deduped = []
    for t in result:
        if t.__name__ not in seen:
            seen.add(t.__name__)
            deduped.append(t)
    return deduped


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
    result = [t for t in result if not _is_hidden_tool(t)]
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
def docstr(what: type | Callable | Any, method_name: str = None, **kwargs):  # pylint: disable=unused-argument
    """Prints the docstring associated with a tool, class, object instance or method.

    Args:
        what: A function, type, class or object instance to get documentation for.
        method_name: Optional method name if what is a class/type.

    Returns:
        None. Prints the documentation directly to console.

    Examples:
        docstr(add)  # Get documentation for the 'add' function
        docstr(user)  # Get documentation for an object's class
        docstr(User, "send_message")  # Get documentation for User.send_message method
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


@tool(system=True)
def panic(**kwargs):  # pylint: disable=unused-argument
    """Escalate the current job to a higher-capability model.

    Use this when the current task is too difficult for the current model.
    This raises the job difficulty by one level and prints the new difficulty.

    Returns:
        None. Updates the current job and prints the current difficulty.
    """
    job = get_current_job()
    if job is None:
        raise RuntimeError("panic() requires a current Statek job context")
    job.panic()
    print(
        f"# Difficulty increased to {job.get_current_difficulty()}. "
        "Continue with the harder task."
    )


@tool(system=True, target={ChatStyle.DIRECT})  # pylint: disable=no-member
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
