import ast
import sys
import inspect
import traceback
import asyncio
import builtins
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union
from contextlib import contextmanager
import dbzero as db0

from statek.exceptions import FutureError, LLM_HarnessError
from statek.future import FutureResult


from statek.exceptions import FutureError
from statek.future import FutureResult
from statek.executors.job import Job, JobStatus, JobDef, parse_warmup_code
from statek.executors.chat_log_item import WarmupLogItem
from statek.statek_push_queue import StatekPushQueue
from statek.llm_api import LLM_API
from statek.llm_harness import get_llm_harness
from statek.settings import get_statek_settings, get_provider_settings, get_statek_logger, statek_log, ChatStyle
from statek.system import inject_context
from statek.utils import (CodeBlock, CallSpec, format_default_llm_repr,
                         get_current_job, get_current_agent, parse_md_dialog,
                         strip_markup)

STATEK_LOGGER = get_statek_logger()


class _MirrorDict(dict):
    """Dict subclass that mirrors writes to a target dict.

    Used as the local namespace in exec() so that variable assignments inside
    compound statements (if/for/while) are immediately visible in the global
    namespace — which is the only namespace that generator expressions and
    comprehensions can see.
    """

    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = target

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self._target is not None:
            self._target[key] = value


def _wrap_param (param):
    if isinstance(param, FutureResult):
        value = param.value
        return value
    return param

def check_for_future_typehint(param, anno):
    """Check if the type hint indicates FutureResult."""
    # Check by name
    if getattr(anno, "__name__", None) == 'FutureResult':
        return True
    # Check if it's the FutureResult class itself
    if anno is FutureResult:
        return True
    # Check if annotation is a string representation
    if str(anno) == 'FutureResult':
        return True
    return False

def _smart_call(func, *args, **kwargs):
    """Wraps arguments unless the target function expects FutureResult."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            param = sig.parameters[name]
            anno = param.annotation
            # Check if type hint matches 'FutureResult' by name or type
            
            # Handle *args (VAR_POSITIONAL)
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                # val is a tuple, wrap each element unless annotated as FutureResult
                if not check_for_future_typehint(param, anno):
                    bound.arguments[name] = tuple(_wrap_param(v) for v in val)
            # Handle **kwargs (VAR_KEYWORD)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # val is a dict, wrap each value unless annotated as FutureResult
                if not check_for_future_typehint(param, anno):
                    bound.arguments[name] = {k: _wrap_param(v) for k, v in val.items()}
            # Handle regular parameters
            elif not check_for_future_typehint(param, anno):
                bound.arguments[name] = _wrap_param(val)
        return func(*bound.args, **bound.kwargs)
    except (ValueError, TypeError):
        # Fallback: wrap everything if signature inspection fails
        return func(*[_wrap_param(a) for a in args], **{k: _wrap_param(v) for k, v in kwargs.items()})
class _ResilientTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        # Visit function (e.g., to wrap the function name itself)
        new_func = self.visit(node.func)

        # Process args: Skip wrapping Names directly so _smart_call handles them; visit others
        new_args = [
            arg if isinstance(arg, ast.Name) else self.visit(arg)
            for arg in node.args
        ]
        new_keywords = [
            ast.keyword(arg=k.arg, value=(k.value if isinstance(k.value, ast.Name) else self.visit(k.value)))
            for k in node.keywords
        ]

        new_node = ast.Call(
            func=ast.Name(id='_smart_call', ctx=ast.Load()),
            args=[new_func] + new_args,
            keywords=new_keywords
        )
        return ast.copy_location(new_node, node)

    def visit_Name(self, node):
        # Wrap variables used in expressions (outside of function arguments handled above)
        if isinstance(node.ctx, ast.Load) and node.id not in {'_wrap_param', '_smart_call', '_fmt_fstring_arg'}:
            new_node = ast.Call(
                func=ast.Name(id='_wrap_param', ctx=ast.Load()),
                args=[node],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        return node

    def visit_FormattedValue(self, node):
        """Transform f-string value expressions to use format_default_llm_repr.

        Only applied when there is no explicit conversion (like !r, !s, !a)
        and no format spec (like :.2f), so explicit formatting is always respected.
        """
        self.generic_visit(node)
        if node.conversion == -1 and node.format_spec is None:
            wrapped = ast.Call(
                func=ast.Name(id='_fmt_fstring_arg', ctx=ast.Load()),
                args=[node.value],
                keywords=[]
            )
            ast.copy_location(wrapped, node)
            node.value = wrapped
        return node

"""Execute a single AST node with custom print function."""
def _fmt_print_arg(arg) -> str:
    """Format a single print() argument.

    Strings are emitted as-is (matching Python's built-in print behaviour).
    All other types use the LLM-friendly representation.
    """
    if isinstance(arg, str):
        return arg
    return format_default_llm_repr(arg)


def custom_print(job, *args, sep=' ', end='\n', **kwargs):
    """Custom print function that writes to job console."""
    output = sep.join(_fmt_print_arg(arg) for arg in args) + end
    job.console_append(output.rstrip('\n'))

def custom_exit(job, status=None):
    """Custom exit function that sets exit status."""
    job.py_env.exit_status = status


@contextmanager
def _setup_execution_context(job: Job, global_context: dict, local_context: dict, print_fn=None):
    """
    Context manager to setup and cleanup execution environment.

    Args:
        job: The Job instance
        global_context: The global execution context dictionary
        local_context: The local execution context dictionary
        print_fn: Optional custom print function; defaults to writing to job console

    Yields:
        tuple: (custom_print_fn, custom_exit_fn) for reference
    """
    # Merge agent's private context if available
    if job.job_def.agent is not None and job.job_def.agent.context is not None:
        global_context.update(job.job_def.agent.context)
    all_direct_tools = list(job.job_def.agent._tools)
    if job.job_def.agent._internal_tools:
        all_direct_tools.extend(job.job_def.agent._internal_tools)
    # Include system tools from the global registry (not stored on agents)
    from statek.system import find_tools  # pylint: disable=import-outside-toplevel
    agent_tool_names = {t.__name__ for t in all_direct_tools}
    for sys_tool in find_tools("SYSTEM"):
        if sys_tool.__name__ not in agent_tool_names:
            all_direct_tools.append(sys_tool)
    for tool in all_direct_tools:
        global_context[tool.__name__] = inject_context(tool, global_context)

    # Create custom functions
    custom_print_fn = print_fn if print_fn is not None else (lambda *args, **kwargs: custom_print(job, *args, **kwargs))
    custom_exit_fn = lambda status=None: custom_exit(job, status)
    
    # Save original built-ins to restore later
    original_print = builtins.print
    original_exit = builtins.exit
    
    # Monkey-patch builtins so they work even in pre-defined functions
    builtins.print = custom_print_fn
    builtins.exit = custom_exit_fn
    
    # Inject into local context (excluding print/exit since they're in builtins)
    local_context['_smart_call'] = _smart_call
    local_context['_wrap_param'] = _wrap_param
    local_context['_fmt_fstring_arg'] = _fmt_print_arg
    global_context['print'] = custom_print_fn
    global_context['exit'] = custom_exit_fn
    global_context['_smart_call'] = _smart_call
    global_context['_wrap_param'] = _wrap_param
    global_context['_fmt_fstring_arg'] = _fmt_print_arg

    # Inject _STATEK_CTX with job-level context (agent, job, etc.)
    statek_ctx = {}
    statek_ctx['job'] = job
    if job.job_def.agent is not None:
        statek_ctx['agent'] = job.job_def.agent
    local_context['_STATEK_CTX'] = statek_ctx
    global_context['_STATEK_CTX'] = statek_ctx

    try:
        yield custom_print_fn, custom_exit_fn
    finally:
        # Restore original built-ins
        builtins.print = original_print
        builtins.exit = original_exit

        # Remove helpers from context
        for key in ['print', 'exit', '_smart_call', '_wrap_param', '_fmt_fstring_arg', '_STATEK_CTX']:
            if key in local_context:
                del local_context[key]


def _is_empty_code(code_str: Optional[str]) -> bool:
    """Check if code string contains no executable statements.

    Returns True when code_str is None, empty, whitespace-only, or contains
    only comments and/or string literals (block comments).
    """
    if not code_str or not code_str.strip():
        return True
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return False
    # After parsing, only Expr nodes with Constant values (string literals / block comments)
    # remain — everything else is a real statement.
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return False
    return True


def _exec_code_body(code_str: str, job: Job, global_context: dict,
                    local_context: dict, output_fn: Callable,
                    error_fn: Callable, print_fn: Callable = None,
                    instr_num: Optional[int] = None):
    """Shared execution body used by exec_step and exec_cli_step.

    Parses *code_str*, transforms it, and executes each statement in order
    inside the provided contexts.  Output produced by expression evaluation
    is forwarded to *output_fn*; errors are forwarded to *error_fn*.

    Args:
        code_str: Python source code to execute.
        job: Job providing the execution context.
        global_context: mutable global namespace for exec/eval.
        local_context: mutable local namespace for exec/eval.
        output_fn: called with a single string when an expression produces a
            non-None result (REPL-style echo).
        error_fn: called with an error message string on exception.
        print_fn: optional custom print function passed to
            ``_setup_execution_context``; when *None* the default
            job-console print is used.
        instr_num: optional instruction index to resume from.
    """
    import types
    initial_local_functions = {
        key for key, value in local_context.items()
        if isinstance(value, types.FunctionType)
    }

    with _setup_execution_context(job, global_context, local_context, print_fn=print_fn):
        tree = ast.parse(code_str)
        _ResilientTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        start_instr = instr_num if instr_num is not None else 0

        for idx, node in enumerate(tree.body):
            if idx < start_instr:
                continue

            try:
                is_expression = isinstance(node, ast.Expr)

                wrapper = ast.Module(body=[node], type_ignores=[])
                code_obj = compile(wrapper, filename="<string>", mode="exec")

                global_context.update(local_context)
                sync_local = _MirrorDict(local_context, target=global_context)

                if is_expression:
                    result = eval(
                        compile(ast.Expression(body=node.value),
                                filename="<string>", mode="eval"),
                        global_context, sync_local)
                    if result is not None:
                        output_fn(format_default_llm_repr(result))
                else:
                    exec(code_obj, global_context, sync_local)

                local_context.update(sync_local)
                global_context.update(local_context)

                newly_defined_functions = {
                    name: func for name, func in local_context.items()
                    if isinstance(func, types.FunctionType)
                    and name not in initial_local_functions
                }
                for function_name in newly_defined_functions:
                    del local_context[function_name]

                if job.py_env.exit_status is not None:
                    break
            except FutureError as e:
                if e.instr_num is None:
                    e.instr_num = idx
                raise
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                error_fn(error_msg)
                raise


async def exec_step(code_str: str, job: Job, instr_num: Optional[int] = None,
                    local_context: Optional[dict] = None) -> bool:
    """
    Execute a single step of code within the job's Python environment.

    The function executes a single step defined as a Python code block within the job's context
    - i.e. globals, locals, console and exit_status. The execution results are reflected in the
    provided job's state.

    Args:
        code: Python code (or expression) to be executed
        job: the Job defining the execution context
        instr_num: optional instruction number to start from (for continuation)

    Returns:
        True if the exit was called (program finished), False otherwise

    Raises:
        FutureError: decorated with instr_num - execution needs to be suspended until
                     continuation criteria are satisfied

    Side effects:
        Console outputs (print results) are appended to _X_console
        local_state might be updated by the program
    """
    statek_log(f"Executing code step (instr_num={instr_num}):\n{code_str}", level='debug')
    if job.py_env.global_state is None:
        global_context = globals()
    else:
        global_context = {key: value for key, value in job.py_env.global_state.items()}
    if local_context is None:
        local_context = dict(job.py_env.local_state) if job.py_env.local_state else {}

    try:
        _exec_code_body(
            code_str, job, global_context, local_context,
            output_fn=lambda s: job.console_append(s),
            error_fn=lambda msg: job.console_append(msg, error_message=msg),
            instr_num=instr_num,
        )
    finally:
        job.py_env.local_state = local_context

    return job.py_env.exit_status is None


async def exec_cli_step(code_str: str, job: Job, console_append: Callable,
                        instr_num: Optional[int] = None,
                        local_context: Optional[dict] = None) -> bool:
    """Execute a code block submitted via a ``python_cli`` tool request.

    Behaves like :func:`exec_step` — same context sharing, temporal-function
    handling, and state mutation — but routes console output (print, expression
    results, errors) to the provided *console_append* callback instead of the
    job's own console.

    Args:
        code_str: Python code block to execute.
        job: the Job providing the execution context.
        console_append: callback receiving each console output line.
        instr_num: optional instruction index to resume from.

    Returns:
        ``True`` if ``exit()`` was called (program finished), ``False``
        otherwise.
    """
    statek_log(f"Executing CLI code step (instr_num={instr_num}):\n{code_str}", level='debug')
    if job.py_env.global_state is None:
        global_context = globals()
    else:
        global_context = {key: value for key, value in job.py_env.global_state.items()}
    if local_context is None:
        local_context = dict(job.py_env.local_state) if job.py_env.local_state else {}

    def cli_print(*args, sep=' ', end='\n', **kwargs):
        output = sep.join(_fmt_print_arg(arg) for arg in args) + end
        console_append(output.rstrip('\n'))

    try:
        _exec_code_body(
            code_str, job, global_context, local_context,
            output_fn=console_append,
            error_fn=console_append,
            print_fn=cli_print,
            instr_num=instr_num,
        )
    finally:
        job.py_env.local_state = local_context

    return job.py_env.exit_status is not None


async def exec_all_steps(code: Union[str, CodeBlock], job: Job,
                         console_append: Callable,
                         instr_num: Optional[Union[int, Tuple[int, int]]] = None,
                         local_context: Optional[dict] = None) -> bool:
    """Execute regular code and any ``python_cli`` tool calls from a code block.

    First executes the regular code (if present) via :func:`exec_step`, then
    iterates over CLI tool calls (see :meth:`CodeBlock.get_cli_tool_calls`)
    and executes each via :func:`exec_cli_step`.  Execution stops after the
    first ``exit()`` call.

    When *instr_num* is an ``int`` it is forwarded to :func:`exec_step` for
    regular-code continuation.  When it is a ``(cli_step, instr)`` tuple,
    regular code is skipped entirely and execution resumes at the indicated
    CLI step and instruction.

    If a :class:`FutureError` is raised from a CLI step, its ``instr_num``
    is rewritten to a ``(cli_step_index, instr_num)`` tuple so that the
    caller can resume at the correct position.

    Args:
        code: plain string or :class:`CodeBlock` to execute.
        job: the Job providing the execution context.
        console_append: callback ``(cli_tool_index, text)`` for CLI output.
        instr_num: optional continuation position.

    Returns:
        ``True`` if ``exit()`` was called, ``False`` otherwise.
    """
    if isinstance(code, str):
        code = CodeBlock(code=code)

    # Determine whether to run regular code and where to start CLI steps
    skip_regular = isinstance(instr_num, tuple)
    regular_instr_num = None if skip_regular else instr_num
    cli_start_idx = instr_num[0] if skip_regular else 0
    cli_instr_num = instr_num[1] if skip_regular else None

    # --- regular code ---
    if not skip_regular and not _is_empty_code(code.code):
        exited = not await exec_step(code.code, job, instr_num=regular_instr_num,
                                     local_context=local_context)
        if exited:
            return True

    # --- CLI tool calls ---
    cli_calls = code.get_cli_tool_calls()
    if not cli_calls:
        return job.py_env.exit_status is not None

    for cli_idx, call_spec in enumerate(cli_calls):
        if cli_idx < cli_start_idx:
            continue

        cli_code = call_spec.kwargs.get("code") if call_spec.kwargs else None
        if _is_empty_code(cli_code):
            cli_instr_num = None
            continue

        step_instr = cli_instr_num if cli_idx == cli_start_idx else None
        cli_instr_num = None  # only applies to the first resumed step

        try:
            exited = await exec_cli_step(
                cli_code, job,
                console_append=lambda s, _idx=cli_idx: console_append(_idx, s),
                instr_num=step_instr,
                local_context=local_context,
            )
        except FutureError as e:
            e.instr_num = (cli_idx, e.instr_num)
            raise

        if exited:
            return True

    return job.py_env.exit_status is not None


async def exec_tool(call_spec: CallSpec, job: Job,
                    local_context: Optional[dict] = None) -> str:
    """Execute a single non-temporal tool invocation within the job's local context.

    Uses _setup_execution_context to inject the same global_context / local_context
    as exec_step (full job state, all tools injected, agent context merged) so that
    tools can freely call other tools. Print output is captured into a private console
    rather than written to the job's console.

    Exceptions are caught, formatted as error messages, and returned — never re-raised.

    Args:
        call_spec: the function call specification
        job: the associated job for context

    Returns:
        The invocation result as console output (newline-joined lines).
        When the return value is not None its repr is included; on error the
        exception type and message are included.
    """
    STATEK_LOGGER.debug(
        "exec_tool: %s args=%r kwargs=%r",
        call_spec.func_name,
        list(call_spec.args) if call_spec.args else [],
        dict(call_spec.kwargs) if call_spec.kwargs else {},
    )

    private_console = []

    def _private_print(*args, sep=' ', end='\n', **kwargs):
        output = sep.join(_fmt_print_arg(arg) for arg in args) + end
        private_console.append(output.rstrip('\n'))

    # Build global and local contexts — mirrors exec_step
    if job.py_env.global_state is None:
        global_context = globals()
    else:
        global_context = {key: value for key, value in job.py_env.global_state.items()}
    if local_context is None:
        local_context = dict(job.py_env.local_state) if job.py_env.local_state else {}

    with _setup_execution_context(job, global_context, local_context, print_fn=_private_print):
        # Re-wrap the original tool with a combined context (global + local) so that
        # _bind_by_name / find_locals can resolve string arguments to actual objects
        # (e.g. docs(what='some_tool') → the actual callable, not the string).
        agent = job.job_def.agent
        original_tool = None
        if agent:
            for t in agent._tools:  # pylint: disable=protected-access
                if t.__name__ == call_spec.func_name:
                    original_tool = t
                    break
            # Also search system tools from the global registry
            if original_tool is None:
                from statek.system import find_tools  # pylint: disable=import-outside-toplevel
                for t in find_tools("SYSTEM"):
                    if t.__name__ == call_spec.func_name:
                        original_tool = t
                        break

        if original_tool is not None:
            combined_ctx = {**global_context, **local_context}
            func = inject_context(original_tool, combined_ctx)
        else:
            func = global_context.get(call_spec.func_name) or local_context.get(call_spec.func_name)

        if func is None:
            return f"NameError: tool '{call_spec.func_name}' not found"

        try:
            args = list(call_spec.args) if call_spec.args else []
            kwargs = dict(call_spec.kwargs) if call_spec.kwargs else {}
            if STATEK_LOGGER.isEnabledFor(10):  # logging.DEBUG
                call_repr = ", ".join(
                    [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
                )
                STATEK_LOGGER.debug("exec_tool calling: %s(%s)", call_spec.func_name, call_repr)
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                private_console.append(format_default_llm_repr(result))
        except Exception as e:  # pylint: disable=broad-exception-caught
            private_console.append(f"{type(e).__name__}: {e}")

    return "\n".join(private_console)


def _log_pending_console(job: Job):
    """Log console output accumulated since the last logged boundary.

    Determines from_pos based on job state:
    - After LLM turns: from last chat_log item's console_pos
    - After warmup: from the next element's console_pos (or current console length)
    - Initial state: from 0
    """
    if job.chat_log:
        last = job.chat_log[-1]
        if isinstance(last, WarmupLogItem):
            # If a following element exists its console_pos marks where the
            # warmup batch ended (it was already logged by _log_console_batch).
            # Otherwise the batch has not been logged yet (e.g. early exit
            # during warmup), so start from the WarmupLogItem's own position.
            last_idx = len(job.chat_log) - 1
            from_pos = (
                job.chat_log[last_idx + 1].console_pos
                if last_idx + 1 < len(job.chat_log)
                else last.console_pos
            )
        else:
            from_pos = last.console_pos
    else:
        from_pos = 0
    to_pos = len(job.py_env.console) if job.py_env.console else 0
    job._log_console_batch(from_pos, to_pos)  # pylint: disable=protected-access


async def handle_md_dialog(llm_resp: str, _local_context: Optional[dict] = None):
    """Dispatch LLM response text to the user via the DialogAgent's send_message.

    Retrieves the current agent from the execution context, verifies it is a
    DialogAgent, parses the response into (body, media) segments and forwards
    each segment to the agent's ``send_message`` callable.

    Args:
        llm_resp: The raw LLM response text to be dispatched.

    Raises:
        TypeError: If the current agent is not a DialogAgent.
    """
    from statek.agents.dialog_agent import DialogAgent  # pylint: disable=import-outside-toplevel

    agent = get_current_agent()
    if not isinstance(agent, DialogAgent):
        raise TypeError(
            f"MD_DIALOG/DIRECT chat style requires a DialogAgent, "
            f"got {type(agent).__name__}."
        )

    for body, media in parse_md_dialog(llm_resp):
        if body or media:
            kwargs = {"body": body}
            if media is not None:
                kwargs["media"] = media
            result = agent.send_message(**kwargs)
            if inspect.iscoroutinefunction(agent.send_message):
                await result


async def run_job_step(job: Job, provider: str = None) -> bool:
    """
    Execute a single step of the agentic pipeline.

    This function performs one iteration of the agent execution loop:
    1. Retrieves and executes pending code
    2. Formulates a prompt and sends it to the LLM API
    3. Processes the response and stores it in the result

    Args:
        job: The job object holding the job's state
        provider: The LLM API provider name (or None for default)

    Returns:
        True if the job has been completed (exit called), False otherwise

    example:
        see: experiments/ai/run_job_step.ipynb
    """

    # Step 0: Check harness constraints before step
    harness = get_llm_harness()
    harness.check_before_step(job)

    # Step 1: If job status is DONE, exit with True
    if job.status == JobStatus.DONE:
        return True

    # Build shared local_context for all context-aware functions in this step
    local_context = dict(job.py_env.local_state) if job.py_env.local_state else {}

    # Step 2: Get next code block pending execution
    code = job.get_next_code_block()

    # Step 3: If code is None, change status to STARTED and go to step #9
    if code is None:
        job.set_status(JobStatus.STARTED)
    else:
        # Step 4: Update status READY -> WARMING_UP or SUSPENDED -> STARTED
        # Store whether we're resuming from SUSPENDED before changing status
        if job.status == JobStatus.READY:
            job.set_status(JobStatus.WARMING_UP)
        elif job.status == JobStatus.SUSPENDED:
            job.set_status(JobStatus.STARTED)

        # Extract the code string (CodeBlock stores code + tool_calls; exec needs the string)
        code_str = code.code if isinstance(code, CodeBlock) else code

        # Create WarmupLogItem before warmup block execution
        if job.status == JobStatus.WARMING_UP:
            block_num = job.warmup_block_num if job.warmup_block_num is not None else 0
            warmup_log_item = WarmupLogItem(
                console_pos=len(job.py_env.console) if job.py_env.console else 0,
                warmup_block_num=block_num
            )
            job.chat_log.append(warmup_log_item)

        # Log warmup block before execution
        if job.status == JobStatus.WARMING_UP and code_str:
            chat_style = get_statek_settings().chat_style
            log_code = f"```python\n{code_str}\n```" if chat_style in (ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG, ChatStyle.DIRECT) else code_str  # pylint: disable=no-member
            job._log(log_code)  # pylint: disable=protected-access

        # Check for empty code submission (no executable code and no tool calls)
        has_tool_calls = isinstance(code, CodeBlock) and code.tool_calls
        if not has_tool_calls and _is_empty_code(code_str):
            error_msg = "Error: no code submitted."
            job.console_append(error_msg, error_message=error_msg)

        # Step 5: Execute regular tool calls (not python_cli) if present and not a continuation
        last_chat_log_item = job.chat_log[-1] if job.chat_log else None
        if isinstance(code, CodeBlock) and code.tool_calls and job.next_instr_num is None:
            regular_calls = code.get_regular_tool_calls()
            if regular_calls:
                for call_spec in regular_calls:
                    result = await exec_tool(call_spec, job, local_context=dict(local_context))
                    if last_chat_log_item is not None:
                        last_chat_log_item.push_tool_result(result)
                    job._log_tool_call_result(call_spec, result)  # pylint: disable=protected-access

        # Step 6: Execute code and CLI tool calls using exec_all_steps
        cli_outputs = {}  # cli_idx -> list of output lines

        def _cli_console_append(cli_idx, text):
            cli_outputs.setdefault(cli_idx, []).append(text)

        try:
            code_block = code if isinstance(code, CodeBlock) else CodeBlock(code=code)
            await exec_all_steps(code_block, job, _cli_console_append, job.next_instr_num,
                                local_context=local_context)

            # Push CLI tool results into tool_log in order
            if cli_outputs and last_chat_log_item is not None:
                for cli_idx in sorted(cli_outputs):
                    last_chat_log_item.push_tool_result("\n".join(cli_outputs[cli_idx]))

            # DIRECT: also echo CLI output to job console so the user sees it
            if job.job_def.chat_style == ChatStyle.DIRECT and cli_outputs:  # pylint: disable=no-member
                for cli_idx in sorted(cli_outputs):
                    for line in cli_outputs[cli_idx]:
                        job.console_append(line)

            # Clear continuation state after successful execution
            job.awaited_result = None
            job.next_instr_num = None
        except FutureError as e:
            # Handle FutureError - suspend job
            job.awaited_result = e.future_result
            job.next_instr_num = e.instr_num
            # Change status STARTED -> SUSPENDED (no change for WARMING_UP)
            if job.status == JobStatus.STARTED:
                job.set_status(JobStatus.SUSPENDED)
            return False
        except Exception as e:
            # Step 7: Handle all other exceptions
            if job.status == JobStatus.WARMING_UP:
                # Critical job definition failure — terminate job and record error
                job.job_def.set_error(e)
                job.set_status(JobStatus.DONE)
                # Provide minimal context for error handlers
                _STATEK_CTX = {'job': job}  # noqa: F841
                handle_critical_error(e)
                return True
            # Non-warmup exception: already printed to console by exec_step
            pass

        # Step 6 & 7: Check if code has finished (exit_status not None)
        if job.py_env.exit_status is not None:
            job.set_status(JobStatus.DONE)
            # Log any pending console output before the exit marker
            _log_pending_console(job)
            job._log(f"exit: {job.py_env.exit_status}")  # pylint: disable=protected-access
            return True

        # Step 8: Handle warmup block progression or transition to STARTED
        if job.status == JobStatus.WARMING_UP:
            # Log this warmup block's console slice
            warmup_batch_from = warmup_log_item.console_pos
            warmup_batch_to = len(job.py_env.console) if job.py_env.console else 0
            job._log_console_batch(warmup_batch_from, warmup_batch_to)  # pylint: disable=protected-access
            if job.has_more_warmup_blocks():
                # Advance to next warmup block, stay in WARMING_UP
                job.advance_warmup_block()
                # Return False to continue execution loop (process next block)
                return False
            else:
                # All warmup blocks completed, transition to STARTED
                job.set_status(JobStatus.STARTED)

    # Step 9: Get LLM API provider
    if provider is None:
        provider = get_statek_settings().default_llm_api_provider
    llm_api = LLM_API.get(provider_name=provider, model=job.model)

    # Step 10: Get next request parameters — log pending console batch first
    _log_pending_console(job)
    request = job.get_next_request()
    # Materialize chat_history generator so it can be consumed by process_request
    if 'chat_history' in request:
        request['chat_history'] = list(request['chat_history'])

    # Step 11: Run the request with LLM API - await response
    response = await llm_api.process_request(**request)

    # add byte stats from LLM API response
    job.total_bytes_sent += response.stats.total_bytes_sent
    job.total_bytes_received += response.stats.total_bytes_received
    job.context_bytes = job.total_bytes_sent + job.total_bytes_received
    if response.stats.cost is not None:
        job.total_cost += response.stats.cost

    # Update session_id if returned by the LLM API
    if response.session_id:
        job.session_id = response.session_id

    # Step 12: Add new log item using append_chat_log
    job.append_chat_log(request, response)

    # Step 13: MD_DIALOG/DIRECT — dispatch LLM response text to user via send_message
    if job.job_def.chat_style in (ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
        await handle_md_dialog(response.text, _local_context=local_context)

    # MD_DIALOG / DIRECT: if LLM returned only text (no code / no tool calls), job is done;
    # if LLM returned code (MD_DIALOG) or tool calls (DIRECT), continue the loop.
    if job.job_def.chat_style in (ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
        if job.job_def.chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
            has_code = bool(response.call_requests)
        else:
            has_code = (response.call_requests
                        or not _is_empty_code(strip_markup(response.text, strict=True)))
        if not has_code:
            custom_exit(job)
            job.set_status(JobStatus.DONE)
            _log_pending_console(job)
            job._log(f"exit: {job.py_env.exit_status}")  # pylint: disable=protected-access
            return True

    # Step 14: Check harness constraints after step
    harness.check_after_step(job)

    # Step 15: Return False
    return False


def process_push_notifications(step_size=100, max_count=500):
    """Process pending push notifications from StatekPushQueues on all open prefixes.

    Finds all StatekPushQueue instances across open prefixes and delivers push
    messages to the target jobs by calling job.push_user_message. Jobs that have
    been deleted are silently ignored.

    Args:
        step_size: Number of notifications to retrieve per batch.
        max_count: Maximum total notifications to process in this call.
    """
    processed = 0

    for prefix in db0.get_prefixes():
        if processed >= max_count:
            break
        queue = db0.find_singleton(StatekPushQueue, prefix.name)
        if queue is None:
            continue
        while processed < max_count:
            batch_size = min(step_size, max_count - processed)
            if queue.is_empty():
                break
            items = queue.pop_from_job_console(batch_size)
            if not items:
                break
            for job_uuid, message in items:
                try:
                    job = db0.fetch(job_uuid)
                    job.push_user_message(message)
                except Exception:  # pylint: disable=broad-except
                    pass
                processed += 1


def unsuspend_jobs():
    """
    Review continuation conditions of suspended jobs and change their status to STARTED
    where the continuation criteria are satisfied.
    
    This function finds all jobs with status SUSPENDED, checks if their awaited_result
    condition is satisfied, and changes their status to STARTED if the condition is met.
    
    Note: This implementation might be slow for a large number of suspended jobs.
    Future versions will introduce a more robust Notifier engine and job expiration conditions.
    """
    # Find all suspended jobs
    suspended_jobs = db0.find(Job, JobStatus.SUSPENDED)
    if len(suspended_jobs) != 0:
        statek_log(f"Found {len(suspended_jobs)} suspended jobs", level='debug')
    for job in suspended_jobs:
        # Check if the job has an awaited_result and if its condition is satisfied
        condition_met = job.awaited_result.check_condition()
        if job.awaited_result is not None and condition_met:
            # Change status from SUSPENDED to STARTED
            job.set_status(JobStatus.STARTED)


def handle_critical_error(error: Optional[Exception] = None) -> None:
    """Handle a critical job-terminating error by invoking registered error handlers.

    Locates the current job from the execution context (via ``_STATEK_CTX``) and
    calls :meth:`~statek.executors.job.Job.notify_handlers` so that every registered
    error handler is invoked with the supplied exception.

    If no job is found in the current context the function is a no-op.

    Args:
        error: The exception that caused the critical failure, or ``None``.
    """
    job = get_current_job()
    if job is not None:
        job.notify_handlers(error=error)


async def job_worker(semaphore, job: Job, provider: str = None):
    async with semaphore:
        _STATEK_CTX = {'job': job}  # noqa: F841 — makes job accessible to handle_critical_error
        if job.job_def.agent is not None:
            _STATEK_CTX['agent'] = job.job_def.agent
        try:
            # Log which agent is running this job
            agent_name = job.job_def.agent.role if job.job_def.agent else "unknown"
            statek_log(f"Agent '{agent_name}' running job {db0.uuid(job)}", level='debug')
            await run_job_step(job, provider)
            # Log cost after each LLM request
            statek_log(f"Agent '{agent_name}' job {db0.uuid(job)} "
                       f"cost: ${job.total_cost:.4f}")
            # Log exit status if job completed successfully
            if job.status == JobStatus.DONE and job.py_env.exit_status is not None:
                exit_msg = f"exit: {job.py_env.exit_status}"
                job.console_append(exit_msg)
        except LLM_HarnessError as e:
            error_msg = f"LLM_HarnessError: {e}"
            statek_log(error_msg, level='debug')
            job.py_env.exit_status = f"Error: {e}"
            job.console_append(error_msg, error_message=error_msg)
            job.set_status(JobStatus.DONE)
            handle_critical_error(e)
        except Exception as e:
            # If job fails, write full stack trace to console and set status to DONE
            import traceback
            error_msg = f"Job {db0.uuid(job)} failed with error: {e}\n{traceback.format_exc()}"
            statek_log(error_msg, level='debug')
            job.console_append(error_msg, error_message=error_msg)
            job.set_status(JobStatus.DONE)
            handle_critical_error(e)

async def run_jobs_loop(max_concurrency: int = 100, provider: str = None,
    start_jobs_func: Callable = None, auto_terminate: bool = False):
    """
    Main loop responsible for processing registered jobs pending execution.
    
    The loop iterates over all active jobs so that none is starved off resources.
    It's assumed that only a single jobs-loop is running in one process.
    
    Args:
        max_concurrency: the execution concurrency level
        provider: the LLM API provider (or None for default)
        start_jobs_func: optional callable for starting new jobs
        auto_terminate: flag indicating if the loop should be terminated once all jobs 
                       have been completed; this flag is most useful for testing
    """
    # Track pending job tasks to avoid exceeding max_concurrency
    pending_tasks = {}  # Dict[Job, asyncio.Task]
    semaphore = asyncio.Semaphore(max_concurrency)
    while True:
        unsuspend_jobs()

        process_push_notifications()

        if start_jobs_func is not None:
            available_capacity = max_concurrency - len(pending_tasks)
            start_jobs_func(available_capacity)
        
        ready_or_started_jobs = db0.filter(lambda found_job: found_job not in pending_tasks,
                                            db0.find(Job, [JobStatus.READY, JobStatus.WARMING_UP, JobStatus.STARTED]))
        # Make sure not to exceed max_concurrency
        if len(pending_tasks) < max_concurrency:
            for job in ready_or_started_jobs:
                # Create task for this job. Add all tasks to pending_tasks
                task = asyncio.create_task(job_worker(semaphore, job, provider))
                pending_tasks[job] = task
        
        if pending_tasks:
            done, pending = await asyncio.wait(
                pending_tasks.values(),
                timeout=0.3,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Clean up completed tasks
            for job, task in list(pending_tasks.items()):
                if task in done:
                    del pending_tasks[job]
        else:
            # No pending tasks, just sleep
            await asyncio.sleep(0.3)
        
        # Check auto_terminate condition after processing
        if auto_terminate:
            # Check if there are any active jobs left (not DONE)
            active_jobs = db0.find(Job, db0.no([JobStatus.DONE]))
            # Also check if we have any tasks that might still be completing
            if len(active_jobs) == 0 and len(pending_tasks) == 0:
                # Double-check after a brief wait to avoid race conditions
                await asyncio.sleep(0.5)
                active_jobs = list(db0.find(Job, [JobStatus.READY, JobStatus.WARMING_UP, JobStatus.STARTED, JobStatus.SUSPENDED]))
                if not active_jobs and not pending_tasks:
                    statek_log("Auto-terminate: all jobs completed, exiting loop", level='debug')
                    break
    return


def find_existing_job_def(
    agent: 'Agent',
    warmup_code: Optional[Union[str, Sequence]],
) -> Optional[JobDef]:
    """Find an existing JobDef matching the given agent and warmup_code.

    Args:
        agent: the Agent the JobDef must be associated with
        warmup_code: the warmup code to match (compared after parsing)

    Returns:
        The first matching JobDef, or None if not found
    """
    parsed = parse_warmup_code(warmup_code)
    for job_def in db0.find(JobDef, db0.as_tag(agent)):
        if job_def.warmup_code == parsed:
            return job_def
    return None


def _make_start_jobs_func(agent, job_def, task_queue_size_func, provider):
    """Create the start_jobs_func callback for run_agentic_loop.

    The returned callable is invoked each loop iteration with available capacity.
    It raises RuntimeError if the job definition has errors (circuit breaker), then
    creates min(capacity, task_queue_size - ready_jobs) new jobs.

    Args:
        agent: the Agent instance
        job_def: the shared JobDef used for new jobs
        task_queue_size_func: callable returning number of queued tasks
        provider: LLM provider name (or None for default)

    Returns:
        Callable[[int], None] — the start_jobs_func callback
    """
    from statek.pyenv import PyEnv  # pylint: disable=import-outside-toplevel

    def start_jobs_func(capacity: int):
        if job_def.has_errors():
            raise RuntimeError(
                f"Job definition for agent '{agent.role}' has errors — aborting agentic loop"
            )

        if capacity <= 0:
            return

        num_awaiting_tasks = task_queue_size_func()

        if num_awaiting_tasks <= 0:
            return

        ready_jobs_count = len(db0.find(Job, db0.as_tag(agent), [JobStatus.READY, JobStatus.WARMING_UP]))

        jobs_to_create = min(capacity, num_awaiting_tasks - ready_jobs_count)

        if jobs_to_create <= 0:
            return

        statek_log(f"Creating {jobs_to_create} new jobs for agent {agent.role}", level='debug')

        provider_settings = get_provider_settings(provider)
        model_to_use = provider_settings.default_model if provider_settings else None

        for _ in range(jobs_to_create):
            Job(
                job_def=job_def,
                model_family=provider or "default",
                model=model_to_use,
                job_status=JobStatus.READY,
                py_env=PyEnv(local_state={}),
            )

    return start_jobs_func


@dataclass
class AgentLoopDef:
    """Definition for a single agent loop within a fleet.

    Attributes:
        agent: the SupervisedAgent instance to run
        warmup_code: initialization code — single block or sequence of blocks
        task_queue_size_func: callable returning the number of queued tasks
    """
    agent: 'SupervisedAgent'
    warmup_code: Union[str, Sequence[str], None]
    task_queue_size_func: Callable


async def run_agentic_loop(agent: 'Agent',
                           warmup_code: Union[str, Sequence[str]],
                           task_queue_size_func: Callable, max_concurrency: int = 100,
                           provider: str = None, auto_terminate: bool = False):
    """
    Helper function to start listening on arriving new tasks (e.g incoming user messages)
    and process them with a specific agent such as Coordinator or MessageDispatcher.

    This function can be used as the agentic system's main loop - where the incoming user
    messages are processed with a specific specialized agent. Internally the function calls
    `run_jobs_loop` and runs indefinitely (unless auto_terminate is True).

    Args:
        agent: the Agent instance (e.g. Coordinator or MessageDispatcher)
        warmup_code: the agent's warmup code - single block or sequence of blocks
                    (e.g. "user, message = fetch_next_message()")
        task_queue_size_func: a function for calculating the number of queued tasks 
                              (e.g. incoming messages) awaiting processing
        max_concurrency: maximum number of concurrent jobs (default: 100)
        provider: the default LLM provider (or None for default)
        auto_terminate: flag indicating if the loop should be terminated once all jobs 
                       have been completed; this flag is most useful for testing
    
    Example warmup code:
        ```
        user, message = fetch_next_message()
        print(message)
        ```
        
    Example task_queue_size function:
        ```
        def task_queue_size() -> int:
            return len(db0.find(Message, MessageStatus.PENDING))
        ```
    """
    statek_log("Starting agentic loop...", level='debug')

    # Reuse an existing matching job definition or create a new one
    job_def = find_existing_job_def(agent, warmup_code)
    if job_def:
        # Clear any previous errors on the job definition they might'be been fixed after process restart
        job_def.clear_errors()
    else:
        parsed_warmup_code = parse_warmup_code(warmup_code)
        job_def = JobDef(
            agent=agent,
            job_params=None,
            warmup_code=parsed_warmup_code,
        )
    
    start_jobs_func = _make_start_jobs_func(agent, job_def, task_queue_size_func, provider)

    await run_jobs_loop(
        max_concurrency=max_concurrency,
        provider=provider,
        start_jobs_func=start_jobs_func,
        auto_terminate=auto_terminate
    )


async def run_agentic_fleet(
    agent_loop_defs: Sequence[AgentLoopDef],
    max_concurrency: int = 100,
    provider: str = None,
    auto_terminate: bool = False,
):
    """
    Start an entire fleet of agents within a single process.

    Creates one JobDef and start_jobs_func per AgentLoopDef, then runs them all
    within a single run_jobs_loop for resource-efficient orchestration.

    Args:
        agent_loop_defs: sequence of AgentLoopDef instances defining agents to run
        max_concurrency: maximum number of concurrent jobs (default: 100)
        provider: the default LLM provider (or None for default)
        auto_terminate: flag indicating if the loop should be terminated once all jobs
                        have been completed; this flag is most useful for testing
    """
    statek_log("Starting agentic fleet...", level='debug')

    start_jobs_funcs = []
    for loop_def in agent_loop_defs:
        agent = loop_def.agent
        warmup_code = loop_def.warmup_code
        task_queue_size_func = loop_def.task_queue_size_func

        job_def = find_existing_job_def(agent, warmup_code)
        if job_def:
            job_def.clear_errors()
        else:
            parsed_warmup_code = parse_warmup_code(warmup_code)
            job_def = JobDef(
                agent=agent,
                job_params=None,
                warmup_code=parsed_warmup_code,
            )

        start_jobs_funcs.append(_make_start_jobs_func(agent, job_def, task_queue_size_func, provider))

    def combined_start_jobs_func(capacity: int):
        for func in start_jobs_funcs:
            func(capacity)

    await run_jobs_loop(
        max_concurrency=max_concurrency,
        provider=provider,
        start_jobs_func=combined_start_jobs_func,
        auto_terminate=auto_terminate,
    )
