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

# pylint: disable=no-member
import ast
import builtins
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import dbzero as db0
from .exceptions import FutureError
from .future import FutureResult, temporal
from .system import tool
from .agents.agent import Agent, SupervisedAgent
from .agents.dialog_agent import DialogAgent
from .chat_style import ChatStyle
from .executors.job import Job, JobStatus, WarmupCodeInput
from .locale import StatekLocale
from .pyenv import PyEnv
from .settings import get_provider_settings as _get_provider_settings
from .settings import get_statek_settings as _get_statek_settings
from .utils import CodeBlock, get_current_job


def _job_params_from_kwargs(agent: Agent, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the JobDef params created by Agent.create_job_def for these kwargs."""
    job_params = dict(kwargs) if kwargs else {}
    if isinstance(agent, DialogAgent):
        job_params.pop("chat_style", None)
    return job_params or None


def _dialog_chat_style(agent: Agent, kwargs: Dict[str, Any]):
    """Return the explicit JobDef chat style DialogAgent.create_job_def will set."""
    if not isinstance(agent, DialogAgent):
        return None
    chat_style = kwargs.get("chat_style")
    if chat_style is None and agent._metadata and 'CHAT_STYLE' in agent._metadata:  # pylint: disable=protected-access
        chat_style = ChatStyle[agent._metadata['CHAT_STYLE']]  # pylint: disable=protected-access
    return chat_style or ChatStyle.MD_DIALOG  # pylint: disable=no-member


def _find_reusable_job_def(
    agent: Agent,
    warmup_code: WarmupCodeInput,
    locale,
    post_processing,
    kwargs: Dict[str, Any],
):
    """Find an existing JobDef matching create_new_job's reusable definition."""
    from statek.executors.utils import find_existing_job_def  # pylint: disable=import-outside-toplevel

    if isinstance(agent, SupervisedAgent):
        combined_warmup_code = agent._combine_warmup_code(warmup_code)  # pylint: disable=protected-access
    else:
        combined_warmup_code = warmup_code
    metadata = agent._metadata or {}  # pylint: disable=protected-access
    return find_existing_job_def(
        agent,
        combined_warmup_code,
        model=metadata.get("MODEL"),
        job_params=_job_params_from_kwargs(agent, kwargs),
        locale=locale,
        chat_style=_dialog_chat_style(agent, kwargs) if isinstance(agent, DialogAgent) else None,
        post_processing=post_processing,
    )


class _ReferencedLocalsCollector(ast.NodeVisitor):
    """Collect external local names referenced by a Python AST."""

    def __init__(self):
        self.builtin_names = set(dir(builtins))
        self.referenced = []
        self.seen = set()
        self.bound = set()

    def add_reference(self, name: str):
        if name in self.bound or name in self.seen or name in self.builtin_names:
            return
        self.seen.add(name)
        self.referenced.append(name)

    def bind_target(self, target):
        if isinstance(target, ast.Name):
            self.bound.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.bind_target(element)
            return
        if isinstance(target, ast.Starred):
            self.bind_target(target.value)

    def visit_store_target_reads(self, target):
        if isinstance(target, ast.Name):
            return
        if isinstance(target, ast.Attribute):
            self.visit(target.value)
            return
        if isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.visit_store_target_reads(element)
            return
        if isinstance(target, ast.Starred):
            self.visit_store_target_reads(target.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.add_reference(node.id)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        elif not isinstance(node.func, ast.Name):
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.visit_store_target_reads(target)
            self.bind_target(target)

    def visit_AnnAssign(self, node):
        if node.annotation is not None:
            self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit_store_target_reads(node.target)
        self.bind_target(node.target)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.add_reference(node.target.id)
        else:
            self.visit(node.target)
        self.visit(node.value)
        self.bind_target(node.target)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self.bind_target(node.target)

    def visit_For(self, node):
        self.visit(node.iter)
        self.bind_target(node.target)
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.bind_target(item.optional_vars)
        for child in node.body:
            self.visit(child)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

    def _visit_comprehension_generators(self, generators):
        for generator in generators:
            self.visit(generator.iter)
            self.bind_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node):
        previous_bound = set(self.bound)
        self._visit_comprehension_generators(node.generators)
        self.visit(node.elt)
        self.bound.clear()
        self.bound.update(previous_bound)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        previous_bound = set(self.bound)
        self._visit_comprehension_generators(node.generators)
        self.visit(node.key)
        self.visit(node.value)
        self.bound.clear()
        self.bound.update(previous_bound)


def get_referenced_locals(code: str) -> Iterable[str]:
    """Return external local names referenced by *code* in appearance order.

    Names bound by the code block itself are not returned. Builtins are also
    skipped because they do not need to be copied into the execution context.
    """
    tree = ast.parse(code)
    collector = _ReferencedLocalsCollector()
    collector.visit(tree)
    return collector.referenced


def get_provider_settings(provider=None):
    """Compatibility shim for tests patching task-level settings access."""
    return _get_provider_settings(provider)


def get_statek_settings():
    """Compatibility shim for tests patching task-level settings access."""
    return _get_statek_settings()


def _copy_warmup_locals(warmup_code: WarmupCodeInput, dest: Dict, caller_locals: Dict) -> None:
    """Copy caller locals referenced by raw or already-parsed warmup blocks."""
    if isinstance(warmup_code, CodeBlock):
        if warmup_code.code:
            copy_locals(warmup_code.code, dest, caller_locals)
        return
    if isinstance(warmup_code, str):
        if warmup_code:
            copy_locals(warmup_code, dest, caller_locals)
        return
    for block in warmup_code:
        code = block.code if isinstance(block, CodeBlock) else block
        if code:
            copy_locals(code, dest, caller_locals)


def copy_locals(code: str, dest: Dict, local_vars: Optional[Dict] = None):
    """
    Identify all locals referenced in a given code block and copy them
    into destination dictionary

    Args:
        code: The dynamic Python code to be analyzed (e.g. `print(user)`
              - copies "user")
        dest: The destination to copy all referenced locals
        local_vars: The explicit local context to copy from - if not provided,
                   the function will use caller's context to retrieve variables
    """
    # If local_vars not provided, get caller's locals
    if local_vars is None:
        caller_frame = inspect.currentframe().f_back
        local_vars = caller_frame.f_locals

    tree = ast.parse(code)

    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node):
            # Capture all Name nodes with Load context (reading variables)
            if isinstance(node.ctx, ast.Load):
                # Copy only the locals that are actually referenced in the code
                if node.id not in dest and node.id in local_vars:
                    dest[node.id] = local_vars[node.id]
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # For attribute access like 'obj.attr', we need to visit 'obj'
            # The 'attr' part is not a variable reference
            self.visit(node.value)
            # Don't call generic_visit to avoid processing 'attr' as a Name

    collector = NameCollector()
    collector.visit(tree)


SubTaskState = db0.enum("SubTaskState", ["WAITING", "STARTED", "COMPLETED", "ERROR"])


@db0.memo(no_default_tags=True)
@dataclass
class TaskError:
    """Error information reported by a completed subtask."""

    err_message: str


@db0.memo
class SubTaskHandler:
    """Handle for a child job delegated as a subtask."""

    def __init__(self, job: Job, id: Optional[Any] = None):  # pylint: disable=redefined-builtin
        """Create a handler for an already-created child job.

        Args:
            job: Child job represented by this handler.
            id: Optional caller-supplied subtask identifier.
        """
        self.job = job
        self.id = id
        self.__is_completed: bool = False
        self.__error: Optional[TaskError] = None
        self.__result: Optional[Any] = None

    @property
    def is_completed(self) -> bool:
        """Return whether the handler has an explicit completion outcome."""
        return self.__is_completed

    @property
    def error(self) -> Optional[TaskError]:
        """Return the subtask error, if completion failed."""
        return self.__error

    @property
    def result(self) -> Optional[Any]:
        """Return the successful completion result, if any."""
        return self.__result

    @property
    def state(self) -> SubTaskState:
        """Resolve the handler state from explicit completion and child job status."""
        if self.__is_completed:
            return SubTaskState.ERROR if self.__error is not None else SubTaskState.COMPLETED
        if self.job.status == JobStatus.READY:
            return SubTaskState.WAITING
        return SubTaskState.STARTED

    def __str__(self) -> str:
        """Return completed results and intentionally raise for unfinished handlers."""
        state = self.state
        if state == SubTaskState.COMPLETED:
            return "" if self.__result is None else str(self.__result)
        if state == SubTaskState.ERROR:
            raise RuntimeError(self.__error.err_message)
        task_id = f" id={self.id}" if self.id is not None else ""
        raise RuntimeError(f"Sub-task{task_id} is not completed: {state}")

    def get_log_message(self) -> str:
        """Return the LLM-facing notification message for this completed handler."""
        task_id = f" id={self.id}" if self.id is not None else ""
        if self.state == SubTaskState.ERROR:
            return f"[Error] sub-task{task_id} failed with {self.__error.err_message}"
        return f"[Notification] sub-task{task_id} completed successfully."

    def complete(self, result: Optional[Any] = None, error: Optional[str] = None) -> None:
        """Record the subtask outcome and notify the parent job when present."""
        if self.__is_completed:
            raise RuntimeError("Sub-task handler is already completed")
        if result is not None and error is not None:
            raise ValueError("Sub-task completion cannot mix result and error")

        self.__result = result
        self.__error = TaskError(error) if error is not None else None
        self.__is_completed = True

        parent_job = self.job.parent_job
        if parent_job is not None:
            parent_job.push_notification(self)


def complete_sub_task(result: Optional[Any] = None, error: Optional[str] = None) -> None:
    """Complete the subtask handler registered in the current child job locals."""
    job = get_current_job()
    if job is None:
        raise RuntimeError("complete_sub_task requires a current job")

    handler = job.py_env.local_state.get("sub_task_handler")
    if handler is None:
        raise RuntimeError("complete_sub_task requires sub_task_handler in current job locals")
    if not isinstance(handler, SubTaskHandler):
        raise TypeError("sub_task_handler must be a SubTaskHandler")

    handler.complete(result=result, error=error)


def create_sub_task(
    job: Job,
    handler_type: Type[SubTaskHandler] = SubTaskHandler,
    **kwargs,
) -> SubTaskHandler:
    """Create a subtask handler for an existing job and inject it into job locals."""
    handler = handler_type(job=job, **kwargs)
    job.add_locals(sub_task_handler=handler)
    return handler


def create_new_job(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    agent: Agent,
    shared_vars: Optional[Dict[str, Any]] = None,
    parent_job: Optional[Job] = None,
    warmup_code: WarmupCodeInput = None,
    locale=None,
    post_processing=None,
    caller_frame=None,
    **kwargs,
) -> Job:
    """Create a ready job with shared locals, inherited locale, and error handlers."""
    shared_vars = shared_vars or {}
    effective_locale = _resolve_child_locale(parent_job, locale)
    job_def = _find_reusable_job_def(
        agent,
        warmup_code,
        effective_locale,
        post_processing,
        kwargs,
    )
    if job_def is None:
        job_def = agent.create_job_def(
            warmup_code=warmup_code,
            locale=effective_locale,
            post_processing=post_processing,
            **kwargs,
        )

    env = PyEnv()
    if warmup_code and caller_frame is None:
        caller_frame = inspect.currentframe().f_back
    if warmup_code and caller_frame is not None:
        caller_locals = caller_frame.f_locals
        _copy_warmup_locals(warmup_code, env.local_state, caller_locals)

    env.local_state.update(shared_vars)

    job = Job(
        job_def=job_def,
        job_status=JobStatus.READY,
        py_env=env,
        parent_job=parent_job,
    )
    if parent_job is not None:
        job.add_error_handlers_from(parent_job)
    return job


@db0.memo
class TaskFutureResult(FutureResult):
    """Future holding associated job"""

    def __init__(self, job: Job, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = job


def get_task_result(task_future: TaskFutureResult) -> Tuple[str, Job]:
    """Retrieve completed Job and exit status or raise FutureError"""
    if task_future.job.status != JobStatus.DONE:
        raise FutureError(task_future)
    return (task_future.job.py_env.exit_status, task_future.job)


def is_job_completed(task_future: TaskFutureResult) -> bool:
    """Check task completion status"""
    return task_future.job.status == JobStatus.DONE


def create_future_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    agent: Agent,
    shared_vars: dict,
    parent_job: Optional[Job],
    warmup_code: WarmupCodeInput = None,
    locale=None,
    caller_frame=None,
    **kwargs,
) -> TaskFutureResult:
    """Create a child job future for temporal utilities.

    Callers may pass ``caller_frame`` when warmup code should copy referenced
    locals from a specific stack frame, as delegate helpers do.
    """
    if warmup_code and caller_frame is None:
        caller_frame = inspect.currentframe().f_back
    job = create_new_job(
        agent,
        shared_vars=shared_vars,
        parent_job=parent_job,
        warmup_code=warmup_code,
        locale=locale,
        caller_frame=caller_frame,
        **kwargs,
    )
    return TaskFutureResult(job, deps=None, state_num=0)


def _resolve_child_locale(parent_job: Optional[Job], locale):
    """Return the locale, inheriting parent_job.locale when locale is empty."""
    if locale is not None or parent_job is None:
        return locale
    return parent_job.job_def.locale


@temporal(complement=get_task_result, condition=is_job_completed)
@tool
def delegate_task(agent: SupervisedAgent,
    warmup_code: WarmupCodeInput = None,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale=None,
    **kwargs) -> TaskFutureResult:
    """Create a new job delegated to given agent.

    Args:
        agent: The `Agent` to delegate task to
        warmup_code: Optional Python code (single block or sequence of blocks)
                    to be executed prior to task start
        parent_job: Optional parent job; when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional ``{var_name: value}`` mapping of variables to
                    be additionally shared with the child job's context.
        locale: Optional locale for job execution.  When omitted and
            parent_job is provided, the parent job's locale is inherited.
        kwargs: job specific parameters for prompt formatting (i.e. job_params)
    """
    # Skip @tool and @temporal decorator frames to reach the actual caller
    caller_frame = inspect.currentframe().f_back.f_back.f_back if warmup_code else None
    return create_future_task(
        agent,
        shared_vars or {},
        parent_job,
        warmup_code=warmup_code,
        locale=locale,
        caller_frame=caller_frame,
        **kwargs,
    )


def get_mute_job_result(future: TaskFutureResult) -> str:
    """Retrieve chat responses from a completed mute job, or exit status on failure."""
    if future.job.status != JobStatus.DONE:
        raise FutureError(future)
    job = future.job
    if job.error is None:
        return "\n".join(job.get_chat_responses())
    if job.py_env.exit_status is not None:
        return job.py_env.exit_status
    return job.error.error_message


@temporal(complement=get_mute_job_result, condition=is_job_completed)
@tool
def delegate_mute_task(agent: SupervisedAgent,
    warmup_code: WarmupCodeInput = None,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale: Optional[StatekLocale] = None,
    **kwargs) -> TaskFutureResult:
    """Create a new job delegated to a mute agent.

    Like :func:`delegate_task`, but the underlying agent is assumed to be mute
    — its messages are not delivered to the end-user. Instead, they are retrieved
    on job completion and returned as the result string.

    Args:
        agent: The `Agent` to delegate task to
        warmup_code: Optional Python code (single block or sequence of blocks)
                    to be executed prior to task start
        parent_job: Optional parent job; when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional ``{var_name: value}`` mapping of variables to
                    be additionally shared with the child job's context.
        locale: Optional locale for job execution.  When omitted and
            parent_job is provided, the parent job's locale is inherited.
        kwargs: job specific parameters for prompt formatting (i.e. job_params)
    """
    # Skip @tool and @temporal decorator frames to reach the actual caller
    caller_frame = inspect.currentframe().f_back.f_back.f_back if warmup_code else None
    return create_future_task(
        agent,
        shared_vars or {},
        parent_job,
        warmup_code=warmup_code,
        locale=locale,
        caller_frame=caller_frame,
        **kwargs,
    )


@temporal(complement=get_mute_job_result, condition=is_job_completed)
@tool
def delegate_mute_dialog(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    agent: DialogAgent,
    message: Any,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale: Optional[StatekLocale] = None,
    **kwargs
) -> TaskFutureResult:
    """Create a new mute dialog job with an initial user message.

    Like :func:`delegate_mute_task`, but for :class:`DialogAgent` jobs that
    start from an incoming user message. The dialog agent's messages are not
    delivered as this function's immediate result; they are collected from the
    completed job and returned as a string.

    Args:
        agent: The dialog agent to delegate the dialog to.
        message: The initial user message. Non-string values are converted by
            the dialog agent's ``message_adapter`` when registered, otherwise
            by ``str(message)``.
        parent_job: Optional parent job — when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional ``{var_name: value}`` mapping of variables to
                    be additionally shared with the child job's context.
        locale: Optional locale for job execution.  When omitted and
            parent_job is provided, the parent job's locale is inherited.
        kwargs: job specific parameters for prompt formatting (i.e. job_params).
    """
    job = start_dialog(
        agent=agent,
        message=message,
        parent_job=parent_job,
        shared_vars=shared_vars,
        locale=locale,
        **kwargs,
    )
    return TaskFutureResult(job, deps=None, state_num=0)


def start_dialog(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    agent: DialogAgent,
    message: Any,
    warmup_code: WarmupCodeInput = None,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale=None,
    **kwargs
) -> Job:
    """Start a dialog job with a DialogAgent and an initial user message.

    Args:
        agent: The DialogAgent to delegate the dialog to.
        message: The initial user message. Non-string values are converted by
            the dialog agent's ``message_adapter`` when registered, otherwise
            by ``str(message)``.
        warmup_code: Optional Python code (single or multiple blocks) to be
                     executed prior to task start.  When provided, all
                     referenced locals of the caller are copied into the
                     new job's context.
        parent_job: Optional parent job — when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional variables to share with the dialog job's
                     context (see :func:`delegate_task`).
        locale: Optional locale for the child job / dialog.  When omitted and
            parent_job is provided, the parent job's locale is inherited.
        kwargs: Optional agent-specific extra arguments (job_params).

    Returns:
        The newly created Job instance.
    """
    caller_frame = inspect.currentframe().f_back if warmup_code else None
    job = create_new_job(
        agent,
        shared_vars=shared_vars,
        parent_job=parent_job,
        warmup_code=warmup_code,
        locale=locale,
        caller_frame=caller_frame,
        **kwargs,
    )
    job.push_user_message(message)

    return job


def submit_new_job(
    agent: SupervisedAgent,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale=None,
    **kwargs
) -> Job:
    """Create a new job for external use.

    Similar to :func:`delegate_task` but intended for external scripts and
    processes.  Accepts actual objects instead of relying on call-stack
    frame inspection.

    Args:
        agent: The agent (must inherit from SupervisedAgent).
        shared_vars: Optional ``{var_name: object}`` mapping of variables
            to share with the job.
        locale: Optional locale for job execution.
        kwargs: Agent-specific job parameters.

    Returns:
        The newly created :class:`Job` instance.
    """
    return create_new_job(
        agent,
        shared_vars=shared_vars,
        locale=locale,
        **kwargs,
    )


def submit_new_jobs_batch(
    agent: SupervisedAgent,
    shared_vars_batch: List[Optional[Dict[str, Any]]],
    locale=None,
    **kwargs
) -> List[Job]:
    """Create multiple jobs with different shared variables in one call.

    Args:
        agent: The agent (must inherit from SupervisedAgent).
        shared_vars_batch: A list of shared_vars entries — one per job.
        locale: Optional locale for job execution.
        kwargs: Agent-specific job parameters (same for all jobs).

    Returns:
        A list of newly created :class:`Job` instances.
    """
    return [
        submit_new_job(agent, shared_vars=sv, locale=locale, **kwargs)
        for sv in shared_vars_batch
    ]
