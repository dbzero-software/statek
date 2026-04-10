# pylint: disable=no-member
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ast
import inspect
import dbzero as db0
from .exceptions import FutureError
from .future import FutureResult, temporal
from .system import tool
from .agents.agent import SupervisedAgent
from .agents.dialog_agent import DialogAgent
from .executors.job import Job, JobStatus
from .pyenv import PyEnv
from .settings import get_provider_settings, get_statek_settings


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


@temporal(complement = get_task_result, condition=is_job_completed)
@tool
def delegate_task(agent: SupervisedAgent,
    warmup_code: Optional[Union[str, Sequence[str]]] = None,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale=None,
    **kwargs) -> TaskFutureResult:
    """Create a new job delegated to given agent.

    Args:
        agent: The `Agent` to delegate task to
        warmup_code: Optional Python code (single block or sequence of blocks)
                    to be executed prior to task start
        parent_job: Optional parent job — when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional ``{var_name: value}`` mapping of variables to
                    be additionally shared with the child job's context.
        locale: Optional locale for job execution.
        kwargs: job specific parameters for prompt formatting (i.e. job_params)
    """

    job_def = agent.create_job_def(
        warmup_code=warmup_code, shared_vars=shared_vars, locale=locale, **kwargs
    )

    env = PyEnv()
    if warmup_code:
        # Go to caller frame (skip decorators)
        caller_frame = inspect.currentframe().f_back.f_back.f_back
        caller_locals = caller_frame.f_locals
        # Copy locals from all warmup blocks
        if isinstance(warmup_code, str):
            copy_locals(warmup_code, env.local_state, caller_locals)
        else:
            for block in warmup_code:
                copy_locals(block, env.local_state, caller_locals)

    if shared_vars:
        env.local_state.update(shared_vars)

    job = Job(
        job_def=job_def,
        model_family=get_statek_settings().default_llm_api_provider,
        model=get_provider_settings().default_model,
        job_status=JobStatus.READY,
        py_env=env
    )

    if parent_job is not None:
        job.add_error_handlers_from(parent_job)

    return TaskFutureResult(job, deps=None, state_num=0)


def start_dialog(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    agent: DialogAgent,
    message: str,
    warmup_code: Optional[Union[str, Sequence[str]]] = None,
    parent_job: Optional[Job] = None,
    shared_vars: Optional[Dict[str, Any]] = None,
    locale=None,
    **kwargs
) -> Job:
    """Start a dialog job with a DialogAgent and an initial user message.

    Args:
        agent: The DialogAgent to delegate the dialog to.
        message: The initial user message.
        warmup_code: Optional Python code (single or multiple blocks) to be
                     executed prior to task start.  When provided, all
                     referenced locals of the caller are copied into the
                     new job's context.
        parent_job: Optional parent job — when provided, the child job
                    inherits the parent's error handlers.
        shared_vars: Optional variables to share with the dialog job's
                     context (see :func:`delegate_task`).
        locale: Optional locale for the child job / dialog.
        kwargs: Optional agent-specific extra arguments (job_params).

    Returns:
        The newly created Job instance.
    """
    job_def = agent.create_job_def(
        warmup_code=warmup_code, shared_vars=shared_vars, locale=locale, **kwargs
    )

    env = PyEnv()
    if warmup_code:
        caller_frame = inspect.currentframe().f_back
        caller_locals = caller_frame.f_locals
        if isinstance(warmup_code, str):
            copy_locals(warmup_code, env.local_state, caller_locals)
        else:
            for block in warmup_code:
                copy_locals(block, env.local_state, caller_locals)

    if shared_vars:
        env.local_state.update(shared_vars)

    job = Job(
        job_def=job_def,
        model_family=get_statek_settings().default_llm_api_provider,
        model=get_provider_settings().default_model,
        job_status=JobStatus.READY,
        py_env=env
    )

    if parent_job is not None:
        job.add_error_handlers_from(parent_job)

    job.push_user_message(message)

    return job


def submit_new_job(
    agent: SupervisedAgent,
    shared_vars: Optional[Dict[str, Any]] = None,
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
        kwargs: Agent-specific job parameters.

    Returns:
        The newly created :class:`Job` instance.
    """
    job_def = agent.create_job_def(shared_vars=shared_vars, **kwargs)

    env = PyEnv()
    if shared_vars:
        env.local_state.update(shared_vars)

    return Job(
        job_def=job_def,
        model_family=get_statek_settings().default_llm_api_provider,
        model=get_provider_settings().default_model,
        job_status=JobStatus.READY,
        py_env=env,
    )


def submit_new_jobs_batch(
    agent: SupervisedAgent,
    shared_vars_batch: List[Optional[Dict[str, Any]]],
    **kwargs
) -> List[Job]:
    """Create multiple jobs with different shared variables in one call.

    Args:
        agent: The agent (must inherit from SupervisedAgent).
        shared_vars_batch: A list of shared_vars entries — one per job.
        kwargs: Agent-specific job parameters (same for all jobs).

    Returns:
        A list of newly created :class:`Job` instances.
    """
    return [
        submit_new_job(agent, shared_vars=sv, **kwargs)
        for sv in shared_vars_batch
    ]
