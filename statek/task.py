# pylint: disable=no-member
from typing import Tuple, Dict, Optional, Type, Iterable, Any, Sequence, Union
import ast
import inspect
import dbzero as db0
from .exceptions import FutureError
from .future import FutureResult, temporal
from .system import tool
from .agents.agent import SupervisedAgent
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


def find_locals(var_type: Optional[Type] = None,
                var_name: Optional[str] = None) -> Iterable[Any]:
    """
    Search through the caller's local context - retrieving variables matching
    a specific type or name. This function is helpful when implementing temporal
    functions which need to be context-aware.

    Args:
        var_type: Optional type to identify local variables by (e.g. SMS_Message or User)
        var_name: Optional variable name to match

    Yields:
        Matching variables from the caller's context. If neither var_type nor var_name 
        is specified, all variables from the local context will be yielded.
    """
    # Search through up to 10 frames up the call stack
    caller_frame = inspect.currentframe().f_back
    frames_to_search = []

    # Collect up to 10 frames
    current_frame = caller_frame
    for _ in range(10):
        if current_frame is None:
            break
        frames_to_search.append(current_frame)
        current_frame = current_frame.f_back

    # Aggregate locals from all frames, with priority to closer frames
    aggregated_locals = {}
    for frame in reversed(frames_to_search):
        frame_locals = frame.f_locals.copy()

        # Check if _local_context is set and extend with it
        if '_local_context' in frame_locals:
            local_context = frame_locals['_local_context']
            if local_context is not None and isinstance(local_context, dict):
                aggregated_locals.update(local_context)

        # Also check if kwargs contains _local_context
        if 'kwargs' in frame_locals and isinstance(frame_locals['kwargs'], dict):
            if '_local_context' in frame_locals['kwargs']:
                local_context = frame_locals['kwargs']['_local_context']
                if local_context is not None and isinstance(local_context, dict):
                    aggregated_locals.update(local_context)

        # Update with frame locals (closer frames override)
        aggregated_locals.update(frame_locals)

    # Iterate through aggregated local variables
    for name, value in aggregated_locals.items():
        # If no filters specified, yield all variables
        if var_type is None and var_name is None:
            yield value
        else:
            # Apply filters
            type_match = var_type is None or isinstance(value, var_type)
            name_match = var_name is None or name == var_name

            if type_match and name_match:
                yield value


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
    **kwargs) -> TaskFutureResult:
    """Create a new job delegated to given agent.

    Args:
        agent: The `Agent` to delegate task to
        warmup_code: Optional Python code (single block or sequence of blocks)
                    to be executed prior to task start
        kwargs: job specific parameters for prompt formatting (i.e. job_params)
    """

    job_def = agent.create_job_def(warmup_code=warmup_code, **kwargs)

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

    job = Job(
        job_def=job_def,
        model_family=get_statek_settings().default_llm_api_provider,
        model=get_provider_settings().default_model,
        job_status=JobStatus.READY,
        py_env=env
    )

    return TaskFutureResult(job, deps=None, state_num=0)
