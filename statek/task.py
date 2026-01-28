# pylint: disable=no-member
from typing import Tuple, Dict, Optional
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


def copy_locals(code: str, scope: Dict, dest: Dict):
    """
    Identify all locals referenced in a given code block and copy them
    into destination dictionary

    Args:
        code: The dynamic Python code to be analyzed
        scope: Local variables passed from execution scope
        dest: The destination to copy all referenced locals 
    """
    tree = ast.parse(code)

    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node):
            # Capture all Name nodes with Load context (reading variables)
            if isinstance(node.ctx, ast.Load):
                # Copy only the locals that are actually referenced in the code
                if node.id not in dest and node.id in scope:
                    dest[node.id] = scope[node.id]
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
def delegate_task(agent: SupervisedAgent, warmup_code: Optional[str] = None,
    **kwargs) -> TaskFutureResult:
    """Create a new job delegated to given agent.
    
    Args: 
        agent: The `Agent` to delegate task to
        warmup_code: Optional Python code to be executed prior to task start
        kwargs: job specific parameters for prompt formatting (i.e. job_params)
    """

    job_def = agent.create_job_def(warmup_code=warmup_code, **kwargs)

    env = PyEnv()
    if warmup_code:
        # Go to caller frame (skip decorators)
        caller_frame = inspect.currentframe().f_back.f_back.f_back
        caller_locals = caller_frame.f_locals
        copy_locals(warmup_code, caller_locals, env.local_state)

    job = Job(
        job_def=job_def,
        model_family=get_statek_settings().default_llm_api_provider,
        model=get_provider_settings().default_model,
        job_status=JobStatus.READY,
        py_env=env
    )

    return TaskFutureResult(job, deps=None, state_num=0)
