import ast
import sys
import inspect
import traceback
import asyncio
import builtins
from typing import Callable
from contextlib import contextmanager
import dbzero as db0


from statek.exceptions import FutureError
from statek.future import FutureResult
from statek.executors.job import Job, JobStatus
from statek.llm_api import LLM_API
from statek.settings import get_statek_settings

def wrap_param(param):
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
                    bound.arguments[name] = tuple(wrap_param(v) for v in val)
            # Handle **kwargs (VAR_KEYWORD)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # val is a dict, wrap each value unless annotated as FutureResult
                if not check_for_future_typehint(param, anno):
                    bound.arguments[name] = {k: wrap_param(v) for k, v in val.items()}
            # Handle regular parameters
            elif not check_for_future_typehint(param, anno):
                bound.arguments[name] = wrap_param(val)
        return func(*bound.args, **bound.kwargs)
    except (ValueError, TypeError):
        # Fallback: wrap everything if signature inspection fails
        return func(*[wrap_param(a) for a in args], **{k: wrap_param(v) for k, v in kwargs.items()})

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
        if isinstance(node.ctx, ast.Load) and node.id not in {'wrap_param', '_smart_call'}:
            new_node = ast.Call(
                func=ast.Name(id='wrap_param', ctx=ast.Load()),
                args=[node],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        return node

"""Execute a single AST node with custom print function."""
def custom_print(job, *args, sep=' ', end='\n', **kwargs):
    """Custom print function that writes to job console."""
    output = sep.join(str(arg) for arg in args) + end
    job.py_env.console_append(output)

def custom_exit(job, status=None):
    """Custom exit function that sets exit status."""
    job.py_env.exit_status = status


@contextmanager
def _setup_execution_context(job: Job, local_context: dict):
    """
    Context manager to setup and cleanup execution environment.
    
    Args:
        job: The Job instance
        local_context: The local execution context dictionary
    
    Yields:
        tuple: (custom_print_fn, custom_exit_fn) for reference
    """
    # Create custom functions
    custom_print_fn = lambda *args, **kwargs: custom_print(job, *args, **kwargs)
    custom_exit_fn = lambda status=None: custom_exit(job, status)
    
    # Save original built-in print to restore later
    original_print = builtins.print
    
    # Monkey-patch builtins.print so it works even in pre-defined functions
    builtins.print = custom_print_fn
    
    # Inject into local context
    local_context['print'] = custom_print_fn
    local_context['_smart_call'] = _smart_call
    local_context['wrap_param'] = wrap_param
    local_context['exit'] = custom_exit_fn
    
    try:
        yield custom_print_fn, custom_exit_fn
    finally:
        # Restore original built-in print
        builtins.print = original_print
        
        # Remove helpers from context
        for key in ['print', 'exit', '_smart_call', 'wrap_param']:
            if key in local_context:
                del local_context[key]


async def exec_step(code_str: str, job: Job) -> bool:
    """
    Execute a single step of code within the job's Python environment.

    Args:
        code: The code string to execute.
        job: The Job instance containing the Python environment.

    Returns:
        True if execution completed without exit signal, False if exit was called.
    """
    # Global and local contexts needs to be dictionaries in order to be used with exec
    if job.py_env.global_state is None:
        global_context = globals()
    else:
        global_context = {key: value for key, value in job.py_env.global_state.items()}
    if job.py_env.local_state is None:
        local_context = {}
    else:
        local_context = {key: value for key, value in job.py_env.local_state.items()}

    # Use context manager to setup and cleanup execution environment
    with _setup_execution_context(job, local_context):
        tree = ast.parse(code_str)
        _ResilientTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        for node in tree.body:
            try:

                wrapper = ast.Module(body=[node], type_ignores=[])
                code_obj = compile(wrapper, filename="<string>", mode="exec")
                exec(code_obj, global_context, local_context)
                # Check for exit signal
                if job.py_env.exit_status is not None:
                    break
                # Save updated local context back to job
            except FutureError:
                continue
    
    job.py_env.local_state = local_context
    return job.py_env.exit_status is None


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
    # Step 1: If job status is DONE, exit with True
    if job.status == JobStatus.DONE:
        return True

    # Step 2: Get next code block pending execution
    code = job.get_next_code_block()

    # Step 3: If code is None, change status to STARTED and go to step #9
    if code is None:
        job.status = JobStatus.STARTED
    else:
        # Step 4: Update status READY -> WARMING_UP or SUSPENDED -> STARTED
        if job.status == JobStatus.READY:
            job.status = JobStatus.WARMING_UP
        elif job.status == JobStatus.SUSPENDED:
            job.status = JobStatus.STARTED

        # Step 5: Execute the code using exec_step
        not_exited = await exec_step(code, job)

        # Step 6 & 7: Check if code has finished (exit_status not None)
        if job.py_env.exit_status is not None:
            job.status = JobStatus.DONE
            return True

        # Step 8: Update status WARMING_UP -> STARTED
        if job.status == JobStatus.WARMING_UP:
            job.status = JobStatus.STARTED

    # Step 9: Get LLM API provider
    if provider is None:
        provider = get_statek_settings().default_llm_api_provider
    llm_api = LLM_API.get(provider_name=provider, model=job.model)

    # Step 10: Get next request parameters
    request = job.get_next_request()

    # Step 11: Run the request with LLM API - await response
    response = await llm_api.process_request(**request)

    # Update session_id if returned by the LLM API
    if response.session_id:
        job.session_id = response.session_id

    # Step 12: Add new log item using append_chat_log
    print(f"LLM Response:\n{response.text}\n{'-'*40}")
    job.append_chat_log(request, response.text)

    # Step 13: Return False
    return False


async def job_worker(semaphore, job: Job, provider: str = None):
    async with semaphore:
        await run_job_step(job, provider)

async def run_jobs_loop(max_concurrency: int = 100, provider: str = None,
    start_jobs_func: Callable = None):
    """
    Main loop responsible for processing registered jobs pending execution.
    
    The loop iterates over all active jobs so that none is starved off resources.
    It's assumed that only a single jobs-loop is running in one process.
    
    Args:
        max_concurrency: the execution concurrency level
        provider: the LLM API provider (or None for default)
        start_jobs_func: optional callable for starting new jobs
    """
    # Track pending job tasks to avoid exceeding max_concurrency
    pending_tasks = {}  # Dict[Job, asyncio.Task]
    ready_or_started_jobs = []
    semaphore = asyncio.Semaphore(max_concurrency)
    while True:
        # Step 1: TODO: Clean up finished tasks from pending_tasks
        
        # Step 2: Call start_jobs_func if provided
        if start_jobs_func is not None:
            available_capacity = max_concurrency - len(pending_tasks)
            start_jobs_func(available_capacity)
        
        # Step 3: Find jobs with status READY or STARTED, excluding jobs already pending
        if len(ready_or_started_jobs) == 0:
            ready_or_started_jobs = db0.filter(lambda job: job not in pending_tasks, 
                                               db0.find(Job, [JobStatus.READY, JobStatus.STARTED]))

    
        # Step 4: Submit run_job_step for jobs that aren't already pending
        # Make sure not to exceed max_concurrency
        if len(pending_tasks) < max_concurrency:
            for job in ready_or_started_jobs:
                # Create task for this job. Add all tasks to pending_tasks
                task = asyncio.create_task(job_worker(semaphore, job, provider))
                pending_tasks[job] = task
        
        # Step 5: Wait for completion of at least 1 job or sleep 300ms
        if pending_tasks:
            # Wait for at least one task to complete or timeout after 300ms
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
