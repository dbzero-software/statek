import ast
import sys
import inspect
import traceback
import asyncio
import builtins
from typing import Callable, Optional
from contextlib import contextmanager
import dbzero as db0

from statek.exceptions import FutureError
from statek.future import FutureResult


from statek.exceptions import FutureError
from statek.future import FutureResult
from statek.executors.job import Job, JobStatus
from statek.llm_api import LLM_API
from statek.settings import get_statek_settings, get_provider_settings, get_statek_logger, statek_log

STATEK_LOGGER = get_statek_logger()

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
        if isinstance(node.ctx, ast.Load) and node.id not in {'_wrap_param', '_smart_call'}:
            new_node = ast.Call(
                func=ast.Name(id='_wrap_param', ctx=ast.Load()),
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
def _setup_execution_context(job: Job, global_context: dict, local_context: dict):
    """
    Context manager to setup and cleanup execution environment.
    
    Args:
        job: The Job instance
        global_context: The global execution context dictionary
        local_context: The local execution context dictionary
    
    Yields:
        tuple: (custom_print_fn, custom_exit_fn) for reference
    """
    # Merge agent's private context if available
    if job.job_def.agent is not None and job.job_def.agent.context is not None:
        global_context.update(job.job_def.agent.context)
    for tool in job.job_def.agent._tools:
        global_context[tool.__name__] = tool
    
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
    local_context['_wrap_param'] = _wrap_param
    local_context['exit'] = custom_exit_fn
    global_context['print'] = custom_print_fn
    global_context['exit'] = custom_exit_fn
    global_context['_smart_call'] = _smart_call
    global_context['_wrap_param'] = _wrap_param
    
    try:
        yield custom_print_fn, custom_exit_fn
    finally:
        # Restore original built-in print
        builtins.print = original_print
        
        # Remove helpers from context
        for key in ['print', 'exit', '_smart_call', '_wrap_param']:
            if key in local_context:
                del local_context[key]


async def exec_step(code_str: str, job: Job, instr_num: Optional[int] = None) -> bool:
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
    # Global and local contexts needs to be dictionaries in order to be used with exec
    statek_log(f"Executing code step (instr_num={instr_num}):\n{code_str}")
    if job.py_env.global_state is None:
        global_context = globals()
    else:
        global_context = {key: value for key, value in job.py_env.global_state.items()}
    if job.py_env.local_state is None:
        local_context = {}
    else:
        local_context = {key: value for key, value in job.py_env.local_state.items()}

    # Use context manager to setup and cleanup execution environment
    try:
        with _setup_execution_context(job, global_context, local_context):
            tree = ast.parse(code_str)
            _ResilientTransformer().visit(tree)
            ast.fix_missing_locations(tree)

            # Determine starting instruction number
            start_instr = instr_num if instr_num is not None else 0
            
            for idx, node in enumerate(tree.body):
                # Skip instructions before the starting point
                if idx < start_instr:
                    continue
                    
                try:
                    wrapper = ast.Module(body=[node], type_ignores=[])
                    code_obj = compile(wrapper, filename="<string>", mode="exec")
                    exec(code_obj, global_context, local_context)
                    # Check for exit signal
                    if job.py_env.exit_status is not None:
                        break
                    # Save updated local context back to job
                except FutureError as e:
                    # Decorate FutureError with instruction number if not already set
                    if e.instr_num is None:
                        e.instr_num = idx
                    # Re-raise the decorated exception
                    raise
    finally:
        # Always save the local state, even if exception is raised
        # This happens after the context manager cleanup, so helper functions are removed
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
        job.set_status(JobStatus.STARTED)
    else:
        # Step 4: Update status READY -> WARMING_UP or SUSPENDED -> STARTED
        # Store whether we're resuming from SUSPENDED before changing status
        resuming_from_suspended = job.status == JobStatus.SUSPENDED
        
        if job.status == JobStatus.READY:
            job.set_status(JobStatus.WARMING_UP)
        elif job.status == JobStatus.SUSPENDED:
            job.set_status(JobStatus.STARTED)

        # Step 5: Execute the code using exec_step
        # Pass next_instr_num if resuming from SUSPENDED
        try:
            instr_num = job.next_instr_num if resuming_from_suspended else None
            not_exited = await exec_step(code, job, instr_num)
            # Clear continuation state after successful execution
            if resuming_from_suspended:
                job.awaited_result = None
                job.next_instr_num = None
        except FutureError as e:
            # Step 5a: Handle FutureError - suspend job
            job.awaited_result = e.future_result
            job.next_instr_num = e.instr_num
            # Change status STARTED -> SUSPENDED (no change for WARMING_UP)
            if job.status == JobStatus.STARTED:
                job.set_status(JobStatus.SUSPENDED)
            return False

        # Step 6 & 7: Check if code has finished (exit_status not None)
        if job.py_env.exit_status is not None:
            job.set_status(JobStatus.DONE)
            return True

        # Step 8: Update status WARMING_UP -> STARTED
        if job.status == JobStatus.WARMING_UP:
            job.set_status(JobStatus.STARTED)

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
    statek_log(f"LLM Response:\n{response.text}")
    job.append_chat_log(request, response.text)

    # Step 13: Return False
    return False


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
        statek_log(f"Found {len(suspended_jobs)} suspended jobs")
    for job in suspended_jobs:
        # Check if the job has an awaited_result and if its condition is satisfied
        condition_met = job.awaited_result.check_condition()
        if job.awaited_result is not None and condition_met:
            # Change status from SUSPENDED to STARTED
            job.set_status(JobStatus.STARTED)


async def job_worker(semaphore, job: Job, provider: str = None):
    async with semaphore:
        try:
            await run_job_step(job, provider)
        except Exception as e:
            # If job fails, set status to DONE
            print(f"Job {db0.uuid(job)} failed with error: {e}")
            traceback.print_exc()
            job.set_status(JobStatus.DONE)

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
    semaphore = asyncio.Semaphore(max_concurrency)
    while True:
        # Step 1: Unsuspend jobs whose continuation conditions are satisfied
        unsuspend_jobs()
        
        # Step 2: Call start_jobs_func if provided
        if start_jobs_func is not None:
            available_capacity = max_concurrency - len(pending_tasks)
            start_jobs_func(available_capacity)
        
        # Step 3: Find jobs with status READY or STARTED, excluding jobs already pending
        ready_or_started_jobs = db0.filter(lambda found_job: found_job not in pending_tasks,
                                            db0.find(Job, [JobStatus.READY, JobStatus.STARTED]))
        # Step 4: Submit run_job_step for jobs that aren't already pending
        # Make sure not to exceed max_concurrency
        if len(pending_tasks) < max_concurrency:
            statek_log("Adding new jobs to pending tasks")
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


async def run_agentic_loop(agent: 'Agent', warmup_code: str,
                           task_queue_size_func: Callable, max_concurrency: int = 100,
                           provider: str = None):
    """
    Helper function to start listening on arriving new tasks (e.g incoming user messages) 
    and process them with a specific agent such as Coordinator or MessageDispatcher. 
    
    This function can be used as the agentic system's main loop - where the incoming user 
    messages are processed with a specific specialized agent. Internally the function calls 
    `run_jobs_loop` and runs indefinitely.
    
    Args:
        agent: the Agent instance (e.g. Coordinator or MessageDispatcher)
        warmup_code: the agent's warmup code (e.g. "user, message = fetch_next_message()")
        task_queue_size_func: a function for calculating the number of queued tasks 
                              (e.g. incoming messages) awaiting processing
        max_concurrency: maximum number of concurrent jobs (default: 100)
        provider: the default LLM provider (or None for default)
    
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
    from statek.executors.job import JobDef
    from statek.pyenv import PyEnv
    statek_log("Starting agentic loop...")
    def start_jobs_func(capacity: int):
        """
        Internal function to create new jobs based on available capacity and pending tasks.
        
        This function:
        1. Calls task_queue_size() to check the number of awaiting tasks
        2. Uses db0.find to identify the number of agent-related jobs in READY or WARMING_UP state
        3. Creates N new agent jobs where N = min(capacity, task_queue_size - ready_jobs)
        """
        if capacity <= 0:
            return
        
        # Get the number of awaiting tasks
        num_awaiting_tasks = task_queue_size_func()
        
        if num_awaiting_tasks <= 0:
            return
        
        # Find the number of jobs related to this agent that are in READY or WARMING_UP state
        ready_jobs = db0.filter(
            lambda job: job.job_def.agent == agent,
            db0.find(Job, [JobStatus.READY, JobStatus.WARMING_UP])
        )
        #FIXME: Change when len of filter returns correct value
        ready_jobs_count = len(list(ready_jobs))
        
        # Calculate how many new jobs we need to create
        # N = min(capacity, task_queue_size - ready_jobs)
        jobs_to_create = min(capacity, num_awaiting_tasks - ready_jobs_count)
        
        if jobs_to_create <= 0:
            return
        
        # Create the new jobs
        statek_log(f"Creating {jobs_to_create} new jobs for agent {agent.role}")
        for _ in range(jobs_to_create):
            # Create job definition
            job_def = JobDef(
                agent=agent,
                job_params=None,
                warmup_code=warmup_code
            )
            
            # Create PyEnv with agent's context if available
            pyenv = PyEnv(local_state={})
            
            # Get model info from settings if provider not specified
            if provider is None:
                settings = get_statek_settings()
                model_to_use = settings.default_model
            else:
                settings = get_provider_settings(provider)
                model_to_use = settings.default_model
            
            # Create the job
            Job(
                job_def=job_def,
                model_family=provider or "default",
                model=model_to_use,
                job_status=JobStatus.READY,
                py_env=pyenv
            )
    
    # Run the jobs loop indefinitely with our custom start_jobs_func
    await run_jobs_loop(
        max_concurrency=max_concurrency,
        provider=provider,
        start_jobs_func=start_jobs_func
    )
