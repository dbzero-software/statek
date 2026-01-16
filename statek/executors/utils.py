import ast
import sys
import inspect
import traceback
import dbzero as db0


from statek.exceptions import FutureError
from statek.executors.job import Job

def wrap_param(param):
    #FIXME: consider making this a proper FutureResult class
    """Wrap parameter to raise FutureError when accessed."""
    return param

def _smart_call(func, *args, **kwargs):
    """Wraps arguments unless the target function expects FutureResult."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            param = sig.parameters[name]
            anno = param.annotation
            # Check if type hint matches 'FutureResult' by name or type
            if getattr(anno, "__name__", str(anno)) != 'FutureResult':
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
    
    # Inject the helper into the execution context


    
    # Inject custom print and exit functions
    local_context['print'] = lambda *args, **kwargs: custom_print(job, *args, **kwargs)
    local_context['_smart_call'] = _smart_call
    local_context['wrap_param'] = wrap_param
    local_context['exit'] = lambda status=None: custom_exit(job, status)
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
    # remove helper from context
    del local_context['print']
    del local_context['exit']
    del local_context['_smart_call']
    del local_context['wrap_param']
    job.py_env.local_state = local_context
    return job.py_env.exit_status is None




