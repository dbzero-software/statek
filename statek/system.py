import functools

def tool(f):
    """
    Marks a function as a tool for LLM agent.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
