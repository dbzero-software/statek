from typing import Any, Callable, Tuple
import functools
import inspect
from .future import get_any_future, get_all_future

def tool(f):
    """Marks a function as a tool for LLM agent."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@tool
def docs(tool_or_class: type | Callable, method_name: str = None):
    """Prints the docstring associated with either a tool, class or its member name.

    Args:
        tool_or_class: A function, tool, or class to get documentation for.
        method_name: Optional method name if tool_or_class is a class. If provided,
                    returns documentation for the specific method of the class.

    Returns:
        None. Prints the documentation directly to console.

    Examples:
        docs(add)  # Get documentation for the 'add' function
        docs(User, "send_message")  # Get documentation for User.send_message method
    """
    # If method_name is provided, get the method from the class
    if method_name is not None:
        if not isinstance(tool_or_class, type):
            print(f"Error: {tool_or_class} is not a class")
            return

        # Get the method from the class
        if not hasattr(tool_or_class, method_name):
            print(f"Error: {tool_or_class.__name__} has no method '{method_name}'")
            return

        target = getattr(tool_or_class, method_name)
    else:
        target = tool_or_class

    # Get the docstring
    docstring = inspect.getdoc(target)

    if docstring:
        print(docstring)
    else:
        if method_name:
            print(f"No docstring found for {tool_or_class.__name__}.{method_name}")
        else:
            name = getattr(tool_or_class, '__name__', str(tool_or_class))
            print(f"No docstring found for {name}")


@tool
def get_any(*args: Any) -> Any:
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
def get_all(*args: Any) -> Tuple[Any]:
    """Waits until evaluation of all given values completes and combines the results.
    
    Args:
        *args: Variable number of values to evaluate.
    
    Returns:
        A tuple containing all the evaluated values, in order.
    
    Examples:
        results = get_all(value1, value2)
    """
    return get_all_future(*args)
