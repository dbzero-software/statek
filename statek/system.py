from typing import Any, Callable, Tuple, Dict
import functools
import inspect
from .future import get_any_future, get_all_future

def tool(f):
    """Marks a function as a tool for LLM agent."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


# pylint: disable=redefined-builtin
def create_tool(tool_name: str, callable: Callable, docstring: str,
                context: Dict, *args, **kwargs) -> Callable:
    """Creates a zero-argument tool function from a callable with bound
    arguments and injects it into context.

    Args:
        tool_name: The name to assign to the created tool function.
        callable: The callable to wrap into a tool.
        docstring: The docstring to assign to the tool.
        context: The context to put the created tool into.
        *args: Positional arguments to bind to the callable.
        **kwargs: Keyword arguments to bind to the callable.

    Returns:
        A zero-argument callable with the specified name and docstring.
    """
    def tool_func():
        # Call the function with bound arguments
        return callable(*args, **kwargs)

    # Set the function name and docstring
    tool_func.__name__ = tool_name
    tool_func.__doc__ = docstring

    # Apply the @tool decorator
    new_tool = tool(tool_func)
    context[tool_name] = new_tool
    return new_tool


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
