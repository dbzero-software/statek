"""Tests for system module."""

# pylint: disable=too-few-public-methods,duplicate-code

from typing import Tuple
import pytest
import dbzero as db0
from statek.system import docs, brief, tool, create_tool, inject_context, find_tools
from statek.future import get_unpack_size, temporal, FutureResult
from statek.docstring import DocstringParseError
from statek.utils import format_callable_decl

class TestDocs:
    """Test cases for docs function."""

    def test_docs_with_function(self, capsys):
        """Test docs with a function that has a docstring."""
        def sample_func():
            """This is a sample function."""

        docs(sample_func)
        captured = capsys.readouterr()
        assert "This is a sample function." in captured.out

    def test_docs_with_class(self, capsys):
        """Test docs with a class that has a docstring."""
        class SampleClass:
            """This is a sample class."""

        docs(SampleClass)
        captured = capsys.readouterr()
        assert "This is a sample class." in captured.out

    def test_docs_with_class_method(self, capsys):
        """Test docs with a class method that has a docstring."""
        class SampleClass:
            """This is a sample class."""
            def sample_method(self):
                """This is a sample method."""

        docs(SampleClass, "sample_method")
        captured = capsys.readouterr()
        assert "This is a sample method." in captured.out

    def test_docs_with_multiline_docstring(self, capsys):
        """Test docs with a multiline docstring."""
        def complex_func():
            """
            This is a complex function.

            It has multiple lines in its docstring.
            Including detailed explanations.
            """

        docs(complex_func)
        captured = capsys.readouterr()
        assert "This is a complex function." in captured.out
        assert "It has multiple lines in its docstring." in captured.out

    def test_docs_with_no_docstring(self):
        """Test docs with a function that has no docstring raises error."""
        def no_doc_func():
            pass

        with pytest.raises(DocstringParseError) as exc_info:
            docs(no_doc_func)
        assert "no docstring" in str(exc_info.value)

    def test_docs_with_class_no_docstring(self):
        """Test docs with a class that has no docstring raises error."""
        class NoDocClass:
            pass

        with pytest.raises(DocstringParseError) as exc_info:
            docs(NoDocClass)
        assert "no docstring" in str(exc_info.value)

    def test_docs_with_method_no_docstring(self):
        """Test docs with a method that has no docstring raises error."""
        class SampleClass:
            """Sample class."""
            def no_doc_method(self):
                pass

        with pytest.raises(DocstringParseError) as exc_info:
            docs(SampleClass, "no_doc_method")
        assert "no docstring" in str(exc_info.value)

    def test_docs_with_non_existent_method(self, capsys):
        """Test docs with a method that doesn't exist."""
        class SampleClass:
            """Sample class."""

        docs(SampleClass, "non_existent_method")
        captured = capsys.readouterr()
        assert "Error: SampleClass has no method 'non_existent_method'" in captured.out

    def test_docs_with_method_on_non_class(self, capsys):
        """Test docs with method_name provided but tool_or_class is not a class."""
        def regular_function():
            """Just a function."""

        docs(regular_function, "some_method")
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "is not a class" in captured.out

    def test_docs_with_staticmethod(self, capsys):
        """Test docs with a static method."""
        class SampleClass:
            """Sample class."""
            @staticmethod
            def static_method():
                """This is a static method."""
                return "static"

        docs(SampleClass, "static_method")
        captured = capsys.readouterr()
        assert "This is a static method." in captured.out

    def test_docs_with_classmethod(self, capsys):
        """Test docs with a class method."""
        class SampleClass:
            """Sample class."""
            @classmethod
            def class_method(cls):
                """This is a class method."""
                return "classmethod"

        docs(SampleClass, "class_method")
        captured = capsys.readouterr()
        assert "This is a class method." in captured.out

    def test_docs_with_builtin_function(self):
        """Test docs with a built-in function raises error (no parseable docstring)."""
        with pytest.raises(DocstringParseError):
            docs(len)

    def test_docs_with_tool_function(self, capsys):
        """Test docs with a function decorated by @tool."""
        @tool
        def decorated_func(x: int, **kwargs):  # pylint: disable=unused-argument
            """This function is decorated.

            Args:
                x (int): An input value.

            Returns:
                None: No return value.
            """

        docs(decorated_func)
        captured = capsys.readouterr()
        assert "This function is decorated." in captured.out
        # kwargs should not appear in output
        assert "kwargs" not in captured.out

    def test_docs_with_temporal_function(self, capsys):
        """Test docs with a temporal function shows complement return type and description."""
        def get_result(fut: FutureResult) -> str:  # pylint: disable=unused-argument
            """Retrieve the completed result.

            Returns:
                str: The final computed value.
            """
            return "result"

        def check_condition(fut: FutureResult) -> bool:  # pylint: disable=unused-argument
            return True

        @temporal(complement=get_result, condition=check_condition)
        def my_temporal_func(x: int, **kwargs) -> FutureResult:  # pylint: disable=unused-argument
            """A temporal function that does something.

            Args:
                x (int): The input value.

            Returns:
                FutureResult: A future result representing the pending task.
            """
            return FutureResult(deps=None, state_num=0)

        docs(my_temporal_func)
        captured = capsys.readouterr()

        # Should show complement's return type (str), not FutureResult
        assert "-> str" in captured.out
        # Should not show FutureResult in signature
        assert "FutureResult" not in captured.out.split('\n')[0]
        # Returns description should come from complement, not the temporal function
        assert "The final computed value" in captured.out
        assert "A future result representing the pending task" not in captured.out
        # Should include the docstring content
        assert "A temporal function" in captured.out
        # Should not show kwargs in signature
        assert "kwargs" not in captured.out


class TestBrief:
    """Test cases for brief function."""

    def test_brief_with_function(self, capsys):
        """Test brief with a function."""
        @tool
        def sample_func(x: int, **kwargs):  # pylint: disable=unused-argument
            """A sample function.

            Args:
                x (int): Input value.

            Returns:
                int: Output value.
            """

        brief(sample_func)
        captured = capsys.readouterr()

        # Brief format: no 'def', no types in signature
        assert "sample_func(x)" in captured.out
        assert "A sample function." in captured.out
        assert "Returns: Output value." in captured.out
        assert "def " not in captured.out

    def test_brief_with_object_instance(self, capsys):
        """Test brief with an object instance gets its class docs."""
        class SampleClass:
            """A sample class for testing.

            Attributes:
                value (int): The stored value.
            """

        obj = SampleClass()
        brief(obj)
        captured = capsys.readouterr()

        assert "SampleClass" in captured.out
        assert "A sample class for testing." in captured.out

    def test_brief_with_class_method(self, capsys):
        """Test brief with a class and method name."""
        class Calculator:
            """A calculator class."""
            def add(self, a: int, b: int) -> int:
                """Add two numbers.

                Args:
                    a (int): First number.
                    b (int): Second number.

                Returns:
                    int: The sum.
                """
                return a + b

        brief(Calculator, "add")
        captured = capsys.readouterr()

        assert "add(a, b)" in captured.out
        assert "Add two numbers." in captured.out
        assert "def " not in captured.out


class TestCreateTool:
    """Test cases for create_tool function."""

    def test_create_tool_basic(self):
        """Test creating a basic tool with args."""
        def add(a, b):
            return a + b

        context = {}
        tool_func = create_tool('add_tool', add, 'Adds two numbers', context, 5, 3)

        assert tool_func.__name__ == 'add_tool'
        assert tool_func.__doc__ == 'Adds two numbers'
        assert tool_func() == 8

        ctx_tool = context['add_tool']

        assert ctx_tool.__name__ == 'add_tool'
        assert ctx_tool.__doc__ == 'Adds two numbers'
        assert ctx_tool() == 8

    def test_create_tool_zero_arguments(self):
        """Test creating a tool from a callable with no arguments."""
        def get_constant():
            return 42

        tool_func = create_tool('constant_tool', get_constant, 'Returns constant', {})

        assert tool_func() == 42

    def test_create_tool_with_mixed_args_kwargs(self):
        """Test creating a tool with both args and kwargs."""
        def format_string(template, name, age):
            return template.format(name=name, age=age)

        tool_func = create_tool(
            'format_tool',
            format_string,
            'Formats a string',
            {},
            'Name: {name}, Age: {age}',
            name='Alice',
            age=30
        )

        assert tool_func() == 'Name: Alice, Age: 30'

    def test_create_tool_with_docs(self, capsys):
        """Test that created tool works with docs function."""
        def sample_func(x):
            return x * 2

        tool_func = create_tool('sample_tool', sample_func, 'Doubles a number', {}, 5)

        docs(tool_func)
        captured = capsys.readouterr()
        assert 'Doubles a number' in captured.out

    def test_create_tool_with_format_callable_decl(self):
        """Test that created tool works with format_callable_decl."""
        def sample_func(x):
            return x * 2

        tool_func = create_tool('my_tool', sample_func, 'My custom tool', {}, 5)

        result = format_callable_decl(tool_func)
        assert 'my_tool' in result
        assert result.startswith('def my_tool()')
        assert 'My custom tool' in result

    def test_create_tool_with_exception(self):
        """Test that exceptions from the callable are properly raised."""
        def divide(a, b):
            return a / b

        tool_func = create_tool('divide_tool', divide, 'Divides numbers', {}, 10, 0)

        with pytest.raises(ZeroDivisionError):
            tool_func()

    def test_create_tool_with_default_arguments(self):
        """Test creating a tool from callable with default arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        tool_func = create_tool('greet_tool', greet, 'Greets someone', {}, 'Alice')

        assert tool_func() == "Hello, Alice!"

    def test_create_tool_exception_if_exists(self):
        """Test creating a basic tool with args."""
        def add(a, b):
            return a + b

        context = {}

        create_tool('add_tool', add, 'Adds two numbers', context, 5, 3)

        with pytest.raises(ValueError):
            create_tool('add_tool', add, 'Adds two numbers', context, 5, 3)

        create_tool('add_tool2', add, 'Adds two numbers', context, 5, 3)

class TestGetUnpackSize:
    """Test cases for get_unpack_size function."""

    def test_get_unpack_size_with_tuple_annotation(self):
        """Test with a function returning a tuple with explicit size."""
        def returns_pair() -> Tuple[int, str]:
            return (1, "hello")

        assert get_unpack_size(returns_pair) == 2

    def test_get_unpack_size_with_no_annotation(self):
        """Test with a function that has no return type annotation."""
        def no_annotation():
            return (1, 2)

        assert get_unpack_size(no_annotation) is None

    def test_get_unpack_size_with_non_tuple_annotation(self):
        """Test with a function returning a non-tuple type."""
        def returns_int() -> int:
            return 42

        assert get_unpack_size(returns_int) is None

    def test_get_unpack_size_with_empty_tuple(self):
        """Test with a function returning an empty tuple annotation."""
        def returns_empty() -> Tuple[()]:
            return ()

        # Tuple[()] is actually just Tuple with no args
        result = get_unpack_size(returns_empty)
        assert result is None or result == 0

    def test_get_unpack_size_with_annotated_lambda(self):
        """Test with an annotated callable."""
        def returns_quad() -> Tuple[int, int, int, int]:
            return (1, 2, 3, 4)

        assert get_unpack_size(returns_quad) == 4

    def test_get_unpack_size_with_builtin_tuple(self):
        """Test with lowercase tuple annotation."""
        def returns_builtin_tuple() -> tuple[int, str]:
            return (1, "hello")

        assert get_unpack_size(returns_builtin_tuple) == 2


class TestToolEnumConversion:
    """Test cases for automatic str → db0.enum conversion in @tool decorator."""

    def test_string_converted_to_enum(self, db0_fixture):  # pylint: disable=unused-argument
        """String arg is converted to the corresponding enum value."""
        SeverityLevel = db0.enum("SeverityLevel", ["INFO", "WARNING", "ERROR"])

        @tool
        def alert(level: SeverityLevel, **kwargs):  # pylint: disable=unused-argument
            return level

        result = alert("INFO")
        assert result is SeverityLevel.INFO

    def test_enum_value_passes_through(self, db0_fixture):  # pylint: disable=unused-argument
        """Direct enum value is not modified."""
        SeverityLevel = db0.enum("SeverityLevel", ["INFO", "WARNING", "ERROR"])

        @tool
        def alert(level: SeverityLevel, **kwargs):  # pylint: disable=unused-argument
            return level

        result = alert(SeverityLevel.INFO)
        assert result is SeverityLevel.INFO

    def test_keyword_arg_conversion(self, db0_fixture):  # pylint: disable=unused-argument
        """Keyword arguments are also converted."""
        SeverityLevel = db0.enum("SeverityLevel", ["INFO", "WARNING", "ERROR"])

        @tool
        def alert(level: SeverityLevel, **kwargs):  # pylint: disable=unused-argument
            return level

        result = alert(level="ERROR")
        assert result is SeverityLevel.ERROR

    def test_invalid_enum_string_raises(self, db0_fixture):  # pylint: disable=unused-argument
        """An invalid enum string raises an exception."""
        SeverityLevel = db0.enum("SeverityLevel", ["INFO", "WARNING", "ERROR"])

        @tool
        def alert(level: SeverityLevel, **kwargs):  # pylint: disable=unused-argument
            return level

        with pytest.raises(Exception):
            alert("INVALID")


class TestToolBindByName:
    """Test cases for binding string arguments to local context variables."""

    def test_string_bound_to_context_variable(self):
        """String arg is resolved to local context variable on type mismatch."""
        class User:
            def __init__(self, name):
                self.name = name

        @tool
        def send_to(recipient: User, message: str, **kwargs):  # pylint: disable=unused-argument
            return recipient

        user = User("Alice")
        wrapped = inject_context(send_to, {"user": user})

        result = wrapped("user", "hello")
        assert result is user

    def test_string_not_bound_when_type_is_str(self):
        """String arg is NOT resolved when parameter type IS str."""
        @tool
        def greet(name: str, **kwargs):  # pylint: disable=unused-argument
            return name

        wrapped = inject_context(greet, {"name": "something else"})

        result = wrapped("name")
        assert result == "name"

    def test_string_not_bound_when_variable_not_in_context(self):
        """String arg is NOT resolved when variable not found in context."""
        class User:
            pass

        @tool
        def send_to(recipient: User, **kwargs):  # pylint: disable=unused-argument
            return recipient

        wrapped = inject_context(send_to, {"admin": User()})

        result = wrapped("user")
        assert result == "user"

    def test_keyword_arg_bound_to_context_variable(self):
        """Keyword string arg is resolved to context variable."""
        class User:
            def __init__(self, name):
                self.name = name

        @tool
        def send_to(recipient: User, message: str, **kwargs):  # pylint: disable=unused-argument
            return recipient

        user = User("Alice")
        wrapped = inject_context(send_to, {"user": user})

        result = wrapped(recipient="user", message="hello")
        assert result is user

    def test_mixed_binding_and_regular_args(self):
        """Only mismatched args are bound, str args stay as-is."""
        class User:
            def __init__(self, name):
                self.name = name

        @tool
        def send_to(recipient: User, message: str, **kwargs):  # pylint: disable=unused-argument
            return (recipient, message)

        user = User("Alice")
        wrapped = inject_context(send_to, {"user": user})

        result = wrapped("user", "hello")
        assert result[0] is user
        assert result[1] == "hello"

    def test_no_context_no_binding(self):
        """Without _local_context, no binding occurs."""
        class User:
            pass

        @tool
        def send_to(recipient: User, **kwargs):  # pylint: disable=unused-argument
            return recipient

        result = send_to("user")
        assert result == "user"


class TestToolSystemFlag:
    """Test cases for system=True parameter on @tool decorator."""

    def test_tool_without_system_flag_is_application(self):
        """@tool without system=True sets tool_system to False."""
        @tool
        def my_app_tool(**kwargs):  # pylint: disable=unused-argument
            """An application tool."""

        assert my_app_tool.tool_system is False

    def test_tool_with_system_true(self):
        """@tool(system=True) sets tool_system to True."""
        @tool(system=True)
        def my_system_tool(**kwargs):  # pylint: disable=unused-argument
            """A system tool."""

        assert my_system_tool.tool_system is True

    def test_tool_with_system_false_explicit(self):
        """@tool(system=False) sets tool_system to False."""
        @tool(system=False)
        def my_app_tool(**kwargs):  # pylint: disable=unused-argument
            """An application tool."""

        assert my_app_tool.tool_system is False

    def test_tool_without_kwargs_raises(self):
        """@tool(system=True) still enforces **kwargs requirement."""
        with pytest.raises(TypeError, match="must accept \\*\\*kwargs"):
            @tool(system=True)
            def bad_tool(x: int):  # pylint: disable=unused-argument
                """Missing **kwargs."""


class TestFindTools:
    """Test cases for find_tools function."""

    def test_find_tools_none_returns_all(self):
        """find_tools(None) returns all registered tools."""
        @tool
        def _app1(**kwargs):  # pylint: disable=unused-argument
            """App tool 1."""

        @tool(system=True)
        def _sys1(**kwargs):  # pylint: disable=unused-argument
            """System tool 1."""

        all_tools = list(find_tools())
        assert _app1 in all_tools
        assert _sys1 in all_tools

    def test_find_tools_system_scope(self):
        """find_tools("SYSTEM") returns only system tools."""
        @tool(system=True)
        def _sys2(**kwargs):  # pylint: disable=unused-argument
            """System tool 2."""

        @tool
        def _app2(**kwargs):  # pylint: disable=unused-argument
            """App tool 2."""

        system_tools = list(find_tools("SYSTEM"))
        assert _sys2 in system_tools
        assert _app2 not in system_tools
        assert all(t.tool_system for t in system_tools)

    def test_find_tools_application_scope(self):
        """find_tools("APPLICATION") returns only non-system tools."""
        @tool
        def _app3(**kwargs):  # pylint: disable=unused-argument
            """App tool 3."""

        @tool(system=True)
        def _sys3(**kwargs):  # pylint: disable=unused-argument
            """System tool 3."""

        app_tools = list(find_tools("APPLICATION"))
        assert _app3 in app_tools
        assert _sys3 not in app_tools
        assert all(not t.tool_system for t in app_tools)

    def test_tools_in_system_scope(self):
        """docs appears in find_tools("SYSTEM")."""
        system_tools = list(find_tools("SYSTEM"))
        assert docs in system_tools
        assert brief in system_tools

    def test_tools_not_in_application_scope(self):
        """docs does not appear in find_tools("APPLICATION")."""
        app_tools = list(find_tools("APPLICATION"))
        assert docs not in app_tools
        assert brief not in app_tools

    def test_list_of_examples_and_show_example_in_system_scope(self):
        """list_of_examples and show_example appear in find_tools("SYSTEM")."""
        # Import triggers registration
        from statek.agents.list_of_examples import list_of_examples, show_example  # pylint: disable=import-outside-toplevel
        system_tools = list(find_tools("SYSTEM"))
        assert list_of_examples in system_tools
        assert show_example in system_tools
