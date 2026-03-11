"""Tests for system module."""

# pylint: disable=too-few-public-methods,duplicate-code

from typing import Tuple
import pytest
import dbzero as db0
from statek.system import (docs, brief, tool, create_tool, inject_context, find_tools,
                           select_tools, error_handler)
from statek.future import get_unpack_size, temporal, FutureResult
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

    def test_docs_with_no_docstring(self, capsys):
        """Test docs with a function that has no docstring prints error."""
        def no_doc_func():
            pass

        docs(no_doc_func)
        captured = capsys.readouterr()
        assert "no docstring" in captured.out

    def test_docs_with_class_no_docstring(self, capsys):
        """Test docs with a class that has no docstring prints error."""
        class NoDocClass:
            pass

        docs(NoDocClass)
        captured = capsys.readouterr()
        assert "no docstring" in captured.out

    def test_docs_with_method_no_docstring(self, capsys):
        """Test docs with a method that has no docstring prints error."""
        class SampleClass:
            """Sample class."""
            def no_doc_method(self):
                pass

        docs(SampleClass, "no_doc_method")
        captured = capsys.readouterr()
        assert "no docstring" in captured.out

    def test_docs_with_non_existent_method(self, capsys):
        """Test docs with a method that doesn't exist."""
        class SampleClass:
            """Sample class."""

        docs(SampleClass, "non_existent_method")
        captured = capsys.readouterr()
        assert "SampleClass has no method 'non_existent_method'" in captured.out

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

    def test_docs_with_builtin_function(self, capsys):
        """Test docs with a built-in function prints error (no parseable docstring)."""
        docs(len)
        captured = capsys.readouterr()
        assert "Error" in captured.out or "error" in captured.out.lower()

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


    def test_docs_with_string_prints_name_error(self, capsys):
        """Test docs with a string prints NameError (likely unresolved name)."""
        docs("unknown_tool")
        captured = capsys.readouterr()
        assert "NameError" in captured.out
        assert "unknown_tool" in captured.out

    def test_docs_with_int_instance(self, capsys):
        """Test docs with an int instance shows int class docs."""
        docs(42)
        captured = capsys.readouterr()
        assert "int" in captured.out


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


    def test_brief_with_string_prints_name_error(self, capsys):
        """Test brief with a string prints NameError (likely unresolved name)."""
        brief("unknown_tool")
        captured = capsys.readouterr()
        assert "NameError" in captured.out
        assert "unknown_tool" in captured.out


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

    def test_create_tool_ignores_extra_positional_args(self):
        """Test that created tool ignores positional args passed by the LLM."""
        def add(a, b):
            return a + b

        tool_func = create_tool('add_tool', add, 'Adds two numbers', {}, 5, 3)

        # LLM may pass positional args that the bound tool doesn't need
        assert tool_func(10) == 8
        assert tool_func(10, 20) == 8

    def test_create_tool_ignores_extra_keyword_args(self):
        """Test that created tool ignores keyword args passed by the LLM."""
        def add(a, b):
            return a + b

        tool_func = create_tool('add_tool', add, 'Adds two numbers', {}, 5, 3)

        assert tool_func(count=10) == 8
        assert tool_func(10, count=20) == 8


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


class TestSelectTools:
    """Test cases for select_tools function."""

    @staticmethod
    def _make_tools():
        """Create a small set of system and application tools for testing."""
        @tool(system=True)
        def sys_a(**kwargs):  # pylint: disable=unused-argument
            """System tool A."""

        @tool(system=True)
        def sys_b(**kwargs):  # pylint: disable=unused-argument
            """System tool B."""

        @tool
        def app_a(**kwargs):  # pylint: disable=unused-argument
            """Application tool A."""

        @tool
        def app_b(**kwargs):  # pylint: disable=unused-argument
            """Application tool B."""

        return sys_a, sys_b, app_a, app_b

    def test_select_tools_system_scope(self):
        """'SYSTEM' scope returns only tools marked system=True."""
        sys_a, sys_b, app_a, app_b = self._make_tools()
        result = select_tools([sys_a, sys_b, app_a, app_b], "SYSTEM")
        assert sys_a in result
        assert sys_b in result
        assert app_a not in result
        assert app_b not in result

    def test_select_tools_application_scope(self):
        """'APPLICATION' scope returns only non-system tools."""
        sys_a, sys_b, app_a, app_b = self._make_tools()
        result = select_tools([sys_a, sys_b, app_a, app_b], "APPLICATION")
        assert app_a in result
        assert app_b in result
        assert sys_a not in result
        assert sys_b not in result

    def test_select_tools_none_scope_returns_all(self):
        """None scope is a pass-through returning every provided tool."""
        sys_a, sys_b, app_a, app_b = self._make_tools()
        all_tools = [sys_a, sys_b, app_a, app_b]
        result = select_tools(all_tools, None)
        assert set(result) == set(all_tools)

    def test_select_tools_all_scope_returns_all(self):
        """"ALL" scope is a pass-through returning every provided tool."""
        sys_a, sys_b, app_a, app_b = self._make_tools()
        all_tools = [sys_a, sys_b, app_a, app_b]
        result = select_tools(all_tools, "ALL")
        assert set(result) == set(all_tools)

    def test_select_tools_preserves_order(self):
        """Output order matches input order for all scopes."""
        sys_a, sys_b, app_a, app_b = self._make_tools()
        ordered = [app_b, sys_a, app_a, sys_b]

        assert select_tools(ordered, "ALL") == ordered
        assert select_tools(ordered, "SYSTEM") == [sys_a, sys_b]
        assert select_tools(ordered, "APPLICATION") == [app_b, app_a]

    def test_select_tools_empty_input_returns_empty(self):
        """Empty input always yields an empty list regardless of scope."""
        assert select_tools([], "SYSTEM") == []
        assert select_tools([], "APPLICATION") == []
        assert select_tools([], "ALL") == []
        assert select_tools([], None) == []

    def test_select_tools_plain_callable_excluded_from_system(self):
        """Plain callables without tool_system are excluded from 'SYSTEM'."""
        def plain():
            """A plain function."""

        result = select_tools([plain], "SYSTEM")
        assert plain not in result

    def test_select_tools_plain_callable_included_in_application(self):
        """Plain callables without tool_system are included in 'APPLICATION'."""
        def plain():
            """A plain function."""

        result = select_tools([plain], "APPLICATION")
        assert plain in result

    def test_select_tools_plain_callable_included_in_all(self):
        """Plain callables without tool_system are included in 'ALL' and None."""
        def plain():
            """A plain function."""

        assert plain in select_tools([plain], "ALL")
        assert plain in select_tools([plain], None)

    def test_select_tools_mixed_with_plain_callable(self):
        """Mix of @tool functions and plain callables is filtered correctly."""
        sys_a, _, app_a, _ = self._make_tools()

        def plain():
            """Plain callable."""

        pool = [sys_a, app_a, plain]

        system_result = select_tools(pool, "SYSTEM")
        assert system_result == [sys_a]

        app_result = select_tools(pool, "APPLICATION")
        assert set(app_result) == {app_a, plain}

    def test_select_tools_all_system_with_system_scope(self):
        """All-system input with SYSTEM scope returns all of them."""
        sys_a, sys_b, _, _ = self._make_tools()
        result = select_tools([sys_a, sys_b], "SYSTEM")
        assert set(result) == {sys_a, sys_b}

    def test_select_tools_all_application_with_application_scope(self):
        """All-application input with APPLICATION scope returns all of them."""
        _, _, app_a, app_b = self._make_tools()
        result = select_tools([app_a, app_b], "APPLICATION")
        assert set(result) == {app_a, app_b}

    def test_select_tools_returns_list(self):
        """Return type is always a list."""
        sys_a, _, _, _ = self._make_tools()
        assert isinstance(select_tools([sys_a], "SYSTEM"), list)
        assert isinstance(select_tools([sys_a], "APPLICATION"), list)
        assert isinstance(select_tools([sys_a], "ALL"), list)
        assert isinstance(select_tools([sys_a], None), list)


# Module-level error handlers for TestToolErrorHandler
# (db0 @memo rejects nested/decorated functions as callable members)
@error_handler
def _eh_noop(context, error=None):  # pylint: disable=unused-argument
    """No-op error handler used by TestToolErrorHandler."""


@tool(error_handler=_eh_noop)
def _sync_tool_with_handler(**kwargs):  # pylint: disable=unused-argument
    """Sync tool that returns a fixed string."""
    return "the_result"


@tool(error_handler=_eh_noop)
async def _async_tool_with_handler(**kwargs):  # pylint: disable=unused-argument
    """Async tool that returns a fixed string."""
    return "async_result"


@tool(error_handler=_eh_noop)
def _tool_no_ctx(**kwargs):  # pylint: disable=unused-argument
    """Tool used without a job in context."""
    return "result"


class TestToolErrorHandler:
    """Tests for the error_handler argument on the @tool decorator."""

    def test_invalid_error_handler_raises_at_decoration_time(self):
        """Passing a callable that is not a valid error handler raises TypeError."""
        def not_a_handler(context, error=None):  # pylint: disable=unused-argument
            pass

        with pytest.raises(TypeError):
            @tool(error_handler=not_a_handler)
            def my_tool(**kwargs):  # pylint: disable=unused-argument
                """A tool."""
                return "value"

    def test_sync_tool_registers_handler_on_job(self, job_factory):
        """Sync tool with error_handler registers it on the current job with the return value."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841  # visible to find_locals via call stack

        result = _sync_tool_with_handler()

        assert result == "the_result"
        assert len(job.error_handlers) == 1
        assert job.error_handlers[0].error_handler is _eh_noop
        assert job.error_handlers[0].context == "the_result"

    @pytest.mark.asyncio
    async def test_async_tool_registers_handler_on_job(self, job_factory):
        """Async tool with error_handler registers it on the current job."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841  # visible to find_locals via call stack

        result = _async_tool_with_handler()

        assert result == "async_result"
        assert len(job.error_handlers) == 1
        assert job.error_handlers[0].error_handler is _eh_noop
        assert job.error_handlers[0].context == "async_result"

    def test_no_job_in_context_does_not_raise(self):
        """Tool with error_handler works normally when no job is available in context."""
        # No _STATEK_CTX in scope — should return normally without raising
        result = _tool_no_ctx()
        assert result == "result"


# ---------------------------------------------------------------------------
# Module-level fixtures for TestToolTemporalErrorHandler
# (db0 @memo rejects nested/decorated functions as callable members)
# ---------------------------------------------------------------------------

@db0.memo
class _TemporalResult(FutureResult):
    """Minimal FutureResult subclass for temporal error-handler tests."""
    def __init__(self, resolved_value):
        super().__init__(deps=None, state_num=0)
        self.resolved_value = resolved_value


def _temporal_complement(fut: _TemporalResult):
    return fut.resolved_value


def _temporal_condition(fut: _TemporalResult):  # pylint: disable=unused-argument
    return True


@error_handler
def _eh_temporal(context, error=None):  # pylint: disable=unused-argument
    """No-op error handler for temporal tests."""


@temporal(complement=_temporal_complement, condition=_temporal_condition)
@tool(error_handler=_eh_temporal)
def _temporal_tool(**kwargs):  # pylint: disable=unused-argument
    """Temporal sync tool with an error handler."""
    return _TemporalResult("resolved")


@temporal(complement=_temporal_complement, condition=_temporal_condition)
@tool(error_handler=_eh_temporal)
async def _async_temporal_tool(**kwargs):  # pylint: disable=unused-argument
    """Temporal async tool with an error handler."""
    return _TemporalResult("async_resolved")


class TestToolTemporalErrorHandler:
    """Tests for error_handler on @tool when the tool returns a FutureResult."""

    def test_handler_not_registered_before_value_access(self, job_factory):
        """Returning a FutureResult defers registration — job has no handlers yet."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841

        future = _temporal_tool()

        assert isinstance(future, FutureResult)
        assert len(job.error_handlers) == 0

    def test_handler_registered_on_value_access(self, job_factory):
        """Accessing .value on the FutureResult registers the handler with resolved value."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841

        future = _temporal_tool()
        value = future.value

        assert value == "resolved"
        assert len(job.error_handlers) == 1
        assert job.error_handlers[0].error_handler is _eh_temporal
        assert job.error_handlers[0].context == "resolved"

    @pytest.mark.asyncio
    async def test_async_temporal_tool_registers_on_value_access(self, job_factory):
        """Async temporal tool also defers registration until .value is accessed."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841

        future = _async_temporal_tool()
        assert len(job.error_handlers) == 0

        value = future.value

        assert value == "async_resolved"
        assert len(job.error_handlers) == 1
        assert job.error_handlers[0].context == "async_resolved"

    def test_handler_registered_only_once(self, job_factory):
        """Accessing .value a second time does not re-register the handler."""
        job = job_factory()
        _STATEK_CTX = {'job': job}  # noqa: F841

        future = _temporal_tool()
        res = future.value  # first access — registers
        res = future.value  # second access — must not add another handler
        assert res == "resolved"
        assert len(job.error_handlers) == 1
