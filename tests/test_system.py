"""Tests for system module."""

# pylint: disable=too-few-public-methods

from typing import Tuple
import pytest
from statek.system import docs, tool, create_tool
from statek.future import get_unpack_size
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
        """Test docs with a function that has no docstring."""
        def no_doc_func():
            pass

        docs(no_doc_func)
        captured = capsys.readouterr()
        assert "No docstring found for no_doc_func" in captured.out

    def test_docs_with_class_no_docstring(self, capsys):
        """Test docs with a class that has no docstring."""
        class NoDocClass:
            pass

        docs(NoDocClass)
        captured = capsys.readouterr()
        assert "No docstring found for NoDocClass" in captured.out

    def test_docs_with_method_no_docstring(self, capsys):
        """Test docs with a method that has no docstring."""
        class SampleClass:
            """Sample class."""
            def no_doc_method(self):
                pass

        docs(SampleClass, "no_doc_method")
        captured = capsys.readouterr()
        assert "No docstring found for SampleClass.no_doc_method" in captured.out

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

    def test_docs_with_builtin_function(self, capsys):
        """Test docs with a built-in function."""
        docs(len)
        captured = capsys.readouterr()
        # Built-in functions have docstrings
        assert "len" in captured.out or "Return the number" in captured.out

    def test_docs_with_tool_function(self, capsys):
        """Test docs with a function decorated by @tool."""
        @tool
        def decorated_func():
            """This function is decorated."""

        docs(decorated_func)
        captured = capsys.readouterr()
        assert "This function is decorated." in captured.out


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
        assert result == 'def my_tool()'

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
