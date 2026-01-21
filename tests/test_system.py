"""Tests for system module."""

# pylint: disable=too-few-public-methods

from statek.system import docs, tool

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
