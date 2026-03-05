"""Tests for the web_ui agents page helper functions."""

from web_ui.pages.agents import _get_tool_info, _get_tool_signature


def _tool_with_full_docs(value: str) -> str:
    """Return the value uppercased.

    Args:
        value (str): The string to uppercase.

    Returns:
        str: The uppercased string.
    """
    return value.upper()


def _tool_no_args() -> None:
    """Do nothing."""


def _tool_no_docstring(x):  # pylint: disable=unused-argument
    pass


def _tool_with_default(name: str, limit: int = 10) -> str:  # pylint: disable=unused-argument
    """Fetch items by name.

    Args:
        name (str): The name to look up.
        limit (int): Maximum results.

    Returns:
        str: Result string.
    """
    return name


class TestGetToolInfo:
    def test_brief_contains_signature_and_description(self):
        brief, _, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'def _tool_with_full_docs' in brief
        assert 'Return the value uppercased.' in brief

    def test_brief_does_not_contain_args_section(self):
        brief, _, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'Args:' not in brief

    def test_full_docs_contains_args_section(self):
        _, full_docs, error = _get_tool_info(_tool_with_full_docs)
        assert error is None
        assert 'def _tool_with_full_docs' in full_docs
        assert 'Args:' in full_docs
        assert 'value' in full_docs
        assert 'Returns:' in full_docs

    def test_no_docstring_returns_none_none(self):
        brief, full_docs, error = _get_tool_info(_tool_no_docstring)
        assert brief is None
        assert full_docs is None
        assert error is not None

    def test_non_callable_returns_none_none(self):
        brief, full_docs, error = _get_tool_info("not_a_function")
        assert brief is None
        assert full_docs is None
        assert error is None


class TestGetToolSignature:
    def test_simple_param(self):
        assert _get_tool_signature(_tool_with_full_docs) == '_tool_with_full_docs(value)'

    def test_no_params(self):
        assert _get_tool_signature(_tool_no_args) == '_tool_no_args()'

    def test_default_param_included(self):
        sig = _get_tool_signature(_tool_with_default)
        assert sig == '_tool_with_default(name, limit)'

    def test_non_callable_returns_str(self):
        result = _get_tool_signature("mytool")  # type: ignore[arg-type]
        assert result == 'mytool'
