"""Tests for CallSpec.format method."""

# pylint: disable=unused-argument

from statek.utils import CallSpec


def test_format_no_args(db0_fixture):
    """No-arg call is formatted as func()."""
    cs = CallSpec(id="S-001", func_name="my_func", args=[], kwargs={})
    assert cs.format() == "my_func()"


def test_format_none_args_treated_as_empty(db0_fixture):
    """None args and kwargs are treated the same as empty."""
    cs = CallSpec(id="S-001", func_name="my_func", args=None, kwargs=None)
    assert cs.format() == "my_func()"


def test_format_positional_args(db0_fixture):
    """Positional args are repr'd and comma-separated."""
    cs = CallSpec(id="S-001", func_name="func", args=["hello", 42], kwargs={})
    assert cs.format() == "func('hello', 42)"


def test_format_kwargs(db0_fixture):
    """Kwargs are formatted as key=repr(val)."""
    cs = CallSpec(id="S-001", func_name="search", args=[], kwargs={"query": "test", "limit": 5})
    result = cs.format()
    assert result.startswith("search(")
    assert "query='test'" in result
    assert "limit=5" in result


def test_format_mixed_args_and_kwargs(db0_fixture):
    """Both positional args and kwargs are included."""
    cs = CallSpec(id="S-001", func_name="func", args=["pos"], kwargs={"k": True})
    result = cs.format()
    assert result.startswith("func('pos'")
    assert "k=True" in result


def test_format_bool_kwarg(db0_fixture):
    """Boolean kwarg is formatted correctly."""
    cs = CallSpec(id="S-001", func_name="toggle", args=[], kwargs={"enabled": False})
    assert cs.format() == "toggle(enabled=False)"
