"""Tests for format_default_llm_repr function."""

# pylint: disable=too-few-public-methods

import dataclasses
from datetime import datetime
from decimal import Decimal

import dbzero as db0

from statek.utils import format_default_llm_repr


# --- Simple scalar values use format_llm_repr (not __str__) ---

def test_int():
    assert format_default_llm_repr(42) == "42"


def test_float():
    assert format_default_llm_repr(3.14) == "3.14"


def test_bool_true():
    assert format_default_llm_repr(True) == "True"


def test_bool_false():
    assert format_default_llm_repr(False) == "False"


def test_string_is_quoted():
    """Strings are quoted (format_llm_repr path), not passed through str()."""
    assert format_default_llm_repr("hello") == '"hello"'


def test_string_empty():
    assert format_default_llm_repr("") == '""'


def test_decimal():
    assert format_default_llm_repr(Decimal("3.14")) == "3.14"


def test_datetime():
    dt = datetime(2026, 1, 1, 12, 0)
    assert format_default_llm_repr(dt) == 'datetime("2026-01-01 12:00")'


# --- Collections use format_llm_repr ---

def test_list():
    assert format_default_llm_repr([1, 2, 3]) == "[1,2,3]"


def test_tuple():
    assert format_default_llm_repr((1, 2)) == "(1,2)"


def test_dict():
    assert format_default_llm_repr({"a": 1}) == '{"a":1}'


def test_set():
    assert format_default_llm_repr({99}) == "{99}"


def test_frozenset():
    assert format_default_llm_repr(frozenset({2})) == "{2}"


# --- Object: __llm_repr__ takes precedence ---

def test_object_with_llm_repr():
    class Widget:
        def __llm_repr__(self):
            return "Widget(custom)"

    assert format_default_llm_repr(Widget()) == "Widget(custom)"


def test_llm_repr_takes_precedence_over_str():
    class Widget:
        def __llm_repr__(self):
            return "llm_result"
        def __str__(self):
            return "str_result"

    assert format_default_llm_repr(Widget()) == "llm_result"


# --- Object: explicit __str__ used when no __llm_repr__ ---

def test_object_with_explicit_str():
    class Widget:
        def __str__(self):
            return "Widget(str)"

    assert format_default_llm_repr(Widget()) == "Widget(str)"


# --- Object: falls back to format_llm_repr ---

def test_plain_object_uses_format_llm_repr():
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    assert format_default_llm_repr(Point(3, 4)) == "Point(x=3,y=4)"


def test_plain_object_no_members():
    class Empty:
        pass

    assert format_default_llm_repr(Empty()) == "Empty()"


def test_dataclass_uses_format_llm_repr():
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    assert format_default_llm_repr(Point(1, 2)) == "Point(x=1,y=2)"


# --- db0.memo objects ---

def test_db0_memo_with_llm_repr(db0_fixture):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str

        def __llm_repr__(self):
            return f"Config[{self.name}]"

    assert format_default_llm_repr(Config(name="test")) == "Config[test]"


def test_db0_memo_without_custom_methods(db0_fixture):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str
        value: int

    assert format_default_llm_repr(Config(name="alpha", value=7)) == 'Config(name="alpha",value=7)'


# --- Exported from statek package ---

def test_exported_from_statek_package():
    from statek import format_default_llm_repr as fn  # pylint: disable=import-outside-toplevel
    assert fn(42) == "42"
