"""Tests for statek_print function."""

# pylint: disable=too-few-public-methods

import dataclasses
import io
from datetime import datetime
from decimal import Decimal

import dbzero as db0

from statek.utils import statek_print


# --- Simple scalar types use format_llm_repr ---

def test_print_int(capsys):
    statek_print(42)
    assert capsys.readouterr().out == "42\n"


def test_print_float(capsys):
    statek_print(3.14)
    assert capsys.readouterr().out == "3.14\n"


def test_print_bool_true(capsys):
    statek_print(True)
    assert capsys.readouterr().out == "True\n"


def test_print_bool_false(capsys):
    statek_print(False)
    assert capsys.readouterr().out == "False\n"


def test_print_string(capsys):
    statek_print("hello")
    assert capsys.readouterr().out == '"hello"\n'


def test_print_decimal(capsys):
    statek_print(Decimal("3.14"))
    assert capsys.readouterr().out == "3.14\n"


def test_print_datetime(capsys):
    dt = datetime(2026, 1, 1, 12, 0)
    statek_print(dt)
    assert capsys.readouterr().out == 'datetime("2026-01-01 12:00")\n'


# --- Collections use format_llm_repr ---

def test_print_list(capsys):
    statek_print([1, 2, 3])
    assert capsys.readouterr().out == "[1,2,3]\n"


def test_print_tuple(capsys):
    statek_print((1, 2))
    assert capsys.readouterr().out == "(1,2)\n"


def test_print_dict(capsys):
    statek_print({"a": 1})
    assert capsys.readouterr().out == '{"a":1}\n'


def test_print_set(capsys):
    statek_print({1})
    assert capsys.readouterr().out == "{1}\n"


def test_print_frozenset(capsys):
    statek_print(frozenset({2}))
    assert capsys.readouterr().out == "{2}\n"


# --- Object: __llm_repr__ takes precedence ---

def test_object_with_llm_repr(capsys):
    class Widget:
        def __llm_repr__(self):
            return "Widget(custom)"

    statek_print(Widget())
    assert capsys.readouterr().out == "Widget(custom)\n"


def test_llm_repr_takes_precedence_over_str(capsys):
    class Widget:
        def __llm_repr__(self):
            return "llm_result"
        def __str__(self):
            return "str_result"

    statek_print(Widget())
    assert capsys.readouterr().out == "llm_result\n"


# --- Object: explicit __str__ used when no __llm_repr__ ---

def test_object_with_explicit_str(capsys):
    class Widget:
        def __str__(self):
            return "Widget(str)"

    statek_print(Widget())
    assert capsys.readouterr().out == "Widget(str)\n"


# --- Object: falls back to format_llm_repr ---

def test_plain_object_uses_format_llm_repr(capsys):
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    statek_print(Point(3, 4))
    assert capsys.readouterr().out == "Point(x=3,y=4)\n"


def test_plain_object_no_members(capsys):
    class Empty:
        pass

    statek_print(Empty())
    assert capsys.readouterr().out == "Empty()\n"


def test_dataclass_uses_format_llm_repr(capsys):
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    statek_print(Point(1, 2))
    assert capsys.readouterr().out == "Point(x=1,y=2)\n"


# --- db0.memo objects ---

def test_db0_memo_with_llm_repr(db0_fixture, capsys):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str

        def __llm_repr__(self):
            return f"Config[{self.name}]"

    statek_print(Config(name="test"))
    assert capsys.readouterr().out == "Config[test]\n"


def test_db0_memo_without_custom_methods(db0_fixture, capsys):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str
        value: int

    statek_print(Config(name="alpha", value=7))
    assert capsys.readouterr().out == 'Config(name="alpha",value=7)\n'


# --- print() compatibility ---

def test_multiple_objects(capsys):
    statek_print(1, 2, 3)
    assert capsys.readouterr().out == "1 2 3\n"


def test_custom_sep(capsys):
    statek_print(1, 2, 3, sep=", ")
    assert capsys.readouterr().out == "1, 2, 3\n"


def test_custom_end(capsys):
    statek_print("hello", end="!")
    assert capsys.readouterr().out == '"hello"!'


def test_empty_end(capsys):
    statek_print("hello", end="")
    assert capsys.readouterr().out == '"hello"'


def test_no_objects(capsys):
    statek_print()
    assert capsys.readouterr().out == "\n"


def test_file_parameter():
    buf = io.StringIO()
    statek_print(42, file=buf)
    assert buf.getvalue() == "42\n"


def test_file_parameter_with_string():
    buf = io.StringIO()
    statek_print("world", file=buf)
    assert buf.getvalue() == '"world"\n'


def test_flush_parameter():
    buf = io.StringIO()
    statek_print(42, file=buf, flush=True)
    assert buf.getvalue() == "42\n"


def test_mixed_types(capsys):
    statek_print(1, "two", [3])
    assert capsys.readouterr().out == '1 "two" [3]\n'
