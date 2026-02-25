"""Tests for format_llm_repr function."""

# pylint: disable=too-few-public-methods

import dataclasses
from datetime import datetime
from decimal import Decimal

import dbzero as db0

from statek.utils import format_llm_repr


# --- Simple scalar values (forwarded to format_value_repr) ---

def test_int():
    assert format_llm_repr(42) == "42"


def test_zero():
    assert format_llm_repr(0) == "0"


def test_negative_int():
    assert format_llm_repr(-7) == "-7"


def test_float():
    assert format_llm_repr(1.5) == "1.5"


def test_string():
    assert format_llm_repr("hello") == '"hello"'


def test_string_empty():
    assert format_llm_repr("") == '""'


def test_bool_true():
    assert format_llm_repr(True) == "True"


def test_bool_false():
    assert format_llm_repr(False) == "False"


def test_bool_not_int():
    assert format_llm_repr(True) != "1"
    assert format_llm_repr(False) != "0"


def test_decimal():
    assert format_llm_repr(Decimal("3.14")) == "3.14"


def test_datetime():
    dt = datetime(2026, 1, 1, 12, 0, 59)
    assert format_llm_repr(dt) == 'datetime("2026-01-01 12:00")'


# --- List formatting ---

def test_list_truncated():
    assert format_llm_repr([1, 2, 3, 4, 5], max_len=3) == "[1,2,3, ...] (5 items total)"


def test_list_not_truncated():
    assert format_llm_repr([1, 2, 3], max_len=3) == "[1,2,3]"


def test_list_exactly_max_len():
    assert format_llm_repr([1, 2], max_len=2) == "[1,2]"


def test_list_empty():
    assert format_llm_repr([]) == "[]"


def test_list_single_item():
    assert format_llm_repr([42]) == "[42]"


def test_list_default_max_len_25():
    items = list(range(30))
    result = format_llm_repr(items)
    expected_items = ",".join(str(i) for i in range(25))
    assert result == f"[{expected_items}, ...] (30 items total)"


def test_list_with_strings():
    assert format_llm_repr(["a", "b"], max_len=5) == '["a","b"]'


# --- Tuple formatting ---

def test_tuple_truncated():
    assert format_llm_repr((1, 2, 3, 4), max_len=2) == "(1,2, ...) (4 items total)"


def test_tuple_not_truncated():
    assert format_llm_repr((10, 20), max_len=5) == "(10,20)"


def test_tuple_empty():
    assert format_llm_repr(()) == "()"


# --- Set formatting ---

def test_set_single_item():
    assert format_llm_repr({99}) == "{99}"


def test_set_empty():
    assert format_llm_repr(set()) == "{}"


# --- Dict formatting ---

def test_dict_truncated():
    result = format_llm_repr({"a": 1, "b": 2, "c": 3}, max_len=1)
    assert result == '{"a":1, ...} (3 items total)'


def test_dict_not_truncated():
    result = format_llm_repr({"a": 1}, max_len=3)
    assert result == '{"a":1}'


def test_dict_empty():
    assert format_llm_repr({}) == "{}"


def test_dict_two_items():
    result = format_llm_repr({"x": 10, "y": 20}, max_len=5)
    assert result == '{"x":10,"y":20}'


def test_dict_max_len_2_of_3():
    result = format_llm_repr({"a": 1, "b": 2, "c": 3}, max_len=2)
    assert result == '{"a":1,"b":2, ...} (3 items total)'


# --- Object formatting ---

def test_object_basic():
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    u = User("alice", 30)
    result = format_llm_repr(u)
    assert result == 'User(name="alice",age=30)'


def test_object_no_members():
    class Empty:
        pass

    result = format_llm_repr(Empty())
    assert result == "Empty()"


def test_object_hide_single():
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    u = User("alice", 30)
    result = format_llm_repr(u, hide=["age"])
    assert result == 'User(name="alice")'


def test_object_hide_multiple():
    class User:
        def __init__(self, name, age, email):
            self.name = name
            self.age = age
            self.email = email

    u = User("alice", 30, "alice@example.com")
    result = format_llm_repr(u, hide=["age", "email"])
    assert result == 'User(name="alice")'


def test_object_expand():
    class Role:
        def __init__(self, name):
            self.name = name

    class User:
        def __init__(self, role):
            self.role = role

    u = User(Role("admin"))
    result = format_llm_repr(u, expand=["role"])
    assert result == 'User(role=Role(name="admin"))'


def test_object_expand_with_other_members():
    class Role:
        def __init__(self, name):
            self.name = name

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    u = User("alice", Role("admin"))
    result = format_llm_repr(u, expand=["role"])
    assert result == 'User(name="alice",role=Role(name="admin"))'


def test_object_show_only():
    class User:
        def __init__(self, name, age, email):
            self.name = name
            self.age = age
            self.email = email

    u = User("alice", 30, "alice@example.com")
    result = format_llm_repr(u, show_only=["name", "age"])
    assert result == 'User(name="alice",age=30)'


def test_object_hide_overrides_show_only():
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    u = User("alice", 30)
    result = format_llm_repr(u, hide=["age"], show_only=["name", "age"])
    assert result == 'User(name="alice")'


def test_object_show_only_nonexistent_member():
    """show_only members that don't exist are silently skipped."""
    class User:
        def __init__(self, name):
            self.name = name

    u = User("bob")
    result = format_llm_repr(u, show_only=["name", "missing"])
    assert result == 'User(name="bob")'


def test_object_member_is_collection():
    """Non-expanded collection members use format_value_repr."""
    class Container:
        def __init__(self, items):
            self.items = items

    c = Container([1, 2, 3])
    result = format_llm_repr(c)
    assert result == "Container(items=<List of 3 items>)"


def test_object_expand_uses_format_llm_repr_for_collection():
    """Expanded collection members use format_llm_repr with max_len."""
    class Container:
        def __init__(self, items):
            self.items = items

    c = Container([1, 2, 3])
    result = format_llm_repr(c, expand=["items"])
    assert result == "Container(items=[1,2,3])"


# --- Dataclass objects ---

def test_dataclass_basic():
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    p = Point(3, 4)
    result = format_llm_repr(p)
    assert result == "Point(x=3,y=4)"


def test_dataclass_hide():
    @dataclasses.dataclass
    class Point:
        x: int
        y: int

    p = Point(3, 4)
    result = format_llm_repr(p, hide=["y"])
    assert result == "Point(x=3)"


# --- db0.memo objects ---

def test_db0_memo_dataclass(db0_fixture):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str
        value: int

    c = Config(name="alpha", value=7)
    result = format_llm_repr(c)
    assert result == 'Config(name="alpha",value=7)'


def test_db0_memo_dataclass_hide(db0_fixture):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Config:
        name: str
        value: int

    c = Config(name="alpha", value=7)
    result = format_llm_repr(c, hide=["value"])
    assert result == 'Config(name="alpha")'


def test_db0_memo_dataclass_expand(db0_fixture):  # pylint: disable=unused-argument
    @db0.memo
    @dataclasses.dataclass
    class Inner:
        x: int

    @db0.memo
    @dataclasses.dataclass
    class Outer:
        label: str
        inner: Inner

    o = Outer(label="test", inner=Inner(x=42))
    result = format_llm_repr(o, expand=["inner"])
    assert result == 'Outer(label="test",inner=Inner(x=42))'
