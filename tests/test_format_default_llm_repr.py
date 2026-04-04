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


def test_string_is_unquoted():
    """Top-level strings are unquoted."""
    assert format_default_llm_repr("hello") == 'hello'


def test_string_empty():
    assert format_default_llm_repr("") == ''


def test_decimal():
    assert format_default_llm_repr(Decimal("3.14")) == "3.14"


def test_datetime():
    dt = datetime(2026, 1, 1, 12, 0)
    assert format_default_llm_repr(dt) == 'datetime("2026-01-01 12:00")'


# --- None values ---

def test_none():
    assert format_default_llm_repr(None) == "None"


def test_none_in_tuple():
    assert format_default_llm_repr((None, None, None)) == "(None,None,None)"


def test_none_in_list():
    assert format_default_llm_repr([None, None]) == "[None,None]"


def test_none_in_dict_value():
    assert format_default_llm_repr({"a": None}) == '{"a":None}'


def test_none_in_set():
    assert format_default_llm_repr({None}) == "{None}"


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


# --- kwargs are forwarded to format_llm_repr ---

def test_plain_object_kwargs_hide():
    """kwargs like hide= are forwarded to format_llm_repr for plain objects."""
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    assert format_default_llm_repr(User("alice", 30), hide=["age"]) == 'User(name="alice")'


def test_plain_object_kwargs_show_only():
    """kwargs like show_only= are forwarded to format_llm_repr for plain objects."""
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    assert format_default_llm_repr(User("alice", 30), show_only=["name"]) == 'User(name="alice")'


def test_collection_kwargs_max_len():
    """max_len= is forwarded for collection types."""
    assert format_default_llm_repr([1, 2, 3, 4, 5], max_len=2) == "[1,2, ...] (5 items total)"


def test_kwargs_not_passed_to_llm_repr_without_var_kw():
    """kwargs are not forwarded to __llm_repr__ if it doesn't accept **kwargs."""
    class Widget:
        def __llm_repr__(self):
            return "Widget(custom)"

    assert format_default_llm_repr(Widget(), hide=["x"]) == "Widget(custom)"


def test_unknown_kwargs_ignored():
    """Unknown kwargs are silently ignored (forwarded to format_llm_repr which ignores them)."""
    class Point:
        def __init__(self, x):
            self.x = x

    assert format_default_llm_repr(Point(1), unknown_kwarg=True) == "Point(x=1)"


# --- Recursion guard: __llm_repr__ delegating back to format_default_llm_repr ---

def test_llm_repr_delegates_to_format_default_no_recursion():
    """__llm_repr__ that calls format_default_llm_repr(self, ...) must not recurse infinitely."""
    class TimeSlot:
        def __init__(self, hour):
            self.hour = hour

    class Event:
        def __init__(self, name, time_slot):
            self.name = name
            self.time_slot = time_slot

        def __llm_repr__(self, **kwargs):
            return format_default_llm_repr(self, expand=["time_slot"], **kwargs)

    result = format_default_llm_repr(Event("standup", TimeSlot(9)))
    assert result == 'Event(name="standup",time_slot=TimeSlot(hour=9))'


def test_llm_repr_expand_kwargs_propagate_to_expanded_field():
    """kwargs (e.g. repeated=True) received by __llm_repr__ propagate into expanded fields."""
    class Meta:
        pass

    class TimeSlot:
        def __init__(self, hour, meta):
            self.hour = hour
            self.meta = meta  # renders as <Object>

    class Event:
        def __init__(self, name, time_slot):
            self.name = name
            self.time_slot = time_slot

        def __llm_repr__(self, **kwargs):
            return format_default_llm_repr(self, expand=["time_slot"], **kwargs)

    # In a list: second Event gets repeated=True, which propagates through expand into time_slot
    events = [
        Event("standup", TimeSlot(9, Meta())),
        Event("retro", TimeSlot(15, Meta())),
    ]
    result = format_default_llm_repr(events)
    assert result == (
        '[Event(name="standup",time_slot=TimeSlot(hour=9,meta=<Object>)),'
        'Event(name="retro",time_slot=TimeSlot(hour=15,...))]'
    )


def test_llm_repr_recursion_guard_kwargs_forwarded():
    """kwargs passed to format_default_llm_repr propagate into the recursive fallback."""
    class Event:
        def __init__(self, name, internal):
            self.name = name
            self.internal = internal

        def __llm_repr__(self, **kwargs):
            return format_default_llm_repr(self, **kwargs)

    result = format_default_llm_repr(Event("x", "secret"), hide=["internal"])
    assert result == 'Event(name="x")'


# --- Exported from statek package ---

def test_exported_from_statek_package():
    from statek import format_default_llm_repr as fn  # pylint: disable=import-outside-toplevel
    assert fn(42) == "42"
