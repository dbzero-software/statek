"""Tests for format_value_repr function."""

# pylint: disable=too-few-public-methods

from datetime import datetime
from decimal import Decimal

from statek.utils import format_value_repr


def test_int():
    assert format_value_repr(2) == "2"


def test_int_zero():
    assert format_value_repr(0) == "0"


def test_negative_int():
    assert format_value_repr(-5) == "-5"


def test_float():
    assert format_value_repr(1.5) == "1.5"


def test_string():
    assert format_value_repr("abc") == '"abc"'


def test_string_empty():
    assert format_value_repr("") == '""'


def test_bool_false():
    assert format_value_repr(False) == "False"


def test_bool_true():
    assert format_value_repr(True) == "True"


def test_bool_not_treated_as_int():
    """bool is a subclass of int, but must be formatted as bool."""
    assert format_value_repr(True) != "1"
    assert format_value_repr(False) != "0"


def test_datetime_minute_precision():
    """datetime is formatted with minute precision and a type wrapper."""
    dt = datetime(2026, 12, 1, 12, 0, 0, 1132)
    assert format_value_repr(dt) == 'datetime("2026-12-01 12:00")'


def test_datetime_different_time():
    dt = datetime(2024, 3, 15, 9, 45)
    assert format_value_repr(dt) == 'datetime("2024-03-15 09:45")'


def test_datetime_ignores_seconds_and_microseconds():
    dt = datetime(2026, 1, 1, 8, 30, 59, 999999)
    assert format_value_repr(dt) == 'datetime("2026-01-01 08:30")'


def test_list():
    assert format_value_repr([1, 2, 3]) == "<List of 3 items>"


def test_list_empty():
    assert format_value_repr([]) == "<List of 0 items>"


def test_list_single_item():
    assert format_value_repr(["x"]) == "<List of 1 items>"


def test_tuple():
    assert format_value_repr((1, 2)) == "<Tuple of 2 items>"


def test_set():
    assert format_value_repr({1, 2, 3}) == "<Set of 3 items>"


def test_dict():
    assert format_value_repr({"a": 1, "b": 2}) == "<Dict of 2 items>"


def test_dict_empty():
    assert format_value_repr({}) == "<Dict of 0 items>"


def test_decimal():
    assert format_value_repr(Decimal("100.00")) == "100.00"


def test_decimal_integer_value():
    assert format_value_repr(Decimal("42")) == "42"


def test_custom_object():
    class UserObject:
        def __init__(self, name):
            self.name = name

    assert format_value_repr(UserObject("adam")) == "<Object>"


def test_custom_object_with_repr():
    """Custom repr does not affect output — still <Object>."""
    class Fancy:
        def __repr__(self):
            return "fancy!"

    assert format_value_repr(Fancy()) == "<Object>"
