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


# --- Inner objects in collections are expanded ---

def test_list_with_objects_expanded():
    class Item:
        def __init__(self, x):
            self.x = x

    items = [Item(1), Item(2)]
    result = format_llm_repr(items)
    assert result == "[Item(x=1),Item(x=2)]"


def test_tuple_with_objects_expanded():
    class Item:
        def __init__(self, x):
            self.x = x

    items = (Item(1), Item(2))
    result = format_llm_repr(items)
    assert result == "(Item(x=1),Item(x=2))"


def test_dict_with_object_values_expanded():
    class Val:
        def __init__(self, n):
            self.n = n

    d = {"a": Val(1)}
    result = format_llm_repr(d)
    assert result == '{"a":Val(n=1)}'


# --- repeated=True kwarg ---

def test_repeated_no_object_fields():
    """repeated=True has no effect when no fields resolve to <Object>."""
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    u = User("alice", 30)
    result = format_llm_repr(u, repeated=True)
    assert result == 'User(name="alice",age=30)'


def test_repeated_with_object_field():
    """repeated=True skips <Object> fields and appends ... instead."""
    class Role:
        pass

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    u = User("alice", Role())
    result = format_llm_repr(u, repeated=True)
    assert result == 'User(name="alice",...)'


def test_repeated_all_object_fields():
    """repeated=True with all fields as <Object> produces ClassName(...)."""
    class Inner:
        pass

    class Outer:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    o = Outer(Inner(), Inner())
    result = format_llm_repr(o, repeated=True)
    assert result == 'Outer(...)'


def test_repeated_false_same_as_default():
    """repeated=False is the same as not passing repeated."""
    class Role:
        pass

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    u = User("alice", Role())
    assert format_llm_repr(u, repeated=False) == format_llm_repr(u)


def test_repeated_scalar_is_no_op():
    """repeated=True on a scalar value has no effect."""
    assert format_llm_repr(42, repeated=True) == "42"
    assert format_llm_repr("hi", repeated=True) == '"hi"'


def test_repeated_unknown_kwargs_ignored():
    """Unknown kwargs are silently ignored."""
    class User:
        def __init__(self, name):
            self.name = name

    u = User("bob")
    result = format_llm_repr(u, unknown_kwarg=True, another=42)
    assert result == 'User(name="bob")'


# --- Collection repeated-type logic ---

def test_list_same_type_second_element_uses_repeated():
    """In a list of same-type objects, the second element is formatted with repeated=True."""
    class Role:
        pass

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    users = [User("Adam", Role()), User("Ela", Role())]
    result = format_llm_repr(users)
    assert result == '[User(name="Adam",role=<Object>),User(name="Ela",...)]'


def test_list_different_types_no_repeated():
    """In a list of different-type objects, each is shown in full."""
    class A:
        def __init__(self, inner):
            self.inner = inner

    class B:
        def __init__(self, inner):
            self.inner = inner

    class Inner:
        pass

    result = format_llm_repr([A(Inner()), B(Inner())])
    assert result == '[A(inner=<Object>),B(inner=<Object>)]'


def test_list_three_same_type_only_first_full():
    """Only the first occurrence of a type is shown in full; the rest use repeated=True."""
    class Role:
        pass

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    users = [User("Adam", Role()), User("Ela", Role()), User("Bob", Role())]
    result = format_llm_repr(users)
    assert result == '[User(name="Adam",role=<Object>),User(name="Ela",...),User(name="Bob",...)]'


def test_tuple_same_type_uses_repeated():
    """Tuples also apply repeated=True for repeated types."""
    class Meta:
        pass

    class Item:
        def __init__(self, val, meta):
            self.val = val
            self.meta = meta

    items = (Item(1, Meta()), Item(2, Meta()))
    result = format_llm_repr(items)
    assert result == '(Item(val=1,meta=<Object>),Item(val=2,...))'


def test_list_scalars_repeated_is_no_op():
    """For scalars, repeated-type tracking doesn't change the output."""
    result = format_llm_repr([1, 2, 3])
    assert result == '[1,2,3]'


def test_dict_same_type_values_uses_repeated():
    """Dict values of the same type use repeated=True from the second occurrence."""
    class Role:
        pass

    class User:
        def __init__(self, name, role):
            self.name = name
            self.role = role

    d = {"a": User("Adam", Role()), "b": User("Ela", Role())}
    result = format_llm_repr(d)
    assert result == '{"a":User(name="Adam",role=<Object>),"b":User(name="Ela",...)}'


# --- __llm_repr__ and __str__ are respected for collection elements ---

def test_list_items_call_llm_repr():
    """Items inside a list call __llm_repr__ if defined."""
    class User:
        def __init__(self, name):
            self.name = name

        def __llm_repr__(self):
            return f"user:{self.name}"

    result = format_llm_repr([User("Alice"), User("Bob")])
    assert result == "[user:Alice,user:Bob]"


def test_tuple_items_call_llm_repr():
    """Items inside a tuple call __llm_repr__ if defined."""
    class User:
        def __init__(self, name):
            self.name = name

        def __llm_repr__(self):
            return f"user:{self.name}"

    result = format_llm_repr((User("Alice"), User("Bob")))
    assert result == "(user:Alice,user:Bob)"


def test_dict_values_call_llm_repr():
    """Dict values call __llm_repr__ if defined."""
    class User:
        def __init__(self, name):
            self.name = name

        def __llm_repr__(self):
            return f"user:{self.name}"

    result = format_llm_repr({"a": User("Alice"), "b": User("Bob")})
    assert result == '{"a":user:Alice,"b":user:Bob}'


def test_list_items_call_explicit_str():
    """Items inside a list call explicit __str__ if defined and no __llm_repr__."""
    class Tag:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return f"#{self.name}"

    result = format_llm_repr([Tag("python"), Tag("dev")])
    assert result == "[#python,#dev]"


def test_list_items_llm_repr_takes_precedence_over_str():
    """__llm_repr__ is preferred over __str__ for collection elements."""
    class Item:
        def __llm_repr__(self):
            return "llm"

        def __str__(self):
            return "str"

    result = format_llm_repr([Item()])
    assert result == "[llm]"
