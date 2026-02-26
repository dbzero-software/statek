"""Tests for docstring parsing utilities."""

import math
from dataclasses import dataclass, field

import dbzero as db0
import pytest

from statek.docstring import (
    parse_docstring,
    format_docstring,
    parse_statek_acl,
    FuncDocString,
    ClassDocString,
    AttrDocString,
    ACL_Item,
    Statek_ACL,
    DocstringParseError
)


def calculate_hypotenuse(a: float, b: float) -> float:
    """Calculates the length of the hypotenuse for a right-angled triangle.

    This function applies the Pythagorean theorem to find the longest side
    of a right-angled triangle given the two shorter sides. The formula used is:

    $c = \\sqrt{a^2 + b^2}$

    Args:
        a (float): The length of side 'a'. Must be a positive number.
        b (float): The length of side 'b'. Must be a positive number.

    Returns:
        float: The calculated length of the hypotenuse.

    Raises:
        ValueError: If either `a` or `b` is negative, as physical lengths
            cannot be less than zero.

    Example:
        >>> calculate_hypotenuse(3.0, 4.0)
        5.0
    """
    if a < 0 or b < 0:
        raise ValueError("Side lengths must be positive.")
    return math.sqrt(a**2 + b**2)


class ParticleSimulator:  # pylint: disable=too-few-public-methods
    """A simulator for tracking 2D particle kinetics.

    This class manages a collection of particle objects and calculates their
    trajectories based on initial velocity and environmental friction.

    Attributes:
        particles (list[dict]): A list of dictionaries, where each dict
            represents a particle's current state (x, y, velocity).
        gravity (float): The constant downward force applied to all particles.
        friction (float): The coefficient of air resistance, defaulting to 0.01.
    """


def simple_func(x: int) -> int:
    """A simple function.

    Args:
        x (int): Input value.

    Returns:
        int: Output value.
    """
    return x * 2


def func_missing_arg_doc(a: int, b: int) -> int:
    """Function with missing argument documentation.

    Args:
        a (int): First argument.

    Returns:
        int: Sum of arguments.
    """
    return a + b


def func_no_docstring(x: int) -> int:  # pylint: disable=unused-argument
    pass


class ClassNoDocstring:  # pylint: disable=too-few-public-methods
    pass


class TestParseFuncDocstring:
    """Test cases for parsing function docstrings."""

    def test_parse_function_basic(self):
        """Test parsing a basic function docstring."""
        result = parse_docstring(simple_func)

        assert isinstance(result, FuncDocString)
        assert result.name == "simple_func"
        assert result.source is simple_func
        assert result.brief_desc == "A simple function."

    def test_parse_function_args(self):
        """Test parsing function arguments."""
        result = parse_docstring(calculate_hypotenuse)

        assert result.args is not None
        assert len(result.args) == 2

        arg_a = result.args[0]
        assert arg_a.name == "a"
        assert arg_a.type == "float"
        assert "length of side" in arg_a.desc

        arg_b = result.args[1]
        assert arg_b.name == "b"
        assert arg_b.type == "float"

    def test_parse_function_returns(self):
        """Test parsing function return documentation."""
        result = parse_docstring(calculate_hypotenuse)

        assert result.returns is not None
        assert result.returns.type == "float"
        assert "hypotenuse" in result.returns.desc

    def test_parse_function_raises(self):
        """Test parsing function raises documentation."""
        result = parse_docstring(calculate_hypotenuse)

        assert result.raises is not None
        assert len(result.raises) == 1

        raise_doc = result.raises[0]
        assert raise_doc.type == "ValueError"
        assert "negative" in raise_doc.desc

    def test_parse_function_example(self):
        """Test parsing function example section."""
        result = parse_docstring(calculate_hypotenuse)

        assert result.example is not None
        assert "calculate_hypotenuse(3.0, 4.0)" in result.example
        assert "5.0" in result.example

    def test_parse_function_full_desc(self):
        """Test parsing full description."""
        result = parse_docstring(calculate_hypotenuse)

        assert "Pythagorean theorem" in result.full_desc
        assert "right-angled triangle" in result.full_desc

    def test_parse_function_missing_arg_raises_error(self):
        """Test that missing argument documentation raises error."""
        with pytest.raises(DocstringParseError) as exc_info:
            parse_docstring(func_missing_arg_doc)

        assert "undocumented arguments" in str(exc_info.value)
        assert "b" in str(exc_info.value)

    def test_parse_function_no_docstring_raises_error(self):
        """Test that missing docstring raises error."""
        with pytest.raises(DocstringParseError) as exc_info:
            parse_docstring(func_no_docstring)

        assert "no docstring" in str(exc_info.value)


class TestParseClassDocstring:
    """Test cases for parsing class docstrings."""

    def test_parse_class_basic(self):
        """Test parsing a basic class docstring."""
        result = parse_docstring(ParticleSimulator)

        assert isinstance(result, ClassDocString)
        assert result.name == "ParticleSimulator"
        assert result.source is ParticleSimulator
        assert "2D particle kinetics" in result.brief_desc

    def test_parse_class_attributes(self):
        """Test parsing class attributes."""
        result = parse_docstring(ParticleSimulator)

        assert result.attrs is not None
        assert len(result.attrs) == 3

        attr_names = {attr.name for attr in result.attrs}
        assert attr_names == {"particles", "gravity", "friction"}

        # Check specific attribute
        particles_attr = next(a for a in result.attrs if a.name == "particles")
        assert particles_attr.type == "list[dict]"
        assert "dictionaries" in particles_attr.desc

    def test_parse_class_full_desc(self):
        """Test parsing class full description."""
        result = parse_docstring(ParticleSimulator)

        assert "collection of particle objects" in result.full_desc
        assert "trajectories" in result.full_desc

    def test_parse_class_no_docstring_raises_error(self):
        """Test that missing docstring raises error."""
        with pytest.raises(DocstringParseError) as exc_info:
            parse_docstring(ClassNoDocstring)

        assert "no docstring" in str(exc_info.value)


class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers together.

        This method performs simple addition.

        Args:
            a (int): First number.
            b (int): Second number.

        Returns:
            int: The sum of a and b.

        Raises:
            TypeError: If inputs are not integers.
        """
        return a + b

    def divide(self, dividend: float, divisor: float) -> float:
        """Divide one number by another.

        Args:
            dividend (float): The number to be divided.
            divisor (float): The number to divide by.

        Returns:
            float: The result of the division.

        Raises:
            ZeroDivisionError: If divisor is zero.
        """
        if divisor == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return dividend / divisor


class TestParseMethodDocstring:
    """Test cases for parsing class method docstrings."""

    def test_parse_method_basic(self):
        """Test parsing a basic method docstring."""
        result = parse_docstring(Calculator.add)

        assert isinstance(result, FuncDocString)
        assert result.name == "add"
        assert result.source is Calculator.add
        assert "Add two numbers" in result.brief_desc

    def test_parse_method_args(self):
        """Test parsing method arguments (excluding self)."""
        result = parse_docstring(Calculator.add)

        assert result.args is not None
        assert len(result.args) == 2

        arg_names = [arg.name for arg in result.args]
        assert "a" in arg_names
        assert "b" in arg_names
        assert "self" not in arg_names

    def test_parse_method_returns(self):
        """Test parsing method return documentation."""
        result = parse_docstring(Calculator.add)

        assert result.returns is not None
        assert result.returns.type == "int"
        assert "sum" in result.returns.desc

    def test_parse_method_raises(self):
        """Test parsing method raises documentation."""
        result = parse_docstring(Calculator.divide)

        assert result.raises is not None
        assert len(result.raises) == 1
        assert result.raises[0].type == "ZeroDivisionError"

    def test_parse_bound_method(self):
        """Test parsing a bound method from an instance."""
        calc = Calculator()
        result = parse_docstring(calc.add)

        assert isinstance(result, FuncDocString)
        assert result.name == "add"
        assert len(result.args) == 2


class TestParseDocstringEdgeCases:
    """Test edge cases for parse_docstring."""

    def test_parse_invalid_type_raises_error(self):
        """Test that invalid input type raises error."""
        with pytest.raises(DocstringParseError) as exc_info:
            parse_docstring("not a class or function")

        assert "Expected a class or callable" in str(exc_info.value)

    def test_parse_lambda_with_no_docstring(self):
        """Test parsing a lambda (no docstring) raises error."""
        with pytest.raises(DocstringParseError):
            parse_docstring(lambda x: x)


class TestFormatDocstring:
    """Test cases for format_docstring function."""

    def test_format_func_brief_no_py_syntax(self):
        """Test brief format without Python syntax."""
        result = format_docstring(parse_docstring(simple_func), brief=True, py_syntax=False)

        assert "simple_func(x)" in result
        assert "A simple function." in result
        assert "def " not in result

    def test_format_func_brief_py_syntax(self):
        """Test brief format with Python syntax."""
        result = format_docstring(parse_docstring(simple_func), brief=True, py_syntax=True)

        assert "def simple_func(x: int) -> int:" in result
        assert '"""A simple function.' in result
        assert "Returns: Output value." in result

    def test_format_func_full_py_syntax(self):
        """Test full format with Python syntax."""
        result = format_docstring(
            parse_docstring(calculate_hypotenuse), brief=False, py_syntax=True
        )

        assert "def calculate_hypotenuse(a: float, b: float) -> float:" in result
        assert "Pythagorean theorem" in result
        assert "Args:" in result
        assert "a (float):" in result
        assert "b (float):" in result
        assert "Raises:" in result
        assert "ValueError:" in result
        assert "Example:" in result
        assert ">>> calculate_hypotenuse(3.0, 4.0)" in result

    def test_format_func_brief_with_returns(self):
        """Test brief format includes return description."""
        result = format_docstring(parse_docstring(simple_func), brief=True, py_syntax=False)

        assert "Returns: Output value." in result

    def test_format_class_brief_py_syntax(self):
        """Test brief class format with Python syntax."""
        result = format_docstring(parse_docstring(ParticleSimulator), brief=True, py_syntax=True)

        assert "class ParticleSimulator:" in result
        assert "2D particle kinetics" in result
        # Brief mode should not include attributes
        assert "Attributes:" not in result

    def test_format_class_full_py_syntax(self):
        """Test full class format with Python syntax."""
        result = format_docstring(parse_docstring(ParticleSimulator), brief=False, py_syntax=True)

        assert "class ParticleSimulator:" in result
        assert "Attributes:" in result
        assert "particles (list[dict]):" in result
        assert "gravity (float):" in result
        assert "friction (float):" in result

    def test_format_class_plain(self):
        """Test class format in plain text."""
        result = format_docstring(parse_docstring(ParticleSimulator), brief=True, py_syntax=False)

        assert "ParticleSimulator" in result
        assert "2D particle kinetics" in result
        assert "class " not in result

    def test_format_func_uses_4_spaces_indent(self):
        """Test that py_syntax output uses 4-space indentation."""
        result = format_docstring(parse_docstring(simple_func), brief=True, py_syntax=True)

        # Check for 4-space indentation, no tabs
        assert '\t' not in result
        lines = result.split('\n')
        for line in lines[1:]:  # Skip first line (def)
            if line.strip():
                assert line.startswith('    '), f"Line should use 4-space indent: {line}"

    def test_format_method_docstring(self):
        """Test formatting a class method docstring."""
        result = format_docstring(parse_docstring(Calculator.add), brief=True, py_syntax=True)

        assert "def add(a: int, b: int) -> int:" in result
        assert "Add two numbers" in result
        # self should not appear in signature
        assert "self" not in result


# --- Dataclass field extraction fixtures ---

@dataclass
class Product:
    """A product in the catalog."""
    name: str = field(metadata={"doc": "The official name of the product"})
    price: float = field(metadata={"doc": "Price in USD"})
    quantity: int = field(default=0, metadata={"doc": "Number of items in stock"})


@db0.memo
@dataclass
class Order:
    """A customer order for tracking purchases."""
    order_id: int = field(metadata={"doc": "Unique order identifier"})
    customer: str = field(metadata={"doc": "Customer name"})
    total: float = field(default=0.0, metadata={"doc": "Total cost in USD"})


@dataclass
class PlainDataclass:
    """A dataclass without doc metadata on fields."""
    x: int = 0
    y: int = 0


class TestParseDataclassFields:
    """Test cases for parsing dataclass fields."""

    def test_parse_dataclass_fields(self):
        """Test that dataclass fields are extracted with name, type, and doc."""
        result = parse_docstring(Product)

        assert isinstance(result, ClassDocString)
        assert result.fields is not None
        assert len(result.fields) == 3

        name_field = result.fields[0]
        assert name_field.name == "name"
        assert name_field.type == "str"
        assert name_field.desc == "The official name of the product"

        price_field = result.fields[1]
        assert price_field.name == "price"
        assert price_field.type == "float"
        assert price_field.desc == "Price in USD"

        quantity_field = result.fields[2]
        assert quantity_field.name == "quantity"
        assert quantity_field.type == "int"
        assert quantity_field.desc == "Number of items in stock"

    def test_parse_memo_dataclass_fields(self):
        """Test that db0.memo-wrapped dataclass fields are extracted."""
        result = parse_docstring(Order)

        assert isinstance(result, ClassDocString)
        assert result.fields is not None
        assert len(result.fields) == 3

        order_id_field = result.fields[0]
        assert order_id_field.name == "order_id"
        assert order_id_field.type == "int"
        assert order_id_field.desc == "Unique order identifier"

        customer_field = result.fields[1]
        assert customer_field.name == "customer"
        assert customer_field.type == "str"
        assert customer_field.desc == "Customer name"

        total_field = result.fields[2]
        assert total_field.name == "total"
        assert total_field.type == "float"
        assert total_field.desc == "Total cost in USD"

    def test_parse_dataclass_without_doc_metadata(self):
        """Test that dataclass without doc metadata produces fields with None desc."""
        result = parse_docstring(PlainDataclass)

        assert result.fields is not None
        assert len(result.fields) == 2
        for f in result.fields:
            assert f.desc is None

    def test_parse_non_dataclass_has_no_fields(self):
        """Test that non-dataclass classes have fields=None."""
        result = parse_docstring(ParticleSimulator)

        assert result.fields is None

    def test_fields_are_attr_docstring_instances(self):
        """Test that field entries are AttrDocString namedtuples."""
        result = parse_docstring(Product)

        for f in result.fields:
            assert isinstance(f, AttrDocString)

    def test_fields_use_resolved_type_hints(self):
        """Test that field types come from get_type_hints, not string annotations."""
        result = parse_docstring(Product)

        # Types should be resolved to their string names
        type_names = [f.type for f in result.fields]
        assert type_names == ["str", "float", "int"]


# --- Fixtures for merged-attributes tests ---

@dataclass
class SensorReading:
    """A reading from a sensor device.

    Attributes:
        value (float): The manually documented sensor value.
        unit (str): Measurement unit (e.g. "celsius").
    """
    value: float = field(metadata={"doc": "Raw sensor output"})
    unit: str = field(metadata={"doc": "Unit from device metadata"})
    timestamp: float = field(default=0.0, metadata={"doc": "Unix epoch seconds"})


class TestFormatDocstringMergedAttributes:
    """Test cases for merged fields/attrs in format_docstring."""

    def test_dataclass_fields_appear_as_attributes(self):
        """Dataclass fields should appear in the Attributes section."""
        result = format_docstring(parse_docstring(Product), brief=False, py_syntax=True)

        assert "Attributes:" in result
        assert "name (str):" in result
        assert "price (float):" in result
        assert "quantity (int):" in result

    def test_attrs_take_precedence_over_fields(self):
        """Explicit attrs win when names overlap with fields."""
        result = format_docstring(parse_docstring(SensorReading), brief=False, py_syntax=True)

        assert "Attributes:" in result
        # Explicit attr descriptions should win for 'value' and 'unit'
        assert "The manually documented sensor value." in result
        assert "Measurement unit" in result
        # 'timestamp' only exists as a field — should still appear
        assert "timestamp (float):" in result
        assert "Unix epoch seconds" in result
        # Field-level descriptions should NOT appear for overlapping names
        assert "Raw sensor output" not in result

    def test_class_with_only_fields_shows_attributes(self):
        """Dataclass with fields but no attrs docstring should still show Attributes."""
        result = format_docstring(parse_docstring(Product), brief=False, py_syntax=True)

        assert "Attributes:" in result
        assert "The official name of the product" in result

    def test_brief_mode_hides_attributes_for_dataclasses(self):
        """Brief mode should not include Attributes section even for dataclasses."""
        result = format_docstring(parse_docstring(Product), brief=True, py_syntax=True)

        assert "Attributes:" not in result

    def test_agent_parameter_accepted(self):
        """format_docstring should accept an agent parameter without error."""
        doc = parse_docstring(Product)
        result = format_docstring(doc, brief=False, py_syntax=True, agent="test_agent")
        assert isinstance(result, str)

        func_doc = parse_docstring(simple_func)
        result = format_docstring(func_doc, brief=False, py_syntax=True, agent="test_agent")
        assert isinstance(result, str)


class TestParseStatekACL:
    """Test cases for parse_statek_acl."""

    def test_grant_all(self):
        """Test granting access to all names with +*."""
        result = parse_statek_acl("+*")

        assert isinstance(result, Statek_ACL)
        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is True
        assert item.name == ""
        assert item.is_prefix is True
        assert item.scope == []

    def test_deny_all(self):
        """Test denying access to all names with -*."""
        result = parse_statek_acl("-*")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is False
        assert item.name == ""
        assert item.is_prefix is True
        assert item.scope == []

    def test_grant_specific_name(self):
        """Test granting access to a specific name."""
        result = parse_statek_acl("+get_slots")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is True
        assert item.name == "get_slots"
        assert item.is_prefix is False
        assert item.scope == []

    def test_deny_specific_name(self):
        """Test denying access to a specific name."""
        result = parse_statek_acl("-get_slots")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is False
        assert item.name == "get_slots"
        assert item.is_prefix is False
        assert item.scope == []

    def test_prefix_pattern(self):
        """Test prefix pattern with trailing wildcard."""
        result = parse_statek_acl("-update_*")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is False
        assert item.name == "update_"
        assert item.is_prefix is True
        assert item.scope == []

    def test_scoped_rule(self):
        """Test rule with agent scope after colon."""
        result = parse_statek_acl("-update_*: MessageDispatcher")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.access is False
        assert item.name == "update_"
        assert item.is_prefix is True
        assert item.scope == ["MessageDispatcher"]

    def test_multiple_scopes(self):
        """Test rule with multiple agent scopes."""
        result = parse_statek_acl("-update_*: MessageDispatcher, Researcher")

        assert len(result.acl) == 1
        item = result.acl[0]
        assert item.scope == ["MessageDispatcher", "Researcher"]

    def test_multiline_acl(self):
        """Test parsing multiple ACL rules."""
        acl_str = """\
+*
-get_slots
-update_*: MessageDispatcher"""
        result = parse_statek_acl(acl_str)

        assert len(result.acl) == 3

        assert result.acl[0] == ACL_Item(access=True, name="", is_prefix=True, scope=[])
        assert result.acl[1] == ACL_Item(access=False, name="get_slots", is_prefix=False, scope=[])
        assert result.acl[2] == ACL_Item(
            access=False, name="update_", is_prefix=True, scope=["MessageDispatcher"]
        )

    def test_header_is_stripped(self):
        """Test that the STATEK-ACL: header line is ignored."""
        acl_str = """\
STATEK-ACL:
    +*
    -get_slots"""
        result = parse_statek_acl(acl_str)

        assert len(result.acl) == 2
        assert result.acl[0].access is True
        assert result.acl[1].name == "get_slots"

    def test_empty_lines_ignored(self):
        """Test that blank lines are skipped."""
        acl_str = """\
+*

-get_slots
"""
        result = parse_statek_acl(acl_str)

        assert len(result.acl) == 2

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace on lines is stripped."""
        acl_str = "   +*   \n   -get_slots   "
        result = parse_statek_acl(acl_str)

        assert len(result.acl) == 2
        assert result.acl[0].access is True
        assert result.acl[1].name == "get_slots"

    def test_empty_input(self):
        """Test parsing empty string returns empty ACL."""
        result = parse_statek_acl("")

        assert isinstance(result, Statek_ACL)
        assert not result.acl

    def test_only_header(self):
        """Test parsing only the header returns empty ACL."""
        result = parse_statek_acl("STATEK-ACL:")

        assert not result.acl

    def test_full_example_from_spec(self):
        """Test the full example from the specification."""
        acl_str = """\
STATEK-ACL:
    +*
    -get_slots
    -update_*: MessageDispatcher"""
        result = parse_statek_acl(acl_str)

        assert len(result.acl) == 3
        assert result.acl[0] == ACL_Item(access=True, name="", is_prefix=True, scope=[])
        assert result.acl[1] == ACL_Item(access=False, name="get_slots", is_prefix=False, scope=[])
        assert result.acl[2] == ACL_Item(
            access=False, name="update_", is_prefix=True, scope=["MessageDispatcher"]
        )


# --- Fixtures for STATEK-ACL in class docstrings ---

class ControlledResource:  # pylint: disable=too-few-public-methods
    """A resource with access control.

    This class provides controlled access to internal state.

    Attributes:
        name (str): The resource name.

    STATEK-ACL:
        +*
        -get_slots
        -update_*: MessageDispatcher
    """


class NoAclClass:  # pylint: disable=too-few-public-methods
    """A simple class without ACL.

    Just a regular class docstring.

    Attributes:
        value (int): Some value.
    """


class AclOnlyClass:  # pylint: disable=too-few-public-methods
    """A class with ACL but no attributes.

    STATEK-ACL:
        -*
        +read_*
    """


class TestParseClassStatekACL:
    """Test cases for parsing STATEK-ACL from class docstrings."""

    def test_class_with_statek_acl(self):
        """Test that STATEK-ACL section is parsed into statek_acl field."""
        result = parse_docstring(ControlledResource)

        assert isinstance(result, ClassDocString)
        assert result.statek_acl is not None
        assert isinstance(result.statek_acl, Statek_ACL)
        assert len(result.statek_acl.acl) == 3

        assert result.statek_acl.acl[0] == ACL_Item(
            access=True, name="", is_prefix=True, scope=[]
        )
        assert result.statek_acl.acl[1] == ACL_Item(
            access=False, name="get_slots", is_prefix=False, scope=[]
        )
        assert result.statek_acl.acl[2] == ACL_Item(
            access=False, name="update_", is_prefix=True, scope=["MessageDispatcher"]
        )

    def test_class_without_statek_acl(self):
        """Test that classes without STATEK-ACL have statek_acl=None."""
        result = parse_docstring(NoAclClass)

        assert isinstance(result, ClassDocString)
        assert result.statek_acl is None

    def test_class_with_acl_preserves_other_sections(self):
        """Test that STATEK-ACL doesn't interfere with other sections."""
        result = parse_docstring(ControlledResource)

        assert result.attrs is not None
        assert len(result.attrs) == 1
        assert result.attrs[0].name == "name"
        assert result.brief_desc == "A resource with access control."
        assert "controlled access" in result.full_desc

    def test_class_with_acl_only(self):
        """Test class with ACL but no attributes section."""
        result = parse_docstring(AclOnlyClass)

        assert result.statek_acl is not None
        assert len(result.statek_acl.acl) == 2
        assert result.statek_acl.acl[0] == ACL_Item(
            access=False, name="", is_prefix=True, scope=[]
        )
        assert result.statek_acl.acl[1] == ACL_Item(
            access=True, name="read_", is_prefix=True, scope=[]
        )
        assert result.attrs is None

    def test_existing_classes_unaffected(self):
        """Test that existing classes still parse correctly with new field."""
        result = parse_docstring(ParticleSimulator)

        assert result.statek_acl is None
        assert result.attrs is not None
        assert len(result.attrs) == 3
