"""Tests for docstring parsing utilities."""

import math
import pytest

from statek.docstring import (
    parse_docstring,
    format_docstring,
    FuncDocString,
    ClassDocString,
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
