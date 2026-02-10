"""Docstring parsing utilities for statek package."""

import inspect
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional, Any, get_type_hints, Union


# Docstring parsing structures
ArgDocString = namedtuple("ArgDocString", ["name", "type", "desc"])
AttrDocString = namedtuple("AttrDocString", ["name", "type", "desc"])
RetDocString = namedtuple("RetDocString", ["type", "desc"])
RaiseDocString = namedtuple("RaiseDocString", ["type", "desc"])


@dataclass
class FuncDocString:
    """Function or a method docstring."""
    source: Any
    name: str
    brief_desc: str
    full_desc: str
    args: Optional[List[ArgDocString]] = None
    returns: Optional[RetDocString] = None
    raises: Optional[List[RaiseDocString]] = None
    example: Optional[str] = None


@dataclass
class ClassDocString:
    """Class docstring."""
    source: Any
    name: str
    brief_desc: str
    full_desc: str
    attrs: Optional[List[AttrDocString]] = None


class DocstringParseError(Exception):
    """Exception raised when docstring parsing fails."""


def parse_docstring(type_or_func: Any) -> FuncDocString | ClassDocString:
    """Parse a docstring from a class or callable into a structured format.

    Args:
        type_or_func: Either a class or a callable (function/method)

    Returns:
        FuncDocString: for functions/methods, ClassDocString for classes

    Raises:
        DocstringParseError: If required docstring elements are missing
    """
    if isinstance(type_or_func, type):
        return _parse_class_docstring(type_or_func)
    elif callable(type_or_func):
        return _parse_func_docstring(type_or_func)
    else:
        raise DocstringParseError(
            f"Expected a class or callable, got {type(type_or_func).__name__}"
        )


def _parse_class_docstring(cls: type) -> ClassDocString:
    """Parse a class docstring into ClassDocString structure."""
    docstring = inspect.getdoc(cls)
    if not docstring:
        raise DocstringParseError(f"Class '{cls.__name__}' has no docstring")

    brief_desc, full_desc, sections = _parse_docstring_sections(docstring)

    attrs = None
    if "Attributes" in sections:
        attrs = _parse_typed_items(sections["Attributes"], AttrDocString)

    return ClassDocString(
        source=cls,
        name=cls.__name__,
        brief_desc=brief_desc,
        full_desc=full_desc,
        attrs=attrs
    )


def _parse_func_docstring(func: Callable) -> FuncDocString:
    """Parse a function/method docstring into FuncDocString structure."""
    docstring = inspect.getdoc(func)
    if not docstring:
        raise DocstringParseError(f"Function '{func.__name__}' has no docstring")

    brief_desc, full_desc, sections = _parse_docstring_sections(docstring)

    args = None
    if "Args" in sections:
        args = _parse_typed_items(sections["Args"], ArgDocString)

    returns = None
    if "Returns" in sections:
        returns = _parse_return_section(sections["Returns"])

    raises = None
    if "Raises" in sections:
        raises = _parse_typed_items(sections["Raises"], RaiseDocString)

    example = None
    if "Example" in sections:
        example = sections["Example"].strip()
    elif "Examples" in sections:
        example = sections["Examples"].strip()

    # Validate that all function arguments are documented
    _validate_args_documented(func, args)

    return FuncDocString(
        source=func,
        name=func.__name__,
        brief_desc=brief_desc,
        full_desc=full_desc,
        args=args,
        returns=returns,
        raises=raises,
        example=example
    )


def _parse_docstring_sections(docstring: str) -> tuple[str, str, dict]:
    """Parse docstring into brief description, full description, and sections.

    Returns:
        Tuple of (brief_desc, full_desc, sections_dict)
    """
    lines = docstring.split('\n')

    # Section headers we recognize
    section_headers = {'Args', 'Arguments', 'Attributes', 'Returns', 'Return',
                       'Raises', 'Raise', 'Example', 'Examples', 'Note', 'Notes'}

    # Find section boundaries
    section_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check if line is a section header (word followed by colon)
        if stripped.rstrip(':') in section_headers and stripped.endswith(':'):
            section_starts.append((i, stripped.rstrip(':')))

    # Extract description (everything before first section)
    if section_starts:
        desc_end = section_starts[0][0]
    else:
        desc_end = len(lines)

    desc_lines = lines[:desc_end]

    # Brief description is the first paragraph
    brief_lines = []
    for line in desc_lines:
        if line.strip() == '':
            break
        brief_lines.append(line.strip())
    brief_desc = ' '.join(brief_lines)

    # Full description is everything up to first section
    full_desc = '\n'.join(line for line in desc_lines).strip()

    # Parse sections
    sections = {}
    for idx, (start_line, section_name) in enumerate(section_starts):
        # Determine end of this section
        if idx + 1 < len(section_starts):
            end_line = section_starts[idx + 1][0]
        else:
            end_line = len(lines)

        # Extract section content (skip the header line)
        section_content = '\n'.join(lines[start_line + 1:end_line])
        sections[section_name] = section_content

    return brief_desc, full_desc, sections


def _parse_typed_items(section_content: str, item_class) -> List:
    """Parse a section containing typed items (Args, Attributes, Raises).

    Handles format: name (type): description
    or for Raises: ExceptionType: description
    """
    items = []
    lines = section_content.split('\n')

    current_name = None
    current_type = None
    current_desc_lines = []

    # Pattern for "name (type): description" or "name: description"
    arg_pattern = re.compile(r'^\s{4,}(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$')
    # Pattern for continuation lines (more indented)
    continuation_pattern = re.compile(r'^\s{8,}(.+)$')

    # Check if this is a 2-field namedtuple (type, desc) like RaiseDocString
    is_two_field = len(item_class._fields) == 2

    for line in lines:
        if not line.strip():
            continue

        arg_match = arg_pattern.match(line)
        if arg_match:
            # Save previous item if exists
            if current_name is not None:
                desc = ' '.join(current_desc_lines).strip()
                if is_two_field:
                    # For RaiseDocString: name is the exception type
                    items.append(item_class(current_name, desc))
                else:
                    items.append(item_class(current_name, current_type, desc))

            current_name = arg_match.group(1)
            current_type = arg_match.group(2)  # May be None
            first_desc = arg_match.group(3).strip()
            current_desc_lines = [first_desc] if first_desc else []
        else:
            cont_match = continuation_pattern.match(line)
            if cont_match and current_name is not None:
                current_desc_lines.append(cont_match.group(1).strip())

    # Don't forget the last item
    if current_name is not None:
        desc = ' '.join(current_desc_lines).strip()
        if is_two_field:
            items.append(item_class(current_name, desc))
        else:
            items.append(item_class(current_name, current_type, desc))

    return items if items else None


def _parse_return_section(section_content: str) -> Optional[RetDocString]:
    """Parse the Returns section into RetDocString.

    Handles format: type: description
    """
    lines = [line for line in section_content.split('\n') if line.strip()]
    if not lines:
        return None

    # Pattern for "type: description"
    ret_pattern = re.compile(r'^\s*(\S+)\s*:\s*(.*)$')

    first_line = lines[0]
    match = ret_pattern.match(first_line)

    if match:
        ret_type = match.group(1)
        desc_parts = [match.group(2).strip()]

        # Collect continuation lines
        for line in lines[1:]:
            desc_parts.append(line.strip())

        return RetDocString(ret_type, ' '.join(desc_parts).strip())

    # If no type specified, treat whole content as description
    return RetDocString(None, ' '.join(line.strip() for line in lines))


def _validate_args_documented(func: Callable, documented_args: Optional[List[ArgDocString]]):
    """Validate that all function arguments are documented.

    Args:
        func: The function to validate
        documented_args: List of documented arguments

    Raises:
        DocstringParseError: If any argument is not documented
    """
    sig = inspect.signature(func)
    func_params = set(sig.parameters.keys())

    # Remove 'self' and 'cls' as they don't need documentation
    func_params.discard('self')
    func_params.discard('cls')

    # Get documented argument names
    documented_names = set()
    if documented_args:
        documented_names = {arg.name for arg in documented_args}

    # Find undocumented arguments
    undocumented = func_params - documented_names
    if undocumented:
        raise DocstringParseError(
            f"Function '{func.__name__}' has undocumented arguments: {', '.join(sorted(undocumented))}"
        )


def format_docstring(docstring: FuncDocString | ClassDocString,
                     brief: bool = False, py_syntax: bool = True) -> str:
    """Format a parsed docstring into a string representation.

    Args:
        docstring: The structured docstring object
        brief: Flag enabling brief-only formatting
        py_syntax: Flag requesting output using Python syntax

    Returns:
        str: The formatted string representation
    """
    if isinstance(docstring, ClassDocString):
        return _format_class_docstring(docstring, brief, py_syntax)
    else:
        return _format_func_docstring(docstring, brief, py_syntax)


def _format_func_docstring(docstring: FuncDocString, brief: bool, py_syntax: bool) -> str:
    """Format a function docstring."""
    # Get signature from source function
    sig_str = _format_signature(docstring.source, docstring.name, py_syntax)

    if py_syntax:
        return _format_func_py_syntax(docstring, sig_str, brief)
    else:
        return _format_func_plain(docstring, sig_str, brief)


def _format_func_plain(docstring: FuncDocString, sig_str: str, brief: bool) -> str:
    """Format function docstring in plain text format."""
    lines = [sig_str]
    lines.append(f"    {docstring.brief_desc}")

    if docstring.returns:
        lines.append(f"    Returns: {docstring.returns.desc}")

    return '\n'.join(lines)


def _format_func_py_syntax(docstring: FuncDocString, sig_str: str, brief: bool) -> str:
    """Format function docstring in Python syntax."""
    lines = [f"def {sig_str}:"]

    # Build docstring content
    doc_lines = []

    if brief:
        # Brief mode: just brief_desc and returns
        doc_lines.append(docstring.brief_desc)
        if docstring.returns:
            doc_lines.append(f"Returns: {docstring.returns.desc}")
    else:
        # Full mode: full_desc, Args, Raises, Example
        if docstring.full_desc:
            doc_lines.append(docstring.full_desc)

        if docstring.args:
            doc_lines.append("Args:")
            for arg in docstring.args:
                type_str = f" ({arg.type})" if arg.type else ""
                doc_lines.append(f"    {arg.name}{type_str}: {arg.desc}")

        if docstring.raises:
            doc_lines.append("Raises:")
            for raise_doc in docstring.raises:
                doc_lines.append(f"    {raise_doc.type}: {raise_doc.desc}")

        if docstring.example:
            doc_lines.append("Example:")
            for example_line in docstring.example.split('\n'):
                doc_lines.append(f"    {example_line}")

    # Format as Python docstring with 4-space indentation
    lines.append('    """' + doc_lines[0])
    if len(doc_lines) > 1:
        for doc_line in doc_lines[1:]:
            lines.append(f"    {doc_line}")
        lines.append('    """')
    else:
        lines[-1] += '"""'

    return '\n'.join(lines)


def _format_class_docstring(docstring: ClassDocString, brief: bool, py_syntax: bool) -> str:
    """Format a class docstring."""
    if py_syntax:
        return _format_class_py_syntax(docstring, brief)
    else:
        return _format_class_plain(docstring, brief)


def _format_class_plain(docstring: ClassDocString, brief: bool) -> str:
    """Format class docstring in plain text format."""
    lines = [docstring.name]
    lines.append(f"    {docstring.brief_desc}")

    return '\n'.join(lines)


def _format_class_py_syntax(docstring: ClassDocString, brief: bool) -> str:
    """Format class docstring in Python syntax."""
    lines = [f"class {docstring.name}:"]

    # Build docstring content
    doc_lines = []

    if brief:
        doc_lines.append(docstring.brief_desc)
    else:
        # Full description
        if docstring.full_desc:
            doc_lines.append(docstring.full_desc)

        # Attributes section
        if docstring.attrs:
            doc_lines.append("")
            doc_lines.append("Attributes:")
            for attr in docstring.attrs:
                type_str = f" ({attr.type})" if attr.type else ""
                doc_lines.append(f"    {attr.name}{type_str}: {attr.desc}")

    # Format as Python docstring
    lines.append('    """' + doc_lines[0])
    if len(doc_lines) > 1:
        for doc_line in doc_lines[1:]:
            lines.append(f"    {doc_line}")
        lines.append('    """')
    else:
        lines[-1] += '"""'

    # In detailed mode, add member functions
    if not brief:
        member_funcs = _get_member_functions(docstring.source)
        for func in member_funcs:
            try:
                func_doc = parse_docstring(func)
                func_formatted = _format_func_docstring(func_doc, brief=True, py_syntax=True)
                # Indent the function definition
                indented_lines = ['    ' + line for line in func_formatted.split('\n')]
                lines.extend(indented_lines)
            except DocstringParseError:
                # Skip functions without proper docstrings
                pass

    return '\n'.join(lines)


def _get_member_functions(cls: type) -> List[Callable]:
    """Get all non-system member functions of a class."""
    members = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_'):
            members.append(method)
    return members


def _format_signature(func: Callable, name: str, include_types: bool) -> str:
    """Format a function signature.

    Args:
        func: The source function
        name: The function name
        include_types: Whether to include type annotations

    Returns:
        Formatted signature string
    """
    sig = inspect.signature(func)

    # Get type hints if needed
    hints = {}
    if include_types:
        try:
            hints = get_type_hints(func)
        except Exception:  # pylint: disable=broad-exception-caught
            hints = getattr(func, "__annotations__", {})

    params = []
    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        if include_types and param_name in hints:
            type_str = _format_type_hint(hints[param_name])
            param_str = f"{param_name}: {type_str}"
        else:
            param_str = param_name

        if param.default is not inspect.Parameter.empty:
            if include_types:
                param_str += f" = {_format_default(param.default)}"
            else:
                param_str += f"={_format_default(param.default)}"

        params.append(param_str)

    params_str = ", ".join(params)

    # Add return type if present
    if include_types and "return" in hints:
        return_str = _format_type_hint(hints["return"])
        return f"{name}({params_str}) -> {return_str}"

    return f"{name}({params_str})"


def _format_type_hint(type_hint) -> str:
    """Format a type hint into a readable string."""
    from typing import get_origin, get_args

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Union types
    if origin is Union:
        formatted_args = [_format_type_hint(arg) for arg in args]
        return " | ".join(formatted_args)

    # Handle generic types
    if origin is not None:
        if args:
            formatted_args = ", ".join(_format_type_hint(arg) for arg in args)
            origin_name = getattr(origin, "__name__", str(origin))
            return f"{origin_name}[{formatted_args}]"
        return getattr(origin, "__name__", str(origin))

    # Handle basic types
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__

    # Fallback
    type_str = str(type_hint)
    if type_str.startswith("typing."):
        type_str = type_str.replace("typing.", "")
    return type_str


def _format_default(default) -> str:
    """Format a default value."""
    if default is None:
        return "None"
    if isinstance(default, str):
        return f'"{default}"'
    return repr(default)
