"""Docstring parsing utilities for statek package."""

import inspect
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional, Any


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
