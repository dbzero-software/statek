"""Docstring parsing utilities for statek package."""  # pylint: disable=too-many-lines

import dataclasses
import inspect
import re
from collections import namedtuple
from dataclasses import dataclass, fields as dataclass_fields
from typing import Callable, Dict, List, Optional, Any, get_type_hints, Union


_VARIANT_SEP_RE = re.compile(r'/([A-Za-z0-9]{0,8}):')


def parse_variants(text: str) -> Dict | str:
    """Parse a multi-variant encoded string into a dict of named variants.

    If the input starts with the ``VARIANTS/`` prefix the string is split on
    ``/NAME:`` delimiters and each segment is returned as a dict entry keyed
    by the lower-cased variant name (empty string for the default variant).
    When the input is not a multi-variant string it is returned unchanged.

    Args:
        text (str): The input string, possibly using multi-variant encoding.

    Returns:
        Dict | str: A dict mapping variant name → text when multi-variant,
            or the original string otherwise.
    """
    stripped = text.lstrip()
    if not stripped.startswith('VARIANTS/'):
        return text

    body = stripped[len('VARIANTS'):]  # starts with /name:
    matches = list(_VARIANT_SEP_RE.finditer(body))
    if not matches:
        return text

    result = {}
    for i, match in enumerate(matches):
        name = match.group(1).lower()
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        result[name] = body[text_start:text_end]
    return result


def get_text_variant(text: str, variant_name: Optional[str]) -> str:
    """Retrieve a named variant from a multi-variant string.

    When the input is a multi-variant string (see :func:`parse_variants`) the
    variant identified by *variant_name* is returned.  If that variant is
    absent the default variant (empty name) is returned instead.  When the
    input is not a multi-variant string it is returned as-is.  Matching on
    *variant_name* is case-insensitive.  When *variant_name* is ``None`` it
    selects the default variant (same as passing an empty string).

    Args:
        text (str): Either a multi-variant or a regular string.
        variant_name (str): The variant name to retrieve (case insensitive),
            or None to select the default variant.

    Returns:
        str: The retrieved text.
    """
    parsed = parse_variants(text)
    if isinstance(parsed, str):
        return parsed
    key = variant_name.lower() if variant_name is not None else ''
    if key in parsed:
        return parsed[key]
    return parsed.get('', '')


# Docstring parsing structures
ACL_Item = namedtuple("ACL_Item", ["access", "name", "is_prefix", "scope"])
ArgDocString = namedtuple("ArgDocString", ["name", "type", "desc"])
AttrDocString = namedtuple("AttrDocString", ["name", "type", "desc"])
RetDocString = namedtuple("RetDocString", ["type", "desc"])
RaiseDocString = namedtuple("RaiseDocString", ["type", "desc"])


@dataclass
class FuncDocString:  # pylint: disable=too-many-instance-attributes
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
    """dataclass fields"""
    fields: Optional[List[AttrDocString]] = None
    """Tag descriptors"""
    tags: Optional[List[str]] = None
    """Explicit attribute declarations in the original docstring"""
    attrs: Optional[List[AttrDocString]] = None
    props: Optional[List[AttrDocString]] = None
    """If not defined, the default ACL from StatekSettings is used"""
    statek_acl: Optional['Statek_ACL'] = None


class DocstringParseError(Exception):
    """Exception raised when docstring parsing fails."""


def parse_docstring(type_or_func: Any, variant_name: str = None) -> FuncDocString | ClassDocString:
    """Parse a docstring from a class or callable into a structured format.

    Args:
        type_or_func: Either a class or a callable (function/method)
        variant_name: Optional variant name (e.g. "tool"). When provided, the
            named variant is extracted from multi-variant docstrings before
            parsing. When None the raw docstring is used.

    Returns:
        FuncDocString: for functions/methods, ClassDocString for classes

    Raises:
        DocstringParseError: If required docstring elements are missing
    """
    if isinstance(type_or_func, type):
        return _parse_class_docstring(type_or_func, variant_name)
    if callable(type_or_func):
        return _parse_func_docstring(type_or_func, variant_name)
    raise DocstringParseError(
        f"Expected a class or callable, got {type(type_or_func).__name__}"
    )


def parse_tool_docstring(type_or_func: Any) -> FuncDocString | ClassDocString:
    """Parse the "tool" variant of a docstring from a class or callable.

    Convenience wrapper around :func:`parse_docstring` that always selects the
    ``"tool"`` variant from multi-variant docstrings.  Falls back to the
    default variant (or the plain docstring) when no ``tool`` variant exists.

    Args:
        type_or_func: Either a class or a callable (function/method)

    Returns:
        FuncDocString: for functions/methods, ClassDocString for classes

    Raises:
        DocstringParseError: If required docstring elements are missing
    """
    return parse_docstring(type_or_func, variant_name="tool")


def _parse_class_docstring(cls: type, variant_name: str = None) -> ClassDocString:
    """Parse a class docstring into ClassDocString structure."""
    docstring = inspect.getdoc(cls)
    if not docstring:
        raise DocstringParseError(f"Class '{cls.__name__}' has no docstring")

    brief_desc, extra_desc, sections = _parse_docstring_sections(docstring)

    attrs = None
    if "Attributes" in sections:
        attrs = _parse_typed_items(sections["Attributes"], AttrDocString)

    tags = None
    if "Tags" in sections:
        tags = _parse_tags_section(sections["Tags"])

    fields = _extract_dataclass_fields(cls)
    props = _extract_class_properties(cls)

    statek_acl = None
    if "STATEK-ACL" in sections:
        statek_acl = parse_statek_acl(sections["STATEK-ACL"])

    brief_desc = get_text_variant(brief_desc, variant_name)
    extra_desc = get_text_variant(extra_desc, variant_name).strip()
    full_desc = (brief_desc + '\n\n' + extra_desc) if extra_desc else brief_desc
    if attrs is not None:
        attrs = [AttrDocString(a.name, a.type,
                               get_text_variant(a.desc, variant_name) if a.desc else a.desc)
                 for a in attrs]
    if fields is not None:
        fields = [AttrDocString(f.name, f.type,
                                get_text_variant(f.desc, variant_name) if f.desc else f.desc)
                  for f in fields]
    if props is not None:
        props = [AttrDocString(p.name, p.type,
                               get_text_variant(p.desc, variant_name) if p.desc else p.desc)
                 for p in props]

    return ClassDocString(
        source=cls,
        name=cls.__name__,
        brief_desc=brief_desc,
        full_desc=full_desc,
        fields=fields,
        tags=tags,
        attrs=attrs,
        props=props,
        statek_acl=statek_acl
    )


def _extract_dataclass_fields(cls: type) -> Optional[List[AttrDocString]]:
    """Extract fields from a dataclass (including db0.memo-wrapped dataclasses).

    Uses dataclasses.fields() for field names/metadata and get_type_hints()
    for resolved type information.

    Returns:
        List of AttrDocString for each field, or None if cls is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        return None

    try:
        hints = get_type_hints(cls)
    except Exception:  # pylint: disable=broad-exception-caught
        hints = {}

    result = []
    for f in dataclass_fields(cls):
        type_hint = hints.get(f.name)
        type_str = _format_type_hint(type_hint) if type_hint else None
        doc = f.metadata.get("doc") if f.metadata else None
        result.append(AttrDocString(name=f.name, type=type_str, desc=doc))

    return result if result else None


def _extract_class_properties(cls: type) -> Optional[List[AttrDocString]]:
    """Extract @property members from a class.

    For each property, extracts the name, return type from the getter's type
    hints, and description from the getter's Returns docstring section.

    Returns:
        List of AttrDocString for each non-private property, or None if none found.
    """
    result = []
    seen: set = set()
    # Walk MRO using vars() to avoid triggering descriptors (e.g. db0.memo attributes)
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if name.startswith('_') or name in seen:
                continue
            seen.add(name)
            if not isinstance(value, property):
                continue

            getter = value.fget
            if getter is None:
                continue

            type_str = None
            try:
                hints = get_type_hints(getter)
                if 'return' in hints:
                    type_str = _format_type_hint(hints['return'])
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            desc = None
            prop_docstring = inspect.getdoc(getter)
            if prop_docstring:
                _, _, sections = _parse_docstring_sections(prop_docstring)
                if 'Returns' in sections:
                    ret = _parse_return_section(sections['Returns'])
                    if ret:
                        desc = ret.desc

            result.append(AttrDocString(name=name, type=type_str, desc=desc))

    return result if result else None


def _parse_func_docstring(func: Callable, variant_name: str = None) -> FuncDocString:
    """Parse a function/method docstring into FuncDocString structure."""
    docstring = inspect.getdoc(func)
    if not docstring:
        raise DocstringParseError(f"Function '{func.__name__}' has no docstring")

    brief_desc, extra_desc, sections = _parse_docstring_sections(docstring)

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

    # For temporal functions, override returns with complement's return type
    if getattr(func, "__is_temporal__", False):
        complement_ret = _extract_complement_return(func)
        if complement_ret is not None:
            returns = complement_ret

    brief_desc = get_text_variant(brief_desc, variant_name)
    extra_desc = get_text_variant(extra_desc, variant_name).strip()
    full_desc = (brief_desc + '\n\n' + extra_desc) if extra_desc else brief_desc
    if args is not None:
        args = [ArgDocString(a.name, a.type,
                             get_text_variant(a.desc, variant_name) if a.desc else a.desc)
                for a in args]
    if returns is not None:
        returns = RetDocString(returns.type,
                               get_text_variant(returns.desc, variant_name)
                               if returns.desc else returns.desc)
    if raises is not None:
        raises = [RaiseDocString(r.type,
                                 get_text_variant(r.desc, variant_name) if r.desc else r.desc)
                  for r in raises]
    if example is not None:
        example = get_text_variant(example, variant_name)

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


def _parse_docstring_sections(docstring: str) -> tuple[str, str, dict]:  # pylint: disable=too-many-locals
    """Parse docstring into brief description, extra description, and sections.

    Returns:
        Tuple of (brief_desc, extra_desc, sections_dict) where extra_desc is
        the description text after the first paragraph (may be a VARIANTS string).
    """
    lines = docstring.split('\n')

    # Section headers we recognize
    section_headers = {'Args', 'Arguments', 'Attributes', 'Returns', 'Return',
                       'Raises', 'Raise', 'Example', 'Examples', 'Note', 'Notes',
                       'Tags', 'STATEK-ACL'}

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

    # Extra description is everything after the first paragraph
    extra_desc = '\n'.join(desc_lines[len(brief_lines):]).strip()

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

    return brief_desc, extra_desc, sections


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


def _parse_tags_section(section_content: str) -> Optional[List[str]]:
    """Parse a Tags section into a list of tag name strings.

    Each non-empty line is a tag entry. If a line contains `` - ``, only the
    part before the separator is used as the tag name; otherwise the whole
    stripped line is the tag name.
    """
    tags = []
    for line in section_content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if ' - ' in stripped:
            tag_name = stripped.split(' - ', 1)[0].strip()
        else:
            tag_name = stripped
        if tag_name:
            tags.append(tag_name)
    return tags if tags else None


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


_SKIP_PARAM_KINDS = {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}


def _validate_args_documented(func: Callable, documented_args: Optional[List[ArgDocString]]):
    """Validate that all function arguments are documented."""
    sig = inspect.signature(func)
    func_params = {
        name for name, p in sig.parameters.items()
        if name not in ('self', 'cls') and p.kind not in _SKIP_PARAM_KINDS
    }
    documented_names = {arg.name for arg in documented_args} if documented_args else set()
    undocumented = func_params - documented_names
    if undocumented:
        raise DocstringParseError(
            f"Function '{func.__name__}' has undocumented arguments: "
            f"{', '.join(sorted(undocumented))}"
        )


def _get_merged_attributes(docstring: ClassDocString) -> Optional[List[AttrDocString]]:
    """Merge dataclass fields, properties, and docstring attrs into a single attributes list.

    Attrs take precedence over both fields and props for entries with the same name.
    Fields come first, then props not already in fields, then attrs-only entries.

    Args:
        docstring: The class docstring containing fields, props, and attrs

    Returns:
        Merged list of AttrDocString, or None if empty
    """
    # Build ordered list of all introspected items (fields then props)
    introspected: List[AttrDocString] = []
    if docstring.fields:
        introspected.extend(docstring.fields)
    if docstring.props:
        seen = {f.name for f in introspected}
        introspected.extend(p for p in docstring.props if p.name not in seen)

    if not introspected and not docstring.attrs:
        return None

    if not introspected:
        return docstring.attrs

    if not docstring.attrs:
        return introspected

    # attrs take precedence by name over introspected items
    merged = {item.name: item for item in introspected}
    for attr in docstring.attrs:
        merged[attr.name] = attr

    # Preserve introspected order, appending attrs-only entries at the end
    introspected_names = [item.name for item in introspected]
    result = [merged[name] for name in introspected_names if name in merged]
    for attr in docstring.attrs:
        if attr.name not in {item.name for item in introspected}:
            result.append(attr)

    return result if result else None


def format_docstring(docstring: FuncDocString | ClassDocString,
                     brief: bool = False, py_syntax: bool = True,
                     agent: str = None,
                     default_acl: 'Optional[Statek_ACL]' = None) -> str:
    """Format a parsed docstring into a string representation.

    Args:
        docstring: The structured docstring object
        brief: Flag enabling brief-only formatting
        py_syntax: Flag requesting output using Python syntax
        agent: Agent identifier for ACL resolution
        default_acl: Fallback ACL when no class-level STATEK-ACL is defined

    Returns:
        str: The formatted string representation
    """
    if isinstance(docstring, ClassDocString):
        return _format_class_docstring(docstring, brief, py_syntax, agent, default_acl)
    return _format_func_docstring(docstring, brief, py_syntax)


def _format_func_docstring(docstring: FuncDocString, brief: bool, py_syntax: bool) -> str:
    """Format a function docstring."""
    # Get signature from source function
    sig_str = _format_signature(docstring.source, docstring.name, py_syntax)

    if py_syntax:
        return _format_func_py_syntax(docstring, sig_str, brief)
    return _format_func_plain(docstring, sig_str, brief)


def _format_func_plain(docstring: FuncDocString, sig_str: str, brief: bool) -> str:  # pylint: disable=unused-argument
    """Format function docstring in plain text format."""
    lines = [sig_str]
    lines.append(f"    {docstring.brief_desc}")

    if docstring.returns:
        lines.append(f"    Returns: {docstring.returns.desc}")

    return '\n'.join(lines)


def _format_func_py_syntax(docstring: FuncDocString, sig_str: str, brief: bool) -> str:  # pylint: disable=too-many-branches
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

        if docstring.returns:
            doc_lines.append(f"Returns: {docstring.returns.desc}")

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


def _format_class_docstring(docstring: ClassDocString, brief: bool, py_syntax: bool,
                             agent: str = None,
                             default_acl: 'Optional[Statek_ACL]' = None) -> str:
    """Format a class docstring."""
    if py_syntax:
        return _format_class_py_syntax(docstring, brief, agent, default_acl)
    return _format_class_plain(docstring, brief)


def _format_class_plain(docstring: ClassDocString, brief: bool) -> str:  # pylint: disable=unused-argument
    """Format class docstring in plain text format."""
    lines = [docstring.name]
    lines.append(f"    {docstring.brief_desc}")

    return '\n'.join(lines)


def _format_class_py_syntax(docstring: ClassDocString, brief: bool,  # pylint: disable=too-many-locals,too-many-branches
                             agent: str = None,
                             default_acl: 'Optional[Statek_ACL]' = None) -> str:
    """Format class docstring in Python syntax."""
    effective_acl = docstring.statek_acl or default_acl

    lines = [f"class {docstring.name}:"]

    # Build docstring content
    doc_lines = []

    if brief:
        doc_lines.append(docstring.brief_desc)
    else:
        # Full description
        if docstring.full_desc:
            doc_lines.append(docstring.full_desc)

        # Tags section
        if docstring.tags:
            doc_lines.append("")
            doc_lines.append("Tags:")
            for tag in docstring.tags:
                doc_lines.append(f"    {tag}")

        # Attributes section (merged from fields and attrs)
        merged_attrs = _get_merged_attributes(docstring)
        if merged_attrs:
            if effective_acl is not None:
                merged_attrs = [a for a in merged_attrs
                                if effective_acl.has_access(a.name, agent)]
            if merged_attrs:
                doc_lines.append("")
                doc_lines.append("Attributes:")
                for attr in merged_attrs:
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
        if effective_acl is not None:
            member_funcs = [f for f in member_funcs
                            if effective_acl.has_access(f.__name__, agent)]
        for func in member_funcs:
            try:
                func_doc = parse_tool_docstring(func)
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
        # For temporal functions, use the complement function's return type
        return_str = _get_effective_return_type(func, hints)
        return f"{name}({params_str}) -> {return_str}"

    return f"{name}({params_str})"


def _extract_complement_return(func: Callable) -> Optional[RetDocString]:
    """Extract the return docstring from a temporal function's complement.

    Builds a RetDocString using the complement's return type and its docstring.

    Args:
        func: A temporal function

    Returns:
        RetDocString: with complement's return type and description, or None
    """
    complement_func = getattr(func, "__temporal_complement__", None)
    if complement_func is None:
        return None

    # Get the complement return type
    ret_type = _extract_complement_return_type(func)
    if ret_type is None:
        return None

    # Get the complement's docstring for the return description
    complement_doc = inspect.getdoc(complement_func)
    desc = None
    if complement_doc:
        _, _, sections = _parse_docstring_sections(complement_doc)
        if "Returns" in sections:
            parsed_ret = _parse_return_section(sections["Returns"])
            if parsed_ret:
                desc = parsed_ret.desc

    return RetDocString(ret_type, desc or ret_type)


def _get_effective_return_type(func: Callable, hints: dict) -> str:
    """Get the effective return type for a function.

    For temporal functions, returns the complement function's return type.
    For regular functions, returns the annotated return type.

    Args:
        func: The function to get return type for
        hints: The type hints dictionary

    Returns:
        Formatted return type string
    """
    # Check if this is a temporal function
    if getattr(func, "__is_temporal__", False):
        complement_type = _extract_complement_return_type(func)
        if complement_type:
            return complement_type

    # Default: use the function's own return type
    return _format_type_hint(hints["return"])


def _extract_complement_return_type(func: Callable) -> Optional[str]:
    """Extract return type from a temporal function's complement function.

    Args:
        func: A temporal function

    Returns:
        Formatted return type string from complement, or None if not found
    """
    # Get complement from the __temporal_complement__ attribute
    if hasattr(func, "__temporal_complement__"):
        complement_func = func.__temporal_complement__
        try:
            complement_hints = get_type_hints(complement_func)
            if "return" in complement_hints:
                return _format_type_hint(complement_hints["return"])
        except (AttributeError, ValueError, TypeError):
            pass

    # Fallback to closure extraction for older temporal functions
    if hasattr(func, "__closure__") and func.__closure__:
        for cell in func.__closure__:
            try:
                complement_func = cell.cell_contents
                if not callable(complement_func):
                    continue
                complement_hints = get_type_hints(complement_func)
                if "return" in complement_hints:
                    return _format_type_hint(complement_hints["return"])
            except (AttributeError, ValueError, TypeError):
                continue

    return None


def _format_type_hint(type_hint) -> str:
    """Format a type hint into a readable string."""
    from typing import get_origin, get_args  # pylint: disable=import-outside-toplevel

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


@dataclass
class Statek_ACL:
    """Parsed STATEK Access Control list."""
    acl: List[ACL_Item]

    def has_access(self, resource_name: str, agent: str = None) -> bool:
        """Determine if access to a resource is granted.

        Iterates the entire ACL; the last matching rule wins. Rules with a
        scope list only match when the given agent is in that list. Without
        an agent, scoped rules are skipped entirely. The default is False.

        Args:
            resource_name (str): The resource (field, method) name to check.
            agent (str): Optional agent name; None matches general rules only.

        Returns:
            bool: True if access is granted, False if denied.
        """
        resolved = False
        for item in self.acl:
            if item.scope:
                if agent is None or agent not in item.scope:
                    continue
            if item.is_prefix:
                if not resource_name.startswith(item.name):
                    continue
            else:
                if resource_name != item.name:
                    continue
            resolved = item.access
        return resolved


def parse_statek_acl(statek_acl: str) -> Statek_ACL:
    """Parse a STATEK-ACL definition string into a structured Statek_ACL.

    Args:
        statek_acl: The ACL definition string, optionally including
            the "STATEK-ACL:" header line

    Returns:
        Statek_ACL: The parsed access control list
    """
    items = []
    for line in statek_acl.split('\n'):
        line = line.strip()
        if not line or line.startswith("STATEK-ACL"):
            continue

        # Determine access
        if line[0] == '+':
            access = True
        elif line[0] == '-':
            access = False
        else:
            continue

        rest = line[1:]

        # Split off optional scope after colon
        scope = []
        if ':' in rest:
            name_part, scope_part = rest.split(':', 1)
            scope = [s.strip() for s in scope_part.split(',') if s.strip()]
        else:
            name_part = rest

        name_part = name_part.strip()

        # Handle wildcard / prefix
        if name_part.endswith('*'):
            is_prefix = True
            name = name_part[:-1]
        else:
            is_prefix = False
            name = name_part

        items.append(ACL_Item(access=access, name=name, is_prefix=is_prefix, scope=scope))

    return Statek_ACL(acl=items)
