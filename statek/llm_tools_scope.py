# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parser for LLM_TOOLS_SCOPE metadata values."""

from dataclasses import dataclass
import re
from typing import List, Optional


_ALLOWED_CATEGORIES = {"SYSTEM", "APPLICATION", "ALL"}
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


@dataclass
class LLM_ToolsScope:
    """Structured representation of an LLM_TOOLS_SCOPE metadata value.

    ``None`` means the corresponding part was not specified. This is distinct
    from an empty list and lets request integration avoid unnecessary explicit
    name resolution.
    """

    category: Optional[str] = None
    additional_tools: Optional[List[str]] = None
    removed_tools: Optional[List[str]] = None


def parse_llm_tools_scope(scope_text: str) -> LLM_ToolsScope:
    """Parse an ``LLM_TOOLS_SCOPE`` metadata value.

    Supported categories are ``SYSTEM``, ``APPLICATION``, and ``ALL``.
    Explicit tool lists start with ``+`` for additions or ``-`` for removals,
    and each list contains comma-delimited tool names.

    Args:
        scope_text: Raw metadata value to parse.

    Returns:
        Parsed ``LLM_ToolsScope``. Empty input returns an unset scope object.

    Raises:
        ValueError: If the scope category or explicit tool-list syntax is invalid.
    """
    text = scope_text.strip()
    if not text:
        return LLM_ToolsScope()

    category, pos = _parse_category(text)
    additional_tools = None
    removed_tools = None

    while pos < len(text):
        pos = _skip_whitespace(text, pos)
        if pos >= len(text):
            break
        marker = text[pos]
        if marker not in "+-":
            raise ValueError(f"Invalid LLM_TOOLS_SCOPE list marker: {marker!r}")
        names, pos = _parse_tool_list(text, pos + 1)
        if marker == "+":
            additional_tools = _append_names(additional_tools, names)
        else:
            removed_tools = _append_names(removed_tools, names)

    return LLM_ToolsScope(
        category=category,
        additional_tools=additional_tools,
        removed_tools=removed_tools,
    )


def _parse_category(text: str) -> tuple[Optional[str], int]:
    """Return normalized category and next parser position."""
    if text[0] in "+-":
        return None, 0

    pos = _find_next_marker(text, 0)
    raw_category = text[:pos].strip()
    category = raw_category.upper()
    if category not in _ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid LLM_TOOLS_SCOPE category: {raw_category!r}")
    return category, pos


def _parse_tool_list(text: str, pos: int) -> tuple[List[str], int]:
    """Parse a comma-delimited tool-name list starting at ``pos``."""
    next_pos = _find_next_marker(text, pos)
    raw_list = text[pos:next_pos]
    names = [_parse_tool_name(raw_name, raw_list) for raw_name in raw_list.split(",")]
    return names, next_pos


def _parse_tool_name(raw_name: str, raw_list: str) -> str:
    """Validate and return one stripped tool name."""
    name = raw_name.strip()
    if not name or not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid LLM_TOOLS_SCOPE tool list: {raw_list!r}")
    return name


def _append_names(existing: Optional[List[str]], names: List[str]) -> List[str]:
    """Append parsed names while preserving ``None`` for unspecified lists."""
    if existing is None:
        return list(names)
    return existing + names


def _find_next_marker(text: str, pos: int) -> int:
    """Return the index of the next add/remove marker or end of string."""
    while pos < len(text) and text[pos] not in "+-":
        pos += 1
    return pos


def _skip_whitespace(text: str, pos: int) -> int:
    """Return the next non-whitespace position."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos
