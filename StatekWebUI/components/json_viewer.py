"""Reusable JSON viewer component for NiceGUI pages."""

from __future__ import annotations

import json
from typing import Any

from ..nicegui_compat import ui


def _coerce_json_value(value: Any) -> Any:
    """Parse JSON strings when possible; otherwise return the original value."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _build_json_editor_props(
    value: Any,
    *,
    mode: str = 'tree',
    read_only: bool = True,
    navigation_bar: bool = True,
) -> dict[str, Any]:
    """Return a stable json_editor configuration for read-heavy inspection flows."""
    if not mode:
        raise ValueError('mode must be a non-empty string')
    return {
        'content': {'json': _coerce_json_value(value)},
        'mode': mode,
        'readOnly': read_only,
        'mainMenuBar': False,
        'navigationBar': navigation_bar,
        'statusBar': False,
    }


def create_json_viewer(
    value: Any,
    *,
    mode: str = 'tree',
    read_only: bool = True,
    height: str = '28rem',
):
    """Create a read-only JSON viewer with expandable/collapsible branches."""
    editor = ui.json_editor(_build_json_editor_props(
        value,
        mode=mode,
        read_only=read_only,
    )).classes('w-full')
    if height:
        editor.style(f'height: {height}')
    return editor
