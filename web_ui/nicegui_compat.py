"""Compatibility helpers for importing web UI modules without NiceGUI installed."""

from __future__ import annotations


class _MissingNiceGUI:
    """Placeholder that raises a clear error when NiceGUI-backed UI code is used."""

    def __getattr__(self, name):
        raise ModuleNotFoundError(
            "nicegui is required to render the web UI. "
            "Install requirements from requirements-ui.txt to use this module."
        )


try:
    from nicegui import app, ui
except ModuleNotFoundError:
    app = _MissingNiceGUI()
    ui = _MissingNiceGUI()
