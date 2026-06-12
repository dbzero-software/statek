"""Compatibility helpers for importing web UI modules without NiceGUI installed."""

from __future__ import annotations


class _MissingNiceGUI:
    """Placeholder that raises a clear error when NiceGUI-backed UI code is used."""

    def __getattr__(self, name):
        raise ModuleNotFoundError(
            "nicegui is required to render the web UI. "
            "Install the package dependencies with `pip install -e .` "
            "or use `pip install -r requirements-ui.txt`."
        )


try:
    from nicegui import app, ui
except ModuleNotFoundError:
    app = _MissingNiceGUI()
    ui = _MissingNiceGUI()
