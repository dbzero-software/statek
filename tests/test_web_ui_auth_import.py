"""Tests for Statek UI shared auth helper resolution."""

import importlib
import sys


def test_statek_auth_resolves_shared_oidc_from_external_paths(tmp_path, monkeypatch):
    helper_path = tmp_path / "web_ui" / "auth" / "nicegui_oidc.py"
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text(
        "class OIDCSettingsBase:\n"
        "    pass\n\n"
        "def setup_oidc_auth():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    original_module = sys.modules.pop("web_ui.auth", None)
    monkeypatch.setenv("STATEK_EXTERNAL_PATHS", str(tmp_path))

    try:
        module = importlib.import_module("web_ui.auth")
    finally:
        sys.modules.pop("web_ui.auth", None)
        if original_module is not None:
            sys.modules["web_ui.auth"] = original_module

    assert module.OIDCSettingsBase.__name__ == "OIDCSettingsBase"
    assert module.setup_oidc_auth() == "ok"
