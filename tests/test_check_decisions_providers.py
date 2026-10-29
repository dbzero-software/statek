"""Offline tests for the opt-in Decisions provider smoke script."""

import asyncio

import pytest

from scripts import check_decisions_providers as smoke
from statek import Decision


def test_load_env_file_and_require_at_least_one_key(tmp_path):
    path = tmp_path / "providers.env"
    path.write_text(
        "# API keys\nOPENROUTER_API_KEY='router'\n"
        'OPENAI_API_KEY="openai"\nTYPESAFE_API_KEY=typesafe\n',
        encoding="utf-8",
    )
    config = smoke.load_env_file(path)
    assert config == {
        "OPENROUTER_API_KEY": "router",
        "OPENAI_API_KEY": "openai",
        "TYPESAFE_API_KEY": "typesafe",
    }
    smoke.validate_config(config)
    smoke.validate_config({"OPENROUTER_API_KEY": "router"})
    with pytest.raises(ValueError, match="No provider API keys"):
        smoke.validate_config({"OPENROUTER_API_KEY": "", "OPENAI_API_KEY": ""})


def test_dry_run_does_not_send_requests(monkeypatch):
    calls = []

    class FakeAPI:
        model = "test-model"

        async def process_request(self, state, questions):
            calls.append((state, questions))
            return {name: Decision(True, 0.9) for name in questions}

    monkeypatch.setattr(smoke.statek, "init", lambda: None)
    monkeypatch.setattr(
        smoke.Decisions_API, "get",
        lambda provider, **kwargs: calls.append((provider, kwargs)) or FakeAPI(),
    )
    config = {
        "OPENROUTER_API_KEY": "router",
        "OPENAI_API_KEY": "openai",
        "TYPESAFE_API_KEY": "typesafe",
    }
    assert asyncio.run(smoke.run_checks(config, dry_run=True)) == 0
    assert len(calls) == 3
    assert [call[0] for call in calls] == ["openrouter", "openai", "typesafe.ai"]


def test_dry_run_initializes_statek_with_dbzero(capsys):
    """Exercise the real startup path without contacting a provider."""
    assert asyncio.run(smoke.run_checks({"OPENAI_API_KEY": "dummy"}, dry_run=True)) == 0
    assert "openai: ready (model=gpt-6-luna)" in capsys.readouterr().out


def test_live_mode_calls_each_provider_once_with_three_questions(monkeypatch):
    calls = []

    class FakeAPI:
        async def process_request(self, state, questions):
            calls.append((state, questions))
            return {name: Decision(True, 0.9) for name in questions}

    monkeypatch.setattr(smoke.statek, "init", lambda: None)
    monkeypatch.setattr(smoke.Decisions_API, "get", lambda *args, **kwargs: FakeAPI())
    config = {
        "OPENROUTER_API_KEY": "router",
        "OPENAI_API_KEY": "openai",
        "TYPESAFE_API_KEY": "typesafe",
    }
    assert asyncio.run(smoke.run_checks(config, dry_run=False)) == 0
    assert len(calls) == 3
    assert all(set(questions) == {"billing", "team", "urgency"}
               for _, questions in calls)


def test_missing_keys_skip_providers_without_factory_calls(monkeypatch, capsys):
    calls = []

    class FakeAPI:
        model = "test-model"

        async def process_request(self, state, questions):
            calls.append((state, questions))
            return {name: Decision(True, 0.9) for name in questions}

    monkeypatch.setattr(smoke.statek, "init", lambda: None)
    monkeypatch.setattr(
        smoke.Decisions_API, "get",
        lambda provider, **kwargs: calls.append((provider, kwargs)) or FakeAPI(),
    )
    config = {"OPENROUTER_API_KEY": "router", "OPENAI_API_KEY": "", "TYPESAFE_API_KEY": " "}
    assert asyncio.run(smoke.run_checks(config, dry_run=False)) == 0
    assert len(calls) == 2
    assert calls[0] == ("openrouter", {"api_key": "router"})
    output = capsys.readouterr().out
    assert "openai: skipped (no API key)" in output
    assert "typesafe.ai: skipped (no API key)" in output
