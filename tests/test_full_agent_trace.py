"""Tests for opt-in full agent execution tracing."""

import json
import logging
from unittest.mock import patch

import dbzero as db0
import pytest

from statek.executors.utils import exec_cli_step
from statek.settings import full_agent_trace


def available_tool() -> str:
    """Return a fixed value for Python CLI tracing tests."""
    return "ok"


@pytest.fixture(autouse=True)
def reset_full_agent_trace_logger():
    """Remove trace handlers so each test owns its temporary trace file."""
    logger = logging.getLogger("statek.full_agent_trace")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_full_agent_trace_is_disabled_without_environment_flag(monkeypatch, tmp_path):
    """Tracing must not emit log entries during normal execution."""
    monkeypatch.delenv("FULL_AGENT_TRACE", raising=False)
    trace_path = tmp_path / "full_agent_trace.jsonl"
    monkeypatch.setenv("FULL_AGENT_TRACE_PATH", str(trace_path))

    full_agent_trace("event", {"value": "visible only when enabled"})

    assert not trace_path.exists()


def test_full_agent_trace_writes_redacted_json_line(monkeypatch, tmp_path):
    """Tracing must preserve diagnostic structure without exposing credentials."""
    monkeypatch.setenv("FULL_AGENT_TRACE", "true")
    trace_path = tmp_path / "full_agent_trace.jsonl"
    monkeypatch.setenv("FULL_AGENT_TRACE_PATH", str(trace_path))

    full_agent_trace("request", {
        "authorization": "Bearer private-token",
        "nested": {
            "api_key": "private-key",
            "access_token": "private-access-token",
            "safe": "retained",
        },
    })

    logged_data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert logged_data["event"] == "request"
    assert logged_data["details"]["authorization"] == "<redacted>"
    assert logged_data["details"]["nested"]["api_key"] == "<redacted>"
    assert logged_data["details"]["nested"]["access_token"] == "<redacted>"
    assert logged_data["details"]["nested"]["safe"] == "retained"


@pytest.mark.asyncio
async def test_python_cli_trace_records_code_context_and_output(monkeypatch, job_factory):
    """Tracing must expose Python CLI payloads and safe context names for diagnosis."""
    monkeypatch.setenv("FULL_AGENT_TRACE", "true")
    job = job_factory(job_params={"goal": "Trace test"})
    job.py_env.local_state = {"available_tool": available_tool}

    with patch("statek.executors.utils.full_agent_trace") as trace:
        await exec_cli_step('print(available_tool())', job, lambda _: None)

    trace.assert_any_call(
        "python_cli.request",
        {
            "job_uuid": str(db0.uuid(job)),  # pylint: disable=no-member
            "code": "print(available_tool())",
            "context_names": ["available_tool"],
        },
    )
    trace.assert_any_call(
        "python_cli.output",
        {"job_uuid": str(db0.uuid(job)), "output": "ok"},  # pylint: disable=no-member
    )
    trace.assert_any_call(
        "python_cli.completed",
        {"job_uuid": str(db0.uuid(job)), "exited": False},  # pylint: disable=no-member
    )
