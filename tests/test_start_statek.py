"""Tests for high-level start_statek helpers."""
# pylint: disable=unused-argument,no-member

from pathlib import Path
from unittest.mock import Mock

import pytest
import dbzero as db0

from statek.agents.agent import Agent, SupervisedAgent
from statek.prompt_config import PromptDef, make_system_prompt
from statek.runner import start_statek, start_statek_async
from statek.settings import StatekSettings
from statek.statek_push_queue import StatekPushQueue


def _settings(**kwargs):
    data = {"prompt_defs": {}, "warmup_defs_dir": None}
    data.update(kwargs)
    return StatekSettings(**data)


def _agent(role="runner-agent"):
    return SupervisedAgent(
        role=role,
        _system_prompt=make_system_prompt("old"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )


@pytest.mark.asyncio
async def test_start_statek_async_uses_single_agent_loop(db0_fixture, monkeypatch):
    agent = _agent()
    calls = {}

    async def fake_loop(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("statek.runner.run_agentic_loop", fake_loop)
    monkeypatch.setattr("statek.runner.StatekClientAPI", Mock())

    await start_statek_async(
        agents=[agent],
        settings=_settings(),
        max_concurrency=7,
        provider="TEST",
    )

    assert calls["agent"] is agent
    assert calls["warmup_code"] is None
    assert calls["queue_prefixes"] == [db0.get_current_prefix().name]
    assert calls["max_concurrency"] == 7
    assert calls["provider"] == "TEST"


@pytest.mark.asyncio
async def test_start_statek_async_uses_fleet_for_multiple_agents(db0_fixture, monkeypatch):
    agent_a = _agent("runner-a")
    agent_b = _agent("runner-b")
    agent_b.update_warmup_def("event = payload")
    calls = {}

    async def fake_fleet(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("statek.runner.run_agentic_fleet", fake_fleet)
    monkeypatch.setattr("statek.runner.StatekClientAPI", Mock())

    await start_statek_async(agents=[agent_a, agent_b], settings=_settings())

    loop_defs = calls["agent_loop_defs"]
    assert [loop_def.agent for loop_def in loop_defs] == [agent_a, agent_b]
    assert loop_defs[0].warmup_code is None
    assert loop_defs[1].warmup_code == "event = payload"


@pytest.mark.asyncio
async def test_start_statek_updates_existing_agents_from_files(
    db0_fixture, temp_dir, monkeypatch
):
    agent = _agent("configured")
    warmup_path = Path(temp_dir)
    (warmup_path / "configured.py").write_text("event = payload", encoding="utf-8")
    prompt_def = PromptDef(
        system=make_system_prompt("new prompt"),
        metadata={"MODEL": "test-model", "DESCRIPTION": "Configured agent"},
    )

    async def fake_loop(**kwargs):
        assert kwargs["agent"] is agent

    monkeypatch.setattr("statek.runner.run_agentic_loop", fake_loop)
    monkeypatch.setattr("statek.runner.StatekClientAPI", Mock())

    await start_statek_async(
        agents=[agent],
        settings=_settings(
            prompt_defs={"configured": prompt_def},
            warmup_defs_dir=str(warmup_path),
        ),
    )

    assert agent.description == "Configured agent"
    assert agent.warmup_def.warmup_code == "event = payload"
    assert list(db0.find(Agent)) == [agent]


@pytest.mark.asyncio
async def test_start_statek_uses_supplied_queue_prefixes(db0_fixture, monkeypatch):
    agent = _agent()
    queue = StatekPushQueue(prefix="queue-prefix")
    calls = {}

    async def fake_loop(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("statek.runner.run_agentic_loop", fake_loop)
    monkeypatch.setattr("statek.runner.StatekClientAPI", Mock())

    await start_statek_async(agents=[agent], push_queues=[queue], settings=_settings())

    assert calls["queue_prefixes"] == ["queue-prefix"]


def test_start_statek_sync_wrapper(db0_fixture, monkeypatch):
    called = {}

    async def fake_start(**kwargs):
        called.update(kwargs)
        return "done"

    monkeypatch.setattr("statek.runner.start_statek_async", fake_start)

    result = start_statek(max_concurrency=3, provider="TEST")

    assert result == "done"
    assert called["max_concurrency"] == 3
    assert called["provider"] == "TEST"


@pytest.mark.asyncio
async def test_start_statek_sync_wrapper_rejects_running_loop(db0_fixture):
    with pytest.raises(RuntimeError, match="start_statek_async"):
        start_statek()
