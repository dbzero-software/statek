"""Tests for process_agent_events."""
# pylint: disable=unused-argument,no-member
import pytest
import dbzero as db0

from statek.agents.agent import SupervisedAgent
from statek.executors.job import Job
from statek.executors.utils import process_agent_events
from statek.prompt_config import make_system_prompt
from statek.statek_push_queue import StatekPushQueue


def _current_queue_prefixes():
    return [db0.get_current_prefix().name]


def _make_agent(role="event-agent", warmup_code="payload = event.payload"):
    agent = SupervisedAgent(
        role=role,
        _system_prompt=make_system_prompt("test"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )
    agent.update_warmup_def(warmup_code)
    return agent


@db0.memo
class _QueuedEvent:
    def __init__(self, payload):
        self.payload = payload


def test_process_agent_events_creates_job_with_event_shared_var(db0_fixture):
    agent = _make_agent()
    event = _QueuedEvent("hello")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(agent, event)

    process_agent_events(agents={agent}, max_count=10, queue_prefixes=_current_queue_prefixes())

    jobs = list(db0.find(Job, db0.as_tag(agent)))
    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_def.agent is agent
    assert job.py_env.local_state["event"] is event


def test_process_agent_events_uses_all_supervised_agents_when_filter_missing(db0_fixture):
    agent = _make_agent()
    event = _QueuedEvent("hello")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(agent, event)

    process_agent_events(max_count=10, queue_prefixes=_current_queue_prefixes())

    jobs = list(db0.find(Job, db0.as_tag(agent)))
    assert len(jobs) == 1
    assert jobs[0].py_env.local_state["event"] is event


def test_process_agent_events_respects_agent_filter(db0_fixture):
    agent_a = _make_agent("agent-a")
    agent_b = _make_agent("agent-b")
    event_a = _QueuedEvent("a")
    event_b = _QueuedEvent("b")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(agent_a, event_a)
    queue.push_to_agent_queue(agent_b, event_b)

    process_agent_events(agents={agent_a}, max_count=10, queue_prefixes=_current_queue_prefixes())

    assert len(db0.find(Job, db0.as_tag(agent_a))) == 1
    assert len(db0.find(Job, db0.as_tag(agent_b))) == 0
    assert queue.pop_from_agent_queue(agent_b, 10) == [event_b]


def test_process_agent_events_respects_max_count(db0_fixture):
    agent = _make_agent()
    queue = StatekPushQueue()
    events = [_QueuedEvent(index) for index in range(3)]
    for event in events:
        queue.push_to_agent_queue(agent, event)

    process_agent_events(agents={agent}, max_count=2, queue_prefixes=_current_queue_prefixes())

    jobs = list(db0.find(Job, db0.as_tag(agent)))
    assert len(jobs) == 2
    assert queue.pop_from_agent_queue(agent, 10) == [events[2]]


def test_process_agent_events_rejects_agents_without_single_referenced_local(db0_fixture):
    agent = _make_agent(warmup_code="print('no external refs')")
    queue = StatekPushQueue()
    event = _QueuedEvent("hello")
    queue.push_to_agent_queue(agent, event)

    with pytest.raises(ValueError, match="exactly one external local"):
        process_agent_events(agents={agent}, max_count=10, queue_prefixes=_current_queue_prefixes())

    assert len(db0.find(Job, db0.as_tag(agent))) == 0
    assert queue.pop_from_agent_queue(agent, 10) == [event]


def test_process_agent_events_ignores_invalid_agent_without_events(db0_fixture):
    invalid_agent = _make_agent("invalid-agent", warmup_code="print('no refs')")
    valid_agent = _make_agent("valid-agent")
    event = _QueuedEvent("hello")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(valid_agent, event)

    process_agent_events(max_count=10, queue_prefixes=_current_queue_prefixes())

    assert len(db0.find(Job, db0.as_tag(invalid_agent))) == 0
    assert len(db0.find(Job, db0.as_tag(valid_agent))) == 1


def test_process_agent_events_validates_all_agents_before_popping(db0_fixture):
    valid_agent = _make_agent("valid-agent")
    invalid_agent = _make_agent("invalid-agent", warmup_code="print('no refs')")
    valid_event = _QueuedEvent("valid")
    invalid_event = _QueuedEvent("invalid")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(valid_agent, valid_event)
    queue.push_to_agent_queue(invalid_agent, invalid_event)

    with pytest.raises(ValueError, match="exactly one external local"):
        process_agent_events(
            agents={valid_agent, invalid_agent},
            max_count=10,
            queue_prefixes=_current_queue_prefixes(),
        )

    assert len(db0.find(Job, db0.as_tag(valid_agent))) == 0
    assert queue.pop_from_agent_queue(valid_agent, 10) == [valid_event]
    assert queue.pop_from_agent_queue(invalid_agent, 10) == [invalid_event]


def test_process_agent_events_uses_only_configured_queue_prefixes(db0_fixture, monkeypatch):
    queue_prefix = db0.get_current_prefix().name
    agent = _make_agent()
    event = _QueuedEvent("hello")
    queue = StatekPushQueue()
    queue.push_to_agent_queue(agent, event)
    db0.open("unrelated-prefix", "rw")
    db0.open(queue_prefix, "rw")
    observed_prefixes = []
    original_find_singleton = db0.find_singleton

    def recording_find_singleton(cls, prefix=None):
        if cls is StatekPushQueue:
            observed_prefixes.append(prefix)
        return original_find_singleton(cls, prefix)

    monkeypatch.setattr(db0, "find_singleton", recording_find_singleton)

    process_agent_events(agents={agent}, max_count=10, queue_prefixes=[queue_prefix])

    jobs = list(db0.find(Job, db0.as_tag(agent), prefix=queue_prefix))
    assert observed_prefixes == [queue_prefix]
    assert len(jobs) == 1
    assert jobs[0].py_env.local_state["event"] is event
