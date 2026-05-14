"""Tests for StatekPushQueue singleton."""
# pylint: disable=no-member,unused-argument,too-few-public-methods
import dbzero as db0
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent, SupervisedAgent
from statek.prompt_config import make_system_prompt
from statek.statek_push_queue import StatekPushQueue


def _make_job(db0_fixture):  # pylint: disable=unused-argument
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("test"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )
    job_def = JobDef(agent=agent)
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=JobStatus.READY)  # pylint: disable=no-member


def _make_job_on_current_prefix():
    """Create a Job (with its own Agent/JobDef) on whatever prefix is currently active."""
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("test"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )
    job_def = JobDef(agent=agent)
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=JobStatus.READY)  # pylint: disable=no-member


def _make_supervised_agent(role="test-agent"):
    return SupervisedAgent(
        role=role,
        _system_prompt=make_system_prompt("test"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )


@db0.memo
class _QueuedMessage:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return f"queued:{self.text}"


def test_statek_push_queue_is_singleton(db0_fixture):
    q1 = StatekPushQueue()
    q2 = StatekPushQueue()
    assert q1 is q2


def test_push_to_job_console_accepts_job_uuid_and_message(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    queue.push_to_job_console(job_uuid=job_uuid, message="hello")


def test_pop_from_job_console_returns_empty_list_when_empty(db0_fixture):
    queue = StatekPushQueue()
    result = queue.pop_from_job_console(10)
    assert result == []


def test_pop_from_job_console_returns_pushed_item(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    queue.push_to_job_console(job_uuid=job_uuid, message="hello")
    result = queue.pop_from_job_console(10)
    assert len(result) == 1
    popped_uuid, popped_msg = result[0]
    assert popped_uuid == job_uuid
    assert popped_msg == "hello"


def test_pop_from_job_console_returns_tuples_of_uuid_and_str(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    queue.push_to_job_console(job_uuid=job_uuid, message="msg")
    result = queue.pop_from_job_console(10)
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 2
    assert isinstance(result[0][0], str)  # db0 UUIDs are strings
    assert isinstance(result[0][1], str)


def test_pop_from_job_console_preserves_memo_message_objects(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    message = _QueuedMessage("object-message")

    queue.push_to_job_console(job_uuid=job_uuid, message=message)

    result = queue.pop_from_job_console(10)
    assert len(result) == 1
    assert result[0] == (job_uuid, message)


def test_pop_from_job_console_respects_count(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    for i in range(5):
        queue.push_to_job_console(job_uuid=job_uuid, message=f"msg{i}")
    result = queue.pop_from_job_console(3)
    assert len(result) == 3


def test_pop_from_job_console_is_fifo(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    for i in range(4):
        queue.push_to_job_console(job_uuid=job_uuid, message=f"msg{i}")
    result = queue.pop_from_job_console(4)
    messages = [msg for _, msg in result]
    assert messages == ["msg0", "msg1", "msg2", "msg3"]


def test_pop_from_job_console_removes_elements(db0_fixture):
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    queue.push_to_job_console(job_uuid=job_uuid, message="first")
    queue.push_to_job_console(job_uuid=job_uuid, message="second")
    first_result = queue.pop_from_job_console(1)
    assert first_result[0][1] == "first"
    second_result = queue.pop_from_job_console(1)
    assert second_result[0][1] == "second"


def test_pop_from_job_console_prefix_none_returns_all(db0_fixture):
    """prefix=None (default) returns entries regardless of job prefix."""
    job = _make_job(db0_fixture)
    job_uuid = db0.uuid(job)
    queue = StatekPushQueue()
    queue.push_to_job_console(job_uuid=job_uuid, message="hello")
    result = queue.pop_from_job_console(10, prefix=None)
    assert len(result) == 1


def test_pop_from_job_console_prefix_name_filters_matching_jobs(db0_fixture):
    """prefix str filters jobs that live on the named prefix."""
    prefix_a = db0.get_current_prefix().name
    queue = StatekPushQueue()
    job_a = _make_job_on_current_prefix()
    job_a_uuid = db0.uuid(job_a)
    db0.commit()
    queue.push_to_job_console(job_uuid=job_a_uuid, message="from-a")

    db0.open("other-prefix", "rw")
    job_b = _make_job_on_current_prefix()
    job_b_uuid = db0.uuid(job_b)
    db0.commit()
    queue.push_to_job_console(job_uuid=job_b_uuid, message="from-b")

    result = queue.pop_from_job_console(10, prefix=prefix_a)
    assert len(result) == 1
    assert result[0][1] == "from-a"


def test_pop_from_job_console_prefix_name_non_matching_jobs_stay_in_queue(db0_fixture):
    """Jobs whose prefix does not match the filter remain in the queue."""
    prefix_a = db0.get_current_prefix().name
    job_a = _make_job_on_current_prefix()
    job_a_uuid = db0.uuid(job_a)
    db0.commit()
    queue = StatekPushQueue()

    db0.open("other-prefix", "rw")
    job_b = _make_job_on_current_prefix()
    job_b_uuid = db0.uuid(job_b)
    db0.commit()

    queue.push_to_job_console(job_uuid=job_a_uuid, message="from-a")
    queue.push_to_job_console(job_uuid=job_b_uuid, message="from-b")

    queue.pop_from_job_console(10, prefix=prefix_a)

    remaining = queue.pop_from_job_console(10)
    assert len(remaining) == 1
    assert remaining[0][1] == "from-b"


def test_pop_from_job_console_prefix_uuid_filters_matching_jobs(db0_fixture):
    """prefix int (UUID) filters jobs that live on the matching prefix."""
    prefix_a_uuid = db0.get_current_prefix().uuid
    job_a = _make_job_on_current_prefix()
    job_a_uuid = db0.uuid(job_a)
    db0.commit()
    queue = StatekPushQueue()

    db0.open("other-prefix", "rw")
    job_b = _make_job_on_current_prefix()
    job_b_uuid = db0.uuid(job_b)
    db0.commit()
    queue.push_to_job_console(job_uuid=job_a_uuid, message="from-a")
    queue.push_to_job_console(job_uuid=job_b_uuid, message="from-b")

    result = queue.pop_from_job_console(10, prefix=prefix_a_uuid)
    assert len(result) == 1
    assert result[0][1] == "from-a"


def test_pop_from_job_console_prefix_uuid_non_matching_stay_in_queue(db0_fixture):
    """Jobs whose prefix UUID does not match remain in the queue."""
    prefix_a = db0.get_current_prefix()
    job_a = _make_job_on_current_prefix()
    job_a_uuid = db0.uuid(job_a)
    db0.commit()
    queue = StatekPushQueue()

    db0.open("other-prefix", "rw")
    job_b = _make_job_on_current_prefix()
    job_b_uuid = db0.uuid(job_b)
    db0.commit()

    queue.push_to_job_console(job_uuid=job_a_uuid, message="from-a")
    queue.push_to_job_console(job_uuid=job_b_uuid, message="from-b")

    queue.pop_from_job_console(10, prefix=prefix_a.uuid)
    remaining = queue.pop_from_job_console(10)
    assert len(remaining) == 1
    assert remaining[0][1] == "from-b"


def test_has_job_returns_true_when_job_uuid_exists(db0_fixture):
    job = _make_job(db0_fixture)
    queue = StatekPushQueue()
    job_uuid = db0.uuid(job)

    queue.push_to_job_console(job_uuid=job_uuid, message="hello")

    assert queue.has_job(job_uuid) is True


def test_has_job_returns_false_when_job_uuid_does_not_exist(db0_fixture):
    job = _make_job(db0_fixture)
    other_job = _make_job(db0_fixture)
    queue = StatekPushQueue()

    queue.push_to_job_console(job_uuid=db0.uuid(job), message="hello")

    assert queue.has_job(db0.uuid(other_job)) is False


def test_has_job_returns_none_when_no_scan_budget_is_available(db0_fixture):
    queue = StatekPushQueue()
    jobs = [_make_job(db0_fixture) for _ in range(3)]

    for index, job in enumerate(jobs):
        queue.push_to_job_console(job_uuid=db0.uuid(job), message=f"msg{index}")

    assert queue.has_job(db0.uuid(jobs[2]), max_scan=0) is None


def test_has_job_does_not_remove_items_from_queue(db0_fixture):
    first_job = _make_job(db0_fixture)
    second_job = _make_job(db0_fixture)
    queue = StatekPushQueue()
    first_uuid = db0.uuid(first_job)
    second_uuid = db0.uuid(second_job)

    queue.push_to_job_console(job_uuid=first_uuid, message="first")
    queue.push_to_job_console(job_uuid=second_uuid, message="second")

    assert queue.has_job(first_uuid) is True
    assert queue.pop_from_job_console(10) == [
        (first_uuid, "first"),
        (second_uuid, "second"),
    ]


def test_pop_from_agent_queue_returns_empty_list_when_empty(db0_fixture):
    queue = StatekPushQueue()
    agent = _make_supervised_agent()

    assert queue.pop_from_agent_queue(agent, 10) == []


def test_has_agent_events_checks_agent_queue_emptiness(db0_fixture):
    queue = StatekPushQueue()
    agent = _make_supervised_agent()

    assert queue.has_agent_events(agent) is False

    queue.push_to_agent_queue(agent, "event")

    assert queue.has_agent_events(agent) is True

    queue.pop_from_agent_queue(agent, 10)

    assert queue.has_agent_events(agent) is False


def test_pop_from_agent_queue_returns_pushed_events(db0_fixture):
    queue = StatekPushQueue()
    agent = _make_supervised_agent()

    queue.push_to_agent_queue(agent, "event")

    assert queue.pop_from_agent_queue(agent, 10) == ["event"]


def test_pop_from_agent_queue_preserves_memo_event_objects(db0_fixture):
    queue = StatekPushQueue()
    agent = _make_supervised_agent()
    event = _QueuedMessage("event-object")

    queue.push_to_agent_queue(agent, event)

    assert queue.pop_from_agent_queue(agent, 10) == [event]


def test_pop_from_agent_queue_respects_count_and_fifo_order(db0_fixture):
    queue = StatekPushQueue()
    agent = _make_supervised_agent()

    for i in range(5):
        queue.push_to_agent_queue(agent, f"event{i}")

    assert queue.pop_from_agent_queue(agent, 3) == [
        "event0",
        "event1",
        "event2",
    ]
    assert queue.pop_from_agent_queue(agent, 10) == ["event3", "event4"]


def test_pop_from_agent_queue_filters_by_agent(db0_fixture):
    queue = StatekPushQueue()
    agent_a = _make_supervised_agent("agent-a")
    agent_b = _make_supervised_agent("agent-b")

    queue.push_to_agent_queue(agent_a, "for-a-1")
    queue.push_to_agent_queue(agent_b, "for-b")
    queue.push_to_agent_queue(agent_a, "for-a-2")

    assert queue.pop_from_agent_queue(agent_a, 10) == ["for-a-1", "for-a-2"]
    assert queue.pop_from_agent_queue(agent_b, 10) == ["for-b"]
