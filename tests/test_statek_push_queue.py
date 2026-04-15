"""Tests for StatekPushQueue singleton."""
# pylint: disable=no-member,unused-argument
import dbzero as db0
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.statek_push_queue import StatekPushQueue


def _make_job(db0_fixture):  # pylint: disable=unused-argument
    agent = Agent(
        role="test",
        _system_prompt="test",
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
        _system_prompt="test",
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )
    job_def = JobDef(agent=agent)
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=JobStatus.READY)  # pylint: disable=no-member


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
