"""Tests for process_push_notifications function."""
# pylint: disable=unused-argument,no-member
import dbzero as db0
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.prompt_config import make_system_prompt
from statek.statek_push_queue import StatekPushQueue
from statek.executors.utils import process_push_notifications


def _current_queue_prefixes():
    return [db0.get_current_prefix().name]


def _make_started_job():
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("test"),
        _metadata={"MODEL": "test-model"},
        _tools=[],
    )
    job_def = JobDef(agent=agent)
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=JobStatus.STARTED)  # pylint: disable=no-member


@db0.memo
class _QueuedMessage:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return f"hello-from-{self.text}"


class TestProcessPushNotifications:

    def test_no_queue_does_nothing(self, db0_fixture):
        # No StatekPushQueue created — should not raise
        process_push_notifications(queue_prefixes=_current_queue_prefixes())

    def test_empty_queue_does_nothing(self, db0_fixture):
        StatekPushQueue()
        process_push_notifications(queue_prefixes=_current_queue_prefixes())

    def test_processes_single_notification(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="hello")

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job.py_env.push_log is not None
        assert job.py_env.push_log[0] == "hello"

    def test_processes_non_string_notification_via_str_conversion(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        message = _QueuedMessage("object")
        queue.push_to_job_console(job_uuid=job_uuid, message=message)

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job.py_env.push_log is not None
        assert job.py_env.push_log[0] == "hello-from-object"
        assert job.contains_ext_ref(message) is True

    def test_processes_non_string_notification_via_job_message_adapter(
        self, db0_fixture
    ):
        job = _make_started_job()
        job.job_def.agent.context["message_adapter"] = (
            lambda message: f"adapted-{message.text}"
        )
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        message = _QueuedMessage("object")
        queue.push_to_job_console(job_uuid=job_uuid, message=message)

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job.py_env.push_log is not None
        assert job.py_env.push_log[0] == "adapted-object"
        assert job.contains_ext_ref(message) is True

    def test_string_notification_is_not_registered_as_ext_ref(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="hello")

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job.py_env.push_log[0] == "hello"
        assert job.contains_ext_ref("hello") is False

    def test_queue_is_empty_after_processing(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="hello")

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        remaining = queue.pop_from_job_console(10)
        assert remaining == []

    def test_processes_multiple_notifications_for_same_job(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="msg1")
        queue.push_to_job_console(job_uuid=job_uuid, message="msg2")

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job.py_env.push_log is not None
        val = job.py_env.push_log[0]
        assert not isinstance(val, str)
        assert "msg1" in val
        assert "msg2" in val

    def test_processes_notifications_for_multiple_jobs(self, db0_fixture):
        job1 = _make_started_job()
        job2 = _make_started_job()
        job1_uuid = db0.uuid(job1)
        job2_uuid = db0.uuid(job2)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job1_uuid, message="for-job1")
        queue.push_to_job_console(job_uuid=job2_uuid, message="for-job2")

        process_push_notifications(queue_prefixes=_current_queue_prefixes())

        assert job1.py_env.push_log[0] == "for-job1"
        assert job2.py_env.push_log[0] == "for-job2"

    def test_filters_notifications_to_requested_prefix_uuid(self, db0_fixture):
        prefix_a = db0.get_current_prefix()
        job_a = _make_started_job()
        job_a_uuid = db0.uuid(job_a)
        queue = StatekPushQueue()

        db0.open("other-prefix", "rw")
        job_b = _make_started_job()
        job_b_uuid = db0.uuid(job_b)

        queue.push_to_job_console(job_uuid=job_a_uuid, message="for-a")
        queue.push_to_job_console(job_uuid=job_b_uuid, message="for-b")

        process_push_notifications(queue_prefixes=[prefix_a.name], job_prefix=prefix_a.uuid)

        assert job_a.py_env.push_log[0] == "for-a"
        assert job_b.py_env.push_log is None
        remaining = queue.pop_from_job_console(10, prefix=None)
        assert remaining == [(job_b_uuid, "for-b")]

    def test_respects_max_count(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        for i in range(10):
            queue.push_to_job_console(job_uuid=job_uuid, message=f"msg{i}")

        process_push_notifications(
            step_size=3,
            max_count=5,
            queue_prefixes=_current_queue_prefixes(),
        )

        remaining = queue.pop_from_job_console(100)
        assert len(remaining) == 5

    def test_ignores_exception_from_push_user_message(self, db0_fixture):
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="hello")

        original = job.__class__.push_user_message

        def raising_push(self, message):  # pylint: disable=unused-argument
            raise RuntimeError("Object has been deleted")

        job.__class__.push_user_message = raising_push

        try:
            # Should not propagate the exception
            process_push_notifications(queue_prefixes=_current_queue_prefixes())
        finally:
            job.__class__.push_user_message = original

    def test_uses_only_configured_queue_prefixes(self, db0_fixture, monkeypatch):
        queue_prefix = db0.get_current_prefix().name
        job = _make_started_job()
        job_uuid = db0.uuid(job)
        queue = StatekPushQueue()
        queue.push_to_job_console(job_uuid=job_uuid, message="hello")
        db0.open("unrelated-prefix", "rw")
        observed_prefixes = []
        original_find_singleton = db0.find_singleton

        def recording_find_singleton(cls, prefix=None):
            if cls is StatekPushQueue:
                observed_prefixes.append(prefix)
            return original_find_singleton(cls, prefix)

        monkeypatch.setattr(db0, "find_singleton", recording_find_singleton)

        process_push_notifications(queue_prefixes=[queue_prefix])

        assert observed_prefixes == [queue_prefix]
        assert job.py_env.push_log[0] == "hello"
