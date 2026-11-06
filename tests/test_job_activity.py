"""Persisted activity tracks conversation mutations, not execution bookkeeping."""
# pylint: disable=no-member,protected-access,redefined-outer-name
from datetime import datetime, timedelta
from functools import partial
from unittest.mock import patch

import dbzero as db0
import pytest

from statek.agents.dialog_agent import RecurringReminder, record_response_delivery
from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import LLM_LogItem, ToolError
from statek.executors.job import Job, JobStatus
from statek.executors.post_processor import PostProcessor
from statek.executors.utils import run_job_step
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from statek.task import SubTaskHandler
from statek.task_difficulty import TaskDifficulty
from statek.utils import CallSpec, CodeBlock
from tests.conftest import DB0_DIR, remove_lock_files


@pytest.fixture
def activity_clock():
    with patch("statek.executors.job.datetime") as clock:
        clock.now.return_value = datetime(2026, 11, 1)
        yield clock


@db0.memo
class ActivityProcessor(PostProcessor):
    def __init__(self, mode="tuple"):
        self.mode = mode

    def process(self, llm_step, job):
        if self.mode == "string":
            return "review"
        if self.mode == "invalid":
            return None
        if self.mode == "unchanged":
            return llm_step
        return ("review", llm_step)


def test_initial_legacy_and_read_only(job_factory):
    job = job_factory()
    assert job.last_activity is None
    del job._Job__last_activity
    assert job.last_activity is None
    assert Job.last_activity.fset is None
    job.set_status(JobStatus.STARTED)
    index = db0.index_of(Job, "_Job__last_activity")
    assert list(db0.find(Job, index.select(job.last_activity, job.last_activity))) == [job]


def test_status_and_bookkeeping(job_factory, activity_clock):
    job = job_factory()
    job.set_status(JobStatus.READY)
    assert job.last_activity is None
    job.set_status(JobStatus.STARTED)
    first = activity_clock.now.return_value
    assert job.last_activity == first
    activity_clock.now.return_value += timedelta(seconds=1)
    job.set_status(JobStatus.STARTED)
    job.next_instr_num = 1
    job.add_locals(example=42)
    job.usage.total_bytes_sent += 10
    job.add_ext_ref(job.job_def)
    job._Job__job_summaries = []
    assert not list(job.get_compacted_chat_log())
    assert not job._process_pending_notifications()
    assert job.last_activity == first
    job.set_status(JobStatus.DONE)
    assert job.last_activity == activity_clock.now.return_value


@pytest.mark.parametrize("style", [
    ChatStyle.CONSOLE, ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG, ChatStyle.DIRECT,
])
def test_user_messages(job_def_factory, activity_clock, style):
    job = Job(job_def_factory(_chat_style=style))
    job.push_user_message("hello")
    assert job.last_activity == activity_clock.now.return_value
    activity_clock.now.return_value += timedelta(seconds=1)
    job.push_user_message("again")
    assert job.last_activity == activity_clock.now.return_value


@pytest.mark.parametrize("subtask", [False, True])
def test_notifications(job_factory, activity_clock, subtask):
    job = job_factory()
    if subtask:
        handler = SubTaskHandler(job_factory())
        with pytest.raises(RuntimeError):
            job.push_notification(handler)
        assert job.last_activity is None
        handler._SubTaskHandler__is_completed = True
        send = partial(job.push_notification, handler)
    else:
        send = partial(job.send_notification, "notice")
    send()
    first = activity_clock.now.return_value
    assert job.last_activity == first
    activity_clock.now.return_value += timedelta(seconds=1)
    job.chat_log.append(LLM_LogItem(console_pos=0))
    assert not job._process_pending_notifications()
    assert job.last_activity == first
    job.chat_log.clear()
    assert job._process_pending_notifications()
    assert job.last_activity == activity_clock.now.return_value
    job.set_status(JobStatus.DONE)
    activity_clock.now.return_value += timedelta(seconds=1)
    send()
    assert job.status == JobStatus.STARTED
    assert job.last_activity == activity_clock.now.return_value


@pytest.mark.parametrize("mutation", ["llm", "reminder", "difficulty", "tuple", "string"])
def test_chat_entries(job_factory, activity_clock, mutation):
    job = job_factory()
    step = LLM_StepData(text="hello", call_requests=None)
    if mutation == "llm":
        job.append_chat_log({}, LLM_Response(step, LLM_Stats(0, 0, None)))
    elif mutation == "reminder":
        assert not job.handle_reminder(RecurringReminder("later", min_dialog_len=5))
        assert job.last_activity is None
        job.handle_reminder(RecurringReminder("now"))
    elif mutation == "difficulty":
        job._set_dynamic_difficulty(TaskDifficulty.high)
    else:
        job.handle_post_processor(ActivityProcessor(mutation), step)
    assert job.last_activity == activity_clock.now.return_value


def test_failed_and_unchanged_operations(job_factory, activity_clock):
    job = job_factory()
    job.push_user_message("hello")
    first = job.last_activity
    activity_clock.now.return_value += timedelta(seconds=1)
    step = LLM_StepData(text="hello", call_requests=None)
    assert job.handle_post_processor(ActivityProcessor("unchanged"), step) == (step, False)
    with pytest.raises(TypeError):
        job.handle_post_processor(ActivityProcessor("invalid"), step)
    with patch.object(Job, "_resolve_user_message", side_effect=ValueError("invalid message")):
        with pytest.raises(ValueError):
            job.push_user_message(object())
    job._set_dynamic_difficulty(job.get_current_difficulty())
    assert not job._process_pending_notifications()
    assert job.last_activity == first


def test_message_wait_completion(job_def_factory, activity_clock):
    job = Job(job_def_factory(_chat_style=ChatStyle.DIRECT))
    job.push_user_message("hello")
    first = job.last_activity
    activity_clock.now.return_value += timedelta(seconds=1)
    with pytest.raises(ValueError):
        with record_response_delivery(job):
            raise ValueError("delivery failed")
    assert job.last_activity == first
    with record_response_delivery(job):
        pass
    assert job.last_activity == activity_clock.now.return_value
    first = job.last_activity
    activity_clock.now.return_value += timedelta(seconds=1)
    with record_response_delivery(job):
        pass
    assert job.last_activity == first


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, "failed"])
async def test_tool_initialization_and_results(job_def_factory, activity_clock, error):
    job = Job(job_def_factory(_chat_style=ChatStyle.DIRECT), job_status=JobStatus.STARTED,
              chat_log=[LLM_LogItem(console_pos=0, llm_resp=CodeBlock(tool_calls=[
                  CallSpec(id="call", func_name="example", args=[], kwargs={}),
                  CallSpec(id="cli", func_name="python_cli", args=[], kwargs={"code": "1"}),
              ]))])

    completed = []

    async def execute_tool(*_args, **_kwargs):
        assert job.last_activity == activity_clock.now.return_value
        activity_clock.now.return_value += timedelta(seconds=1)
        return "result", error

    async def execute_code(_code, _job, console_append, *_args, **kwargs):
        assert job.last_activity == activity_clock.now.return_value
        result = job.last_chat_log_item.tool_log[0]
        assert isinstance(result, ToolError) if error else result == "result"
        activity_clock.now.return_value += timedelta(seconds=1)
        console_append(0, "output")
        if error:
            kwargs["error_append"](0, error)
        completed.append(True)

    with patch("statek.executors.utils.exec_tool", side_effect=execute_tool), \
         patch("statek.executors.utils.exec_all_steps", side_effect=execute_code), \
         patch("statek.executors.utils.LLM_API.get", side_effect=ValueError("stop before LLM")):
        with pytest.raises(ValueError, match="stop before LLM"):
            await run_job_step(job)
    assert completed == [True]
    assert job.last_activity == activity_clock.now.return_value
    result = job.last_chat_log_item.tool_log[1]
    assert isinstance(result, ToolError) if error else result == "output"


@pytest.mark.asyncio
async def test_warmup_entry(job_def_factory, activity_clock):
    job = Job(job_def_factory(warmup_code="x = 1"), job_status=JobStatus.WARMING_UP)

    async def execute_code(*_args, **_kwargs):
        assert job.last_activity == activity_clock.now.return_value
        assert len(job.chat_log) == 1

    with patch("statek.executors.utils.exec_all_steps", side_effect=execute_code), \
         patch("statek.executors.utils.LLM_API.get", side_effect=ValueError("stop before LLM")):
        with pytest.raises(ValueError, match="stop before LLM"):
            await run_job_step(job)


def test_index_updates_sorting_and_reopening(job_factory, activity_clock):
    older, newer = job_factory(), job_factory()
    older.set_status(JobStatus.STARTED)
    first = activity_clock.now.return_value
    activity_clock.now.return_value += timedelta(seconds=1)
    newer.push_user_message("hello")
    second = activity_clock.now.return_value
    index = db0.index_of(Job, "_Job__last_activity")
    assert list(db0.find(Job, index.select(first, first))) == [older]
    assert list(index.sort(db0.find(Job))) == [older, newer]
    assert list(index.sort(db0.find(Job), desc=True)) == [newer, older]
    activity_clock.now.return_value += timedelta(seconds=1)
    older.send_notification("updated")
    assert not list(db0.find(Job, index.select(first, first)))
    assert list(index.sort(db0.find(Job))) == [newer, older]
    ids = [db0.uuid(newer), db0.uuid(older)]
    db0.close()
    remove_lock_files(DB0_DIR)
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    index = db0.index_of(Job, "_Job__last_activity")
    restored = list(index.sort(db0.find(Job)))
    assert [db0.uuid(job) for job in restored] == ids
    assert [job.last_activity for job in restored] == [second, activity_clock.now.return_value]
