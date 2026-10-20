"""Message-local wait completion independent of execution log entries."""
# pylint: disable=no-member

from datetime import datetime, timedelta
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import UserLogItem, finish_message_waits
from statek.agents.dialog_agent import record_response_delivery
from statek.exceptions import LLM_HarnessError
from statek.executors.job import Job
from statek.executors.utils import job_worker


@pytest.mark.usefixtures("db0_fixture")
def test_first_wait_completion_wins() -> None:
    """Later completions cannot replace the first, even with a reversed clock."""
    start = datetime(2026, 10, 19)
    message = UserLogItem("hello", start)
    finish_message_waits([message], start + timedelta(seconds=5))
    finish_message_waits([message], start + timedelta(seconds=3))
    assert message.wait_finished_at == start + timedelta(seconds=5)


def test_delivery_excludes_messages_arriving_during_send(job_factory) -> None:
    """Capture eligible messages before sending, then finish only their waits."""
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.push_user_message("initial")
    with record_response_delivery(job):
        job.push_user_message("late")
    assert job.chat_log[0].wait_finished_at is not None
    assert job.chat_log[1].wait_finished_at is None


def test_failed_send_does_not_finish_wait(job_factory) -> None:
    """A failed delivery is not a completed response."""
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.push_user_message("initial")
    with pytest.raises(RuntimeError):
        with record_response_delivery(job):
            raise RuntimeError("delivery failed")
    assert job.chat_log[0].wait_finished_at is None


def test_overlapping_sends_preserve_first_completion(job_factory) -> None:
    """An earlier-started send cannot overwrite a faster completion."""
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.push_user_message("initial")
    start = job.created_at
    with patch("statek.agents.dialog_agent.datetime", wraps=datetime) as clock:
        with record_response_delivery(job):
            with record_response_delivery(job):
                clock.now.return_value = start + timedelta(seconds=5)
            clock.now.return_value = start + timedelta(seconds=20)
    assert list(job.get_response_times()) == [(start, 5.0)]


@pytest.mark.parametrize("style", [ChatStyle.DIRECT, ChatStyle.MD_DIALOG])
def test_constructor_normalizes_initial_message(job_factory, style) -> None:
    """Current constructor inputs retain content, position and creation time."""
    original = job_factory()
    original.job_def.set_chat_style(style)
    start = datetime(2026, 10, 19)
    job = Job(job_def=original.job_def, chat_log=["hello"], created_at=start)
    assert len(job.chat_log) == 1
    assert isinstance(job.chat_log[0], UserLogItem)
    assert job.chat_log[0].message == "hello"
    assert job.chat_log[0].timestamp == start
    assert list(job.get_response_times()) == [(start, None)]


def test_cancelled_send_leaves_wait_pending(job_factory) -> None:
    """Cancellation must not be recorded as successful delivery."""
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    job.push_user_message("initial")
    with pytest.raises(asyncio.CancelledError):
        with record_response_delivery(job):
            raise asyncio.CancelledError()
    assert job.chat_log[0].wait_finished_at is None


@pytest.mark.asyncio
async def test_harness_closure_wins_over_inflight_send(job_factory) -> None:
    """A late send neither overwrites a terminal closure nor closes a resumed message."""
    job = job_factory()
    job.job_def.set_chat_style(ChatStyle.DIRECT)
    start = job.created_at
    job.push_user_message("initial")
    with patch("statek.agents.dialog_agent.datetime", wraps=datetime) as clock, \
         patch("statek.executors.utils.datetime", new=clock), \
         patch("statek.executors.utils.run_job_step", new_callable=AsyncMock) as step:
        step.side_effect = LLM_HarnessError("limit")
        with record_response_delivery(job):
            clock.now.return_value = start + timedelta(seconds=5)
            await job_worker(asyncio.Semaphore(1), job)
            assert job.push_user_message("resumed")
            clock.now.return_value = start + timedelta(seconds=20)
    assert job.chat_log[0].wait_finished_at == start + timedelta(seconds=5)
    assert job.chat_log[-2].wait_finished_at is None
