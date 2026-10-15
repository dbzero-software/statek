"""Tests for handle_critical_error and its integration with job_worker."""

# pylint: disable=too-few-public-methods,no-member

import asyncio
from datetime import datetime, timedelta
from typing import Callable
from unittest.mock import AsyncMock, patch

import dbzero as db0
import pytest
import statek

from statek.chat_style import ChatStyle
from statek.settings import StatekSettings
from statek.llm_harness import LLM_Harness
from statek.exceptions import LLM_HarnessError
from statek.pyenv import Error, ErrorKind
from statek.executors.job import Job, JobDefError, JobStatus
from statek.executors.job import record_response_delivery
from statek.executors.chat_log_item import LLM_LogItem, UserLogItem
from statek.executors.utils import handle_critical_error, job_worker
from statek.system import error_handler
from statek.utils import _statek_ctx_scope
from tests.conftest import DB0_DIR


# Module-level capture list — mutations visible across calls
_call_log = []
_application_call_log = []


@error_handler
def _capture(context, error=None):  # pylint: disable=unused-argument
    """Append received (context, error) to the module-level capture list."""
    _call_log.append((context, error))


@error_handler
def _capture_application_error(context, error=None):
    """Capture the job's application-level critical-error notification."""
    _application_call_log.append((context, error))


class TestHandleCriticalError:
    """Unit tests for handle_critical_error."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("post_step", [False, True])
    async def test_real_harness_limits_end_pending_wait(self, job_factory, post_step):
        """Both actual pre-step and post-step limit failures use terminal timing."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)
        t0 = datetime(2026, 10, 15, 12)
        job.created_at = t0
        job.chat_log.append("initial")
        harness = (
            LLM_Harness(None, 100, 100, -1) if post_step
            else LLM_Harness(-1, 100, 100, None)
        )
        if post_step:
            step_context = patch("statek.executors.utils.run_job_step", new_callable=AsyncMock)
        else:
            step_context = patch("statek.executors.utils.get_llm_harness", return_value=harness)
        with patch("statek.executors.job.datetime", wraps=datetime) as clock, step_context as step:
            clock.now.return_value = t0 + timedelta(seconds=5)
            if post_step:
                step.side_effect = lambda *args: harness.check_after_step(job)
            await job_worker(asyncio.Semaphore(1), job)
        assert job.status == JobStatus.DONE
        assert list(job.get_response_times()) == [(t0, 5.0)]
        assert job.py_env.exceptions[len(job.py_env.console) - 1].kind == ErrorKind.HARNESS

    @pytest.mark.asyncio
    @pytest.mark.parametrize("style", [ChatStyle.DIRECT, ChatStyle.MD_DIALOG])
    @pytest.mark.parametrize("already_answered", [False, True])
    async def test_harness_failure_closes_wait_before_handlers_and_survives_resume(
        self, job_factory, style, already_answered,
    ):
        """Terminal failure resolves pending waits, never replacing a first response."""
        statek.init(StatekSettings(prompt_defs={}))
        job = job_factory()
        job.job_def.set_chat_style(style)
        t0 = datetime(2026, 10, 15, 12)
        job.created_at = t0
        job.chat_log.append("initial")
        with patch("statek.executors.job.datetime", wraps=datetime) as clock:
            if already_answered:
                clock.now.return_value = t0 + timedelta(seconds=2)
                with record_response_delivery(job):
                    pass
            job.chat_log.append(UserLogItem(message="queued", timestamp=t0 + timedelta(seconds=3)))
            expected = [(t0, 2.0 if already_answered else 5.0),
                        (t0 + timedelta(seconds=3), 2.0)]

            def check_handlers(error):
                assert isinstance(error, LLM_HarnessError)
                assert list(job.get_response_times()) == expected
                clock.now.return_value = t0 + timedelta(seconds=100)

            clock.now.return_value = t0 + timedelta(seconds=5)
            with patch("statek.executors.utils.run_job_step", new_callable=AsyncMock) as step, \
                 patch("statek.executors.utils.handle_critical_error", side_effect=check_handlers):
                step.side_effect = LLM_HarnessError("budget exhausted")
                await job_worker(asyncio.Semaphore(1), job)
            assert job.status == JobStatus.DONE
            assert list(job.get_response_times()) == expected
            identifier = db0.uuid(job)
            db0.close()
            db0.init(DB0_DIR, read_write=True)
            db0.open("test_prefix", "rw")
            job = db0.fetch(identifier)
            assert list(job.get_response_times()) == expected
            assert job.push_user_message("continue") is True
            t2 = t0 + timedelta(seconds=800)
            job.chat_log[-2].timestamp = t2
            clock.now.return_value = t2 + timedelta(seconds=2)
            with record_response_delivery(job):
                pass
        assert list(job.get_response_times()) == expected + [(t2, 2.0)]

    def test_notifies_job_handlers_with_error(self, job_factory):
        """When Statek context is active, the registered handler is invoked with error."""
        _call_log.clear()
        job = job_factory()
        job.add_error_handler(_capture, "ctx")

        exc = RuntimeError("boom")
        with _statek_ctx_scope({'job': job}):
            handle_critical_error(exc)

        assert len(_call_log) == 1
        assert _call_log[0][1] is exc

    def test_bound_application_handler_uses_existing_handler_collection(
        self, job_factory
    ):
        """Latest-input binding replaces itself without adding state to Job."""
        _call_log.clear()
        _application_call_log.clear()
        job = job_factory()
        job.job_def.agent.context["critical_error_handler"] = (
            _capture_application_error
        )
        job.add_error_handler(_capture, "ordinary-context")
        job.bind_critical_error_context("first-message")
        job.bind_critical_error_context("latest-message")

        assert not hasattr(job, "_Job__critical_error_handler")
        assert len(job.error_handlers) == 2

        exc = RuntimeError("boom")
        with _statek_ctx_scope({'job': job}):
            handle_critical_error(exc)
            handle_critical_error(exc)

        assert _call_log == [("ordinary-context", exc)]
        assert _application_call_log == [("latest-message", exc)]
        assert job.error_handlers == []

    def test_bound_application_handler_survives_reopen(self, job_factory):
        """The standard persisted handler collection retains the active binding."""
        _application_call_log.clear()
        job = job_factory()
        job.job_def.agent.context["critical_error_handler"] = (
            _capture_application_error
        )
        job.bind_critical_error_context("latest-message")
        identifier = db0.uuid(job)

        db0.close()
        db0.init(DB0_DIR, read_write=True)
        db0.open("test_prefix", "rw")
        restored = db0.fetch(identifier)
        failure = RuntimeError("after reopen")
        restored.notify_handlers(error=failure)

        assert _application_call_log == [("latest-message", failure)]

    def test_no_job_in_context_does_not_raise(self):
        """handle_critical_error is a no-op when no job is found in context."""
        # No _STATEK_CTX in scope — must not raise
        handle_critical_error(RuntimeError("ignored"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message", [
        "Maximum token usage exceeded: 55320/50002.0",
        "Maximum number of exceptions exceeded: 8/6.0",
        "Maximum consecutive exceptions exceeded: 8/6.0",
        "Maximum number of turns exceeded: 101/100.0",
        "Token budget exhausted",
        "",
    ])
    @pytest.mark.parametrize("with_execution_errors", [False, True])
    async def test_job_worker_notifies_handlers_on_harness_error(
        self, job_factory: Callable[..., Job], message: str, with_execution_errors: bool,
    ) -> None:
        """job_worker calls handle_critical_error when LLM_HarnessError is raised."""
        _call_log.clear()
        job = job_factory()
        for index in range(3 if with_execution_errors else 1):
            job.chat_log.append(LLM_LogItem(console_pos=len(job.py_env.console or [])))
            if with_execution_errors:
                execution_error = f"ValueError: failure {index}"
                job.console_append(
                    execution_error, error=Error(ErrorKind.EXECUTION, execution_error),
                )
        previous_errors = {
            key: error.message for key, error in (job.py_env.exceptions or {}).items()
        }
        job.add_error_handler(_capture, "ctx")
        semaphore = asyncio.Semaphore(1)

        exc = LLM_HarnessError(message)
        with patch('statek.executors.utils.run_job_step', new_callable=AsyncMock) as mock_step:
            mock_step.side_effect = exc
            await job_worker(semaphore, job)

        assert len(_call_log) == 1
        assert _call_log[0][1] is exc
        assert job.status == JobStatus.DONE
        assert isinstance(job.error, JobDefError)
        assert job.error.error_message == message
        assert job.py_env.exit_status == f"Error: {message}"
        error_msg = f"LLM_HarnessError: {message}"
        assert job.py_env.console[-1] == error_msg
        assert {key: error.message for key, error in job.py_env.exceptions.items()} == {
            **previous_errors, len(job.py_env.console) - 1: error_msg,
        }
        assert job.exception_count == (3 if with_execution_errors else 0)
        assert job.max_consecutive_exceptions == (3 if with_execution_errors else 0)
        assert job.py_env.exceptions[len(job.py_env.console) - 1].kind == ErrorKind.HARNESS

    @pytest.mark.asyncio
    async def test_job_worker_notifies_handlers_on_generic_exception(self, job_factory):
        """job_worker calls handle_critical_error for any other critical exception."""
        _call_log.clear()
        job = job_factory()
        job.chat_log.append(LLM_LogItem(console_pos=0))
        job.add_error_handler(_capture, "ctx")
        semaphore = asyncio.Semaphore(1)

        exc = ValueError("unexpected failure")
        with patch('statek.executors.utils.run_job_step', new_callable=AsyncMock) as mock_step:
            mock_step.side_effect = exc
            await job_worker(semaphore, job)

        assert len(_call_log) == 1
        assert _call_log[0][1] is exc
        assert job.status == JobStatus.DONE
        assert isinstance(job.error, JobDefError)
        assert job.error.error_message == "unexpected failure"
        assert job.exception_count == 1
        assert job.py_env.exceptions[0].kind == ErrorKind.EXECUTION

    @pytest.mark.asyncio
    async def test_job_worker_notifies_bound_application_handler(self, job_factory):
        """A worker-level critical exception reports the captured application context."""
        _application_call_log.clear()
        job = job_factory()
        job.job_def.agent.context["critical_error_handler"] = (
            _capture_application_error
        )
        job.bind_critical_error_context("latest-message")
        semaphore = asyncio.Semaphore(1)

        exc = ValueError("unexpected failure")
        with patch('statek.executors.utils.run_job_step', new_callable=AsyncMock) as mock_step:
            mock_step.side_effect = exc
            await job_worker(semaphore, job)

        assert _application_call_log == [("latest-message", exc)]
