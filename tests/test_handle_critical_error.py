"""Tests for handle_critical_error and its integration with job_worker."""

# pylint: disable=too-few-public-methods,no-member

import asyncio
from typing import Callable
from unittest.mock import AsyncMock, patch

import pytest

from statek.exceptions import LLM_HarnessError
from statek.pyenv import Error, ErrorKind
from statek.executors.job import Job, JobDefError, JobStatus
from statek.executors.chat_log_item import LLM_LogItem
from statek.executors.utils import handle_critical_error, job_worker
from statek.system import error_handler
from statek.utils import _statek_ctx_scope


# Module-level capture list — mutations visible across calls
_call_log = []


@error_handler
def _capture(context, error=None):  # pylint: disable=unused-argument
    """Append received (context, error) to the module-level capture list."""
    _call_log.append((context, error))


class TestHandleCriticalError:
    """Unit tests for handle_critical_error."""

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
