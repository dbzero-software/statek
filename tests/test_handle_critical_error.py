"""Tests for handle_critical_error and its integration with job_worker."""

# pylint: disable=too-few-public-methods,no-member

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from statek.exceptions import LLM_HarnessError
from statek.executors.job import JobDefError, JobStatus
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
    async def test_job_worker_notifies_handlers_on_harness_error(self, job_factory):
        """job_worker calls handle_critical_error when LLM_HarnessError is raised."""
        _call_log.clear()
        job = job_factory()
        job.add_error_handler(_capture, "ctx")
        semaphore = asyncio.Semaphore(1)

        with patch('statek.executors.utils.run_job_step', new_callable=AsyncMock) as mock_step:
            mock_step.side_effect = LLM_HarnessError("too many turns")
            await job_worker(semaphore, job)

        assert len(_call_log) == 1
        assert isinstance(_call_log[0][1], LLM_HarnessError)
        assert job.status == JobStatus.DONE
        assert isinstance(job.error, JobDefError)
        assert job.error.error_message == "too many turns"

    @pytest.mark.asyncio
    async def test_job_worker_notifies_handlers_on_generic_exception(self, job_factory):
        """job_worker calls handle_critical_error for any other critical exception."""
        _call_log.clear()
        job = job_factory()
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
