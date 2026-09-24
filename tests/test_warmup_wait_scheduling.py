"""Regression tests for scheduling warmups that await a FutureResult."""
# pylint: disable=no-member

import asyncio
from contextlib import suppress
from dataclasses import dataclass

import dbzero as db0
import pytest

from statek.exceptions import FutureError
from statek.executors.chat_log_item import WarmupLogItem
from statek.executors.job import Job, JobStatus
from statek.executors.utils import run_job_step, run_jobs_loop, unsuspend_jobs
from statek.future import FutureResult


@db0.memo
@dataclass
class WaitState:
    """Mutable dependency used to control and count future readiness checks."""

    ready: bool = False
    value: str = "done"
    condition_checks: int = 0


def _check_wait_state(future_result: FutureResult) -> bool:
    future_result.deps.condition_checks += 1
    return future_result.deps.ready


def _fetch_wait_state(future_result: FutureResult) -> str:
    if not future_result.deps.ready:
        raise FutureError(future_result=future_result)
    return future_result.deps.value


def _controlled_future(wait_state: WaitState) -> FutureResult:
    future = FutureResult(deps=wait_state, state_num=0)
    future.set_complement_functions(
        complement=_fetch_wait_state,
        condition=_check_wait_state,
    )
    return future


def _warmup_items(job: Job) -> list[WarmupLogItem]:
    return [item for item in job.chat_log if isinstance(item, WarmupLogItem)]


async def _wait_until_done(job: Job) -> None:
    while job.status != JobStatus.DONE:
        await asyncio.sleep(0.01)


class TestWarmupWaitReadiness:
    """Waiting warmups remain dormant until their recorded future is ready."""

    @pytest.mark.asyncio
    async def test_direct_calls_do_not_reexecute_unresolved_warmup(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Direct callers cannot bypass the continuation readiness gate."""
        wait_state = WaitState()
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(
                warmup_code="attempts = attempts + 1\nresult = future_value"
            ),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state["attempts"] = 0
        job.py_env.local_state["future_value"] = future

        assert await run_job_step(job) is False
        initial_history = list(job.chat_log)
        initial_console = list(job.py_env.console or [])

        assert await run_job_step(job) is False
        assert await run_job_step(job) is False

        assert job.status == JobStatus.WARMING_UP
        assert job.awaited_result is future
        assert job.next_instr_num == 1
        assert job.py_env.local_state["attempts"] == 1
        assert list(job.chat_log) == initial_history
        assert list(job.py_env.console or []) == initial_console
        assert wait_state.condition_checks == 2

    @pytest.mark.asyncio
    async def test_scheduler_skips_blocked_warmup_and_checks_release_once(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """A blocked warmup does not consume a worker or starve another ready job."""
        wait_state = WaitState()
        future = _controlled_future(wait_state)
        blocked_job = Job(
            job_def=job_def_factory(
                warmup_code="attempts = attempts + 1\nresult = future_value\nexit(result)"
            ),
            job_status=JobStatus.READY,
        )
        blocked_job.py_env.local_state["attempts"] = 0
        blocked_job.py_env.local_state["future_value"] = future
        assert await run_job_step(blocked_job) is False

        ready_job = Job(
            job_def=job_def_factory(warmup_code='exit("ready job completed")'),
            job_status=JobStatus.READY,
        )
        initial_history_length = len(_warmup_items(blocked_job))
        scheduler_cycles = 0
        repeated_wait_observed = asyncio.Event()

        def record_scheduler_cycle(_available_capacity: int) -> None:
            nonlocal scheduler_cycles
            scheduler_cycles += 1
            if scheduler_cycles >= 3:
                repeated_wait_observed.set()

        loop_task = asyncio.create_task(
            run_jobs_loop(
                max_concurrency=1,
                queue_prefixes=[db0.get_current_prefix().name],
                start_jobs_func=record_scheduler_cycle,
            )
        )
        try:
            await asyncio.wait_for(repeated_wait_observed.wait(), timeout=2)

            assert ready_job.status == JobStatus.DONE
            assert blocked_job.status == JobStatus.WARMING_UP
            assert blocked_job.awaited_result is future
            assert blocked_job.py_env.local_state["attempts"] == 1
            assert len(_warmup_items(blocked_job)) == initial_history_length

            checks_before_release = wait_state.condition_checks
            wait_state.ready = True
            await asyncio.wait_for(_wait_until_done(blocked_job), timeout=2)

            assert blocked_job.py_env.exit_status == "done"
            assert blocked_job.py_env.local_state["attempts"] == 1
            assert blocked_job.awaited_result is None
            assert blocked_job.next_instr_num is None
            assert wait_state.condition_checks == checks_before_release + 1
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    def test_ready_suspended_job_is_released_and_clears_awaited_result(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Ordinary suspended jobs retain their existing transition to STARTED."""
        wait_state = WaitState(ready=True)
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(warmup_code=None),
            job_status=JobStatus.SUSPENDED,
        )
        job.awaited_result = future
        job.next_instr_num = 0

        unsuspend_jobs()

        assert job.status == JobStatus.STARTED
        assert job.awaited_result is None
        assert job.next_instr_num == 0
        assert wait_state.condition_checks == 1
