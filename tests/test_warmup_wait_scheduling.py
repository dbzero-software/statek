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
from statek.utils import CallSpec, CodeBlock
from tests.conftest import DB0_DIR


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


def _reopen_job(job: Job) -> Job:
    identifier = db0.uuid(job)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    return db0.fetch(identifier)


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


class TestPersistedWarmupWait:
    """Warmup continuation state remains correct across dbzero reopen."""

    @pytest.mark.asyncio
    async def test_unresolved_wait_survives_reopen_and_resumes_exact_instruction(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Restart keeps the recorded future and skips completed instructions."""
        wait_state = WaitState()
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(
                warmup_code=(
                    "attempts = attempts + 1\n"
                    "result = future_value\n"
                    "exit(result)"
                )
            ),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state["attempts"] = 0
        job.py_env.local_state["future_value"] = future

        assert await run_job_step(job) is False
        restored = _reopen_job(job)
        restored_future = restored.awaited_result

        assert restored.status == JobStatus.WARMING_UP
        assert restored.next_instr_num == 1
        assert restored_future is restored.py_env.local_state["future_value"]
        assert await run_job_step(restored) is False
        assert restored.py_env.local_state["attempts"] == 1
        assert len(_warmup_items(restored)) == 1

        restored_future.deps.ready = True
        assert await run_job_step(restored) is True

        assert restored.status == JobStatus.DONE
        assert restored.py_env.exit_status == "done"
        assert restored.py_env.local_state["attempts"] == 1
        assert restored.awaited_result is None
        assert restored.next_instr_num is None
        assert len(_warmup_items(restored)) == 1

    @pytest.mark.asyncio
    async def test_released_continuation_survives_reopen_until_execution(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """A persisted release latch retains its exact resume instruction."""
        wait_state = WaitState()
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(
                warmup_code=(
                    "attempts = attempts + 1\n"
                    "result = future_value\n"
                    "exit(result)"
                )
            ),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state["attempts"] = 0
        job.py_env.local_state["future_value"] = future

        assert await run_job_step(job) is False
        wait_state.ready = True
        unsuspend_jobs()
        assert job.awaited_result is None
        assert job.next_instr_num == 1

        restored = _reopen_job(job)

        assert restored.status == JobStatus.WARMING_UP
        assert restored.awaited_result is None
        assert restored.next_instr_num == 1
        assert await run_job_step(restored) is True
        assert restored.py_env.local_state["attempts"] == 1
        assert restored.next_instr_num is None


class TestWarmupWaitCompatibility:
    """Continuation release remains compatible with repeated waits and old history."""

    @pytest.mark.asyncio
    async def test_resumed_instruction_can_wait_on_a_second_future(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """A new FutureError replaces the released continuation state."""
        first_state = WaitState(value="first")
        second_state = WaitState(value="second")
        first_future = _controlled_future(first_state)
        second_future = _controlled_future(second_state)
        job = Job(
            job_def=job_def_factory(
                warmup_code=(
                    "attempts = attempts + 1\n"
                    "first = first_future\n"
                    "between = between + 1\n"
                    "second = second_future\n"
                    "exit(second)"
                )
            ),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state.update({
            "attempts": 0,
            "between": 0,
            "first_future": first_future,
            "second_future": second_future,
        })

        assert await run_job_step(job) is False
        assert job.awaited_result is first_future
        assert job.next_instr_num == 1

        first_state.ready = True
        assert await run_job_step(job) is False

        assert job.awaited_result is second_future
        assert job.next_instr_num == 3
        assert job.py_env.local_state["attempts"] == 1
        assert job.py_env.local_state["between"] == 1
        assert len(_warmup_items(job)) == 1

        assert await run_job_step(job) is False
        assert job.py_env.local_state["between"] == 1

        second_state.ready = True
        assert await run_job_step(job) is True
        assert job.py_env.exit_status == "second"
        assert job.awaited_result is None
        assert job.next_instr_num is None
        assert len(_warmup_items(job)) == 1

    @pytest.mark.asyncio
    async def test_existing_duplicate_history_reuses_latest_item_without_cleanup(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Previously duplicated history stays intact and does not grow again."""
        cli_call = CallSpec(
            id="CLI-OLD",
            func_name="python_cli",
            kwargs={
                "code": (
                    'print("before")\n'
                    "result = future_value\n"
                    'print("after", result)'
                )
            },
        )
        wait_state = WaitState(value="ready")
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(
                warmup_code=[CodeBlock(code=None, tool_calls=[cli_call]), 'exit("done")']
            ),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state["future_value"] = future

        assert await run_job_step(job) is False
        original = _warmup_items(job)[0]
        duplicate = WarmupLogItem(
            console_pos=original.console_pos,
            warmup_block_num=original.warmup_block_num,
            tool_log=["before"],
        )
        job.chat_log.append(duplicate)

        wait_state.ready = True
        assert await run_job_step(job) is False

        assert _warmup_items(job) == [original, duplicate]
        assert original.tool_log == ["before"]
        assert duplicate.tool_log == ["before\nafter ready"]

    @pytest.mark.asyncio
    async def test_auto_terminate_waits_for_unresolved_warmup(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """The jobs loop remains active until a waiting warmup can finish."""
        wait_state = WaitState()
        future = _controlled_future(wait_state)
        job = Job(
            job_def=job_def_factory(warmup_code="result = future_value\nexit(result)"),
            job_status=JobStatus.READY,
        )
        job.py_env.local_state["future_value"] = future
        assert await run_job_step(job) is False

        scheduler_cycles = 0
        waiting_observed = asyncio.Event()

        def record_scheduler_cycle(_available_capacity: int) -> None:
            nonlocal scheduler_cycles
            scheduler_cycles += 1
            if scheduler_cycles >= 2:
                waiting_observed.set()

        loop_task = asyncio.create_task(
            run_jobs_loop(
                max_concurrency=1,
                queue_prefixes=[db0.get_current_prefix().name],
                start_jobs_func=record_scheduler_cycle,
                auto_terminate=True,
            )
        )
        try:
            await asyncio.wait_for(waiting_observed.wait(), timeout=2)
            assert loop_task.done() is False
            assert job.status == JobStatus.WARMING_UP
            assert len(_warmup_items(job)) == 1

            wait_state.ready = True
            await asyncio.wait_for(loop_task, timeout=3)

            assert job.status == JobStatus.DONE
            assert job.py_env.exit_status == "done"
            assert len(_warmup_items(job)) == 1
        finally:
            if not loop_task.done():
                loop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_task
