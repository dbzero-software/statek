"""Tests for run_job_step resumption after FutureError."""
# pylint: disable=no-member,R0903

from dataclasses import dataclass

import pytest
import dbzero as db0

from statek.executors.job import Job, JobStatus
from statek.executors.utils import run_job_step
from statek.future import FutureResult
from statek.exceptions import FutureError


@db0.memo
@dataclass
class MemoObject:
    value: int = 0


def _check_condition_false(_):
    return False


def _fetch_result_not_ready(future_result):
    raise FutureError(future_result=future_result)


def _check_condition_true(_):
    return True


def _fetch_result_from_deps(self):
    return self.deps.value


def create_future_not_ready():
    """Create a FutureResult that raises FutureError when accessed."""
    future = FutureResult(deps=MemoObject(value=0), state_num=0)
    future.set_complement_functions(
        complement=_fetch_result_not_ready,
        condition=_check_condition_false
    )
    return future


def create_future_ready(value=42):
    """Create a ready FutureResult with a specific value."""
    future = FutureResult(deps=MemoObject(value=value), state_num=0)
    future.set_complement_functions(
        complement=_fetch_result_from_deps,
        condition=_check_condition_true
    )
    return future


class TestRunJobStepFutureErrorResumption:
    """Test that run_job_step resumes execution from the exact line that threw FutureError."""

    @pytest.mark.asyncio
    async def test_resumes_from_future_error_line(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Test that execution resumes from the exact line that threw FutureError,
        skipping already-executed instructions."""
        # Code block with multiple statements: the FutureError occurs on the
        # third instruction (idx=2). On resume, instructions 0 and 1 must be
        # skipped so side-effects (like the counter increment) don't repeat.
        code = (
            'counter = counter + 1\n'      # idx 0 - should run once
            'before_flag = True\n'          # idx 1 - should run once
            'result = future_val\n'         # idx 2 - raises FutureError first time
            'after_flag = True'             # idx 3 - runs only after resume
        )
        job_def = job_def_factory(warmup_code=[code, 'exit("ok")'])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        # Seed the environment with initial values
        future_not_ready = create_future_not_ready()
        job.py_env.local_state['counter'] = 0
        job.py_env.local_state['before_flag'] = False
        job.py_env.local_state['after_flag'] = False
        job.py_env.local_state['future_val'] = future_not_ready

        # First run: executes idx 0 (counter=1), idx 1 (before_flag=True),
        # then idx 2 raises FutureError
        result1 = await run_job_step(job)
        assert result1 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state['counter'] == 1
        assert job.py_env.local_state['before_flag'] is True
        assert job.py_env.local_state['after_flag'] is False
        assert job.awaited_result is future_not_ready
        assert job.next_instr_num == 2  # suspended at idx 2
        assert job.warmup_block_num is None  # block did not advance (None means block 0)

        # Make the future ready and replace it in env
        future_ready = create_future_ready(99)
        job.py_env.local_state['future_val'] = future_ready

        # Second run: resumes from idx 2, skipping idx 0 and 1.
        # counter must stay 1 (not incremented again).
        result2 = await run_job_step(job)
        assert result2 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state['counter'] == 1  # NOT 2 — proves skip
        assert job.py_env.local_state['result'] == 99
        assert job.py_env.local_state['after_flag'] is True
        assert job.awaited_result is None
        assert job.next_instr_num is None
        assert job.warmup_block_num == 1  # block advanced

        # Final block: exit
        result3 = await run_job_step(job)
        assert result3 is True
        assert job.status == JobStatus.DONE
