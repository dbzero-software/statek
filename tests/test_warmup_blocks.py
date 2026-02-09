"""Tests for warmup_code multiple blocks feature."""

import pytest
from dataclasses import dataclass
import dbzero as db0

from statek.executors.job import Job, JobDef, JobStatus
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


class TestRunJobStepMultipleBlocks:
    """Test cases for run_job_step with multiple warmup blocks."""

    @pytest.mark.asyncio
    async def test_run_job_step_multiple_blocks_progression(self, job_def_factory, db0_fixture):  # pylint: disable=unused-argument
        """Test run_job_step progresses through all warmup blocks."""
        job_def = job_def_factory(warmup_code=['x = 1', 'y = x + 1', 'exit("complete")'])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        # First block: x = 1
        result1 = await run_job_step(job)
        assert result1 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state.get('x') == 1
        assert job.warmup_block_num == 1

        # Second block: y = x + 1
        result2 = await run_job_step(job)
        assert result2 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state.get('y') == 2
        assert job.warmup_block_num == 2

        # Third block: exit("complete")
        result3 = await run_job_step(job)
        assert result3 is True
        assert job.status == JobStatus.DONE
        assert job.py_env.exit_status == "complete"

    @pytest.mark.asyncio
    async def test_run_job_step_multiple_blocks_continuation_after_future_error(
        self, job_def_factory, db0_fixture
    ):  # pylint: disable=unused-argument
        """Test continuation from warmup block after FutureError."""
        # Create warmup blocks where second block will suspend on FutureResult
        job_def = job_def_factory(warmup_code=[
            'x = 1',
            'result = future_val\nprint(result)',
            'exit("done")'
        ])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        # First block executes successfully
        result1 = await run_job_step(job)
        assert result1 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state.get('x') == 1
        assert job.warmup_block_num == 1

        # Inject not-ready future before second block
        future_not_ready = create_future_not_ready()
        job.py_env.local_state['future_val'] = future_not_ready

        # Second block should suspend (stay in WARMING_UP, FutureError caught)
        result2 = await run_job_step(job)
        assert result2 is False
        assert job.status == JobStatus.WARMING_UP
        assert job.awaited_result is future_not_ready
        assert job.next_instr_num == 0  # Suspended at result = future_val instruction
        # warmup_block_num should NOT advance since block didn't complete
        assert job.warmup_block_num == 1

        # Make the future ready
        future_ready = create_future_ready(42)
        job.py_env.local_state['future_val'] = future_ready

        # Resume second block - should complete and advance
        result3 = await run_job_step(job)
        assert result3 is False
        assert job.status == JobStatus.WARMING_UP
        assert "42" in job.py_env.console[0]
        assert job.warmup_block_num == 2

        # Third block: exit
        result4 = await run_job_step(job)
        assert result4 is True
        assert job.status == JobStatus.DONE
        assert job.py_env.exit_status == "done"
