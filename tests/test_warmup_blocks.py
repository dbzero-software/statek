"""Tests for warmup_code multiple blocks feature."""
# pylint: disable=no-member

from dataclasses import dataclass

import pytest
import dbzero as db0

from statek.executors.job import Job, JobStatus, parse_warmup_code
from statek.executors.utils import run_job_step
from statek.future import FutureResult
from statek.exceptions import FutureError
from statek.chat_style import ChatStyle
from statek.utils import CodeBlock


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


class TestParseWarmupCode:
    """Test cases for parse_warmup_code function."""

    def test_parse_none(self):
        """Test parsing None returns None."""
        assert parse_warmup_code(None) is None

    def test_parse_single_string_no_separator(self):
        """Test parsing single string without separator returns string."""
        code = "x = 1\nprint(x)"
        result = parse_warmup_code(code)
        assert result == code

    def test_parse_string_with_separator(self):
        """Test parsing string with 10+ dash separator returns list."""
        code = """x = 1
# ----------
y = 2
# ----------
z = 3"""
        result = parse_warmup_code(code)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "x = 1"
        assert result[1] == "y = 2"
        assert result[2] == "z = 3"

    def test_parse_string_with_long_separator(self):
        """Test parsing with more than 10 dashes works."""
        code = """block1
# --------------------
block2"""
        result = parse_warmup_code(code)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_string_short_separator_not_split(self):
        """Test that fewer than 10 dashes doesn't split."""
        code = """x = 1
# ---------
y = 2"""
        result = parse_warmup_code(code)
        # 9 dashes should not split
        assert isinstance(result, str)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_parse_sequence_returns_list(self):
        """Test parsing sequence returns list."""
        code = ["block1", "block2", "block3"]
        result = parse_warmup_code(code)
        assert result == ["block1", "block2", "block3"]

    def test_parse_strips_whitespace(self):
        """Test that blocks are stripped of whitespace."""
        code = """  x = 1
# ------------
  y = 2  """
        result = parse_warmup_code(code)
        assert result[0] == "x = 1"
        assert result[1] == "y = 2"

    def test_parse_hidden_metadata_preserved_on_block(self, db0_fixture):  # pylint: disable=unused-argument
        """Hidden warmup metadata is preserved on the parsed warmup block."""
        result = parse_warmup_code("#STATEK: hidden = True\nprint('hidden')")
        assert isinstance(result, CodeBlock)
        assert result.code == "print('hidden')"
        assert result.metadata == {"hidden": True}


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
    async def test_warmup_console_positions_recorded_after_each_block(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """_warmup_end_positions returns the console length after each completed block."""
        job_def = job_def_factory(warmup_code=[
            'print("block0")',
            'print("block1")',
            'exit("done")',
        ])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        # After first block: one position derived from the next element
        await run_job_step(job)
        positions = job._warmup_end_positions()  # pylint: disable=protected-access
        assert len(positions) == 1
        assert positions[0] == len(job.py_env.console)

        # After second block: two positions recorded
        await run_job_step(job)
        positions = job._warmup_end_positions()  # pylint: disable=protected-access
        assert len(positions) == 2
        assert positions[1] == len(job.py_env.console)

    @pytest.mark.asyncio
    async def test_warmup_console_positions_interleave_correctly(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """Each block's console output is bounded by consecutive WarmupLogItem end positions."""
        job_def = job_def_factory(warmup_code=[
            'print("from block0")',
            'print("from block1")',
            'exit("done")',
        ])
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY
        )

        await run_job_step(job)  # block 0
        await run_job_step(job)  # block 1

        positions = job._warmup_end_positions()  # pylint: disable=protected-access
        pos0 = positions[0]
        pos1 = positions[1]

        # Console lines for block 0 are before pos0
        block0_output = job.py_env.console[:pos0]
        assert any("block0" in line for line in block0_output)

        # Console lines for block 1 are between pos0 and pos1
        block1_output = job.py_env.console[pos0:pos1]
        assert any("block1" in line for line in block1_output)

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

    @pytest.mark.asyncio
    async def test_direct_warmup_plain_code_executes(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """In DIRECT mode, plain Python warmup blocks must still execute.

        Regression: exec_all_steps used to skip ALL plain code in DIRECT mode,
        which dropped warmup print() output and left console_pos=0 across all
        WarmupLogItems, causing empty tool results in get_chat_history.
        """
        job_def = job_def_factory(warmup_code=[
            'print("hello from warmup")',
            'exit("done")',
        ])
        job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,
        )

        await run_job_step(job)
        assert any("hello from warmup" in line for line in job.py_env.console)
        positions = job._warmup_end_positions()  # pylint: disable=protected-access
        assert positions[0] == len(job.py_env.console)
        assert positions[0] > 0

    @pytest.mark.asyncio
    async def test_hidden_warmup_block_executes_normally(
        self, job_def_factory, db0_fixture  # pylint: disable=unused-argument
    ):
        """Hidden warmup blocks execute and advance like regular warmup blocks."""
        parsed_warmup = parse_warmup_code(
            "#STATEK: hidden = True\n"
            "x = 41\n"
            "print('hidden ran')\n"
            "# ----------\n"
            "y = x + 1"
        )
        job_def = job_def_factory(warmup_code=parsed_warmup)
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,
        )

        result = await run_job_step(job)

        assert result is False
        assert job.status == JobStatus.WARMING_UP
        assert job.py_env.local_state["x"] == 41
        assert any("hidden ran" in line for line in job.py_env.console)
        assert job.warmup_block_num == 1
