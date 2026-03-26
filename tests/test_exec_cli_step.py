"""Tests for exec_cli_step function."""

import pytest

from statek.executors.utils import exec_cli_step


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


class TestExecCliStep:
    """Test cases for exec_cli_step function."""

    def create_job(self, job_factory, job_params=None):
        """Helper to create job with default params."""
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_print_goes_to_callback(self, job_factory):
        """Print output is sent to the console_append callback, not the job console."""
        job = self.create_job(job_factory)
        outputs = []
        await exec_cli_step('print("hello")', job, outputs.append)
        assert outputs == ["hello"]
        assert not job.py_env.console

    @pytest.mark.asyncio
    async def test_returns_true_on_exit(self, job_factory):
        """Returns True when exit() is called."""
        job = self.create_job(job_factory)
        result = await exec_cli_step('exit("done")', job, lambda _: None)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_normally(self, job_factory):
        """Returns False when code finishes without exit()."""
        job = self.create_job(job_factory)
        result = await exec_cli_step('x = 1', job, lambda _: None)
        assert result is False

    @pytest.mark.asyncio
    async def test_shares_job_state(self, job_factory):
        """Execution modifies job local_state like exec_step does."""
        job = self.create_job(job_factory)
        await exec_cli_step('x = 42', job, lambda _: None)
        assert job.py_env.local_state.get('x') == 42
