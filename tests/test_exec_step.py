"""Tests for exec_step function."""

import pytest
import dbzero as db0

from statek.executors.utils import exec_step

@db0.memo
class MemoObject:  # pylint: disable=too-few-public-methods
    value: int = 0


class TestExecStep:  # pylint: disable=too-few-public-methods
    """Test cases for exec_step function."""

    @pytest.mark.asyncio
    async def test_exec_step_simple_print(self, job_factory):
        """Test exec_step with simple print statement."""        
        simple_job = job_factory(description="Test job", goal="Test goal")        
        code = 'print("Hello, World!")'

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert "Hello, World!" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_variable_assignment(self, job_factory):
        """Test exec_step with variable assignment."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = 'x = 42'

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.local_state.get('x') == 42

    @pytest.mark.asyncio
    async def test_exec_step_multiple_statements(self, job_factory):
        """Test exec_step with multiple statements."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = '''
x = 10
y = 20
z = x + y
print(f"Result: {z}")
'''

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.local_state.get('x') == 10
        assert simple_job.py_env.local_state.get('y') == 20
        assert simple_job.py_env.local_state.get('z') == 30
        assert simple_job.py_env.console is not None
        assert "Result: 30" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_with_exit(self, job_factory):
        """Test exec_step with exit call."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = 'exit("completed")'

        result = await exec_step(code, simple_job)

        assert result is False
        assert simple_job.py_env.exit_status == "completed"

    @pytest.mark.asyncio
    async def test_exec_step_preserves_state(self, job_factory):
        """Test that exec_step preserves state across calls."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code1 = 'counter = 0'
        code2 = 'counter += 1'
        code3 = 'print(counter)'

        await exec_step(code1, simple_job)
        await exec_step(code2, simple_job)
        result = await exec_step(code3, simple_job)

        assert result is True
        assert simple_job.py_env.local_state.get('counter') == 1
        assert "1" in simple_job.py_env.console[-1]

    @pytest.mark.asyncio
    async def test_exec_step_print_with_separator(self, job_factory):
        """Test exec_step with print using custom separator."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = 'print("a", "b", "c", sep="-")'

        result = await exec_step(code, simple_job)

        assert result is True
        assert "a-b-c" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_with_db0_objects(self, job_factory):
        """Test exec_step finishes execution on exit call."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = '''memo_object.value = 15'''
        obj = MemoObject()
        simple_job.py_env.local_state = {'memo_object': obj}
        result = await exec_step(code, simple_job)

        assert result is True
        assert obj.value == 15

    @pytest.mark.asyncio
    async def test_exec_step_finishing_on_exit(self, job_factory):
        """Test exec_step finishes execution on exit call."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = '''print("Start")
exit("Success")
print("This should not run")'''
        result = await exec_step(code, simple_job)

        assert result is False
        assert simple_job.py_env.exit_status == "Success"
        assert any("Start" in line for line in simple_job.py_env.console)
        assert not any("This should not run" in line for line in simple_job.py_env.console)
