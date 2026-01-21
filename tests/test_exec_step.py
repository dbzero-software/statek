"""Tests for exec_step function."""

from dataclasses import dataclass
import pytest
import dbzero as db0

from statek.executors.utils import exec_step
from statek.future import FutureResult, temporal
from statek.exceptions import FutureError


@db0.memo
@dataclass
class MemoObject:  # pylint: disable=too-few-public-methods
    value: int = 0


def _check_condition_false(_):
    """Check condition that always returns False."""
    return False


def _fetch_result_not_ready(_):
    """Fetch result that raises FutureError."""
    raise FutureError("Not ready")


def _check_condition_true(_):
    """Check condition that always returns True."""
    return True


def _fetch_result_ready(_):
    """Fetch result that returns 42."""
    return 42


def _fetch_result_from_deps(self):
    """Generic fetch result that returns the value from deps."""
    return self.deps.value


def local_print(some_argument):
    """Local print function that just prints the argument."""
    print("Local print called with argument:")
    print(some_argument)


# Define temporal functions as globals
@temporal(complement=_fetch_result_ready, condition=_check_condition_true)
def get_value():
    """Temporal function that returns a ready FutureResult."""
    return FutureResult(
        deps=MemoObject(value=42),
        state_num=0
    )


@temporal(complement=_fetch_result_not_ready, condition=_check_condition_false)
def get_value_not_ready():
    """Temporal function that returns a not-ready FutureResult."""
    return FutureResult(
        deps=MemoObject(value=0),
        state_num=0
    )


def function_with_future_typehint(future_param: FutureResult):
    """Function that expects a FutureResult parameter (should not be unwrapped)."""
    # Check if we received a FutureResult object
    if isinstance(future_param, FutureResult):
        print("Received FutureResult object")
        return True
    print(f"Received unwrapped value: {future_param}")
    return False


def function_without_typehint(param):
    """Function without type hint (parameter should be unwrapped)."""
    print(f"Received value: {param}, type: {type(param).__name__}")
    return param


def function_with_args(*args):
    """Function with *args (should unwrap each arg)."""
    result = []
    for i, arg in enumerate(args):
        print(f"Arg {i}: {arg}, type: {type(arg).__name__}")
        result.append(arg)
    return result


def function_with_kwargs(**kwargs):
    """Function with **kwargs (should unwrap each value)."""
    result = {}
    for key, value in kwargs.items():
        print(f"Kwarg {key}: {value}, type: {type(value).__name__}")
        result[key] = value
    return result


def function_with_mixed(regular_param, *args, **kwargs):
    """Function with regular param, *args, and **kwargs."""
    print(f"Regular: {regular_param}")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
    return {'regular': regular_param, 'args': args, 'kwargs': kwargs}


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
    async def test_exec_step_print_from_function(self, job_factory):
        """Test exec_step with simple print statement."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        code = 'local_print("Hello, World!")'
        simple_job.py_env.local_state = {'local_print': local_print}
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 2
        assert "Local print called with argument:" in simple_job.py_env.console[0]
        assert "Hello, World!" in simple_job.py_env.console[1]

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

    @pytest.mark.asyncio
    async def test_exec_step_print_with_future_result(self, job_factory):
        """Test exec_step with print statement accepting FutureResult."""

        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create a FutureResult that initially raises FutureError
        future_not_ready = FutureResult(
            deps=MemoObject(value=0),
            state_num=0
        )
        future_not_ready.set_complement_functions(
            complement=_fetch_result_not_ready,
            condition=_check_condition_false
        )

        simple_job.py_env.local_state = {'future_val': future_not_ready, 'local_print': local_print}
        code = 'print(future_val)'

        # First execution should skip the print due to FutureError
        result = await exec_step(code, simple_job)

        assert result is True
        # Console should be empty since print was skipped
        assert simple_job.py_env.console is None

        # Now create a FutureResult that's ready
        future_ready = FutureResult(
            deps=MemoObject(value=42),
            state_num=0
        )
        future_ready.set_complement_functions(
            complement=_fetch_result_ready,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {'future_val': future_ready, 'local_print': local_print}

        # Second execution should print the value
        result = await exec_step(code, simple_job)

        assert result is True
        assert len(simple_job.py_env.console) == 1
        assert "42" in simple_job.py_env.console[0]


    @pytest.mark.asyncio
    async def test_exec_step_local_function_with_future_result(self, job_factory):
        """Test exec_step with local function statement accepting FutureResult."""

        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create a FutureResult that initially raises FutureError
        future_not_ready = FutureResult(
            deps=MemoObject(value=0),
            state_num=0
        )
        future_not_ready.set_complement_functions(
            complement=_fetch_result_not_ready,
            condition=_check_condition_false
        )

        simple_job.py_env.local_state = {'future_val': future_not_ready, 'local_print': local_print}
        code = 'local_print(future_val)'

        # First execution should skip the print due to FutureError
        result = await exec_step(code, simple_job)

        assert result is True
        # Console should be empty since print was skipped
        assert simple_job.py_env.console is None

        # Now create a FutureResult that's ready
        future_ready = FutureResult(
            deps=MemoObject(value=42),
            state_num=0
        )
        future_ready.set_complement_functions(
            complement=_fetch_result_ready,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {'future_val': future_ready, 'local_print': local_print}

        # Second execution should print the value
        result = await exec_step(code, simple_job)

        assert result is True
        assert len(simple_job.py_env.console) == 2
        assert "42" in simple_job.py_env.console[1]

    @pytest.mark.asyncio
    async def test_exec_step_temporal_function_print(self, job_factory):
        """Test exec_step with temporal function and print."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        simple_job.py_env.local_state = {'get_value': get_value}

        code = '''result = get_value()
print(result)'''

        # Execute and check that print works with temporal function result
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert "42" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_temporal_function_not_ready(self, job_factory):
        """Test exec_step with temporal function that's not ready."""
        simple_job = job_factory(description="Test job", goal="Test goal")
        simple_job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        simple_job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}

        code = '''result = get_value_not_ready()
print(result)'''

        # Execute - should skip the print due to FutureError
        result = await exec_step(code, simple_job)

        assert result is True
        # Console should be empty since print was skipped
        assert simple_job.py_env.console is None

    @pytest.mark.asyncio
    async def test_exec_step_future_typehint_not_unwrapped(self, job_factory):
        """Test that function with FutureResult typehint receives unwrapped FutureResult."""
        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create a ready FutureResult
        future_ready = FutureResult(
            deps=MemoObject(value=42),
            state_num=0
        )
        future_ready.set_complement_functions(
            complement=_fetch_result_ready,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {
            'function_with_future_typehint': function_with_future_typehint,
            'future_val': future_ready
        }

        code = '''result = function_with_future_typehint(future_val)'''

        # Execute - function should receive FutureResult object, not unwrapped value
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Received FutureResult object" in simple_job.py_env.console[0]
        assert simple_job.py_env.local_state.get('result') is True

    @pytest.mark.asyncio
    async def test_exec_step_no_typehint_unwrapped(self, job_factory):
        """Test that function without typehint receives unwrapped value."""
        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create a ready FutureResult
        future_ready = FutureResult(
            deps=MemoObject(value=42),
            state_num=0
        )
        future_ready.set_complement_functions(
            complement=_fetch_result_ready,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {
            'function_without_typehint': function_without_typehint,
            'future_val': future_ready
        }

        code = '''result = function_without_typehint(future_val)'''

        # Execute - function should receive unwrapped value (42), not FutureResult
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Received value: 42" in simple_job.py_env.console[0]
        assert simple_job.py_env.local_state.get('result') == 42

    @pytest.mark.asyncio
    async def test_exec_step_function_with_args(self, job_factory):
        """Test that function with *args receives unwrapped values."""
        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create multiple FutureResult objects
        future1 = FutureResult(deps=MemoObject(value=10), state_num=0)
        future1.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future2 = FutureResult(deps=MemoObject(value=20), state_num=0)
        future2.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future3 = FutureResult(deps=MemoObject(value=30), state_num=0)
        future3.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {
            'function_with_args': function_with_args,
            'f1': future1,
            'f2': future2,
            'f3': future3
        }

        code = '''result = function_with_args(f1, f2, f3)'''

        # Execute - function should receive unwrapped values
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Arg 0: 10" in simple_job.py_env.console[0]
        assert "Arg 1: 20" in simple_job.py_env.console[1]
        assert "Arg 2: 30" in simple_job.py_env.console[2]
        assert simple_job.py_env.local_state.get('result') == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_exec_step_function_with_kwargs(self, job_factory):
        """Test that function with **kwargs receives unwrapped values."""
        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create FutureResult objects
        future_a = FutureResult(deps=MemoObject(value=100), state_num=0)
        future_a.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future_b = FutureResult(deps=MemoObject(value=200), state_num=0)
        future_b.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {
            'function_with_kwargs': function_with_kwargs,
            'fa': future_a,
            'fb': future_b
        }

        code = '''result = function_with_kwargs(x=fa, y=fb)'''

        # Execute - function should receive unwrapped values
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Kwarg x: 100" in simple_job.py_env.console[0]
        assert "Kwarg y: 200" in simple_job.py_env.console[1]
        assert simple_job.py_env.local_state.get('result') == {'x': 100, 'y': 200}

    @pytest.mark.asyncio
    async def test_exec_step_function_with_mixed_params(self, job_factory):
        """Test that function with mixed parameters receives unwrapped values."""
        simple_job = job_factory(description="Test job", goal="Test goal")

        # Create FutureResult objects
        future_reg = FutureResult(deps=MemoObject(value=1), state_num=0)
        future_reg.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future_arg1 = FutureResult(deps=MemoObject(value=2), state_num=0)
        future_arg1.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future_arg2 = FutureResult(deps=MemoObject(value=3), state_num=0)
        future_arg2.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        future_kw = FutureResult(deps=MemoObject(value=4), state_num=0)
        future_kw.set_complement_functions(
            complement=_fetch_result_from_deps,
            condition=_check_condition_true
        )

        simple_job.py_env.local_state = {
            'function_with_mixed': function_with_mixed,
            'fr': future_reg,
            'fa1': future_arg1,
            'fa2': future_arg2,
            'fk': future_kw
        }

        code = '''result = function_with_mixed(fr, fa1, fa2, key=fk)'''

        # Execute - function should receive all unwrapped values
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        retrieved_result = simple_job.py_env.local_state.get('result')
        assert retrieved_result['regular'] == 1
        assert retrieved_result['args'] == (2, 3)
        assert retrieved_result['kwargs'] == {'key': 4}
