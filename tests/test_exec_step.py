"""Tests for exec_step function."""
# pylint: disable=too-many-lines

from dataclasses import dataclass
from datetime import date, datetime
import pytest
import dbzero as db0

from statek.executors.utils import exec_step
from statek.future import FutureResult, FutureElement, temporal
from statek.exceptions import FutureError
from statek.system import tool
from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus


@db0.memo
@dataclass
class MemoObject:  # pylint: disable=too-few-public-methods
    value: int = 0

@db0.memo
class MemoWithMethod:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

@db0.memo
class MemoWithValueAndReady:
    def __init__(self, value, ready):
        self.value = value
        self.ready = ready


def _check_condition_false(_):
    """Check condition that always returns False."""
    return False


def _fetch_result_not_ready(future_result):
    """Fetch result that raises FutureError."""
    raise FutureError(future_result=future_result)


def _check_condition_true(_):
    """Check condition that always returns True."""
    return True


def _fetch_result_ready(_):
    """Fetch result that returns 42."""
    return 42


def _fetch_result_from_deps(self):
    """Generic fetch result that returns the value from deps."""
    return self.deps.value


def _fetch_result_when_deps_ready(self):
    """Return the dependency value only after deps.ready is set."""
    if self.deps.ready:
        return self.deps.value
    raise FutureError(future_result=self)


def _fetch_result_pair_from_deps(self) -> tuple[MemoWithMethod, MemoWithMethod]:
    """Fetch a typed tuple result from deps for FutureResult unpacking."""
    if self.deps.ready:
        return self.deps.value
    self.deps.ready = True  # Simulate that the result becomes ready after first check
    raise FutureError(future_result=self)


def local_print(some_argument):
    """Local print function that just prints the argument."""
    print("Local print called with argument:")
    print(some_argument)


def compute(a, b, c):
    """Function that computes sum of three arguments."""
    return a + b + c


_pending_expression_future = None
_pending_expression_call_count = 0


def get_pending_expression_value():
    """Module-level helper because dbzero cannot persist nested callables."""
    global _pending_expression_call_count  # pylint: disable=global-statement
    _pending_expression_call_count += 1
    return _pending_expression_future


# Helper functions for creating FutureResults
def create_future(value, complement, condition):
    """Create a FutureResult with specified value and complement functions.

    Args:
        value: The value to store in the MemoObject deps
        complement: The complement function to fetch the result
        condition: The condition function to check readiness

    Returns:
        FutureResult configured with the provided parameters
    """
    future = FutureResult(
        deps=MemoObject(value=value),
        state_num=0
    )
    future.set_complement_functions(
        complement=complement,
        condition=condition
    )
    return future


def create_future_not_ready():
    """Create a FutureResult that raises FutureError when accessed."""
    return create_future(
        value=0,
        complement=_fetch_result_not_ready,
        condition=_check_condition_false
    )


def create_future_ready(value=42):
    """Create a ready FutureResult with a specific value."""
    return create_future(
        value=value,
        complement=_fetch_result_from_deps,
        condition=_check_condition_true
    )


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


@temporal(complement=_fetch_result_pair_from_deps, condition=_check_condition_false)
def get_two_values():
    """Temporal function that returns a FutureResult resolving to two values."""
    return FutureResult(
        deps=MemoWithValueAndReady(value=(MemoWithMethod(7), MemoWithMethod(9)), ready=False),
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


# Default job parameters used across tests
DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


@db0.memo
@dataclass
class _LLMReprTestObj:  # pylint: disable=too-few-public-methods
    """Object with __llm_repr__ for implicit expression output tests."""
    name: str = "test"

    def __llm_repr__(self):
        return f"custom:{self.name}"


# pylint: disable=too-many-public-methods
class TestExecStep:
    """Test cases for exec_step function."""

    def create_job(self, job_factory, job_params=None):
        """Helper to create job with default params."""
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    @pytest.mark.asyncio
    async def test_exec_step_simple_print(self, job_factory):
        """Test exec_step with simple print statement."""
        simple_job = self.create_job(job_factory)
        code = 'print("Hello, World!")'

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Hello, World!" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_print_from_function(self, job_factory):
        """Test exec_step with simple print statement."""
        simple_job = self.create_job(job_factory)
        code = 'local_print("Hello, World!")'
        simple_job.py_env.local_state = {'local_print': local_print}
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "Local print called with argument:" in simple_job.py_env.console[0]
        assert "Hello, World!" in simple_job.py_env.console[1]

    @pytest.mark.asyncio
    async def test_exec_step_variable_assignment(self, job_factory):
        """Test exec_step with variable assignment."""
        simple_job = self.create_job(job_factory)
        code = 'x = 42'

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.local_state.get('x') == 42

    @pytest.mark.asyncio
    async def test_exec_step_multiple_statements(self, job_factory):
        """Test exec_step with multiple statements."""
        simple_job = self.create_job(job_factory)
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
        simple_job = self.create_job(job_factory)
        code = 'exit("completed")'

        result = await exec_step(code, simple_job)

        assert result is False
        assert simple_job.py_env.exit_status == "completed"

    @pytest.mark.asyncio
    async def test_exec_step_preserves_state(self, job_factory):
        """Test that exec_step preserves state across calls."""
        simple_job = self.create_job(job_factory)
        code1 = 'counter = 0'
        code2 = 'counter += 1'
        code3 = 'print(counter)'

        await exec_step(code1, simple_job)
        await exec_step(code2, simple_job)
        result = await exec_step(code3, simple_job)

        assert result is True
        assert simple_job.py_env.local_state.get('counter') == 1
        assert "1" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_print_with_separator(self, job_factory):
        """Test exec_step with print using custom separator."""
        simple_job = self.create_job(job_factory)
        code = 'print("a", "b", "c", sep="-")'

        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console[0] == 'a-b-c'

    @pytest.mark.asyncio
    async def test_exec_with_db0_objects(self, job_factory):
        """Test exec_step finishes execution on exit call."""
        simple_job = self.create_job(job_factory)
        code = '''memo_object.value = 15'''
        obj = MemoObject()
        simple_job.py_env.local_state = {'memo_object': obj}
        result = await exec_step(code, simple_job)

        assert result is True
        assert obj.value == 15

    @pytest.mark.asyncio
    async def test_exec_step_finishing_on_exit(self, job_factory):
        """Test exec_step finishes execution on exit call."""
        simple_job = self.create_job(job_factory)
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

        simple_job = self.create_job(job_factory)

        # Create a FutureResult that initially raises FutureError
        future_not_ready = create_future_not_ready()

        simple_job.py_env.local_state = {'future_val': future_not_ready, 'local_print': local_print}
        code = 'print(future_val)'

        # First execution should raise FutureError with instr_num
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        # Verify the exception has instr_num set
        assert exc_info.value.instr_num == 0
        assert exc_info.value.future_result is future_not_ready
        # Console should be empty since print was not executed
        assert simple_job.py_env.console is None

        # Now create a FutureResult that's ready
        future_ready = create_future_ready()

        simple_job.py_env.local_state = {'future_val': future_ready, 'local_print': local_print}

        # Second execution should print the value
        result = await exec_step(code, simple_job)

        assert result is True
        assert "42" in simple_job.py_env.console[0]


    @pytest.mark.asyncio
    async def test_exec_step_local_function_with_future_result(self, job_factory):
        """Test exec_step with local function statement accepting FutureResult."""

        simple_job = self.create_job(job_factory)

        # Create a FutureResult that initially raises FutureError
        future_not_ready = create_future_not_ready()

        simple_job.py_env.local_state = {'future_val': future_not_ready, 'local_print': local_print}
        code = 'local_print(future_val)'

        # First execution should raise FutureError with instr_num
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        # Verify the exception has instr_num set
        assert exc_info.value.instr_num == 0
        assert exc_info.value.future_result is future_not_ready
        # Console should be empty since print was not executed
        assert simple_job.py_env.console is None

        # Now create a FutureResult that's ready
        future_ready = create_future_ready()

        simple_job.py_env.local_state = {'future_val': future_ready, 'local_print': local_print}

        # Second execution should print the value
        result = await exec_step(code, simple_job)

        assert result is True
        assert "42" in simple_job.py_env.console[1]

    @pytest.mark.asyncio
    async def test_exec_step_temporal_function_print(self, job_factory):
        """Test exec_step with temporal function and print."""
        simple_job = self.create_job(job_factory)
        simple_job.py_env.local_state = {'get_value': get_value}

        code = '''result = get_value()
print(result)'''

        # Execute and check that print works with temporal function result
        result = await exec_step(code, simple_job)

        assert result is True
        assert simple_job.py_env.console is not None
        assert "42" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_temporal_function_not_ready(self, job_factory):
        """Test exec_step with temporal function that's not ready."""
        simple_job = self.create_job(job_factory)
        simple_job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}
        simple_job.py_env.local_state = {'get_value_not_ready': get_value_not_ready}

        code = '''result = get_value_not_ready()
print(result)'''

        # Execute - should raise FutureError with instr_num on the second statement (print)
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        # Verify the exception has instr_num set to the second instruction (1)
        # The first instruction assigns the FutureResult, the second tries to print it
        assert exc_info.value.instr_num == 1
        # Console should be empty since print was not executed
        assert simple_job.py_env.console is None

    @pytest.mark.asyncio
    async def test_exec_step_temporal_function_unpacks_to_future_elements(self, job_factory):
        """Temporal FutureResult tuple unpacking stores FutureElement instances."""
        simple_job = self.create_job(job_factory)
        simple_job.py_env.local_state = {'get_two_values': get_two_values}

        code = '''left, right = get_two_values()\nprint(left.get_value(), right.get_value())'''

        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        assert exc_info.value.instr_num == 1
        assert simple_job.py_env.console is None
        assert isinstance(simple_job.py_env.local_state.get('left'), FutureElement)
        assert isinstance(simple_job.py_env.local_state.get('right'), FutureElement)

        result = await exec_step(code, simple_job, instr_num=1)

        assert result is True
        assert simple_job.py_env.console is not None
        assert any('7 9' in line for line in simple_job.py_env.console)

    @pytest.mark.asyncio
    async def test_exec_step_continuation_with_instr_num(self, job_factory):
        """Test exec_step continuation from specific instruction using instr_num."""
        simple_job = self.create_job(job_factory)

        # Multi-statement code
        code = '''x = 1
y = 2
print(x)
print(y)
z = 3'''

        # Execute first time and check all statements execute
        result = await exec_step(code, simple_job)
        assert result is True
        assert len(simple_job.py_env.console) == 2
        assert "1" in simple_job.py_env.console[0]
        assert "2" in simple_job.py_env.console[1]

        # Clear console for next test
        simple_job.py_env.console = None

        # Now execute starting from instruction 2 (third statement: print(x))
        result = await exec_step(code, simple_job, instr_num=2)
        assert result is True
        # Should only execute print(x), print(y), and z = 3
        assert len(simple_job.py_env.console) == 2
        assert "1" in simple_job.py_env.console[0]
        assert "2" in simple_job.py_env.console[1]
        assert simple_job.py_env.local_state['z'] == 3

    @pytest.mark.asyncio
    async def test_exec_step_continuation_after_future_error(self, job_factory):
        """Test exec_step continuation after FutureError using instr_num."""
        simple_job = self.create_job(job_factory)

        # Create a FutureResult that initially raises FutureError
        future_not_ready = create_future_not_ready()

        # Multi-statement code where second statement will raise FutureError
        code = '''x = 1
print(future_val)
y = 2'''

        simple_job.py_env.local_state = {'future_val': future_not_ready}

        # First execution should raise FutureError at instruction 1
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        assert exc_info.value.instr_num == 1
        # x should be assigned but console should be empty
        assert simple_job.py_env.local_state['x'] == 1
        assert simple_job.py_env.console is None

        # Now make the future ready
        future_ready = create_future_ready()
        simple_job.py_env.local_state['future_val'] = future_ready

        # Continue from instruction 1 (where error occurred)
        result = await exec_step(code, simple_job, instr_num=1)
        assert result is True
        # Now print should succeed and y should be assigned
        assert "42" in simple_job.py_env.console[0]
        assert simple_job.py_env.local_state['y'] == 2

    @pytest.mark.asyncio
    async def test_exec_step_future_error_in_function_argument(self, job_factory):
        """Test FutureError raised in function argument evaluation."""
        simple_job = self.create_job(job_factory)

        # Create a FutureResult that initially raises FutureError
        future_not_ready = create_future_not_ready()

        simple_job.py_env.local_state = {
            'future_val': future_not_ready,
            'compute': compute
        }

        # Code where FutureError occurs in middle of argument evaluation
        code = '''x = 1
result = compute(10, future_val, 30)
y = 2'''

        # First execution should raise FutureError at instruction 1 (compute call)
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        assert exc_info.value.instr_num == 1
        assert exc_info.value.future_result is future_not_ready
        # First instruction executed, second failed, third not reached
        assert simple_job.py_env.local_state['x'] == 1
        assert 'result' not in simple_job.py_env.local_state
        assert 'y' not in simple_job.py_env.local_state

        # Now make the future ready
        future_ready = create_future_ready(20)
        simple_job.py_env.local_state['future_val'] = future_ready

        # Continue from instruction 1 (the compute call)
        result = await exec_step(code, simple_job, instr_num=1)
        assert result is True
        # Now compute should succeed with value 20 and y should be assigned
        assert simple_job.py_env.local_state['result'] == 60  # 10 + 20 + 30
        assert simple_job.py_env.local_state['y'] == 2

    @pytest.mark.asyncio
    async def test_exec_step_future_error_in_nested_expression(self, job_factory):
        """Test FutureError raised in nested expression evaluation."""
        simple_job = self.create_job(job_factory)

        # Create a FutureResult that initially raises FutureError
        future_not_ready = create_future_not_ready()

        simple_job.py_env.local_state = {'future_val': future_not_ready}

        # Code with nested expression containing FutureResult
        code = '''a = 5
b = a + future_val * 2
c = 3'''

        # First execution should raise FutureError at instruction 1
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        assert exc_info.value.instr_num == 1
        assert exc_info.value.future_result is future_not_ready
        assert simple_job.py_env.local_state['a'] == 5
        assert 'b' not in simple_job.py_env.local_state
        assert 'c' not in simple_job.py_env.local_state

        # Now make the future ready
        future_ready = create_future_ready(10)
        simple_job.py_env.local_state['future_val'] = future_ready

        # Continue from instruction 1
        result = await exec_step(code, simple_job, instr_num=1)
        assert result is True
        assert simple_job.py_env.local_state['b'] == 25  # 5 + 10 * 2
        assert simple_job.py_env.local_state['c'] == 3

    @pytest.mark.asyncio
    async def test_exec_step_multiple_future_errors_in_sequence(self, job_factory):
        """Test multiple FutureErrors in different instructions - step-by-step continuation."""
        simple_job = self.create_job(job_factory)

        # Create two FutureResults that raise FutureError
        future_not_ready_1 = create_future_not_ready()
        future_not_ready_2 = create_future_not_ready()

        simple_job.py_env.local_state = {
            'future_val_1': future_not_ready_1,
            'future_val_2': future_not_ready_2
        }

        code = '''x = 1
y = 2
print(future_val_1)
z = 3
print(future_val_2)
w = 4'''

        # First execution should raise FutureError at instruction 2 (first print)
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job)

        assert exc_info.value.instr_num == 2
        assert exc_info.value.future_result is future_not_ready_1
        assert simple_job.py_env.local_state['x'] == 1
        assert simple_job.py_env.local_state['y'] == 2

        # Make first future ready but second still not ready
        future_ready_1 = create_future_ready()
        simple_job.py_env.local_state['future_val_1'] = future_ready_1

        # Continue from instruction 2, should now fail at instruction 4 (second print)
        with pytest.raises(FutureError) as exc_info:
            await exec_step(code, simple_job, instr_num=2)

        assert exc_info.value.instr_num == 4
        assert exc_info.value.future_result is future_not_ready_2
        assert simple_job.py_env.local_state['z'] == 3
        assert 'w' not in simple_job.py_env.local_state
        # Should have printed 42
        assert len(simple_job.py_env.console) == 1
        assert "42" in simple_job.py_env.console[0]

        # Make second future ready
        future_ready_2 = create_future_ready(99)
        simple_job.py_env.local_state['future_val_2'] = future_ready_2

        # Continue from instruction 4, should complete successfully
        result = await exec_step(code, simple_job, instr_num=4)
        assert result is True
        assert simple_job.py_env.local_state['w'] == 4
        # Should now have both prints
        assert len(simple_job.py_env.console) == 2
        assert "42" in simple_job.py_env.console[0]
        assert "99" in simple_job.py_env.console[1]

    @pytest.mark.asyncio
    async def test_exec_step_future_typehint_not_unwrapped(self, job_factory):
        """Test that function with FutureResult typehint receives unwrapped FutureResult."""
        simple_job = self.create_job(job_factory)

        # Create a ready FutureResult
        future_ready = create_future_ready()

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
        simple_job = self.create_job(job_factory)

        # Create a ready FutureResult
        future_ready = create_future_ready()

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
        simple_job = self.create_job(job_factory)

        # Create multiple FutureResult objects
        future1 = create_future_ready(10)
        future2 = create_future_ready(20)
        future3 = create_future_ready(30)

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
        simple_job = self.create_job(job_factory)

        # Create FutureResult objects
        future_a = create_future_ready(100)
        future_b = create_future_ready(200)

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
        simple_job = self.create_job(job_factory)

        # Create FutureResult objects
        future_reg = create_future_ready(1)
        future_arg1 = create_future_ready(2)
        future_arg2 = create_future_ready(3)
        future_kw = create_future_ready(4)

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

    @pytest.mark.asyncio
    async def test_exec_step_nested_function_calls(self, job_factory):
        """Test that multiple functions can be defined and call each other."""
        simple_job = self.create_job(job_factory)

        # Define helper functions in one code block
        code_define = '''
def helper_a(x):
    return x * 2

def helper_b(x):
    return x + 10

def main_function(x):
    temp = helper_a(x)
    return helper_b(temp)

result = main_function(5)
print(f"Result: {result}")
'''
        result = await exec_step(code_define, simple_job)
        assert result is True
        assert simple_job.py_env.local_state.get('result') == 20  # (5 * 2) + 10
        assert "Result: 20" in simple_job.py_env.console[0]


    @pytest.mark.asyncio
    async def test_exec_step_with_agent_context(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that agent's private context is merged into global execution context."""
        # Define a custom function to be available in agent's context
        def custom_tool():
            return "agent_context_value"

        # Create agent with private context
        agent = Agent(
            role="test",
            _system_prompt="Test agent with context",
            _metadata={"MODEL": "test-model"},
            _tools=[],
            _X__context={"custom_tool": custom_tool, "agent_var": 42}
        )

        # Create job with this agent
        job_def = JobDef(
            agent=agent,
            job_params=None,
            warmup_code=None
        )
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY  # pylint: disable=no-member
        )

        # Execute code that uses agent's context
        code = """
result = custom_tool()
value = agent_var
"""
        await exec_step(code, job)

        # Verify that agent's context was accessible
        assert job.py_env.local_state.get('result') == "agent_context_value"
        assert job.py_env.local_state.get('value') == 42

    @pytest.mark.asyncio
    async def test_exec_step_expression_output(self, job_factory):
        """Test that standalone expressions print their result like in REPL."""
        simple_job = self.create_job(job_factory)

        # Test simple expression
        code = """a = 5
a"""
        await exec_step(code, simple_job)

        # Verify the expression result was printed
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert "5" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_expression_none_printed(self, job_factory):
        """Test that expressions evaluating to None print 'None'."""
        simple_job = self.create_job(job_factory)

        code = """result = None
result"""
        await exec_step(code, simple_job)

        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert simple_job.py_env.console[0] == "None"

    @pytest.mark.asyncio
    async def test_exec_step_expression_string(self, job_factory):
        """Test that string expressions are printed unquoted at top level."""
        simple_job = self.create_job(job_factory)

        code = """text = "hello"
text"""
        await exec_step(code, simple_job)

        # Verify the string was printed without quotes
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert simple_job.py_env.console[0] == 'hello'

    @pytest.mark.asyncio
    async def test_exec_step_expression_uses_llm_repr(self, job_factory):
        """Test that standalone expressions use format_default_llm_repr, not repr."""
        simple_job = self.create_job(job_factory)
        obj = _LLMReprTestObj(name="slot")
        simple_job.py_env.local_state = {'obj': obj}

        code = """obj"""
        await exec_step(code, simple_job)

        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert simple_job.py_env.console[0] == "custom:slot"

    @pytest.mark.asyncio
    async def test_exec_step_expression_list_uses_llm_repr_for_items(self, job_factory):
        """Test that list expressions format nested items via __llm_repr__."""
        simple_job = self.create_job(job_factory)
        simple_job.py_env.local_state = {
            'items': [_LLMReprTestObj(name="anna"), _LLMReprTestObj(name="jan")]
        }

        await exec_step("items", simple_job)

        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert simple_job.py_env.console[0] == "[custom:anna,custom:jan]"

    @pytest.mark.asyncio
    async def test_exec_step_expression_complex(self, job_factory):
        """Test that complex expressions are evaluated and printed."""
        simple_job = self.create_job(job_factory)

        code = """x = 10
y = 20
x + y"""
        await exec_step(code, simple_job)

        # Verify the expression result was printed
        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert "30" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_temporal_expression_prints_value(self, job_factory):
        """Standalone temporal function call prints the unwrapped FutureResult value."""
        simple_job = self.create_job(job_factory)
        simple_job.py_env.local_state = {'get_value': get_value}

        await exec_step('get_value()', simple_job)

        assert simple_job.py_env.console is not None
        assert len(simple_job.py_env.console) == 1
        assert "42" in simple_job.py_env.console[0]

    @pytest.mark.asyncio
    async def test_exec_step_pending_temporal_expression_print_resume_does_not_reexecute(
            self, job_factory):
        """Resume pending implicit expression output without re-running expression code."""
        global _pending_expression_future  # pylint: disable=global-statement
        global _pending_expression_call_count  # pylint: disable=global-statement
        simple_job = self.create_job(job_factory)
        future = FutureResult(deps=MemoWithValueAndReady(77, False), state_num=0)
        future.set_complement_functions(
            complement=_fetch_result_when_deps_ready,
            condition=_check_condition_true,
        )
        _pending_expression_future = future
        _pending_expression_call_count = 0

        simple_job.py_env.local_state = {
            'get_pending_expression_value': get_pending_expression_value
        }

        with pytest.raises(FutureError) as exc_info:
            await exec_step('get_pending_expression_value()', simple_job)

        assert exc_info.value.instr_num == 0
        assert simple_job.py_env.future_result is future
        assert simple_job.py_env.console is None
        assert _pending_expression_call_count == 1

        future.deps.ready = True
        await exec_step(
            'get_pending_expression_value()', simple_job,
            instr_num=exc_info.value.instr_num
        )

        assert simple_job.py_env.future_result is None
        assert simple_job.py_env.console == ["77"]
        assert _pending_expression_call_count == 1

    @pytest.mark.asyncio
    async def test_exec_step_local_vars_in_comprehension(self, job_factory):
        """Test that local variables are accessible in list comprehensions."""
        simple_job = self.create_job(job_factory)

        # Test list comprehension accessing local variables
        code = """keywords = ['a', 'b', 'c']
cur_norm = 'abc'
result = [kw for kw in keywords if kw in cur_norm]"""
        await exec_step(code, simple_job)

        # Verify the comprehension executed successfully
        assert simple_job.py_env.local_state.get('result') == ['a', 'b', 'c']

    @pytest.mark.asyncio
    async def test_exec_step_local_vars_in_generator(self, job_factory):
        """Test that local variables are accessible in generator expressions."""
        simple_job = self.create_job(job_factory)

        # Test generator expression with any() accessing local variables
        code = """keywords = ['x', 'y', 'z']
cur_norm = 'abc'
has_match = any(kw in cur_norm for kw in keywords if kw)"""
        await exec_step(code, simple_job)

        # Verify the generator expression executed successfully
        assert simple_job.py_env.local_state.get('has_match') is False

        # Test with matching keywords
        code2 = """keywords = ['a', 'b', 'c']
cur_norm = 'abc'
has_match = any(kw in cur_norm for kw in keywords if kw)"""
        await exec_step(code2, simple_job)

        assert simple_job.py_env.local_state.get('has_match') is True

    @pytest.mark.asyncio
    async def test_exec_step_nested_comprehension_with_locals(self, job_factory):
        """Test nested comprehensions accessing multiple local variables."""
        simple_job = self.create_job(job_factory)

        code = """text_a = 'hello world'
text_b = 'world peace'
keywords = ['world', 'peace', 'love']
matches = [(kw, text_a, text_b) for kw in keywords if kw in text_a or kw in text_b]"""
        await exec_step(code, simple_job)

        # Verify nested comprehension executed successfully
        matches = simple_job.py_env.local_state.get('matches')
        assert len(matches) == 2
        assert matches[0][0] == 'world'
        assert matches[1][0] == 'peace'


# ---------------------------------------------------------------------------
# Module-level async @tool functions for async-tool-in-exec_step tests.
# Must be at module level: @tool requires **kwargs and db0 disallows decorated
# nested functions.
# ---------------------------------------------------------------------------

@tool
async def _async_exec_tool_return(value: str, **kwargs):  # pylint: disable=unused-argument
    """Async tool that returns a value based on its input."""
    return f"async_result: {value}"


@tool
async def _async_exec_tool_print_and_return(x: int, **kwargs):  # pylint: disable=unused-argument
    """Async tool that prints a message and returns double the input."""
    print(f"async_print: {x}")
    return x * 2


@tool
async def _async_exec_tool_error(**kwargs):  # pylint: disable=unused-argument
    """Async tool that always raises a RuntimeError."""
    raise RuntimeError("async tool error")


class TestExecStepWithAsyncTools:
    """Tests for exec_step executing code that calls async @tool functions.

    When a tool is decorated with @tool and its underlying function is async,
    the decorator wraps it with nest_asyncio so it can be called synchronously
    from within exec()-executed code even though exec_step itself runs inside
    a running event loop.
    """

    def _make_job(self, tools):
        """Create a minimal job whose agent has thesq given tools registered."""
        agent = Agent(
            role="async_tools_exec_test",
            _system_prompt="Test",
            _metadata={"MODEL": "test-model"},
            _tools=tools,
        )
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        return Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,  # pylint: disable=no-member
        )

    @pytest.mark.asyncio
    async def test_async_tool_return_value_assigned_in_local_state(self, db0_fixture):  # pylint: disable=unused-argument
        """Return value of an async @tool is correctly assigned via exec_step."""
        job = self._make_job([_async_exec_tool_return])

        await exec_step('result = _async_exec_tool_return("world")', job)

        assert job.py_env.local_state.get("result") == "async_result: world"

    @pytest.mark.asyncio
    async def test_async_tool_print_captured_in_job_console(self, db0_fixture):  # pylint: disable=unused-argument
        """Print calls inside an async @tool are routed to the job console."""
        job = self._make_job([_async_exec_tool_print_and_return])

        await exec_step("value = _async_exec_tool_print_and_return(21)", job)

        assert job.py_env.local_state.get("value") == 42
        assert job.py_env.console is not None
        assert any("async_print: 21" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_async_tool_exception_written_to_console_and_reraised(self, db0_fixture):  # pylint: disable=unused-argument
        """An exception from an async @tool is logged to the job console and re-raised."""
        job = self._make_job([_async_exec_tool_error])

        with pytest.raises(RuntimeError, match="async tool error"):
            await exec_step("_async_exec_tool_error()", job)

        assert job.py_env.console is not None
        assert any("RuntimeError" in line for line in job.py_env.console)
        assert any("async tool error" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_multiple_async_tool_calls_in_one_exec_step(self, db0_fixture):  # pylint: disable=unused-argument
        """Multiple async @tool calls within a single code block are all executed."""
        job = self._make_job([_async_exec_tool_return])
        code = (
            '_async_exec_tool_return("first")\n'
            'b = _async_exec_tool_return("second")'
        )

        await exec_step(code, job)

        assert job.py_env.local_state.get("b") == "async_result: second"


class TestExecStepStatekCtx:
    """Tests for _STATEK_CTX injection in exec_step execution context."""

    @pytest.mark.asyncio
    async def test_statek_ctx_available_in_exec_step(self, job_factory):
        """_STATEK_CTX is accessible as a variable during code execution."""
        job = job_factory()
        await exec_step('_ctx = _STATEK_CTX', job)
        assert job.py_env.local_state.get('_ctx') is not None

    @pytest.mark.asyncio
    async def test_statek_ctx_agent_is_job_agent(self, job_factory):
        """_STATEK_CTX['agent'] refers to the job's agent."""
        job = job_factory()
        await exec_step('_agent = _STATEK_CTX["agent"]', job)
        assert job.py_env.local_state.get('_agent') is job.job_def.agent

    @pytest.mark.asyncio
    async def test_statek_ctx_not_persisted_after_exec_step(self, job_factory):
        """_STATEK_CTX is not saved to local_state after execution."""
        job = job_factory()
        await exec_step('x = 1', job)
        assert '_STATEK_CTX' not in job.py_env.local_state

    @pytest.mark.asyncio
    async def test_perm_ctx_created_on_demand_persists_after_exec_step(self, job_factory):
        """_PERM_CTX is created only when requested and remains in PyEnv locals."""
        job = job_factory()
        await exec_step('from statek.utils import perm_ctx_set\nperm_ctx_set(key="value")', job)
        assert job.py_env.local_state["_PERM_CTX"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_perm_ctx_not_persisted_until_created(self, job_factory):
        """_PERM_CTX is absent from PyEnv locals until a tool creates it."""
        job = job_factory()
        await exec_step('x = 1', job)
        assert '_PERM_CTX' not in job.py_env.local_state

    @pytest.mark.asyncio
    async def test_exec_step_preserves_job_local_state_added_during_execution(self, job_factory):
        """Direct job local-state writes are not dropped by execution-local writeback."""
        job = job_factory()
        await exec_step(
            '_STATEK_CTX["job"].py_env.local_state["external"] = "keep"\n'
            'x = 1',
            job,
        )
        assert job.py_env.local_state["external"] == "keep"
        assert job.py_env.local_state["x"] == 1


# ---------------------------------------------------------------------------
# Module-level @tool for string-to-name resolution tests.
# Must be module-level: @tool requires **kwargs and db0 disallows decorated
# nested functions.
# ---------------------------------------------------------------------------

@tool
def _resolve_type_tool(what: type, **kwargs):  # pylint: disable=unused-argument
    """Receives a type argument and prints its name.

    Args:
        what: The type argument to inspect.

    Returns:
        None.
    """
    if isinstance(what, type):
        print(f"resolved:{what.__name__}")
    else:
        print(f"got_string:{what}")


class TestExecStepStringNameResolution:
    """Tests for _bind_by_name resolving string class names from agent context."""

    def _make_job(self, tools, context=None):
        """Create a minimal job whose agent has the given tools and context."""
        agent = Agent(
            role="string_resolution_test",
            _system_prompt="Test",
            _metadata={"MODEL": "test-model"},
            _tools=tools,
            _X__context=context or {},
        )
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        return Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,  # pylint: disable=no-member
        )

    @pytest.mark.asyncio
    async def test_tool_resolves_class_name_string_from_agent_context(self, db0_fixture):  # pylint: disable=unused-argument
        """A tool with type annotation resolves a string class name via agent context."""
        job = self._make_job(
            tools=[_resolve_type_tool],
            context={"MemoObject": MemoObject},
        )

        await exec_step('_resolve_type_tool("MemoObject")', job)

        assert job.py_env.console is not None
        assert any("resolved:MemoObject" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_tool_passes_string_when_name_not_in_context(self, db0_fixture):  # pylint: disable=unused-argument
        """When the name is not in context, the string is passed as-is."""
        job = self._make_job(tools=[_resolve_type_tool])  # no context with MemoObject

        await exec_step('_resolve_type_tool("MemoObject")', job)

        assert job.py_env.console is not None
        assert any("got_string:MemoObject" in line for line in job.py_env.console)


# ---------------------------------------------------------------------------
# Module-level @tool for ISO date/datetime parsing tests.
# ---------------------------------------------------------------------------

@tool
def _date_tool(when: date, **kwargs):  # pylint: disable=unused-argument
    """Receives a date argument and prints it.

    Args:
        when: The date value.

    Returns:
        None.
    """
    print(f"date:{when!r}")


@tool
def _datetime_tool(when: datetime, **kwargs):  # pylint: disable=unused-argument
    """Receives a datetime argument and prints it.

    Args:
        when: The datetime value.

    Returns:
        None.
    """
    print(f"datetime:{when!r}")


@tool
def _date_or_datetime_tool(when: date | datetime, **kwargs):  # pylint: disable=unused-argument
    """Receives a date or datetime argument and prints its type and value.

    Args:
        when: The date or datetime value.

    Returns:
        None.
    """
    print(f"type:{type(when).__name__} value:{when!r}")


class TestExecStepIsoDateParsing:
    """Tests for _bind_by_name parsing ISO date/datetime strings."""

    def _make_job(self, tools, context=None):
        agent = Agent(
            role="iso_date_parse_test",
            _system_prompt="Test",
            _metadata={"MODEL": "test-model"},
            _tools=tools,
            _X__context=context or {},
        )
        job_def = JobDef(agent=agent, job_params=None, warmup_code=None)
        return Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.READY,  # pylint: disable=no-member
        )

    @pytest.mark.asyncio
    async def test_plain_date_annotation_parses_iso_string(self, db0_fixture):  # pylint: disable=unused-argument
        """A string ISO date is converted to date when annotation is date."""
        job = self._make_job(tools=[_date_tool])
        await exec_step('_date_tool("2026-03-10")', job)
        assert job.py_env.console is not None
        assert any("date:datetime.date(2026, 3, 10)" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_plain_datetime_annotation_parses_iso_string(self, db0_fixture):  # pylint: disable=unused-argument
        """A string ISO datetime is converted to datetime when annotation is datetime."""
        job = self._make_job(tools=[_datetime_tool])
        await exec_step('_datetime_tool("2026-03-10T12:30:00")', job)
        assert job.py_env.console is not None
        assert any("datetime:datetime.datetime(2026, 3, 10, 12, 30)" in line
                   for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_union_date_or_datetime_date_string_uses_first_match(self, db0_fixture):  # pylint: disable=unused-argument
        """For date | datetime union, a date-only string is parsed as date (first in union)."""
        job = self._make_job(tools=[_date_or_datetime_tool])
        await exec_step('_date_or_datetime_tool("2026-03-10")', job)
        assert job.py_env.console is not None
        assert any("type:date" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_union_datetime_string_falls_back_to_datetime(  # pylint: disable=unused-argument
            self, db0_fixture):
        """For date | datetime union, a full datetime string falls back to datetime."""
        job = self._make_job(tools=[_date_or_datetime_tool])
        await exec_step('_date_or_datetime_tool("2026-03-10T15:00:00")', job)
        assert job.py_env.console is not None
        assert any("type:datetime" in line for line in job.py_env.console)

    @pytest.mark.asyncio
    async def test_non_date_string_still_resolves_via_find_locals(self, db0_fixture):  # pylint: disable=unused-argument
        """Non-ISO strings for date annotation still fall back to find_locals resolution."""
        d = date(2025, 1, 1)
        job = self._make_job(tools=[_date_tool], context={"my_date": d})
        await exec_step('_date_tool("my_date")', job)
        assert job.py_env.console is not None
        assert any("date:datetime.date(2025, 1, 1)" in line for line in job.py_env.console)


# ---------------------------------------------------------------------------
# Mock objects and helpers for weekend duty calendar tests
# ---------------------------------------------------------------------------

@db0.memo
class _MockTimeSlot:
    """Mock time slot with a start_time."""
    def __init__(self, start_time):
        self.start_time = start_time


@db0.memo
class _MockSlot:
    """Mock duty slot wrapping a time slot."""
    def __init__(self, start_time):
        self.time_slot = _MockTimeSlot(start_time)


@db0.memo
class _MockUserCalendar:
    """Mock user calendar that returns pre-configured slots."""
    def __init__(self, slots):
        self._slots = slots

    def get_slots(self, start, end):  # pylint: disable=unused-argument
        return self._slots


# Module-level state used by the mock tool functions below.
# Each test sets these before calling exec_step.
_duty_calendar_return = None
_duty_answers = []


def _get_user_calendar_mock():
    """Mock get_user_calendar that returns _duty_calendar_return."""
    return _duty_calendar_return


def _answer_mock(content):
    """Mock answer that appends to _duty_answers."""
    _duty_answers.append(content)


# The multi-line code block under test
_WEEKEND_DUTY_CODE = '''\
from datetime import date
import calendar as cal

year = 2026
months = [3, 4]
target_weekdays = {5, 6}

range_start = date(year, months[0], 1)
last_month = months[-1]
range_end = date(year, last_month, cal.monthrange(year, last_month)[1])

user_calendar = get_user_calendar()

if user_calendar is None:
    answer("Nie znaleziono kalendarza dyżurów dla użytkownika.")
else:
    user_slots = user_calendar.get_slots(range_start, range_end)
    matching_slots = [s for s in user_slots if s.time_slot.start_time.weekday() in target_weekdays]

    if matching_slots:
        DAYS_PL = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
        dates_str = ", ".join(
            f"{DAYS_PL[s.time_slot.start_time.weekday()]} {s.time_slot.start_time.strftime('%d-%m-%Y')}"
            for s in matching_slots
        )
        answer(f"Tak, masz weekendowe dyżury w następujące dni: {dates_str}.")
    else:
        answer("Nie masz dyżurów weekendowych w marcu ani kwietniu 2026 roku.")
'''


class TestExecStepWeekendDutyCalendar:
    """Tests for exec_step executing weekend duty calendar code."""

    def create_job(self, job_factory, job_params=None):
        """Helper to create job with default params."""
        return job_factory(job_params=job_params or DEFAULT_JOB_PARAMS)

    def _setup(self, job, calendar_return):
        """Configure module-level mocks and inject them into the job."""
        global _duty_calendar_return, _duty_answers  # pylint: disable=global-statement
        _duty_calendar_return = calendar_return
        _duty_answers = []
        job.py_env.local_state = {
            'get_user_calendar': _get_user_calendar_mock,
            'answer': _answer_mock,
        }

    @pytest.mark.asyncio
    async def test_no_calendar_returns_not_found_message(self, job_factory):
        """When get_user_calendar returns None, answer reports no calendar found."""
        job = self.create_job(job_factory)
        self._setup(job, calendar_return=None)

        await exec_step(_WEEKEND_DUTY_CODE, job)

        assert len(_duty_answers) == 1
        assert "Nie znaleziono kalendarza" in _duty_answers[0]

    @pytest.mark.asyncio
    async def test_no_weekend_slots_returns_empty_message(self, job_factory):
        """When calendar has no weekend slots, answer reports no weekend duties."""
        job = self.create_job(job_factory)
        weekday_slots = [
            _MockSlot(date(2026, 3, 2)),   # Monday
            _MockSlot(date(2026, 3, 11)),   # Wednesday
            _MockSlot(date(2026, 4, 6)),    # Monday
        ]
        self._setup(job, calendar_return=_MockUserCalendar(weekday_slots))

        await exec_step(_WEEKEND_DUTY_CODE, job)

        assert len(_duty_answers) == 1
        assert "Nie masz dyżurów weekendowych" in _duty_answers[0]

    @pytest.mark.asyncio
    async def test_weekend_slots_listed_in_answer(self, job_factory):
        """When calendar has weekend slots, answer lists them with Polish day names."""
        job = self.create_job(job_factory)
        weekend_slots = [
            _MockSlot(date(2026, 3, 7)),    # Saturday
            _MockSlot(date(2026, 3, 15)),   # Sunday
            _MockSlot(date(2026, 4, 25)),   # Saturday
        ]
        self._setup(job, calendar_return=_MockUserCalendar(weekend_slots))

        await exec_step(_WEEKEND_DUTY_CODE, job)

        assert len(_duty_answers) == 1
        answer_text = _duty_answers[0]
        assert "Tak, masz weekendowe dyżury" in answer_text
        assert "sobota 07-03-2026" in answer_text
        assert "niedziela 15-03-2026" in answer_text
        assert "sobota 25-04-2026" in answer_text

    @pytest.mark.asyncio
    async def test_mixed_slots_only_weekends_in_answer(self, job_factory):
        """When calendar has both weekday and weekend slots, only weekends are reported."""
        job = self.create_job(job_factory)
        mixed_slots = [
            _MockSlot(date(2026, 3, 2)),    # Monday
            _MockSlot(date(2026, 3, 7)),    # Saturday
            _MockSlot(date(2026, 3, 10)),   # Tuesday
            _MockSlot(date(2026, 4, 5)),    # Sunday
        ]
        self._setup(job, calendar_return=_MockUserCalendar(mixed_slots))

        await exec_step(_WEEKEND_DUTY_CODE, job)

        assert len(_duty_answers) == 1
        answer_text = _duty_answers[0]
        assert "sobota 07-03-2026" in answer_text
        assert "niedziela 05-04-2026" in answer_text
        assert "poniedziałek" not in answer_text
        assert "wtorek" not in answer_text
