# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable
from unittest.mock import Mock, patch
import pytest

from statek.task import copy_locals, delegate_task
from statek.executors.job import JobStatus
from statek.exceptions import FutureError

class TestCopyLocals:
    """Tests for copy_locals function."""

    def test_copy_locals_function_call(self):
        """Test copying variables used in function calls."""
        code = "result = func(x, y, z=w)"
        scope_locals = {'func': print, 'x': 1, 'y': 2, 'w': 3, 'unused': 999}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'func': print, 'x': 1, 'y': 2, 'w': 3}


    def test_copy_locals_attribute_access(self):
        """Test that attribute access copies the object, not the attribute name."""
        class MockObj:
            value = 42

        obj = MockObj()
        code = "result = obj.value"
        scope_locals = {'obj': obj, 'value': 100}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'obj': obj}

    def test_copy_locals_method_call(self):
        """Test copying variables in method calls."""
        code = "obj.method(arg1, arg2)"
        scope_locals = {'obj': object(), 'arg1': 'a', 'arg2': 'b', 'unused': 'c'}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert 'obj' in dest
        assert 'arg1' in dest
        assert 'arg2' in dest
        assert 'unused' not in dest


    def test_copy_locals_expressions(self):
        """Test copying variables used in complex expressions."""
        code = "result = (x + y) * z / w"
        scope_locals = {'x': 1, 'y': 2, 'z': 3, 'w': 4, 'unused': 5}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'x': 1, 'y': 2, 'z': 3, 'w': 4}


    def test_copy_locals_subscript(self):
        """Test copying variables used in subscript operations."""
        code = "result = data[key]"
        scope_locals = {'data': {'foo': 'bar'}, 'key': 'foo', 'unused': 'x'}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'data': {'foo': 'bar'}, 'key': 'foo'}


    def test_copy_locals_list_comprehension(self):
        """Test copying variables used in list comprehensions."""
        code = "result = [x for item in items]"
        scope_locals = {'x': 10, 'items': [1, 2, 3], 'unused': 999}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'x': 10, 'items': [1, 2, 3]}
        # 'item' is local to the comprehension, should not be copied


    def test_copy_locals_multiple_statements(self):
        """Test copying variables across multiple statements."""
        code = (
            "x = a + b\n"
            "y = c * d\n"
            "z = x + y\n"
        )

        scope_locals = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'x': 10, 'y': 20}
        dest = {}

        copy_locals(code, scope_locals, dest)
        # Should include a, b, c, d from first two lines
        # and x, y from third line (referencing existing scope_locals)
        assert 'a' in dest
        assert 'b' in dest
        assert 'c' in dest
        assert 'd' in dest
        assert 'x' in dest
        assert 'y' in dest


    def test_copy_locals_syntax_error(self):
        """Test that syntax errors are handled gracefully."""
        code = "this is not valid python"
        scope_locals = {'x': 1, 'y': 2}
        dest = {}

        # Should not raise an exception
        with pytest.raises(SyntaxError):
            copy_locals(code, scope_locals, dest)


    def test_copy_locals_empty_code(self):
        """Test with empty code string."""
        code = ""
        scope_locals = {'x': 1, 'y': 2}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert not dest


    def test_copy_locals_no_references(self):
        """Test code that doesn't reference any variables."""
        code = "x = 42"
        scope_locals = {'y': 1, 'z': 2}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert not dest # x is being assigned, not referenced


    def test_copy_locals_builtins_not_copied(self):
        """Test that built-in names are not copied from scope_locals."""
        code = "result = len(items) + max(values)"
        scope_locals = {'items': [1, 2, 3], 'values': [5, 10], 'len': 'custom', 'max': 'custom'}
        dest = {}

        copy_locals(code, scope_locals, dest)
        # Should copy items and values, and also len/max if they're in scope_locals
        assert 'items' in dest
        assert 'values' in dest
        # If len and max are in scope_locals, they'll be copied
        assert 'len' in dest
        assert 'max' in dest


    def test_copy_locals_conditional_expression(self):
        """Test copying variables in conditional expressions."""
        code = "result = x if condition else y"
        scope_locals = {'x': 1, 'y': 2, 'condition': True, 'unused': 3}
        dest = {}

        copy_locals(code, scope_locals, dest)
        assert dest == {'x': 1, 'y': 2, 'condition': True}


class TestDelegateTask:
    """Tests for delegate_task function."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings functions."""
        with patch('statek.task.get_statek_settings') as mock_statek, \
             patch('statek.task.get_provider_settings') as mock_provider:

            mock_statek_settings = Mock()
            mock_statek_settings.default_llm_api_provider = "OPENAI"
            mock_statek.return_value = mock_statek_settings

            mock_provider_settings = Mock()
            mock_provider_settings.default_model = "gpt-4"
            mock_provider.return_value = mock_provider_settings

            yield {
                'statek': mock_statek,
                'provider': mock_provider,
                'statek_settings': mock_statek_settings,
                'provider_settings': mock_provider_settings
            }

    def test_delegate_task_creates_job_with_correct_parameters(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Test delegate_task creates a Job with correct parameters."""
        prompt = "Process this data"
        result = delegate_task(supervised_agent, prompt)

        # Verify result is TaskFutureResult with a Job
        assert result.job.model_family == "OPENAI"
        assert result.job.model == "gpt-4"
        assert result.job.job_def.agent is supervised_agent
        assert result.job.job_def.prompt() == prompt

    def test_delegate_task_with_warmup_code_copies_referenced_locals(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Test delegate_task copies referenced local variables when warmup_code is provided."""
        prompt = "Process this data"
        warmup_code = "result = x + y"

        # Simulate calling delegate_task from a function with local variables
        x = 10
        y = 20
        unused_var = 999

        result = delegate_task(supervised_agent, prompt, warmup_code=warmup_code)

        # Verify local_state contains referenced variables
        assert result.job.py_env.local_state["x"] == 10
        assert result.job.py_env.local_state["y"] == 20

        # Verify unused variable was not copied
        assert 'unused_var' not in result.job.py_env.local_state

    def test_delegate_task_with_kwargs(self, db0_fixture, supervised_agent, mock_settings):
        """Test delegate_task passes kwargs to agent.create_job_def."""
        prompt = "Process {data_type} for {user}"

        result = delegate_task(
            supervised_agent,
            prompt,
            data_type="orders",
            user="Alice"
        )

        assert result.job.job_def.context["data_type"] == "orders"
        assert result.job.job_def.context["user"] == "Alice"
        assert result.job.job_def.prompt() == "Process orders for Alice"

    def test_delegate_task_future_result(self, db0_fixture, supervised_agent, mock_settings):
        """Test delegate_task passes kwargs to agent.create_job_def."""
        prompt = "Process this data"

        result = delegate_task(supervised_agent, prompt)

        assert result.check_condition() is False

        with pytest.raises(FutureError):
            _ = result.value

        result.job.status = JobStatus.DONE
        assert result.check_condition()

        result.job.py_env.exit_status = "OK"
        assert result.value == ("OK", result.job)
