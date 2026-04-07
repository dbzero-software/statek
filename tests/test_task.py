# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable
from unittest.mock import Mock, patch
import pytest

from statek.task import (
    copy_locals, delegate_task, start_dialog,
    submit_new_job, submit_new_jobs_batch,
)
from statek.executors.job import Job, JobStatus
from statek.executors.chat_log_item import UserLogItem
from statek.agents.dialog_agent import DialogAgent
from statek.chat_style import ChatStyle
from statek.exceptions import FutureError

def _noop_error_handler(context, error=None):
    """Minimal error handler for tests."""


class TestCopyLocals:
    """Tests for copy_locals function."""

    def test_copy_locals_function_call(self):
        """Test copying variables used in function calls."""
        code = "result = func(x, y, z=w)"
        scope_locals = {'func': print, 'x': 1, 'y': 2, 'w': 3, 'unused': 999}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert dest == {'func': print, 'x': 1, 'y': 2, 'w': 3}


    def test_copy_locals_attribute_access(self):
        """Test that attribute access copies the object, not the attribute name."""
        class MockObj:
            value = 42

        obj = MockObj()
        code = "result = obj.value"
        scope_locals = {'obj': obj, 'value': 100}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert dest == {'obj': obj}

    def test_copy_locals_method_call(self):
        """Test copying variables in method calls."""
        code = "obj.method(arg1, arg2)"
        scope_locals = {'obj': object(), 'arg1': 'a', 'arg2': 'b', 'unused': 'c'}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert 'obj' in dest
        assert 'arg1' in dest
        assert 'arg2' in dest
        assert 'unused' not in dest


    def test_copy_locals_expressions(self):
        """Test copying variables used in complex expressions."""
        code = "result = (x + y) * z / w"
        scope_locals = {'x': 1, 'y': 2, 'z': 3, 'w': 4, 'unused': 5}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert dest == {'x': 1, 'y': 2, 'z': 3, 'w': 4}


    def test_copy_locals_subscript(self):
        """Test copying variables used in subscript operations."""
        code = "result = data[key]"
        scope_locals = {'data': {'foo': 'bar'}, 'key': 'foo', 'unused': 'x'}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert dest == {'data': {'foo': 'bar'}, 'key': 'foo'}


    def test_copy_locals_list_comprehension(self):
        """Test copying variables used in list comprehensions."""
        code = "result = [x for item in items]"
        scope_locals = {'x': 10, 'items': [1, 2, 3], 'unused': 999}
        dest = {}

        copy_locals(code, dest, scope_locals)
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

        copy_locals(code, dest, scope_locals)
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
            copy_locals(code, dest, scope_locals)


    def test_copy_locals_empty_code(self):
        """Test with empty code string."""
        code = ""
        scope_locals = {'x': 1, 'y': 2}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert not dest


    def test_copy_locals_no_references(self):
        """Test code that doesn't reference any variables."""
        code = "x = 42"
        scope_locals = {'y': 1, 'z': 2}
        dest = {}

        copy_locals(code, dest, scope_locals)
        assert not dest # x is being assigned, not referenced


    def test_copy_locals_builtins_not_copied(self):
        """Test that built-in names are not copied from scope_locals."""
        code = "result = len(items) + max(values)"
        scope_locals = {'items': [1, 2, 3], 'values': [5, 10], 'len': 'custom', 'max': 'custom'}
        dest = {}

        copy_locals(code, dest, scope_locals)
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

        copy_locals(code, dest, scope_locals)
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
        result = delegate_task(supervised_agent)

        # Verify result is TaskFutureResult with a Job
        assert result.job.model_family == "OPENAI"
        assert result.job.model == "gpt-4"
        assert result.job.job_def.agent is supervised_agent
        # Prompt comes from agent's __prompt_template
        assert result.job.job_def.prompt() == "Test task"

    def test_delegate_task_with_warmup_code_copies_referenced_locals(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Test delegate_task copies referenced local variables when warmup_code is provided."""
        warmup_code = "result = x + y"

        # Simulate calling delegate_task from a function with local variables
        x = 10
        y = 20
        unused_var = 999

        result = delegate_task(supervised_agent, warmup_code=warmup_code)

        # Verify local_state contains referenced variables
        assert result.job.py_env.local_state["x"] == 10
        assert result.job.py_env.local_state["y"] == 20

        # Verify unused variable was not copied
        assert 'unused_var' not in result.job.py_env.local_state

    def test_delegate_task_with_kwargs(self, db0_fixture, mock_settings):
        """Test delegate_task passes kwargs as job_params to agent.create_job_def."""
        from statek.agents.agent import SupervisedAgent  # pylint: disable=import-outside-toplevel

        # Create agent with prompt template that uses job_params
        agent = SupervisedAgent(
            role="test",
            _system_prompt="Test agent",
            _metadata={'prompt_template': 'Process {data_type} for {user}'},
            _tools=[]
        )

        result = delegate_task(
            agent,
            data_type="orders",
            user="Alice"
        )

        assert result.job.job_def.job_params["data_type"] == "orders"
        assert result.job.job_def.job_params["user"] == "Alice"
        assert result.job.job_def.prompt() == "Process orders for Alice"

    def test_delegate_task_future_result(self, db0_fixture, supervised_agent, mock_settings):
        """Test delegate_task returns proper future result."""
        result = delegate_task(supervised_agent)

        assert result.check_condition() is False

        with pytest.raises(FutureError):
            _ = result.value

        result.job.set_status(JobStatus.DONE)
        assert result.check_condition()

        result.job.py_env.exit_status = "OK"
        assert result.value == ("OK", result.job)


    def test_delegate_task_with_parent_job_copies_error_handlers(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Error handlers from parent_job are inherited by the child job."""
        parent_result = delegate_task(supervised_agent)
        parent_job = parent_result.job
        parent_job.add_error_handler(_noop_error_handler, "ctx")

        child_result = delegate_task(supervised_agent, parent_job=parent_job)
        assert len(child_result.job.error_handlers) == 1
        assert child_result.job.error_handlers[0].error_handler is _noop_error_handler

    def test_delegate_task_without_parent_job_has_no_handlers(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Without parent_job, the child job has no error handlers."""
        result = delegate_task(supervised_agent)
        assert len(result.job.error_handlers) == 0

    def test_delegate_task_dict_shared_vars_populates_local_state(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Dict shared_vars populate the child job's local_state with the
        provided names."""
        result = delegate_task(
            supervised_agent, shared_vars={"alpha": 42, "label": "test"}
        )
        assert result.job.py_env.local_state["alpha"] == 42
        assert result.job.py_env.local_state["label"] == "test"

    def test_delegate_task_list_shared_vars_derives_names_from_type(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """List shared_vars derive variable names from each value's type."""
        result = delegate_task(supervised_agent, shared_vars=[supervised_agent])
        assert (
            result.job.py_env.local_state["supervisedagent"]
            is supervised_agent
        )

    def test_delegate_task_dict_shared_vars_reported_in_job_params(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Dict shared_vars surface their names in the job_def's job_params."""
        result = delegate_task(
            supervised_agent, shared_vars={"alpha": 1, "beta": 2}
        )
        assert result.job.job_def.job_params["shared_vars"] == ["alpha", "beta"]

    def test_delegate_task_shared_vars_no_print_warmup_generated(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """shared_vars should not produce ad-hoc print() warmup_code."""
        result = delegate_task(supervised_agent, shared_vars={"alpha": 1})
        assert result.job.job_def.warmup_code is None

    def test_delegate_task_shared_vars_combines_with_warmup_code(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """shared_vars and warmup_code can be supplied together."""
        x = 5  # picked up by warmup_code via frame inspection
        result = delegate_task(
            supervised_agent,
            warmup_code="result = x",
            shared_vars={"alpha": 99},
        )
        assert result.job.py_env.local_state["x"] == 5
        assert result.job.py_env.local_state["alpha"] == 99

    def test_delegate_task_no_shared_vars_keeps_local_state_empty(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Without shared_vars or warmup_code, local_state stays empty."""
        result = delegate_task(supervised_agent)
        assert not result.job.py_env.local_state


def _make_send_message(body: str, media=None):
    """Mock send_message for DialogAgent tests.

    Args:
        body: The message text.
        media: Optional media attachment.
    """
    return f"sent: {body}"


class TestStartDialog:
    """Tests for start_dialog function."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings functions."""
        with patch('statek.task.get_statek_settings') as mock_statek, \
             patch('statek.task.get_provider_settings') as mock_provider:

            mock_statek_settings = Mock()
            mock_statek_settings.default_llm_api_provider = "OPENAI"
            mock_statek_settings.chat_style = None
            mock_statek.return_value = mock_statek_settings

            mock_provider_settings = Mock()
            mock_provider_settings.default_model = "gpt-4"
            mock_provider.return_value = mock_provider_settings

            yield

    def test_returns_job(self, db0_fixture, mock_settings):
        """start_dialog returns a Job instance."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hello")
        assert isinstance(job, Job)

    def test_job_agent_is_dialog_agent(self, db0_fixture, mock_settings):
        """Job's agent is the supplied DialogAgent."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hi")
        assert job.job_def.agent is agent

    def test_initial_message_in_chat_log(self, db0_fixture, mock_settings):
        """Initial message is pushed into the job via push_user_message."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hello world")
        # First message stored as str in chat_log (MD_DIALOG) or push_log
        has_msg = any(
            (isinstance(item, str) and item == "hello world")
            or (isinstance(item, UserLogItem)
                and item.message == "hello world")
            for item in job.chat_log
        ) or (
            job.py_env.push_log is not None
            and "hello world" in str(job.py_env.push_log)
        )
        assert has_msg

    def test_kwargs_passed_as_job_params(self, db0_fixture, mock_settings):
        """Extra kwargs become job_params on the JobDef."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(
            agent, message="hi", topic="weather")
        assert job.job_def.job_params["topic"] == "weather"

    def test_start_dialog_with_parent_job_copies_error_handlers(
        self, db0_fixture, mock_settings
    ):
        """Error handlers from parent_job are inherited by the dialog job."""
        agent = DialogAgent(send_message=_make_send_message)
        parent_job = start_dialog(agent, message="parent")
        parent_job.add_error_handler(_noop_error_handler, "ctx")

        child_job = start_dialog(agent, message="child", parent_job=parent_job)
        assert len(child_job.error_handlers) == 1
        assert child_job.error_handlers[0].error_handler is _noop_error_handler

    def test_start_dialog_dict_shared_vars_populates_local_state(
        self, db0_fixture, mock_settings
    ):
        """Dict shared_vars populate the dialog job's local_state."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(
            agent, message="hi", shared_vars={"alpha": 42, "label": "test"}
        )
        assert job.py_env.local_state["alpha"] == 42
        assert job.py_env.local_state["label"] == "test"

    def test_start_dialog_list_shared_vars_derives_names_from_type(
        self, db0_fixture, mock_settings
    ):
        """List shared_vars derive variable names from each value's type."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hi", shared_vars=[agent])
        assert job.py_env.local_state["dialogagent"] is agent

    def test_start_dialog_dict_shared_vars_reported_in_job_params(
        self, db0_fixture, mock_settings
    ):
        """Dict shared_vars surface their names in the job_def's job_params."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(
            agent, message="hi", shared_vars={"alpha": 1, "beta": 2}
        )
        assert job.job_def.job_params["shared_vars"] == ["alpha", "beta"]

    def test_start_dialog_shared_vars_no_print_warmup_generated(
        self, db0_fixture, mock_settings
    ):
        """shared_vars should not produce ad-hoc print() warmup_code."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hi", shared_vars={"alpha": 1})
        assert job.job_def.warmup_code is None

    def test_start_dialog_shared_vars_combines_with_warmup_code(
        self, db0_fixture, mock_settings
    ):
        """shared_vars and warmup_code can be supplied together."""
        agent = DialogAgent(send_message=_make_send_message)
        x = 5  # picked up by warmup_code via frame inspection
        job = start_dialog(
            agent,
            message="hi",
            warmup_code="result = x",
            shared_vars={"alpha": 99},
        )
        assert job.py_env.local_state["x"] == 5
        assert job.py_env.local_state["alpha"] == 99

    def test_start_dialog_no_shared_vars_keeps_local_state_empty(
        self, db0_fixture, mock_settings
    ):
        """Without shared_vars or warmup_code, local_state stays empty."""
        agent = DialogAgent(send_message=_make_send_message)
        job = start_dialog(agent, message="hi")
        assert not job.py_env.local_state


class TestDialogAgentCreateJobDefChatStyle:
    """Tests for DialogAgent.create_job_def chat_style parameter."""

    def test_default_chat_style_is_md_dialog(self, db0_fixture):
        """Without explicit chat_style, jobs default to MD_DIALOG."""
        agent = DialogAgent(send_message=_make_send_message)
        job_def = agent.create_job_def()
        assert job_def.chat_style == ChatStyle.MD_DIALOG

    def test_explicit_chat_style_overrides_default(self, db0_fixture):
        """Passing chat_style overrides the MD_DIALOG default."""
        agent = DialogAgent(send_message=_make_send_message)
        job_def = agent.create_job_def(chat_style=ChatStyle.DIRECT)
        assert job_def.chat_style == ChatStyle.DIRECT


class TestResolveSharedVars:
    """Tests for the _resolve_shared_vars helper."""

    def test_none_returns_empty(self):
        from statek.task import _resolve_shared_vars  # pylint: disable=import-outside-toplevel
        assert _resolve_shared_vars(None) == {}

    def test_empty_dict_returns_empty(self):
        from statek.task import _resolve_shared_vars  # pylint: disable=import-outside-toplevel
        assert _resolve_shared_vars({}) == {}

    def test_dict_form(self):
        from statek.task import _resolve_shared_vars  # pylint: disable=import-outside-toplevel
        obj_a = object()
        obj_b = object()
        result = _resolve_shared_vars({"alpha": obj_a, "beta": obj_b})
        assert result == {"alpha": obj_a, "beta": obj_b}

    def test_list_form_derives_name_from_type(self):
        from statek.task import _resolve_shared_vars  # pylint: disable=import-outside-toplevel
        class Invoice:
            pass

        inv = Invoice()
        result = _resolve_shared_vars([inv])
        assert result == {"invoice": inv}


class TestSubmitNewJob:
    """Tests for submit_new_job function."""

    @pytest.fixture
    def mock_settings(self):
        with patch('statek.task.get_statek_settings') as mock_statek, \
             patch('statek.task.get_provider_settings') as mock_provider:
            mock_statek_settings = Mock()
            mock_statek_settings.default_llm_api_provider = "OPENAI"
            mock_statek.return_value = mock_statek_settings
            mock_provider_settings = Mock()
            mock_provider_settings.default_model = "gpt-4"
            mock_provider.return_value = mock_provider_settings
            yield

    def test_no_shared_vars(self, db0_fixture, supervised_agent, mock_settings):
        """Creates a job with empty local_state when no shared_vars given."""
        job = submit_new_job(supervised_agent)
        assert isinstance(job, Job)
        assert not job.py_env.local_state

    def test_dict_shared_vars(self, db0_fixture, supervised_agent, mock_settings):
        """Dict shared_vars populate local_state with correct names."""
        data = {"count": 42, "label": "test"}
        job = submit_new_job(supervised_agent, shared_vars=data)
        assert job.py_env.local_state["count"] == 42
        assert job.py_env.local_state["label"] == "test"

    def test_list_shared_vars(self, db0_fixture, supervised_agent, mock_settings):
        """List shared_vars derive names from object type."""
        job = submit_new_job(supervised_agent, shared_vars=[supervised_agent])
        assert job.py_env.local_state["supervisedagent"] is supervised_agent

    def test_kwargs_forwarded_as_job_params(self, db0_fixture, mock_settings):
        """Extra kwargs become job_params on the JobDef."""
        from statek.agents.agent import SupervisedAgent  # pylint: disable=import-outside-toplevel
        agent = SupervisedAgent(
            role="test",
            _system_prompt="Test",
            _metadata={'prompt_template': 'Handle {kind}'},
            _tools=[]
        )
        job = submit_new_job(agent, kind="invoice")
        assert job.job_def.job_params["kind"] == "invoice"
        assert job.job_def.prompt() == "Handle invoice"


class TestSubmitNewJobsBatch:
    """Tests for submit_new_jobs_batch function."""

    @pytest.fixture
    def mock_settings(self):
        with patch('statek.task.get_statek_settings') as mock_statek, \
             patch('statek.task.get_provider_settings') as mock_provider:
            mock_statek_settings = Mock()
            mock_statek_settings.default_llm_api_provider = "OPENAI"
            mock_statek.return_value = mock_statek_settings
            mock_provider_settings = Mock()
            mock_provider_settings.default_model = "gpt-4"
            mock_provider.return_value = mock_provider_settings
            yield

    def test_creates_multiple_jobs(self, db0_fixture, supervised_agent, mock_settings):
        """Creates one job per shared_vars entry."""
        batch = [
            {"x": 1},
            {"x": 2},
            None,
        ]
        jobs = submit_new_jobs_batch(supervised_agent, batch)
        assert len(jobs) == 3
        assert jobs[0].py_env.local_state["x"] == 1
        assert jobs[1].py_env.local_state["x"] == 2
        assert not jobs[2].py_env.local_state
