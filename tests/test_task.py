# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable
import builtins
from unittest.mock import Mock, patch
import dbzero as db0
import pytest

from statek.task import (
    TaskFutureResult, copy_locals, create_future_task, delegate_task,
    delegate_mute_dialog, delegate_mute_task, start_dialog, submit_new_job,
    submit_new_jobs_batch, create_sub_task, SubTaskHandler, SubTaskState,
    TaskError, complete_sub_task, create_new_job, get_referenced_locals,
)
from statek.executors.chat_log_item import LLM_LogItem
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.chat_log_item import UserLogItem
from statek.agents.agent import WarmupDef
from statek.agents.dialog_agent import DialogAgent, RecurringReminder, RecursiveReminder
from statek.chat_style import ChatStyle
from statek.exceptions import FutureError
from statek.locale import StatekLocale, StatekLangCode, StatekCountryCode
from statek.prompt_config import make_system_prompt
from statek.utils import CodeBlock, _statek_ctx_scope

def _noop_error_handler(context, error=None):
    """Minimal error handler for tests."""


def _send_dialog_body(body, **kwargs):  # pylint: disable=unused-argument
    """Minimal dialog sender for tests."""


def _run_with_current_job(job, func):
    """Run func while job is visible through Statek context."""
    with _statek_ctx_scope({"job": job}):
        return func()


@db0.memo
class CustomSubTaskHandler(SubTaskHandler):
    """Custom subtask handler for tests."""


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


class TestGetReferencedLocals:
    """Tests for get_referenced_locals function."""

    def test_returns_external_names_in_appearance_order(self):
        code = (
            "user = message.sender\n"
            "print(user)\n"
            "send(user, channel)\n"
        )

        assert list(get_referenced_locals(code)) == ["message", "channel"]

    def test_deduplicates_repeated_references(self):
        code = "result = message.sender + message.body"

        assert list(get_referenced_locals(code)) == ["message"]

    def test_ignores_names_bound_by_assignments(self):
        code = (
            "x = a + b\n"
            "y = x + c\n"
            "z = y\n"
        )

        assert list(get_referenced_locals(code)) == ["a", "b", "c"]

    def test_assignment_targets_can_reference_external_objects(self):
        code = (
            "obj.value = value\n"
            "data[key] = obj.value\n"
        )

        assert list(get_referenced_locals(code)) == ["value", "obj", "data", "key"]

    def test_handles_destructuring_assignments(self):
        code = (
            "user, text = message.payload\n"
            "result = format_message(user, text, locale)\n"
        )

        assert list(get_referenced_locals(code)) == ["message", "locale"]

    def test_ignores_builtin_names(self):
        code = "result = len(items) + max(values)"

        assert list(get_referenced_locals(code)) == ["items", "values"]

    def test_ignores_direct_function_callees(self):
        code = "result = transform(message, locale=locale)"

        assert list(get_referenced_locals(code)) == ["message", "locale"]

    def test_method_call_receivers_are_referenced(self):
        code = "result = client.transform(message)"

        assert list(get_referenced_locals(code)) == ["client", "message"]

    def test_handles_comprehension_targets_as_local(self):
        code = "result = [item.value + offset for item in items if item.enabled]"

        assert list(get_referenced_locals(code)) == ["items", "offset"]

    def test_comprehension_targets_do_not_leak_to_following_code(self):
        code = (
            "result = [item for item in items]\n"
            "print(item)\n"
        )

        assert list(get_referenced_locals(code)) == ["items", "item"]

    def test_syntax_error_is_propagated(self):
        with pytest.raises(SyntaxError):
            list(get_referenced_locals("this is not valid python"))


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
        assert result.job.model_family is None
        assert result.job.model == "test-model"
        assert result.job.job_def.model_family is None
        assert result.job.job_def.model == "test-model"
        assert result.job.job_def.agent is supervised_agent

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
            _system_prompt=make_system_prompt("Test agent"),
            _metadata={'MODEL': 'test-model'},
            _tools=[]
        )

        result = delegate_task(
            agent,
            data_type="orders",
            user="Alice"
        )

        assert result.job.job_def.job_params["data_type"] == "orders"
        assert result.job.job_def.job_params["user"] == "Alice"

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

    def test_delegate_task_with_parent_job_stores_parent_job(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Child task jobs retain the actual parent Job object."""
        parent_result = delegate_task(supervised_agent)

        child_result = delegate_task(supervised_agent, parent_job=parent_result.job)

        assert child_result.job.parent_job is parent_result.job

    def test_delegate_task_without_parent_job_has_no_parent_job(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Jobs created without a parent expose parent_job as None."""
        result = delegate_task(supervised_agent)

        assert result.job.parent_job is None

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

    def test_delegate_task_dict_shared_vars_not_reported_in_job_params(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Dict shared_vars stay local to the child job state."""
        result = delegate_task(
            supervised_agent, shared_vars={"alpha": 1, "beta": 2}
        )
        assert result.job.job_def.job_params is None

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

    def test_delegate_task_locale_forwarded_to_job_def(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """locale parameter is forwarded to the JobDef."""

        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        result = delegate_task(supervised_agent, locale=locale)
        assert result.job.job_def.locale is locale

    def test_delegate_task_no_locale_defaults_to_none(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Without locale, JobDef.locale is None."""
        result = delegate_task(supervised_agent)
        assert result.job.job_def.locale is None

    def test_delegate_task_inherits_parent_locale_when_unspecified(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """With parent_job, locale defaults to the parent's locale."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        parent_result = delegate_task(supervised_agent, locale=parent_locale)

        child_result = delegate_task(supervised_agent, parent_job=parent_result.job)

        assert child_result.job.job_def.locale is parent_locale

    def test_delegate_task_explicit_locale_overrides_parent_locale(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """An explicit child locale takes precedence over parent_job.locale."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        child_locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.GB,
        )
        parent_result = delegate_task(supervised_agent, locale=parent_locale)

        child_result = delegate_task(
            supervised_agent, parent_job=parent_result.job, locale=child_locale
        )

        assert child_result.job.job_def.locale is child_locale

    def test_delegate_task_none_locale_inherits_parent_locale(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """locale=None inherits parent_job.locale."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        parent_result = delegate_task(supervised_agent, locale=parent_locale)

        child_result = delegate_task(
            supervised_agent, parent_job=parent_result.job, locale=None
        )

        assert child_result.job.job_def.locale is parent_locale


class TestCreateFutureTask:
    """Tests for create_future_task utility."""

    def test_creates_raw_future_with_shared_vars(
        self, db0_fixture, supervised_agent
    ):
        """create_future_task creates a ready child job and returns its future."""
        result = create_future_task(
            supervised_agent,
            shared_vars={"alpha": 42, "label": "test"},
            parent_job=None,
        )

        assert isinstance(result, TaskFutureResult)
        assert result.job.status == JobStatus.READY
        assert result.job.job_def.agent is supervised_agent
        assert result.job.job_def.job_params is None
        assert result.job.py_env.local_state["alpha"] == 42
        assert result.job.py_env.local_state["label"] == "test"
        assert result.job.parent_job is None

    def test_inherits_parent_locale_and_error_handlers(
        self, db0_fixture, supervised_agent
    ):
        """Parent locale and error handlers are propagated to the child job."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        parent = create_future_task(
            supervised_agent,
            shared_vars={},
            parent_job=None,
        ).job
        parent.job_def.locale = locale
        parent.add_error_handler(_noop_error_handler, "ctx")

        result = create_future_task(
            supervised_agent,
            shared_vars={},
            parent_job=parent,
        )

        assert result.job.job_def.locale is locale
        assert result.job.parent_job is parent
        assert len(result.job.error_handlers) == 1
        assert result.job.error_handlers[0].error_handler is _noop_error_handler

    def test_explicit_locale_overrides_parent_locale(
        self, db0_fixture, supervised_agent
    ):
        """create_future_task can override the inherited parent locale."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        child_locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.GB,
        )
        parent = create_future_task(
            supervised_agent,
            shared_vars={},
            parent_job=None,
            locale=parent_locale,
        ).job

        result = create_future_task(
            supervised_agent,
            shared_vars={},
            parent_job=parent,
            locale=child_locale,
        )

        assert result.job.job_def.locale is child_locale

    def test_warmup_code_copies_referenced_locals(
        self, db0_fixture, supervised_agent
    ):
        """create_future_task supports warmup code with caller locals."""
        x = 10
        unused_var = 999

        result = create_future_task(
            supervised_agent,
            shared_vars={"alpha": 42},
            parent_job=None,
            warmup_code="result = x + alpha",
        )

        assert result.job.job_def.warmup_code == "result = x + alpha"
        assert result.job.py_env.local_state["x"] == 10
        assert result.job.py_env.local_state["alpha"] == 42
        assert "unused_var" not in result.job.py_env.local_state

    def test_kwargs_forwarded_as_job_params(
        self, db0_fixture, supervised_agent
    ):
        """create_future_task forwards extra kwargs to create_job_def."""
        result = create_future_task(
            supervised_agent,
            shared_vars={"alpha": 42},
            parent_job=None,
            data_type="orders",
            user="Alice",
        )

        assert result.job.job_def.job_params["data_type"] == "orders"
        assert result.job.job_def.job_params["user"] == "Alice"


class TestCreateNewJob:
    """Tests for the shared job-construction helper."""

    def test_creates_ready_job_with_shared_vars_and_parent_state(
        self, db0_fixture, supervised_agent
    ):
        """create_new_job matches create_future_task job construction behavior."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        parent = create_future_task(
            supervised_agent,
            shared_vars={},
            parent_job=None,
            locale=parent_locale,
        ).job
        parent.add_error_handler(_noop_error_handler, "ctx")

        job = create_new_job(
            supervised_agent,
            shared_vars={"alpha": 42},
            parent_job=parent,
            topic="orders",
        )

        assert isinstance(job, Job)
        assert job.status == JobStatus.READY
        assert job.job_def.agent is supervised_agent
        assert job.job_def.locale is parent_locale
        assert job.job_def.job_params["topic"] == "orders"
        assert "shared_vars" not in job.job_def.job_params
        assert job.py_env.local_state["alpha"] == 42
        assert job.parent_job is parent
        assert len(job.error_handlers) == 1
        assert job.error_handlers[0].error_handler is _noop_error_handler

    def test_explicit_locale_overrides_parent_locale(self, db0_fixture, supervised_agent):
        """create_new_job preserves current locale override behavior."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        child_locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.GB,
        )
        parent = create_new_job(supervised_agent, locale=parent_locale)

        job = create_new_job(supervised_agent, parent_job=parent, locale=child_locale)

        assert job.job_def.locale is child_locale

    def test_warmup_code_copies_referenced_caller_locals(
        self, db0_fixture, supervised_agent
    ):
        """create_new_job copies warmup-code locals from the caller frame."""
        x = 10
        unused_var = 999

        job = create_new_job(
            supervised_agent,
            shared_vars={"alpha": 42},
            warmup_code="result = x + alpha",
        )

        assert job.job_def.warmup_code == "result = x + alpha"
        assert job.py_env.local_state["x"] == 10
        assert job.py_env.local_state["alpha"] == 42
        assert "unused_var" not in job.py_env.local_state

    def test_code_block_warmup_copies_referenced_caller_locals(
        self, db0_fixture, supervised_agent
    ):
        """create_new_job copies caller locals referenced by dynamic CodeBlock warmup."""
        external_value = 10
        unused_var = 999

        job = create_new_job(
            supervised_agent,
            warmup_code=[
                CodeBlock(code="result = external_value + 1"),
                CodeBlock(code=None),
                CodeBlock(code=""),
            ],
        )

        assert job.py_env.local_state["external_value"] == 10
        assert "unused_var" not in job.py_env.local_state

    def test_reuses_job_def_with_agent_code_block_warmup(
        self, db0_fixture, supervised_agent
    ):
        """Agent warmup_def entries that are CodeBlocks do not crash reuse lookup."""
        hidden_block = CodeBlock(code="init_shared_context(user)", metadata={"hidden": True})
        supervised_agent.warmup_def = WarmupDef(warmup_code=["x = 1", hidden_block])

        first = create_new_job(supervised_agent)
        second = create_new_job(supervised_agent)

        assert first.job_def is second.job_def
        assert first.job_def.warmup_code == ["x = 1", hidden_block]

    def test_reuses_matching_job_def(self, db0_fixture, supervised_agent):
        """Repeated identical jobs reuse one JobDef."""
        first = create_new_job(
            supervised_agent,
            shared_vars={"alpha": 1},
            topic="orders",
        )
        second = create_new_job(
            supervised_agent,
            shared_vars={"alpha": 2},
            topic="orders",
        )

        assert second.job_def is first.job_def
        assert len(db0.find(JobDef, db0.as_tag(supervised_agent))) == 1
        assert first.py_env.local_state["alpha"] == 1
        assert second.py_env.local_state["alpha"] == 2

    def test_creates_distinct_job_def_for_different_static_inputs(
        self, db0_fixture, supervised_agent
    ):
        """Warmup, params, and locale remain part of JobDef identity."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        base = create_new_job(supervised_agent, topic="orders")
        different_warmup = create_new_job(
            supervised_agent,
            warmup_code="x = 1",
            topic="orders",
        )
        different_params = create_new_job(supervised_agent, topic="staff")
        different_locale = create_new_job(supervised_agent, topic="orders", locale=locale)

        assert different_warmup.job_def is not base.job_def
        assert different_params.job_def is not base.job_def
        assert different_locale.job_def is not base.job_def
        assert len(db0.find(JobDef, db0.as_tag(supervised_agent))) == 4

    def test_reuses_dialog_job_def_with_same_chat_style(self, db0_fixture):
        """Dialog job reuse includes the resolved chat style."""
        agent = DialogAgent(
            send_message=_send_dialog_body,
            _metadata={"MODEL": "test-model"},
        )

        first = create_new_job(agent, chat_style=ChatStyle.DIRECT)
        second = create_new_job(agent, chat_style=ChatStyle.DIRECT)
        third = create_new_job(agent, chat_style=ChatStyle.MD_DIALOG)

        assert second.job_def is first.job_def
        assert third.job_def is not first.job_def
        assert len(db0.find(JobDef, db0.as_tag(agent))) == 2


class TestSubTaskHandler:
    """Tests for sub-task handler primitives."""

    def test_state_maps_ready_job_to_waiting(self, job_factory):
        """A handler for a READY child job is waiting."""
        child = job_factory()

        handler = SubTaskHandler(job=child, id="child-1")

        assert handler.state == SubTaskState.WAITING

    def test_has_no_status_property(self, job_factory):
        """The public API exposes state, not status."""
        handler = SubTaskHandler(job=job_factory())

        assert not hasattr(type(handler), "status")
        with pytest.raises(AttributeError):
            getattr(handler, "status")

    def test_state_maps_active_job_states_to_started(self, job_factory):
        """Active child job states are exposed as STARTED."""
        child = job_factory()
        handler = SubTaskHandler(job=child)

        for status in (
            JobStatus.WARMING_UP,
            JobStatus.STARTED,
            JobStatus.SUSPENDED,
            JobStatus.DONE,
        ):
            child.set_status(status)
            assert handler.state == SubTaskState.STARTED

    def test_state_prefers_explicit_success_completion(self, job_factory):
        """Explicit handler completion overrides child job status."""
        child = job_factory()
        handler = SubTaskHandler(job=child)
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access

        assert handler.state == SubTaskState.COMPLETED

    def test_state_prefers_explicit_error_completion(self, job_factory):
        """Explicit handler errors override child job status."""
        child = job_factory()
        handler = SubTaskHandler(job=child)
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access
        handler._SubTaskHandler__error = TaskError("failed")  # pylint: disable=protected-access

        assert handler.state == SubTaskState.ERROR

    def test_str_returns_completed_result(self, job_factory):
        """String conversion exposes the successful completion result."""
        child = job_factory()
        handler = SubTaskHandler(job=child)
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access
        handler._SubTaskHandler__result = "done"  # pylint: disable=protected-access

        assert str(handler) == "done"

    def test_str_returns_empty_string_for_success_without_result(self, job_factory):
        """A successful no-result subtask stringifies to an empty string."""
        child = job_factory()
        handler = SubTaskHandler(job=child)
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access

        assert str(handler) == ""

    def test_str_raises_for_pending_handler(self, job_factory):
        """String conversion raises until the subtask is completed."""
        child = job_factory()
        handler = SubTaskHandler(job=child, id="child-1")

        with pytest.raises(RuntimeError, match="child-1"):
            str(handler)

    def test_str_raises_for_error_handler(self, job_factory):
        """String conversion raises with the task error message on failure."""
        child = job_factory()
        handler = SubTaskHandler(job=child)
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access
        handler._SubTaskHandler__error = TaskError("failed")  # pylint: disable=protected-access

        with pytest.raises(RuntimeError, match="failed"):
            str(handler)

    def test_get_log_message_reports_success(self, job_factory):
        """Successful completions produce an LLM-facing notification."""
        handler = SubTaskHandler(job=job_factory(), id="child-1")
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access

        assert handler.get_log_message() == (
            "[Notification] sub-task id=child-1 completed successfully."
        )

    def test_get_log_message_reports_success_without_id(self, job_factory):
        """Successful completions omit the id fragment when no id exists."""
        handler = SubTaskHandler(job=job_factory())
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access

        assert handler.get_log_message() == (
            "[Notification] sub-task completed successfully."
        )

    def test_get_log_message_reports_error(self, job_factory):
        """Errored completions produce an LLM-facing error notification."""
        handler = SubTaskHandler(job=job_factory(), id="child-1")
        handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access
        handler._SubTaskHandler__error = TaskError("failed")  # pylint: disable=protected-access

        assert handler.get_log_message() == "[Error] sub-task id=child-1 failed with failed"

    def test_complete_success_stores_result_and_notifies_parent(self, job_factory):
        """Successful completion stores the result and pushes to the parent job."""
        parent = job_factory()
        child = job_factory()
        child.parent_job = parent
        handler = SubTaskHandler(job=child, id="child-1")

        handler.complete(result="done")

        assert handler.is_completed is True
        assert handler.state == SubTaskState.COMPLETED
        assert handler.result == "done"
        assert handler.error is None
        assert parent.find_sub_task_handler(id="child-1") is handler

    def test_complete_success_without_result(self, job_factory):
        """A no-result completion is a successful completed subtask."""
        handler = SubTaskHandler(job=job_factory(), id="child-1")

        handler.complete()

        assert handler.is_completed is True
        assert handler.state == SubTaskState.COMPLETED
        assert handler.result is None
        assert str(handler) == ""

    def test_complete_error_stores_task_error_and_notifies_parent(self, job_factory):
        """Errored completion stores TaskError and still notifies the parent."""
        parent = job_factory()
        child = job_factory()
        child.parent_job = parent
        handler = SubTaskHandler(job=child, id="child-1")

        handler.complete(error="failed")

        assert handler.is_completed is True
        assert handler.state == SubTaskState.ERROR
        assert handler.error.err_message == "failed"
        assert handler.result is None
        assert parent.find_sub_task_handler(id="child-1") is handler

    def test_complete_rejects_double_completion(self, job_factory):
        """Handlers cannot be completed twice."""
        handler = SubTaskHandler(job=job_factory(), id="child-1")
        handler.complete(result="done")

        with pytest.raises(RuntimeError, match="already completed"):
            handler.complete(result="again")

        assert handler.result == "done"

    def test_complete_rejects_mixed_result_and_error(self, job_factory):
        """Completion cannot mix a result with an error outcome."""
        handler = SubTaskHandler(job=job_factory(), id="child-1")

        with pytest.raises(ValueError, match="result and error"):
            handler.complete(result="done", error="failed")

        assert handler.is_completed is False


class TestCompleteSubTask:
    """Tests for the child-job completion convenience helper."""

    def test_forwards_to_current_job_subtask_handler(self, job_factory):
        """complete_sub_task locates sub_task_handler in current job locals."""
        parent = job_factory()
        child = job_factory()
        child.parent_job = parent
        handler = create_sub_task(child, id="child-1")

        _run_with_current_job(child, lambda: complete_sub_task(result="done"))

        assert handler.is_completed is True
        assert handler.result == "done"
        assert parent.find_sub_task_handler(id="child-1") is handler

    def test_requires_current_job(self, job_factory):
        """complete_sub_task requires Statek job execution context."""
        create_sub_task(job_factory(), id="child-1")

        with pytest.raises(RuntimeError, match="current job"):
            complete_sub_task(result="done")

    def test_requires_subtask_handler_local(self, job_factory):
        """complete_sub_task fails clearly when no handler is in locals."""
        child = job_factory()

        with pytest.raises(RuntimeError, match="sub_task_handler"):
            _run_with_current_job(child, lambda: complete_sub_task(result="done"))


class TestCreateSubTask:
    """Tests for create_sub_task utility."""

    def test_wraps_job_and_injects_handler_local(self, job_factory):
        """create_sub_task wraps an existing job and injects sub_task_handler."""
        child = job_factory()

        handler = create_sub_task(child, id="child-1")

        assert isinstance(handler, SubTaskHandler)
        assert handler.job is child
        assert handler.id == "child-1"
        assert child.py_env.local_state["sub_task_handler"] is handler

    def test_supports_custom_handler_type(self, job_factory):
        """create_sub_task can construct custom handler subclasses."""
        child = job_factory()

        handler = create_sub_task(child, handler_type=CustomSubTaskHandler, id="custom")

        assert isinstance(handler, CustomSubTaskHandler)
        assert handler.id == "custom"
        assert child.py_env.local_state["sub_task_handler"] is handler


class TestDelegateMuteTask:
    """Tests for delegate_mute_task and get_mute_job_result."""

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
            yield

    def test_result_returns_chat_responses_on_success(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """On successful completion, value is the agent's chat responses joined by newlines."""
        result = delegate_mute_task(supervised_agent)
        result.job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp="Hello"))
        result.job.chat_log.append(LLM_LogItem(console_pos=1, llm_resp="World"))
        result.job.set_status(JobStatus.DONE)
        assert result.value == "Hello\nWorld"

    def test_result_returns_error_on_failure(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """On failure, value is the exit_status if set, otherwise the error message."""
        from statek.executors.job import JobDefError  # pylint: disable=import-outside-toplevel
        result = delegate_mute_task(supervised_agent)
        result.job.set_status(JobStatus.DONE)
        try:
            raise RuntimeError("something broke")
        except RuntimeError as exc:
            result.job.error = JobDefError(exc, collect_traceback=False)
        # Without exit_status, falls back to error_message
        assert result.value == "something broke"
        # exit_status takes precedence when set
        result.job.py_env.exit_status = "aborted: quota exceeded"
        assert result.value == "aborted: quota exceeded"

    def test_delegate_mute_task_inherits_parent_locale_when_unspecified(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Mute task locale defaults to parent_job.locale."""
        parent_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        parent_result = delegate_mute_task(supervised_agent, locale=parent_locale)

        child_result = delegate_mute_task(
            supervised_agent, parent_job=parent_result.job
        )

        assert child_result.job.job_def.locale is parent_locale
        assert child_result.job.parent_job is parent_result.job


class TestDelegateMuteDialog:
    """Tests for delegate_mute_dialog."""

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

    def test_creates_dialog_job_with_initial_message(self, db0_fixture, mock_settings):
        """delegate_mute_dialog creates a dialog job and pushes the initial user message."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        result = delegate_mute_dialog(agent, message="hello")

        assert isinstance(result.job, Job)
        assert result.job.job_def.agent is agent
        assert result.job.chat_log[0] == "hello"
        assert result.job.status == JobStatus.READY

    def test_non_string_initial_message_uses_message_adapter(
        self, db0_fixture, mock_settings
    ):
        """delegate_mute_dialog forwards non-string messages to start_dialog."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        agent.context["message_adapter"] = lambda msg: f"adapted-{msg.value}"

        result = delegate_mute_dialog(agent, message=MessageForAdapter("object"))

        assert result.job.chat_log[0] == "adapted-object"

    def test_result_returns_chat_responses_on_success(self, db0_fixture, mock_settings):
        """On completion, delegate_mute_dialog resolves to user-facing chat responses."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        result = delegate_mute_dialog(agent, message="hello")

        result.job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp="# Answer"))
        result.job.chat_log.append(LLM_LogItem(console_pos=1, llm_resp="# Follow-up"))
        result.job.set_status(JobStatus.DONE)

        assert result.value == "# Answer\n# Follow-up"

    def test_forwards_parent_shared_vars_locale_and_kwargs(
        self, db0_fixture, mock_settings
    ):
        """delegate_mute_dialog forwards dialog job construction arguments."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.GB,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        parent = start_dialog(agent, message="parent")
        parent.add_error_handler(_noop_error_handler, "ctx")

        result = delegate_mute_dialog(
            agent,
            message="child",
            parent_job=parent,
            shared_vars={"alpha": 42},
            locale=locale,
            topic="weather",
        )

        assert result.job.error_handlers[0].error_handler is _noop_error_handler
        assert result.job.parent_job is parent
        assert result.job.py_env.local_state["alpha"] == 42
        assert result.job.job_def.locale is locale
        assert result.job.job_def.job_params["topic"] == "weather"
        assert "shared_vars" not in result.job.job_def.job_params

    def test_delegate_mute_dialog_inherits_parent_locale_when_unspecified(
        self, db0_fixture, mock_settings
    ):
        """Mute dialog locale defaults to parent_job.locale."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        parent = start_dialog(agent, message="parent", locale=locale)

        result = delegate_mute_dialog(agent, message="child", parent_job=parent)

        assert result.job.job_def.locale is locale


_recorded_send_calls = []


def _recording_send_message(body: str, media=None):
    """Module-level send_message that records invocations."""
    _recorded_send_calls.append((body, media))
    return "ok"


def _make_send_message(body: str, media=None):
    """Mock send_message for DialogAgent tests.

    Args:
        body: The message text.
        media: Optional media attachment.
    """
    return f"sent: {body}"


class MessageForAdapter:
    """Message object used by dialog startup adapter tests."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"fallback-{self.value}"


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
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hello")
        assert isinstance(job, Job)

    def test_job_agent_is_dialog_agent(self, db0_fixture, mock_settings):
        """Job's agent is the supplied DialogAgent."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hi")
        assert job.job_def.agent is agent

    def test_initial_message_in_chat_log(self, db0_fixture, mock_settings):
        """Initial message is pushed into the job via push_user_message."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
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

    def test_non_string_initial_message_uses_message_adapter(
        self, db0_fixture, mock_settings
    ):
        """Non-string initial messages are adapted through push_user_message."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        agent.context["message_adapter"] = lambda msg: f"adapted-{msg.value}"

        job = start_dialog(agent, message=MessageForAdapter("object"))

        assert job.chat_log[0] == "adapted-object"

    def test_non_string_initial_message_falls_back_to_str(
        self, db0_fixture, mock_settings
    ):
        """Non-string initial messages fall back to str(message)."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        job = start_dialog(agent, message=MessageForAdapter("object"))

        assert job.chat_log[0] == "fallback-object"

    def test_kwargs_passed_as_job_params(self, db0_fixture, mock_settings):
        """Extra kwargs become job_params on the JobDef."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(
            agent, message="hi", topic="weather")
        assert job.job_def.job_params["topic"] == "weather"

    def test_start_dialog_with_parent_job_copies_error_handlers(
        self, db0_fixture, mock_settings
    ):
        """Error handlers from parent_job are inherited by the dialog job."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        parent_job = start_dialog(agent, message="parent")
        parent_job.add_error_handler(_noop_error_handler, "ctx")

        child_job = start_dialog(agent, message="child", parent_job=parent_job)
        assert len(child_job.error_handlers) == 1
        assert child_job.error_handlers[0].error_handler is _noop_error_handler
        assert child_job.parent_job is parent_job

    def test_start_dialog_dict_shared_vars_populates_local_state(
        self, db0_fixture, mock_settings
    ):
        """Dict shared_vars populate the dialog job's local_state."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(
            agent, message="hi", shared_vars={"alpha": 42, "label": "test"}
        )
        assert job.py_env.local_state["alpha"] == 42
        assert job.py_env.local_state["label"] == "test"

    def test_start_dialog_dict_shared_vars_not_reported_in_job_params(
        self, db0_fixture, mock_settings
    ):
        """Dict shared_vars stay local to the dialog job state."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(
            agent, message="hi", shared_vars={"alpha": 1, "beta": 2}
        )
        assert job.job_def.job_params is None

    def test_start_dialog_shared_vars_no_print_warmup_generated(
        self, db0_fixture, mock_settings
    ):
        """shared_vars should not produce ad-hoc print() warmup_code."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hi", shared_vars={"alpha": 1})
        assert job.job_def.warmup_code is None

    def test_start_dialog_shared_vars_combines_with_warmup_code(
        self, db0_fixture, mock_settings
    ):
        """shared_vars and warmup_code can be supplied together."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
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
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hi")
        assert not job.py_env.local_state

    def test_start_dialog_locale_forwarded_to_job_def(
        self, db0_fixture, mock_settings
    ):
        """locale parameter is forwarded to the JobDef."""

        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.GB,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hello", locale=locale)
        assert job.job_def.locale is locale

    def test_start_dialog_no_locale_defaults_to_none(
        self, db0_fixture, mock_settings
    ):
        """Without locale, JobDef.locale is None."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job = start_dialog(agent, message="hi")
        assert job.job_def.locale is None

    def test_start_dialog_inherits_parent_locale_when_unspecified(
        self, db0_fixture, mock_settings
    ):
        """With parent_job, dialog locale defaults to the parent's locale."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        parent_job = start_dialog(agent, message="parent", locale=locale)

        child_job = start_dialog(agent, message="child", parent_job=parent_job)

        assert child_job.job_def.locale is locale

    def test_start_dialog_none_locale_inherits_parent_locale(
        self, db0_fixture, mock_settings
    ):
        """locale=None inherits parent_job.locale."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        parent_job = start_dialog(agent, message="parent", locale=locale)

        child_job = start_dialog(
            agent, message="child", parent_job=parent_job, locale=None
        )

        assert child_job.job_def.locale is locale


class TestDialogAgentAddAnswerTool:
    """Tests for DialogAgent's add_answer_tool parameter."""

    def test_answer_tool_registered_by_default(self, db0_fixture):
        """By default, an 'answer' tool is registered on the agent."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        tool_names = [t.__name__ for t in agent.all_tools]
        assert "answer" in tool_names

    def test_answer_tool_absent_when_disabled(self, db0_fixture):
        """When add_answer_tool=False, no 'answer' tool is registered."""
        agent = DialogAgent(
            send_message=_make_send_message,
            add_answer_tool=False,
            _metadata={"MODEL": "test-model"},
        )
        tool_names = [t.__name__ for t in agent.all_tools]
        assert "answer" not in tool_names

    def test_answer_tool_forwards_body_and_media_and_exits(self, db0_fixture):
        """answer forwards body/media to send_message then signals exit."""
        _recorded_send_calls.clear()
        agent = DialogAgent(
            send_message=_recording_send_message, _metadata={"MODEL": "test-model"}
        )
        answer = next(t for t in agent.all_tools if t.__name__ == "answer")

        exit_calls = []
        original_exit = builtins.exit
        builtins.exit = lambda status=None: exit_calls.append(status)
        try:
            answer(body="final", media="img.png")
        finally:
            builtins.exit = original_exit

        assert _recorded_send_calls == [("final", "img.png")]
        assert exit_calls == ["Success"]


class TestDialogAgentReminder:
    """Tests for DialogAgent reminder configuration."""

    def test_reminder_defaults_to_none(self, db0_fixture):
        """DialogAgent has no reminder until one is configured."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        assert agent.reminder is None

    def test_set_new_reminder_creates_recurring_reminder_by_default(self, db0_fixture):
        """set_new_reminder stores a recurring reminder by default."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        reminder = agent.set_new_reminder("Use report_outcome.")

        assert isinstance(reminder, RecurringReminder)
        assert reminder.text == "Use report_outcome."
        assert agent.reminder is reminder

    def test_set_new_reminder_passes_recurring_reminder_kwargs(self, db0_fixture):
        """set_new_reminder forwards implementation-specific reminder properties."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        reminder = agent.set_new_reminder("Use report_outcome.", min_dialog_len=3)

        assert isinstance(reminder, RecurringReminder)
        assert reminder.min_dialog_len == 3
        assert agent.reminder is reminder

    def test_set_new_reminder_accepts_recursive_type_alias(self, db0_fixture):
        """set_new_reminder accepts the old RECURSIVE type as an alias."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        reminder = agent.set_new_reminder("Use report_outcome.", type="RECURSIVE")

        assert isinstance(reminder, RecurringReminder)
        assert agent.reminder is reminder

    def test_recursive_reminder_alias_is_available(self, db0_fixture):
        """RecursiveReminder remains as a backward-compatible class alias."""
        reminder = RecursiveReminder(text="Use report_outcome.")

        assert isinstance(reminder, RecurringReminder)

    def test_set_new_reminder_rejects_base_reminder_type(self, db0_fixture):
        """set_new_reminder does not instantiate the base reminder type."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        with pytest.raises(ValueError, match="Unsupported reminder type"):
            agent.set_new_reminder("Follow up.", type="REMINDER")

    def test_set_new_reminder_rejects_unknown_type(self, db0_fixture):
        """Unknown reminder types fail explicitly."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        with pytest.raises(ValueError, match="Unsupported reminder type"):
            agent.set_new_reminder("Follow up.", type="UNKNOWN")

    def test_set_reminder_accepts_same_prefix_reminder(self, db0_fixture):
        """set_reminder stores a preconfigured reminder on the same prefix."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        reminder = RecurringReminder(text="Use report_outcome.", min_dialog_len=3)

        stored = agent.set_reminder(reminder)

        assert stored is reminder
        assert agent.reminder is reminder

    def test_set_reminder_rejects_non_reminder(self, db0_fixture):
        """set_reminder only accepts Reminder-derived instances."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})

        with pytest.raises(TypeError, match="reminder must be a Reminder"):
            agent.set_reminder("Use report_outcome.")

    def test_set_reminder_rejects_different_prefix_reminder(self, db0_fixture):
        """set_reminder rejects reminders stored outside the agent prefix."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        db0.open("other_prefix", "rw")
        reminder = RecurringReminder(text="Use report_outcome.")

        with pytest.raises(ValueError, match="same db0 prefix"):
            agent.set_reminder(reminder)


class TestDialogAgentCreateJobDefChatStyle:
    """Tests for DialogAgent.create_job_def chat_style parameter."""

    def test_default_chat_style_is_md_dialog(self, db0_fixture):
        """Without explicit chat_style, jobs default to MD_DIALOG."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job_def = agent.create_job_def()
        assert job_def.chat_style == ChatStyle.MD_DIALOG

    def test_explicit_chat_style_overrides_default(self, db0_fixture):
        """Passing chat_style overrides the MD_DIALOG default."""
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job_def = agent.create_job_def(chat_style=ChatStyle.DIRECT)
        assert job_def.chat_style == ChatStyle.DIRECT

    def test_locale_forwarded_through_dialog_agent(self, db0_fixture):
        """DialogAgent.create_job_def forwards locale to the JobDef."""

        locale = StatekLocale(
            lang_code=StatekLangCode.FR,
            country_code=StatekCountryCode.FR,
        )
        agent = DialogAgent(send_message=_make_send_message, _metadata={"MODEL": "test-model"})
        job_def = agent.create_job_def(locale=locale)
        assert job_def.locale is locale


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
        assert job.job_def.job_params is None

    def test_kwargs_forwarded_as_job_params(self, db0_fixture, mock_settings):
        """Extra kwargs become job_params on the JobDef."""
        from statek.agents.agent import SupervisedAgent  # pylint: disable=import-outside-toplevel
        agent = SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("Test"),
            _metadata={'MODEL': 'test-model'},
            _tools=[]
        )
        job = submit_new_job(agent, kind="invoice")
        assert job.job_def.job_params["kind"] == "invoice"

    def test_locale_forwarded_to_job_def(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """locale parameter is forwarded to the JobDef."""
        locale = StatekLocale(
            lang_code=StatekLangCode.FR,
            country_code=StatekCountryCode.FR,
        )

        job = submit_new_job(supervised_agent, locale=locale)

        assert job.job_def.locale is locale

    def test_no_locale_defaults_to_none(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Without locale, JobDef.locale is None."""
        job = submit_new_job(supervised_agent)

        assert job.job_def.locale is None


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

    def test_locale_forwarded_to_each_job_def(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """locale parameter is forwarded to every created JobDef."""
        locale = StatekLocale(
            lang_code=StatekLangCode.DE,
            country_code=StatekCountryCode.DE,
        )

        jobs = submit_new_jobs_batch(supervised_agent, [{"x": 1}, None], locale=locale)

        assert len(jobs) == 2
        assert all(job.job_def.locale is locale for job in jobs)

    def test_batch_no_locale_defaults_to_none(
        self, db0_fixture, supervised_agent, mock_settings
    ):
        """Without locale, every batch JobDef.locale is None."""
        jobs = submit_new_jobs_batch(supervised_agent, [{"x": 1}, None])

        assert all(job.job_def.locale is None for job in jobs)
