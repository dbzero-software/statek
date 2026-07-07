"""Tests for Job class."""

# pylint: disable=no-member

import json
import types
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock
import dbzero as db0
import pytest
from tests.conftest import create_chat_log_item, set_warmup_positions
from statek.executors.job import (
    DialogItem,
    Job,
    JobDef,
    JobDefError,
    JobStatus,
    TaskDifficulty,
    parse_model_metadata,
)
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats, OpenRouter_API
from statek.model_pricing import set_model_pricing
from statek.model_name import ModelName, parse_model_name
from statek.chat_history import ChatRole, ContentSource, format_chat_history_item
from statek.agents.dialog_agent import RecurringReminder
from statek.executors.chat_log_item import (
    LLM_LogItem,
    PostProcessedItem,
    ReminderLogItem,
    SubTaskLogItem,
    UserLogItem,
    WarmupLogItem,
)
from statek.executors.post_processor import PostProcessor
from statek.settings import ChatStyle, LLM_API_Settings
from statek.locale import StatekLocale, StatekLangCode, StatekCountryCode
from statek.prompt_config import make_system_prompt, parse_system_prompt
from statek.utils import CodeBlock, CallSpec, _statek_ctx_scope
from statek.system import find_sub_task_handler
from statek.task import SubTaskHandler, TaskError


def _run_with_current_job(job, func):
    """Run func while job is visible through Statek context."""
    with _statek_ctx_scope({"job": job}):
        return func()


def _completed_subtask_handler(job, subtask_id=None, result=None, error=None):
    """Create a completed handler fixture before the public complete API exists."""
    handler = SubTaskHandler(job=job, id=subtask_id)
    handler._SubTaskHandler__is_completed = True  # pylint: disable=protected-access
    if error is not None:
        handler._SubTaskHandler__error = TaskError(error)  # pylint: disable=protected-access
    else:
        handler._SubTaskHandler__result = result  # pylint: disable=protected-access
    return handler


@db0.memo
class JobExtRefThing:
    """Memo object used by Job external-reference tests."""

    def __init__(self, value):
        self.value = value


class MessageForAdapter:
    """Message object used by push_user_message adapter tests."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"fallback-{self.value}"


@db0.memo
class CountActionsPostProcessor(PostProcessor):
    """Post-processor test double used for count_llm_actions exclusions."""

    def process(self, llm_step: LLM_StepData, job: Job) -> LLM_StepData:
        """Return the input step unchanged."""
        del job
        return llm_step


class TestJobDefError:
    """Test cases for JobDefError class."""

    def _make_raised_error(self, msg="something went wrong"):
        """Return an exception that has been raised (has a traceback)."""
        try:
            raise ValueError(msg)
        except ValueError as exc:
            return exc

    def test_error_message_is_set(self, db0_fixture):  # pylint: disable=unused-argument
        """error_message is set to the string representation of the exception."""
        error = self._make_raised_error("boom")
        jde = JobDefError(error)
        assert jde.error_message == "boom"

    def test_traceback_collected_by_default(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is a non-empty sequence of strings when collect_traceback=True."""
        error = self._make_raised_error("oops")
        jde = JobDefError(error)
        assert jde.traceback is not None
        assert len(jde.traceback) > 0
        assert all(isinstance(s, str) for s in jde.traceback)

    def test_traceback_not_collected_when_disabled(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is None when collect_traceback=False."""
        error = self._make_raised_error("oops")
        jde = JobDefError(error, collect_traceback=False)
        assert jde.traceback is None

    def test_traceback_none_when_no_traceback_on_exception(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is None when exception was never raised (no __traceback__)."""
        error = ValueError("never raised")
        jde = JobDefError(error)
        assert jde.traceback is None


class TestJobWithError:
    """Test cases for Job.error field."""

    def test_job_error_is_none_by_default(self, job_factory):
        """Job.error is None when created without error."""
        job = job_factory()
        assert job.error is None

    def test_job_error_can_be_set(self, job_factory, db0_fixture):  # pylint: disable=unused-argument
        """Job.error can be set to a JobDefError instance."""
        try:
            raise RuntimeError("job failed")
        except RuntimeError as exc:
            err = JobDefError(exc)
            job = job_factory()
            job.error = err
            assert job.error is err
        assert job.error.error_message == "job failed"


class TestJobAddLocals:
    """Tests for injecting additional local values into an existing job."""

    def test_add_locals_merges_values(self, job_factory):
        """add_locals adds and overwrites values in py_env.local_state."""
        job = job_factory()
        job.py_env.local_state = {"existing": 1}

        job.add_locals(existing=2, added=3)

        assert job.py_env.local_state == {"existing": 2, "added": 3}

    def test_add_locals_initializes_missing_local_state(self, job_factory):
        """add_locals creates local_state when it is missing."""
        job = job_factory()
        job.py_env.local_state = None

        job.add_locals(answer=42)

        assert job.py_env.local_state == {"answer": 42}


class TestJobDef:
    """Test cases for JobDef class."""

    def test_parse_model_metadata_returns_single_model(self):
        """A model without difficulty labels is returned unchanged."""
        assert parse_model_metadata("gpt-5.4-mini") == "gpt-5.4-mini"

    def test_parse_model_name_returns_bare_model(self):
        """Bare model names populate only the model component."""
        assert parse_model_name("gpt-5.4") == ModelName(None, None, "gpt-5.4")

    def test_parse_model_name_returns_model_family_and_model(self):
        """Two-part names are interpreted as model-family/model."""
        assert parse_model_name("openai/gpt-5.4") == ModelName(None, "openai", "gpt-5.4")

    def test_parse_model_name_returns_provider_family_and_model(self):
        """Three-part names are interpreted as provider/family/model."""
        assert parse_model_name("openrouter/openai/gpt-5.4") == ModelName(
            "openrouter", "openai", "gpt-5.4"
        )

    def test_parse_model_name_converts_empty_components_to_none(self):
        """Empty path components become None."""
        assert parse_model_name("openai//gpt-5.4") == ModelName("openai", None, "gpt-5.4")

    @pytest.mark.usefixtures("db0_fixture")
    def test_parse_model_metadata_returns_complete_difficulty_mapping(self):
        """Combined labels populate every TaskDifficulty exactly once."""
        result = parse_model_metadata("L:gpt-5.4-nano,MH:gpt-5.4-mini")
        assert result == {
            TaskDifficulty.low: "gpt-5.4-nano",
            TaskDifficulty.medium: "gpt-5.4-mini",
            TaskDifficulty.high: "gpt-5.4-mini",
        }

    @pytest.mark.usefixtures("db0_fixture")
    def test_parse_model_metadata_rejects_partial_difficulty_mapping(self):
        """Either all difficulty levels or no labels must be configured."""
        with pytest.raises(ValueError, match="all difficulty levels"):
            parse_model_metadata("L:gpt-5.4-nano,H:gpt-5.4")

    @pytest.mark.usefixtures("db0_fixture")
    def test_parse_model_metadata_rejects_duplicate_difficulty(self):
        """A difficulty level cannot be assigned twice."""
        with pytest.raises(ValueError, match="Duplicate"):
            parse_model_metadata("LM:gpt-5.4-mini,MH:gpt-5.4")

    def test_job_def_parses_model_metadata_on_creation(self, agent):
        """JobDef stores parsed difficulty mappings in metadata."""
        job_def = JobDef(
            agent=agent,
            metadata={"MODEL": "LM:gpt-5.4-mini,H:gpt-5.4"},
        )
        assert job_def.metadata["MODEL"] == {
            TaskDifficulty.low: "gpt-5.4-mini",
            TaskDifficulty.medium: "gpt-5.4-mini",
            TaskDifficulty.high: "gpt-5.4",
        }

    def test_job_def_model_formats_difficulty_mapping_in_enum_order(self, agent):
        """Difficulty mappings are formatted in enum order."""
        job_def = JobDef(
            agent=agent,
            metadata={"MODEL": "H:large,L:small,M:medium"},
        )
        assert job_def.model == "L:small,M:medium,H:large"

    def test_job_def_model_family_comes_from_model_name(self, agent):
        """Model family is derived from MODEL only."""
        job_def = JobDef(
            agent=agent,
            metadata={"MODEL": "openrouter/openai/gpt-5.4"},
        )
        assert job_def.model_family == "openai"

def test_job_system_prompt_delegates_to_agent(agent_factory, job_def_factory):
    """system_prompt delegates to agent.system_prompt with job params and difficulty."""
    agent = agent_factory(system_prompt="You are a {role} assistant")
    job_def = job_def_factory(job_params={"role": "helpful"})
    job_def.agent = agent
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    assert job.system_prompt() == "You are a helpful assistant"


def test_job_system_prompt_without_job_params(agent_factory, job_def_factory):
    """system_prompt returns the formatted prompt when no job params are present."""
    agent = agent_factory(system_prompt="You are a test assistant")
    job_def = job_def_factory(job_params=None)
    job_def.agent = agent
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    assert job.system_prompt() == "You are a test assistant"


def test_job_system_prompt_no_agent(job_def_factory):
    """system_prompt returns empty string when the job definition has no agent."""
    job_def = job_def_factory()
    job_def.agent = None
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    assert job.system_prompt() == ""


def test_job_system_prompt_resolves_shared_var_names(agent_factory, job_def_factory):
    """system_prompt resolves {shared_var_names} from job_params shared_vars."""
    agent = agent_factory(system_prompt="Variables: {shared_var_names}")
    job_def = job_def_factory(job_params={"shared_vars": ["user", "message"]})
    job_def.agent = agent
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    assert job.system_prompt() == "Variables: user, message"


def test_job_system_prompt_allows_difficulty_override(agent):
    """system_prompt can be formatted for an explicit difficulty."""
    agent.update_system_prompt(parse_system_prompt(
        "Intro.\n\n"
        "--- low: Scope ---\nLow instructions.\n\n"
        "--- high: Scope ---\nHigh instructions."
    ))
    job_def = JobDef(
        agent=agent,
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "high",
        },
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    result = job.system_prompt(TaskDifficulty.low)

    assert "Low instructions." in result
    assert "High instructions." not in result


class TestJob:
    """Test cases for Job class."""

    def test_contains_ext_ref_false_before_add(self, job_factory):
        """Job does not report unrelated memo objects as external refs."""
        job = job_factory()
        ext_ref = JobExtRefThing("message")
        assert job.contains_ext_ref(ext_ref) is False
        assert job._Job__ext_ref is None  # pylint: disable=protected-access

    def test_add_ext_ref_registers_memo_object(self, job_factory):
        """Job remembers memo objects registered as external refs."""
        job = job_factory()
        ext_ref = JobExtRefThing("message")
        job.add_ext_ref(ext_ref)
        assert job.contains_ext_ref(ext_ref) is True

    def test_add_ext_ref_is_weak_reference(self, job_factory):
        """External refs do not increase the referenced object's db0 refcount."""
        job = job_factory()
        ext_ref = JobExtRefThing("message")
        refcount = db0.getrefcount(ext_ref)
        job.add_ext_ref(ext_ref)
        assert db0.getrefcount(ext_ref) == refcount

    def test_contains_ext_ref_false_for_non_memo(self, job_factory):
        """contains_ext_ref returns False for non-memo values."""
        job = job_factory()
        job.add_ext_ref(JobExtRefThing("message"))
        assert job.contains_ext_ref("message") is False
        assert job.contains_ext_ref(object()) is False

    def test_add_ext_ref_ignores_non_memo(self, job_factory):
        """Adding a non-memo value is harmless."""
        job = job_factory()
        job.add_ext_ref("message")
        assert job.contains_ext_ref("message") is False

    def test_get_next_prompt_first_prompt_empty_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console is empty."""
        job = job_factory()
        result = job.get_next_prompt()

        # Should return the system_prompt since console is empty
        assert result == "Test agent"

    def test_get_next_prompt_first_prompt_with_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console has content."""
        job = job_factory()

        # Add some console output
        job.py_env.console_append("Output line 1")
        job.py_env.console_append("Output line 2")

        result = job.get_next_prompt()

        # Should include the system_prompt and all console outputs from position 0
        expected = "Test agent\n> Output line 1\n> Output line 2"
        assert result == expected

    def test_get_next_prompt_subsequent_prompt_from_console_pos(self, job_factory):
        """Test get_next_prompt when chat_log has entries."""
        job = job_factory()

        # Setup console with multiple outputs
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")
        job.py_env.console_append("Output 3")

        # Add a chat log item that processed first 2 console entries
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Some LLM response"))

        result = job.get_next_prompt()

        # Should only include console outputs from position 2 onwards
        expected = "> Output 3"
        assert result == expected

    def test_get_next_prompt_subsequent_prompt_no_new_console(self, job_factory):
        """Test get_next_prompt when no new console output since last chat."""
        job = job_factory()

        # Setup console
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        # Add chat log item that already processed all console entries
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Response"))

        result = job.get_next_prompt()

        # Should return empty string as there's no new console output
        assert result == ""

    def test_get_next_prompt_multiple_chat_items(self, job_factory):
        """Test get_next_prompt uses the last chat log item's console_pos."""
        job = job_factory()

        # Setup console with multiple outputs
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4", "Out5"]

        # Add multiple chat log items
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp2"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="resp3"))

        result = job.get_next_prompt()

        # Should use the last chat item's console_pos (4)
        expected = "> Out5"
        assert result == expected

    def test_get_next_prompt_push_log_none_no_change(self, job_factory):
        """No push_log → behaviour is unchanged."""
        job = job_factory()
        job.py_env.console_append("Out1")
        result = job.get_next_prompt()
        assert result == "Test agent\n> Out1"

    def test_get_next_prompt_first_prompt_push_log_appended(self, job_factory):
        """First prompt: push_log message is appended after console output."""
        job = job_factory()
        job.py_env.console_append("Out1")
        job.push_user_message("user message")  # key=1
        result = job.get_next_prompt()
        assert "Out1" in result
        assert "user message" in result

    def test_get_next_prompt_first_prompt_push_log_order(self, job_factory):
        """First prompt: push_log message appears after console output."""
        job = job_factory()
        job.py_env.console_append("Out1")
        job.push_user_message("user message")
        result = job.get_next_prompt()
        assert result.index("Out1") < result.index("user message")

    def test_get_next_prompt_subsequent_prompt_push_log_at_from_pos(self, job_factory):
        """Subsequent prompt: push_log entry at from_pos is included."""
        job = job_factory()
        job.py_env.console = ["c1", "c2"]
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp"))
        job.push_user_message("pushed msg")  # key=2 (console len=2)
        result = job.get_next_prompt()
        assert "pushed msg" in result

    def test_get_next_prompt_subsequent_prompt_push_log_before_from_pos_excluded(self, job_factory):
        """Subsequent prompt: push_log entry with key < from_pos is excluded."""
        job = job_factory()
        job.py_env.console = ["c1", "c2"]
        job.push_user_message("early msg")     # key=2 (console len=2 at push time)
        job.py_env.console.append("c3")      # console grows to 3
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp"))
        result = job.get_next_prompt()
        assert "early msg" not in result

    def test_get_next_prompt_push_log_list_values_included(self, job_factory):
        """Multiple pushes at same position (stored as list) are all included."""
        job = job_factory()
        job.push_user_message("msg1")  # key=0
        job.push_user_message("msg2")  # key=0, becomes list
        result = job.get_next_prompt()
        assert "msg1" in result
        assert "msg2" in result

    def test_get_next_prompt_push_log_has_language_hint(self, job_def_factory):
        """push_log messages include language hint when locale is non-EN."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.push_user_message("Podaj grafik")
        result = job.get_next_prompt()
        assert "Podaj grafik (PAMIĘTAJ:" in result

    def test_get_next_prompt_push_log_no_hint_for_en(self, job_def_factory):
        """push_log messages have no hint when locale language is EN."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.push_user_message("Show schedule")
        result = job.get_next_prompt()
        assert "Show schedule" in result
        assert "(PAMIĘTAJ:" not in result
        assert "(ERINNERUNG:" not in result

    def test_get_next_prompt_push_log_no_hint_when_disabled(self, job_def_factory):
        """AUTO_LANG_HINT: False disables language hint on push_log messages."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(
            locale=locale,
            metadata={"MODEL": "test-model", "AUTO_LANG_HINT": "False"},
        )
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.push_user_message("Podaj grafik")
        result = job.get_next_prompt()
        assert "Podaj grafik" in result
        assert "(PAMIĘTAJ:" not in result

    def test_get_next_prompt_push_log_list_has_hint_on_each(self, job_def_factory):
        """Each message in a multi-push list gets the language hint."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.push_user_message("msg1")
        job.push_user_message("msg2")
        result = job.get_next_prompt()
        assert "msg1 (PAMIĘTAJ:" in result
        assert "msg2 (PAMIĘTAJ:" in result


class TestJobGetChatHistory:
    """Test cases for Job.get_chat_history method.

    ``get_chat_history`` yields conversational ``ChatHistoryItem`` objects
    only; the agent system prompt is passed separately via
    ``get_next_request``.
    """

    def test_get_chat_history_empty_chat_log(self, job_factory):
        """Empty chat_log: no conversational history is yielded."""
        job = job_factory()
        history = list(job.get_chat_history())
        assert not history

    def test_get_chat_history_single_chat_item(self, job_factory):
        """Single LLM turn at console_pos=0 with subsequent console output."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="LLM response 1"))
        # Console output produced AFTER the LLM responded.
        job.py_env.console = ["Output 1", "Output 2"]

        history = list(job.get_chat_history())

        # [ASSISTANT(resp1), USER(console for resp1)]
        assert len(history) == 2
        assert history[0].role == ChatRole.ASSISTANT
        assert history[0].content == "LLM response 1"
        assert history[0].content_src == ContentSource.ASSISTANT
        assert history[1].role == ChatRole.USER
        assert history[1].content == "Output 1\nOutput 2"
        assert history[1].content_src == ContentSource.CONSOLE

    def test_get_chat_history_multiple_chat_items(self, job_factory):
        """Multiple LLM turns each followed by their own console slice."""
        job = job_factory()
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4", "Out5"]
        # First turn at console_pos 0 (no prior output) — its execution
        # produced Out1, Out2. Second turn starts at 2, produces Out3, Out4.
        # Third turn starts at 4, produces Out5.
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp2"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="resp3"))

        history = list(job.get_chat_history())

        # [asst1, console1, asst2, console2, asst3, console3]
        assert len(history) == 6
        assert history[0].content == "resp1"
        assert history[1].content == "Out1\nOut2"
        assert history[1].content_src == ContentSource.CONSOLE
        assert history[2].content == "resp2"
        assert history[3].content == "Out3\nOut4"
        assert history[4].content == "resp3"
        assert history[5].content == "Out5"

    def test_get_chat_history_initial_user_message_from_push_log(self, job_factory):
        """A push_log entry at console position 0 becomes the initial USER item."""
        job = job_factory()
        job.py_env.push_log = {0: "Hello, please do X"}

        history = list(job.get_chat_history())

        assert len(history) == 1
        assert history[0].role == ChatRole.USER
        assert history[0].content == "Hello, please do X"
        assert history[0].content_src == ContentSource.USER

    def test_get_chat_history_push_log_initial_has_language_hint(self, job_def_factory):
        """push_log[0] in chat_history gets the language hint appended."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.push_log = {0: "Podaj grafik"}

        history = list(job.get_chat_history())
        assert len(history) == 1
        assert "Podaj grafik (PAMIĘTAJ:" in history[0].content

    def test_get_chat_history_push_log_yields_has_language_hint(self, job_def_factory):
        """push_log entries in _yield_pushes get the language hint appended."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.console = ["c1", "c2"]
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp"))
        job.py_env.push_log = {1: "Follow-up question"}

        history = list(job.get_chat_history())
        push_items = [h for h in history if h.content_src == ContentSource.USER]
        assert any("Follow-up question (PAMIĘTAJ:" in h.content for h in push_items)

    def test_get_chat_history_push_log_no_hint_when_disabled(self, job_def_factory):
        """AUTO_LANG_HINT: False disables hint in chat_history push_log items."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(
            locale=locale,
            metadata={"MODEL": "test-model", "AUTO_LANG_HINT": "False"},
        )
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.push_log = {0: "Podaj grafik"}

        history = list(job.get_chat_history())
        assert history[0].content == "Podaj grafik"

    def test_get_chat_history_initial_user_message_from_chat_log_str(self, job_factory):
        """A leading str entry in chat_log feeds into the initial USER item."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        job.push_user_message("hi there")

        history = list(job.get_chat_history())

        assert len(history) == 1
        assert history[0].role == ChatRole.USER
        assert history[0].content == "hi there"

    def test_get_chat_history_initial_chat_log_str_has_language_hint(self, job_def_factory):
        """Leading chat_log str gets the language hint when locale is non-EN."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        job.push_user_message("Podaj grafik")

        history = list(job.get_chat_history())

        assert len(history) == 1
        assert history[0].role == ChatRole.USER
        assert "Podaj grafik (PAMIĘTAJ:" in history[0].content

    def test_get_chat_history_initial_chat_log_str_no_hint_when_disabled(self, job_def_factory):
        """AUTO_LANG_HINT disables the hint for a leading chat_log str."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(
            locale=locale,
            metadata={"MODEL": "test-model", "AUTO_LANG_HINT": "False"},
        )
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        job.push_user_message("Podaj grafik")

        history = list(job.get_chat_history())

        assert len(history) == 1
        assert history[0].content == "Podaj grafik"


def test_get_current_difficulty_returns_static_metadata(job_def_factory):
    """Metadata difficulty is used when no dynamic difficulty is stored."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.low
    assert job._Job__last_difficulty is None  # pylint: disable=protected-access


def test_get_current_difficulty_returns_static_settings_default(job_def_factory):
    """Settings default is static when metadata does not define difficulty."""
    settings = MagicMock(statek_default_difficulty="H")
    job_def = job_def_factory(metadata={"MODEL": "L:small,M:medium,H:large"})
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    with patch("statek.executors.job.get_statek_settings", return_value=settings):
        assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.high

    assert job._Job__last_difficulty is None  # pylint: disable=protected-access


def test_get_current_difficulty_uses_example_difficulty(job_def_factory):
    """Example difficulty is dynamic and is stored as the last resolved difficulty."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 3}

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.high,
    ) as mock_get_example_difficulty:
        assert job.get_current_difficulty() == TaskDifficulty.high

    mock_get_example_difficulty.assert_called_once_with("test", 3)
    assert job._Job__last_difficulty == TaskDifficulty.high  # pylint: disable=protected-access


def test_get_current_difficulty_uses_registered_job_perm_ctx_example_id(job_def_factory):
    """The registered job's PyEnv context supplies the last example ID."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 8}

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.medium,
    ) as mock_get_example_difficulty:
        assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.medium

    mock_get_example_difficulty.assert_called_once_with("test", 8)
    assert job._Job__last_difficulty == TaskDifficulty.medium  # pylint: disable=protected-access


def test_get_current_difficulty_example_does_not_downgrade_last_dynamic(
    job_def_factory,
):
    """Lower example difficulty does not downgrade a previously resolved dynamic value."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 4}
    job._Job__last_difficulty = TaskDifficulty.high  # pylint: disable=protected-access

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.low,
    ):
        assert job.get_current_difficulty() == TaskDifficulty.high

    assert job._Job__last_difficulty == TaskDifficulty.high  # pylint: disable=protected-access


def test_get_current_difficulty_example_can_upgrade_last_dynamic(
    job_def_factory,
):
    """Higher example difficulty upgrades the stored dynamic value."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 5}
    job._Job__last_difficulty = TaskDifficulty.low  # pylint: disable=protected-access

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.medium,
    ):
        assert job.get_current_difficulty() == TaskDifficulty.medium

    assert job._Job__last_difficulty == TaskDifficulty.medium  # pylint: disable=protected-access


def test_get_current_difficulty_missing_example_difficulty_falls_back_to_metadata(
    job_def_factory,
):
    """Missing example difficulty falls through to static metadata."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "medium",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 6}

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=None,
    ):
        assert job.get_current_difficulty() == TaskDifficulty.medium

    assert job._Job__last_difficulty is None  # pylint: disable=protected-access


def test_get_current_difficulty_ignores_non_job_task_difficulty_attribute(
    job_def_factory,
):
    """Job difficulty state is only stored in __last_difficulty."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.task_difficulty = TaskDifficulty.high

    assert job._Job__last_difficulty is None  # pylint: disable=protected-access
    assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.low


def test_panic_increases_low_difficulty_to_medium(job_def_factory):
    """panic raises the current difficulty by one level."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    _run_with_current_job(job, job.panic)

    assert job._Job__last_difficulty == TaskDifficulty.medium  # pylint: disable=protected-access
    assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.medium


def test_panic_increases_medium_difficulty_to_high(job_def_factory):
    """panic raises medium difficulty to high."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "medium",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    _run_with_current_job(job, job.panic)

    assert job._Job__last_difficulty == TaskDifficulty.high  # pylint: disable=protected-access
    assert _run_with_current_job(job, job.get_current_difficulty) == TaskDifficulty.high


def test_panic_raises_when_already_high(job_def_factory):
    """panic fails once difficulty is already at the maximum."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "high",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    with pytest.raises(RuntimeError, match="already at high difficulty"):
        _run_with_current_job(job, job.panic)


def test_get_current_model_returns_plain_model(job_def_factory):
    """MODEL without difficulty labels is returned unchanged."""
    job_def = job_def_factory(metadata={"MODEL": "test-model"})
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    assert job.get_current_model() == "test-model"
    assert job._Job__last_difficulty is None  # pylint: disable=protected-access


def test_get_current_model_uses_current_difficulty(job_def_factory):
    """MODEL difficulty mapping is resolved through get_current_difficulty."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "medium",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    assert _run_with_current_job(job, job.get_current_model) == "medium"
    assert job._Job__last_difficulty is None  # pylint: disable=protected-access


def test_get_current_model_uses_example_dynamic_difficulty(job_def_factory):
    """MODEL lookup resolves example difficulty through get_current_difficulty."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 1}

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.medium,
    ):
        assert job.get_current_model() == "medium"

    assert job._Job__last_difficulty == TaskDifficulty.medium  # pylint: disable=protected-access


def test_usage_pricing_uses_concrete_difficulty_model(job_def_factory):
    """Usage cost uses the selected MODEL entry, not the raw difficulty mapping."""
    set_model_pricing("openai", "gpt-5.4-mini", Decimal("0.40"), Decimal("1.60"))
    job_def = job_def_factory(
        metadata={
            "PROVIDER": "openai",
            "MODEL": "LM:gpt-5.4-mini,H:gpt-4o",
            "DEFAULT_DIFFICULTY": "medium",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    job.usage.total_input_tokens = 1_000_000

    assert job.usage.pricing.is_valid is True
    assert job.usage.total_cost == pytest.approx(0.40)


def test_sync_usage_pricing_preserves_tokens_for_existing_usage(job_def_factory):
    """Older jobs with token counts can be re-priced without losing usage data."""
    set_model_pricing("openai", "gpt-5.4-mini", Decimal("0.40"), Decimal("1.60"))
    job_def = job_def_factory(
        metadata={
            "PROVIDER": "openai",
            "MODEL": "LM:gpt-5.4-mini,H:gpt-4o",
            "DEFAULT_DIFFICULTY": "medium",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
    job.usage.pricing = set_model_pricing(
        "openai", "L:bad,M:bad,H:bad", Decimal("0"), Decimal("0")
    )
    job.usage.total_input_tokens = 1_000_000

    job._sync_usage_pricing()  # pylint: disable=protected-access

    assert job.usage.total_input_tokens == 1_000_000
    assert job.usage.total_cost == pytest.approx(0.40)


def test_get_next_request_formats_system_prompt_with_current_difficulty(agent):
    """The outgoing system prompt uses Job.get_current_difficulty for section selection."""
    agent.update_system_prompt(parse_system_prompt(
        "Intro.\n\n"
        "--- low: Scope ---\nLow instructions.\n\n"
        "--- high: Scope ---\nHigh instructions."
    ))
    job_def = JobDef(
        agent=agent,
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "high",
        },
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    request = job.get_next_request()

    assert "High instructions." in request["system_prompt"]
    assert "Low instructions." not in request["system_prompt"]


def test_get_next_request_uses_last_dynamic_difficulty(
    job_def_factory,
):
    """Request construction keeps the dynamic high-water mark for examples."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "low",
        }
    )
    job = Job(
        job_def=job_def,
        job_status=JobStatus.STARTED,  # pylint: disable=no-member
    )
    job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 2}
    job._Job__last_difficulty = TaskDifficulty.high  # pylint: disable=protected-access

    with patch(
        "statek.executors.job._get_example_difficulty_for_job",
        return_value=TaskDifficulty.low,
    ):
        request = _run_with_current_job(job, job.get_next_request)

    assert request["model"] == "large"
    assert request["metadata"]["MODEL"][TaskDifficulty.high] == "large"
    assert request["metadata"]["MODEL"][TaskDifficulty.low] == "small"
    assert request["metadata"]["MODEL"][TaskDifficulty.medium] == "medium"
    assert job._Job__last_difficulty == TaskDifficulty.high  # pylint: disable=protected-access


def test_get_next_request_does_not_rewrite_model_metadata(job_def_factory):
    """Request construction leaves MODEL metadata as the JobDef snapshot."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "L:small,M:medium,H:large",
            "DEFAULT_DIFFICULTY": "high",
        }
    )
    original_model_metadata = dict(job_def.metadata["MODEL"])
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    request = _run_with_current_job(job, job.get_next_request)

    assert request["model"] == "large"
    assert request["metadata"]["MODEL"] == original_model_metadata
    assert job_def.metadata["MODEL"] == original_model_metadata


def test_get_next_request_uses_provider_from_model_name(job_def_factory):
    """The model string can override provider selection and preserve the family when needed."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "openrouter/openai/gpt-5.4",
            "PROVIDER": "OPENAI",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    request = _run_with_current_job(job, job.get_next_request)

    assert request["model"] == "openai/gpt-5.4"


def test_get_next_request_discards_model_family_for_non_family_provider(job_def_factory):
    """Providers such as OpenAI receive only the concrete model identifier."""
    job_def = job_def_factory(
        metadata={
            "MODEL": "openai/openai/gpt-5.4",
        }
    )
    job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

    request = _run_with_current_job(job, job.get_next_request)

    assert request["model"] == "gpt-5.4"


class TestJobGetNextRequest:
    """Test cases for Job.get_next_request method.

    ``get_next_request`` carries ``system_prompt`` separately from
    conversational ``chat_history``.
    """

    def test_get_next_request_keys(self, job_factory):
        """The request dict carries chat_history plus a dedicated system_prompt key."""
        job = job_factory()
        request = job.get_next_request()

        assert "chat_history" in request
        assert request["system_prompt"] == "Test agent"
        assert request["model"] == "test-model"
        assert "prompt" not in request
        assert "metadata" in request
        assert "available_tools" in request
        assert "session_id" not in request

    def test_get_next_request_chat_history_excludes_system(self, job_factory):
        """The system prompt is not embedded in chat_history."""
        job = job_factory()
        history = list(job.get_next_request()["chat_history"])

        assert not history

    def test_get_next_request_with_chat_history(self, job_factory):
        """chat_history contains ChatHistoryItems for SYSTEM + each turn."""
        job = job_factory()
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="Response 1"))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Response 2"))

        history = list(job.get_next_request()["chat_history"])

        # [asst1, console1, asst2] — asst2 has no following console output
        assert len(history) == 3
        assert history[0].content == "Response 1"
        assert history[1].content == "Out1\nOut2"
        assert history[1].content_src == ContentSource.CONSOLE
        assert history[2].content == "Response 2"

    def test_get_next_request_structure(self, job_factory):
        """get_next_request returns a dict whose chat_history is a lazy generator."""
        job = job_factory()
        request = job.get_next_request()

        assert isinstance(request, dict)
        assert "prompt" not in request
        assert request["system_prompt"] == "Test agent"
        assert isinstance(request["chat_history"], types.GeneratorType)
        assert "session_id" not in request

    def test_get_next_request_no_chat_style_when_none(self, job_factory):
        """get_next_request omits CHAT_STYLE from metadata when chat_style is None."""
        job = job_factory()
        job.job_def.set_chat_style(None)
        request = job.get_next_request()
        assert "CHAT_STYLE" not in request["metadata"]

    def test_get_next_request_empty_console_no_history(self, job_factory):
        """No console, no chat_log: chat_history is empty."""
        job = job_factory()
        history = list(job.get_next_request()["chat_history"])
        assert not history

    def test_get_next_request_appends_language_rule_for_non_en(self, job_def_factory):
        """System prompt includes language rule when locale has non-EN language."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        request = job.get_next_request()
        assert request["system_prompt"].startswith("Test agent")
        assert "Test agent" in request["system_prompt"]
        # The language rule should be appended after the base prompt
        assert len(request["system_prompt"]) > len("Test agent")

    def test_get_next_request_no_language_rule_for_en(self, job_def_factory):
        """System prompt is unchanged when locale language is EN."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        request = job.get_next_request()
        assert request["system_prompt"] == "Test agent"

    def test_get_next_request_no_language_rule_when_no_locale(self, job_factory):
        """System prompt is unchanged when no locale is set."""
        job = job_factory()
        request = job.get_next_request()
        assert request["system_prompt"] == "Test agent"

    def test_get_next_request_no_language_rule_when_disabled(self, job_def_factory):
        """AUTO_LANG_RULE: False in metadata disables the language rule."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(
            locale=locale,
            metadata={"MODEL": "test-model", "AUTO_LANG_RULE": "False"},
        )
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        request = job.get_next_request()
        assert request["system_prompt"] == "Test agent"

    def test_get_next_request_uses_model_frozen_on_job_def_creation(
        self, db0_fixture  # pylint: disable=unused-argument
    ):
        """MODEL is frozen on JobDef creation and reused in later requests."""
        from statek.agents.agent import SupervisedAgent  # pylint: disable=import-outside-toplevel

        agent = SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("Test agent"),
            _metadata={"MODEL": "deepseek/deepseek-v3.2"},
            _tools=[],
        )
        job_def = agent.create_job_def()
        job = Job(job_def=job_def, job_status=JobStatus.READY)

        agent.update_metadata({"MODEL": "openai/gpt-4o"})
        request = job.get_next_request()

        assert job.job_def.model == "deepseek/deepseek-v3.2"
        assert job.model == "deepseek/deepseek-v3.2"
        assert request["model"] == "deepseek/deepseek-v3.2"
        assert request["metadata"]["MODEL"] == "deepseek/deepseek-v3.2"

    def test_get_next_request_extracts_temperature_from_metadata(
        self, job_def_factory
    ):
        """TEMPERATURE metadata is exposed as an explicit request parameter."""
        job_def = job_def_factory(metadata={"MODEL": "test-model", "TEMPERATURE": "0.3"})
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )

        request = job.get_next_request()

        assert request["temperature"] == 0.3
        assert request["metadata"]["TEMPERATURE"] == "0.3"

    def test_get_next_request_extracts_enable_reasoning_from_metadata(
        self, job_def_factory
    ):
        """REASONING metadata is exposed as an explicit boolean request parameter."""
        job_def = job_def_factory(metadata={"MODEL": "test-model", "REASONING": "true"})
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )

        request = job.get_next_request()

        assert request["enable_reasoning"] is True
        assert request["metadata"]["REASONING"] == "true"

    def test_get_next_request_uses_job_def_metadata_snapshot_after_agent_update(
        self, db0_fixture  # pylint: disable=unused-argument
    ):
        """Existing jobs keep the metadata captured by their JobDef."""
        from statek.agents.agent import SupervisedAgent  # pylint: disable=import-outside-toplevel

        agent = SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("Test agent"),
            _metadata={"MODEL": "test-model", "TEMPERATURE": "0.3", "REASONING": "true"},
            _tools=[],
        )
        job_def = agent.create_job_def()
        agent.update_metadata({"MODEL": "test-model", "TEMPERATURE": "0.1"})

        job = Job(job_def=job_def, job_status=JobStatus.READY)
        request = job.get_next_request()

        assert request["temperature"] == 0.3
        assert request["enable_reasoning"] is True
        assert request["metadata"]["TEMPERATURE"] == "0.3"
        assert request["metadata"]["REASONING"] == "true"

    def test_get_next_request_uses_metadata_default_difficulty_model(
        self, job_def_factory
    ):
        """DEFAULT_DIFFICULTY selects the concrete model from MODEL mapping."""
        job_def = job_def_factory(
            metadata={
                "MODEL": "L:small,M:medium,H:large",
                "DEFAULT_DIFFICULTY": "high",
            }
        )
        job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member

        request = job.get_next_request()

        assert job._Job__last_difficulty is None  # pylint: disable=protected-access
        assert request["model"] == "large"
        assert request["metadata"]["MODEL"][TaskDifficulty.high] == "large"
        assert request["metadata"]["MODEL"][TaskDifficulty.low] == "small"
        assert request["metadata"]["MODEL"][TaskDifficulty.medium] == "medium"

    def test_get_next_request_allows_static_default_difficulty_downgrade(
        self, job_def_factory
    ):
        """Static DEFAULT_DIFFICULTY definitions are returned as configured."""
        job_def = job_def_factory(
            metadata={
                "MODEL": "L:small,M:medium,H:large",
                "DEFAULT_DIFFICULTY": "high",
            }
        )
        job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
        assert _run_with_current_job(job, job.get_next_request)["model"] == "large"

        job.job_def.metadata["DEFAULT_DIFFICULTY"] = "low"
        request = _run_with_current_job(job, job.get_next_request)

        assert job._Job__last_difficulty is None  # pylint: disable=protected-access
        assert request["model"] == "small"


class TestJobGetRequestData:
    """Test cases for Job.get_request_data."""

    def test_get_request_data_reconstructs_first_and_second_turn(self, job_factory):
        """Historical request data is rebuilt from the append-only job state."""
        job = job_factory()

        job.py_env.console_append("Step 1 output")
        request1 = job.get_next_request()
        job.append_chat_log(request1, LLM_Response(
            step_data=LLM_StepData(text="code_block_1", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        job.py_env.console_append("Step 2 output")
        job.py_env.console_append("Step 2 more output")
        request2 = job.get_next_request()
        job.append_chat_log(request2, LLM_Response(
            step_data=LLM_StepData(text="code_block_2", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        historical_1 = job.get_request_data(0)
        historical_2 = job.get_request_data(1)

        assert historical_1["system_prompt"] == "Test agent"
        assert not list(historical_1["chat_history"])

        history_2 = list(historical_2["chat_history"])
        assert [item.content for item in history_2] == [
            "code_block_1",
            "Step 2 output\nStep 2 more output",
        ]
        assert history_2[1].content_src == ContentSource.CONSOLE
        assert historical_2["model"] == "test-model"

    def test_get_request_data_rejects_out_of_range_turn(self, job_factory):
        """Missing historical turns raise IndexError."""
        job = job_factory()

        with pytest.raises(IndexError, match="turn_num"):
            job.get_request_data(0)

        request = job.get_next_request()
        job.append_chat_log(request, LLM_Response(
            step_data=LLM_StepData(text="resp", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        with pytest.raises(IndexError, match="turn_num"):
            job.get_request_data(-1)
        with pytest.raises(IndexError, match="turn_num"):
            job.get_request_data(1)

    def test_get_request_data_reconstructs_multi_tool_turn_for_preview(
        self, job_factory, db0_fixture
    ):
        del db0_fixture
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)

        job.push_user_message("moj grafik na kwiecien")
        turns = [
            (
                0,
                CallSpec(
                    id="STATEK-WARMUP-000",
                    func_name="python_cli",
                    kwargs={"code": "warmup()"},
                ),
                "Current date and time",
            ),
            (
                1,
                CallSpec(id="STATEK-001", func_name="list_of_examples", kwargs={}),
                "# Example ID: Example name",
            ),
            (
                2,
                CallSpec(id="STATEK-002", func_name="show_example", kwargs={}),
                "# --- EXAMPLE: Showing a monthly schedule calendar ---",
            ),
            (
                3,
                CallSpec(
                    id="call_render_april",
                    func_name="python_cli",
                    kwargs={"code": "render_april()"},
                ),
                "2026-04-01",
            ),
            (
                4,
                CallSpec(id="call_panic", func_name="panic", kwargs={}),
                "# Difficulty increased to medium. Continue with the harder task.",
            ),
            (
                5,
                CallSpec(
                    id="call_render_final",
                    func_name="python_cli",
                    kwargs={"code": "answer('kwiecien')"},
                ),
                (
                    "log: answer(body='Oto twoj grafik dyzurow na kwiecien 2026.', "
                    "media='private/calendar.svg')"
                ),
            ),
        ]
        for console_pos, call_spec, result in turns:
            item = create_chat_log_item(
                console_pos=console_pos,
                llm_resp=CodeBlock(code=None, tool_calls=[call_spec]),
            )
            item.tool_log = [result]
            job.chat_log.append(item)
        job.chat_log.append(UserLogItem(message="i jeszcze na maj"))
        job.chat_log.append(create_chat_log_item(console_pos=6, llm_resp="dummy second turn"))
        job.py_env.console = [result for _, _, result in turns]

        historical = job.get_request_data(6)
        history = list(historical["chat_history"])
        payload = OpenRouter_API(LLM_API_Settings(
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-key",
        )).preview_request(**{**historical, "chat_history": history})

        messages = payload["messages"]
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "user",
        ]
        assert messages[2]["tool_calls"][0]["id"] == "STATEK-WARMUP-000"
        assert messages[3]["tool_call_id"] == "STATEK-WARMUP-000"
        assert messages[4]["tool_calls"][0]["id"] == "STATEK-001"
        assert messages[5]["tool_call_id"] == "STATEK-001"
        assert messages[6]["tool_calls"][0]["id"] == "STATEK-002"
        assert messages[7]["tool_call_id"] == "STATEK-002"
        assert messages[8]["tool_calls"][0]["id"] == "call_render_april"
        assert messages[9]["tool_call_id"] == "call_render_april"
        assert messages[10]["tool_calls"][0]["id"] == "call_panic"
        assert messages[11]["tool_call_id"] == "call_panic"
        assert messages[12]["tool_calls"][0]["id"] == "call_render_final"
        assert messages[13]["tool_call_id"] == "call_render_final"
        assert messages[14]["content"] == "i jeszcze na maj"

    def test_get_next_request_prefers_last_dynamic_difficulty_for_example(
        self, job_def_factory
    ):
        """Example resolution keeps the stored dynamic high-water mark."""
        job_def = job_def_factory(
            metadata={
                "MODEL": "L:small,M:medium,H:large",
                "DEFAULT_DIFFICULTY": "low",
            }
        )
        job = Job(job_def=job_def, job_status=JobStatus.READY)  # pylint: disable=no-member
        job.py_env.local_state["_PERM_CTX"] = {"last_example_id": 7}
        job._Job__last_difficulty = TaskDifficulty.high  # pylint: disable=protected-access

        with patch(
            "statek.executors.job._get_example_difficulty_for_job",
            return_value=TaskDifficulty.low,
        ):
            request = job.get_next_request()

        assert job._Job__last_difficulty == TaskDifficulty.high  # pylint: disable=protected-access
        assert request["model"] == "large"

    def test_last_response_empty_chat_log(self, job_factory):
        """Test last_response returns None when chat_log is empty."""
        job = job_factory()
        assert job.chat_log == []
        assert job.last_response is None

    def test_last_response_with_chat_log(self, job_factory):
        """Test last_response returns the llm_resp from the last chat log item."""
        job = job_factory()

        # Add chat log items
        job.chat_log.append(create_chat_log_item(
            console_pos=0, llm_resp="print('first response')"
        ))
        job.chat_log.append(create_chat_log_item(
            console_pos=1, llm_resp="print('second response')"
        ))

        assert job.last_response == "print('second response')"

    def test_last_response_returns_code_block(self, job_factory):
        """last_response returns a CodeBlock when llm_resp is a CodeBlock."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        result = job.last_response

        assert isinstance(result, CodeBlock)
        assert result.code == "x = 1"
        assert result.tool_calls[0].func_name == "my_tool"


class TestJobGetNextCodeBlock:
    """Tests for Job.get_next_code_block — covering str and CodeBlock returns."""

    def test_returns_none_when_done(self, job_factory):
        """Returns None when job status is DONE."""
        job = job_factory()
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member

        assert job.get_next_code_block() is None

    def test_returns_str_warmup_when_ready(self, job_factory):
        """Returns plain string warmup block when status is READY."""
        job = job_factory(warmup_code="x = 1")

        result = job.get_next_code_block()

        assert result == "x = 1"
        assert isinstance(result, str)

    def test_returns_code_block_warmup_when_ready(self, job_factory):
        """Returns CodeBlock warmup block when status is READY and warmup is a CodeBlock."""
        call_spec = CallSpec(id="W-001", func_name="setup_tool", args=[], kwargs={})
        block = CodeBlock(code="setup()", tool_calls=[call_spec])
        job = job_factory(warmup_code=block)

        result = job.get_next_code_block()

        assert isinstance(result, CodeBlock)
        assert result.code == "setup()"

    def test_returns_str_from_last_response(self, job_factory):
        """Returns plain string last_response when status is STARTED."""
        job = job_factory()
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 42"))

        result = job.get_next_code_block()

        assert result == "x = 42"
        assert isinstance(result, str)

    def test_returns_code_block_from_last_response(self, job_factory):
        """Returns CodeBlock last_response when status is STARTED and llm_resp is a CodeBlock."""
        job = job_factory()
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        result = job.get_next_code_block()

        assert isinstance(result, CodeBlock)
        assert result.code == "x = 1"
        assert result.tool_calls[0].func_name == "my_tool"


class TestJobGetNextPromptWithWarmup:
    """Test get_next_prompt when warmup was executed."""

    def test_no_warmup_behavior_unchanged(self, job_factory):
        """Without warmup_code the prompt is unchanged."""
        job = job_factory()
        job.py_env.console = ["line1", "line2"]

        result = job.get_next_prompt()

        assert result == "Test agent\n> line1\n> line2"

    def test_warmup_prompt_is_last_block_console_output(self, job_factory):
        """With warmup, prompt is the console output OF the last warmup block."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["last block output"]
        set_warmup_positions(job, [1])  # block 0 produced 1 line

        result = job.get_next_prompt()

        assert "> last block output" in result

    def test_warmup_prompt_excludes_template_and_warmup_code(self, job_factory):
        """With warmup, template and code are NOT in the prompt (they're in chat_history)."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["last block output"]
        set_warmup_positions(job, [1])

        result = job.get_next_prompt()

        assert "Test task" not in result
        assert "x = 1" not in result

    def test_warmup_multi_block_prompt_is_last_block_output(self, job_factory):
        """With multiple warmup blocks, prompt is ONLY the last block's console output."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])  # block0->1 line, block1->1 line

        result = job.get_next_prompt()

        assert "> out2" in result

    def test_warmup_multi_block_prompt_excludes_earlier_blocks(self, job_factory):
        """Earlier warmup block outputs are NOT in the prompt (they're in chat_history)."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])

        result = job.get_next_prompt()

        assert "block1" not in result
        assert "block2" not in result
        assert "> out1" not in result


class TestJobGetChatHistoryWithWarmup:
    """Test get_chat_history yields warmup as ASSISTANT + USER (console) pairs."""

    def test_warmup_no_chat_log_single_block(self, job_factory):
        """One warmup block, no LLM turns: [asst(code), user(console)]."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup output"]
        set_warmup_positions(job, [1])

        history = list(job.get_chat_history())

        assert len(history) == 2
        assert history[0].role == ChatRole.ASSISTANT
        assert history[0].content == "x = 1"
        assert history[0].content_src == ContentSource.SYSTEM
        assert history[1].role == ChatRole.USER
        assert history[1].content == "warmup output"
        assert history[1].content_src == ContentSource.CONSOLE

    def test_warmup_no_chat_log_two_blocks(self, job_factory):
        """Two warmup blocks: [asst1, user1, asst2, user2]."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])

        history = list(job.get_chat_history())

        assert len(history) == 4
        assert history[0].content == "block1"
        assert history[0].content_src == ContentSource.SYSTEM
        assert history[1].content == "out1"
        assert history[1].content_src == ContentSource.CONSOLE
        assert history[2].content == "block2"
        assert history[3].content == "out2"

    def test_warmup_with_chat_log(self, job_factory):
        """Warmup followed by an LLM turn: [asst_w, user_w, asst_llm, user_llm]."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup out", "post-warmup out"]
        set_warmup_positions(job, [1])
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="resp1"))

        history = list(job.get_chat_history())

        assert len(history) == 4
        assert history[0].content == "x = 1"        # warmup asst
        assert history[1].content == "warmup out"   # warmup user
        assert history[2].content == "resp1"        # llm asst
        assert history[3].content == "post-warmup out"  # llm user

    def test_tool_only_warmup_keeps_system_content_source(self, job_factory):
        """Tool-only warmup assistant items should still be marked as SYSTEM."""
        call_spec = CallSpec(id="T", func_name="docstr", args=["topic"])
        job = job_factory(warmup_code=CodeBlock(code=None, tool_calls=[call_spec]))
        warmup_item = WarmupLogItem(console_pos=0, warmup_block_num=0, tool_log="tool result")
        job.chat_log.append(warmup_item)

        history = list(job.get_chat_history())

        assert history[0].role == ChatRole.ASSISTANT
        assert history[0].content is None
        assert history[0].content_src == ContentSource.SYSTEM
        assert history[0].tool_calls == [call_spec]
        assert history[1].role == ChatRole.TOOL
        assert history[1].content == "tool result"

    def test_python_cli_output_as_tool_not_duplicate_user(self, job_factory):
        """python_cli console output is attributed to a TOOL message only —
        no duplicate USER console dump following it.  This reproduces the
        production bug where every python_cli tool call produced an empty
        <CONSOLE_OUTPUT> tool-result followed by a USER message carrying
        the actual output."""
        call_spec = CallSpec(id="call_1", func_name="python_cli",
                             kwargs={"code": "render_calendar()"})
        job = job_factory()
        # tool_log aligned with tool_calls; console advanced by the same output.
        item = create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[call_spec]),
        )
        item.tool_log = ["private/calendar.svg"]
        job.chat_log.append(item)
        job.py_env.console = ["private/calendar.svg"]

        history = list(job.get_chat_history())

        roles = [h.role for h in history]
        assert roles == [ChatRole.ASSISTANT, ChatRole.TOOL]
        assert history[1].content == "private/calendar.svg"

    def test_warmup_not_in_subsequent_messages(self, job_factory):
        """Warmup code does not appear in console items after the first LLM turn."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup out", "after-first-llm"]
        set_warmup_positions(job, [1])
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="resp1"))

        history = list(job.get_chat_history())

        # Last user item is the console produced by the first LLM turn.
        last_user = history[-1]
        assert last_user.role == ChatRole.USER
        assert last_user.content == "after-first-llm"
        assert "x = 1" not in (last_user.content or "")

    def test_hidden_warmup_excluded_from_next_request_history(self, job_factory):
        """Hidden warmup code and console output are excluded from LLM-facing history."""
        hidden = CodeBlock(code="hidden_setup()", metadata={"hidden": True})
        job = job_factory(warmup_code=[hidden, "visible_setup()"])
        job.py_env.console = ["hidden output", "visible output"]
        set_warmup_positions(job, [1, 2])

        history = list(job.get_next_request()["chat_history"])
        serialized = "\n".join(str(item.content) for item in history)

        assert "hidden_setup" not in serialized
        assert "hidden output" not in serialized
        assert "visible_setup" in serialized
        assert "visible output" in serialized

    def test_hidden_warmup_tool_results_excluded_from_history(self, job_factory):
        """Hidden warmup tool calls and their results are excluded from chat history."""
        hidden_call = CallSpec(id="STATEK-001", func_name="hidden_tool", kwargs={})
        visible_call = CallSpec(id="STATEK-002", func_name="visible_tool", kwargs={})
        hidden = CodeBlock(
            code="hidden_tool()",
            tool_calls=[hidden_call],
            metadata={"hidden": True},
        )
        visible = CodeBlock(code="visible_tool()", tool_calls=[visible_call])
        job = job_factory(warmup_code=[hidden, visible])
        hidden_item = WarmupLogItem(console_pos=0, warmup_block_num=0)
        hidden_item.tool_log = ["hidden result"]
        visible_item = WarmupLogItem(console_pos=0, warmup_block_num=1)
        visible_item.tool_log = ["visible result"]
        job.chat_log.extend([hidden_item, visible_item])

        history = list(job.get_chat_history())
        serialized = json.dumps([
            {
                "content": item.content,
                "tool_calls": [cs.func_name for cs in item.tool_calls]
                if isinstance(item.tool_calls, list) else None,
            }
            for item in history
        ])

        assert "hidden_tool" not in serialized
        assert "hidden result" not in serialized
        assert "visible_tool" in serialized
        assert "visible result" in serialized

    def test_preview_request_excludes_hidden_warmup_payload(self, job_factory):
        """Provider preview payloads do not contain hidden warmup content."""
        hidden = CodeBlock(code="hidden_setup()", metadata={"hidden": True})
        job = job_factory(warmup_code=[hidden, "visible_setup()"])
        job.py_env.console = ["hidden output", "visible output"]
        set_warmup_positions(job, [1, 2])
        request = job.get_next_request()
        history = list(request["chat_history"])

        payload = OpenRouter_API(LLM_API_Settings(
            api_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-key",
        )).preview_request(**{**request, "chat_history": history})
        serialized = json.dumps(payload)

        assert "hidden_setup" not in serialized
        assert "hidden output" not in serialized
        assert "visible_setup" in serialized
        assert "visible output" in serialized

    def test_llm_initial_step_size_excludes_hidden_warmup_text(self, job_factory):
        """Hidden warmup code is excluded from LLM-facing token estimates."""
        hidden = CodeBlock(code="h" * 40, metadata={"hidden": True})
        visible = "v" * 20
        job = job_factory(warmup_code=[hidden, visible])

        assert job.get_llm_initial_step_size() == ((len("Test agent") + len(visible)) // 4, 0)


class TestJobSetStatus:  # pylint: disable=too-few-public-methods
    """Test cases for Job.set_status method."""

    def test_set_status_initial(self, job_factory):
        """Test setting initial job status."""
        job = job_factory()

        # Initial status should be READY
        assert job.status == JobStatus.READY  # pylint: disable=no-member
        assert len(db0.find(Job, JobStatus.READY)) == 1  # pylint: disable=no-member

        # Change status to STARTED
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member

        # Verify status is updated
        assert job.status == JobStatus.STARTED  # pylint: disable=no-member

        # Verify tags are updated
        assert len(db0.find(Job, JobStatus.READY)) == 0  # pylint: disable=no-member
        assert len(db0.find(Job, JobStatus.STARTED)) == 1  # pylint: disable=no-member


class TestJobAppendChatLog:
    """Test cases for Job.append_chat_log method."""

    def test_append_chat_log_empty_console(self, job_factory):
        """Test append_chat_log with empty console."""
        job = job_factory()

        request = job.get_next_request()
        llm_resp = LLM_Response(
            step_data=LLM_StepData(text="print('hello')", call_requests=None),
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
        )
        job.append_chat_log(request, llm_resp)

        assert len(job.chat_log) == 1
        assert job.chat_log[0].llm_resp == "print('hello')"
        assert job.chat_log[0].console_pos == 0

    def test_append_chat_log_with_console_output(self, job_factory):
        """Test append_chat_log with console output."""
        job = job_factory()

        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")
        job.py_env.console_append("Output 3")

        request = job.get_next_request()
        llm_resp = LLM_Response(
            step_data=LLM_StepData(text="x = 5", call_requests=None),
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
        )
        job.append_chat_log(request, llm_resp)

        assert len(job.chat_log) == 1
        assert job.chat_log[0].console_pos == 3
        assert job.chat_log[0].llm_resp == "x = 5"

    def test_append_chat_log_multiple_times(self, job_factory):
        """Test append_chat_log called multiple times."""
        job = job_factory()

        job.py_env.console_append("Step 1 output")
        request1 = job.get_next_request()
        job.append_chat_log(request1, LLM_Response(
            step_data=LLM_StepData(text="code_block_1", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        job.py_env.console_append("Step 2 output")
        job.py_env.console_append("Step 2 more output")
        request2 = job.get_next_request()
        job.append_chat_log(request2, LLM_Response(
            step_data=LLM_StepData(text="code_block_2", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        job.py_env.console_append("Step 3 output")
        request3 = job.get_next_request()
        job.append_chat_log(request3, LLM_Response(
            step_data=LLM_StepData(text="code_block_3", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        ))

        assert len(job.chat_log) == 3

        assert job.chat_log[0].console_pos == 1
        assert job.chat_log[0].llm_resp == "code_block_1"

        assert job.chat_log[1].console_pos == 3
        assert job.chat_log[1].llm_resp == "code_block_2"

        assert job.chat_log[2].console_pos == 4
        assert job.chat_log[2].llm_resp == "code_block_3"


class TestAppendChatLogDirect:
    """Tests for append_chat_log with ChatStyle.DIRECT."""

    def test_direct_job_override_wins_over_global_chat_style(self, job_factory):
        """Job chat_style override must control append_chat_log formatting."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member

        request = job.get_next_request()
        llm_resp = LLM_Response(
            step_data=LLM_StepData(
                text="Oto Twoj grafik na kwiecien 2026 roku.",
                call_requests=None,
            ),
            stats=LLM_Stats(0, 0, None),
        )
        with patch(
            'statek.executors.job.get_statek_settings',
            return_value=mock_settings
        ):
            job.append_chat_log(request, llm_resp)

        assert job.chat_log[0].llm_resp == "Oto Twoj grafik na kwiecien 2026 roku."

    def test_direct_discards_code_from_response(self, job_factory):
        """DIRECT style: keep only dialog text, ignoring fenced code blocks."""
        job = job_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.DIRECT  # pylint: disable=no-member

        request = job.get_next_request()
        llm_resp = LLM_Response(
            step_data=LLM_StepData(
                text="```python\nx = 42\n```\nHello user!",
                call_requests=None,
            ),
            stats=LLM_Stats(0, 0, None),
        )
        with patch(
            'statek.executors.job.get_statek_settings',
            return_value=mock_settings
        ):
            job.append_chat_log(request, llm_resp)

        assert job.chat_log[0].llm_resp == "Hello user!"

    def test_direct_preserves_tool_calls(self, job_factory):
        """DIRECT style: tool calls are preserved; code holds dialog text only."""
        from statek.llm_api import CallParams  # pylint: disable=import-outside-toplevel
        job = job_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.DIRECT  # pylint: disable=no-member

        request = job.get_next_request()
        call = CallParams(call_id="c1", name="python_cli", args=[], kwargs={"code": "x=1"})
        llm_resp = LLM_Response(
            step_data=LLM_StepData(text="```python\nx = 42\n```", call_requests=[call]),
            stats=LLM_Stats(0, 0, None),
        )
        with patch(
            'statek.executors.job.get_statek_settings',
            return_value=mock_settings
        ):
            job.append_chat_log(request, llm_resp)

        stored = job.chat_log[0].llm_resp
        assert isinstance(stored, CodeBlock)
        # Fenced code is stripped — no dialog text outside fences
        assert stored.code is None
        assert len(stored.tool_calls) == 1

    def test_direct_prose_with_tool_calls_stores_dialog_text(self, job_factory):
        """DIRECT: prose response with tool calls stores dialog in code field.

        The dialog text is stored in ``CodeBlock.code`` for chat history
        reconstruction.  ``exec_all_steps`` skips code execution in DIRECT
        mode so the prose is never passed to ``ast.parse``.
        """
        from statek.llm_api import CallParams  # pylint: disable=import-outside-toplevel
        job = job_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.DIRECT  # pylint: disable=no-member

        request = job.get_next_request()
        call = CallParams(call_id="c1", name="python_cli", args=[],
                          kwargs={"code": "x=1"})
        llm_resp = LLM_Response(
            step_data=LLM_StepData(
                text="You have 3 preference points remaining for April.",
                call_requests=[call],
            ),
            stats=LLM_Stats(0, 0, None),
        )
        with patch(
            'statek.executors.job.get_statek_settings',
            return_value=mock_settings
        ):
            job.append_chat_log(request, llm_resp)

        stored = job.chat_log[0].llm_resp
        assert isinstance(stored, CodeBlock)
        assert stored.code == "You have 3 preference points remaining for April."
        assert len(stored.tool_calls) == 1

    def test_direct_plain_text_response_stored_as_none(self, job_factory):
        """DIRECT style: plain text response is preserved in chat_log."""
        job = job_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.DIRECT  # pylint: disable=no-member

        request = job.get_next_request()
        llm_resp = LLM_Response(
            step_data=LLM_StepData(text="Hello, how can I help?", call_requests=None),
            stats=LLM_Stats(0, 0, None),
        )
        with patch(
            'statek.executors.job.get_statek_settings',
            return_value=mock_settings
        ):
            job.append_chat_log(request, llm_resp)

        assert job.chat_log[0].llm_resp == "Hello, how can I help?"


class TestPushUserMessageDirect:
    """Tests for push_user_message with ChatStyle.DIRECT."""

    def test_direct_first_message_stored_as_str(self, job_factory):
        """DIRECT style: first message stored as plain str in chat_log."""
        job = job_factory()
        job.job_def.set_chat_style(
            ChatStyle.DIRECT)  # pylint: disable=no-member
        job.push_user_message("hello")
        assert job.chat_log[0] == "hello"

    def test_direct_subsequent_message_stored_as_user_log_item(
        self, job_factory
    ):
        """DIRECT style: subsequent messages stored as UserLogItem."""
        job = job_factory()
        job.job_def.set_chat_style(
            ChatStyle.DIRECT)  # pylint: disable=no-member
        job.push_user_message("first")
        job.push_user_message("second")
        assert isinstance(job.chat_log[1], UserLogItem)
        assert job.chat_log[1].message == "second"

    def test_uses_agent_message_adapter_for_non_string_message(
        self, job_factory
    ):
        """Non-string messages are resolved through the agent message_adapter."""
        job = job_factory()
        job.job_def.set_chat_style(
            ChatStyle.DIRECT)  # pylint: disable=no-member
        job.job_def.agent.context["message_adapter"] = (
            lambda msg: f"adapted-{msg.value}"
        )

        job.push_user_message(MessageForAdapter("object"))

        assert job.chat_log[0] == "adapted-object"

    def test_falls_back_to_str_when_message_adapter_missing(self, job_factory):
        """Non-string messages fall back to str(message)."""
        job = job_factory()
        job.job_def.set_chat_style(
            ChatStyle.DIRECT)  # pylint: disable=no-member

        job.push_user_message(MessageForAdapter("object"))

        assert job.chat_log[0] == "fallback-object"


class TestPushUserMessageNumCompletions:
    """Tests for num_completions tracking in push_user_message."""

    def test_num_completions_starts_as_none(self, job_factory):
        """New job has num_completions=None."""
        job = job_factory()
        assert job.num_completions is None

    def test_done_to_started_sets_num_completions_to_1(self, job_factory):
        """First DONE->STARTED transition sets num_completions from None to 1."""
        job = job_factory()
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member
        job.push_user_message("hi")
        assert job.num_completions == 1

    def test_second_completion_increments_to_2(self, job_factory):
        """Second DONE->STARTED transition increments num_completions to 2."""
        job = job_factory()
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member
        job.push_user_message("first")
        # Job is now STARTED; transition back to DONE
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member
        job.push_user_message("second")
        assert job.num_completions == 2

    def test_no_increment_when_not_done(self, job_factory):
        """push_user_message on a non-DONE job does not change num_completions."""
        job = job_factory()
        assert job.status == JobStatus.READY  # pylint: disable=no-member
        job.push_user_message("msg")
        assert job.num_completions is None

    def test_returns_true_on_done_to_started(self, job_factory):
        """push_user_message returns True when DONE->STARTED transition occurs."""
        job = job_factory()
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member
        assert job.push_user_message("hi") is True

    def test_returns_false_when_not_done(self, job_factory):
        """push_user_message returns False when no transition occurs."""
        job = job_factory()
        assert job.push_user_message("hi") is False

    def test_done_to_started_clears_exit_status_and_appends_pending_llm(
        self, job_factory
    ):
        """DONE->STARTED clears exit_status and records an awaited LLM turn."""
        job = job_factory()
        job.py_env.exit_status = "done"
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member

        assert job.push_user_message("hi") is True

        assert job.status == JobStatus.STARTED  # pylint: disable=no-member
        assert job.py_env.exit_status is None
        assert isinstance(job.chat_log[-1], LLM_LogItem)
        assert job.chat_log[-1].llm_resp is None


class TestJobDefErrors:
    """Tests for JobDef.set_error, get_errors, has_errors."""

    def _make_raised_error(self, msg="something went wrong"):
        try:
            raise ValueError(msg)
        except ValueError as exc:
            return exc

    def test_has_errors_false_by_default(self, job_def_factory):
        """has_errors returns False when no errors have been set."""
        job_def = job_def_factory()
        assert job_def.has_errors() is False

    def test_get_errors_empty_by_default(self, job_def_factory):
        """get_errors yields nothing when no errors have been set."""
        job_def = job_def_factory()
        assert not list(job_def.get_errors())

    def test_set_error_creates_job_def_error(self, job_def_factory):
        """set_error creates a JobDefError associated with the job definition."""
        job_def = job_def_factory()
        error = self._make_raised_error("boom")
        job_def.set_error(error)
        errors = list(job_def.get_errors())
        assert len(errors) == 1
        assert isinstance(errors[0], JobDefError)
        assert errors[0].error_message == "boom"

    def test_set_error_has_errors_true(self, job_def_factory):
        """has_errors returns True after set_error is called."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("oops"))
        assert job_def.has_errors() is True

    def test_set_error_collects_traceback_by_default(self, job_def_factory):
        """set_error collects traceback by default."""
        job_def = job_def_factory()
        error = self._make_raised_error("traceback test")
        job_def.set_error(error)
        errors = list(job_def.get_errors())
        assert errors[0].traceback is not None
        assert len(errors[0].traceback) > 0

    def test_set_error_no_traceback_when_disabled(self, job_def_factory):
        """set_error does not collect traceback when collect_traceback=False."""
        job_def = job_def_factory()
        error = self._make_raised_error("no tb")
        job_def.set_error(error, collect_traceback=False)
        errors = list(job_def.get_errors())
        assert errors[0].traceback is None

    def test_set_error_multiple_errors(self, job_def_factory):
        """set_error can be called multiple times; all errors are retrievable."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("first"))
        job_def.set_error(self._make_raised_error("second"))
        errors = list(job_def.get_errors())
        assert len(errors) == 2
        messages = {e.error_message for e in errors}
        assert messages == {"first", "second"}

    def test_errors_isolated_between_job_defs(self, job_def_factory):
        """Errors set on one JobDef are not visible from another."""
        job_def1 = job_def_factory()
        job_def2 = job_def_factory()
        job_def1.set_error(self._make_raised_error("only for def1"))
        assert not list(job_def2.get_errors())
        assert job_def2.has_errors() is False

    def test_clear_errors_on_empty_job_def_does_not_raise(self, job_def_factory):
        """clear_errors does nothing and does not raise when there are no errors."""
        job_def = job_def_factory()
        job_def.clear_errors()
        assert job_def.has_errors() is False

    def test_clear_errors_removes_single_error(self, job_def_factory):
        """clear_errors removes a single error so has_errors returns False."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("boom"))
        job_def.clear_errors()
        assert job_def.has_errors() is False
        assert not list(job_def.get_errors())

    def test_clear_errors_removes_multiple_errors(self, job_def_factory):
        """clear_errors removes all errors when multiple errors were set."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("first"))
        job_def.set_error(self._make_raised_error("second"))
        job_def.set_error(self._make_raised_error("third"))
        job_def.clear_errors()
        assert not list(job_def.get_errors())

    def test_clear_errors_is_idempotent(self, job_def_factory):
        """clear_errors can be called twice without raising."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("once"))
        job_def.clear_errors()
        job_def.clear_errors()
        assert job_def.has_errors() is False

    def test_clear_errors_does_not_affect_other_job_def(self, job_def_factory):
        """clear_errors on one JobDef does not remove errors from another."""
        job_def1 = job_def_factory()
        job_def2 = job_def_factory()
        job_def1.set_error(self._make_raised_error("def1 error"))
        job_def2.set_error(self._make_raised_error("def2 error"))
        job_def1.clear_errors()
        assert job_def1.has_errors() is False
        assert job_def2.has_errors() is True

    def test_set_error_after_clear_errors(self, job_def_factory):
        """set_error can be used again after clear_errors."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("old"))
        job_def.clear_errors()
        job_def.set_error(self._make_raised_error("new"))
        errors = list(job_def.get_errors())
        assert len(errors) == 1
        assert errors[0].error_message == "new"


class TestJobDefUpdateWarmupCode:
    """Tests for JobDef.update_warmup_code."""

    def test_update_applies_new_value(self, job_def_factory):
        """warmup_code is updated when the parsed new value differs from current."""
        job_def = job_def_factory(warmup_code=None)
        job_def.update_warmup_code("x = 1")
        assert job_def.warmup_code == "x = 1"

    def test_update_none_clears_existing(self, job_def_factory):
        """Passing None clears an existing warmup_code value."""
        job_def = job_def_factory(warmup_code="x = 1")
        job_def.update_warmup_code(None)
        assert job_def.warmup_code is None

    def test_no_update_when_value_identical(self, job_def_factory):
        """warmup_code is not reassigned when the parsed value equals the current one."""
        job_def = job_def_factory(warmup_code="x = 1")
        job_def.update_warmup_code("x = 1")
        assert job_def.warmup_code == "x = 1"

    def test_no_update_when_none_stays_none(self, job_def_factory):
        """Calling update_warmup_code(None) on a None field does nothing."""
        job_def = job_def_factory(warmup_code=None)
        job_def.update_warmup_code(None)
        assert job_def.warmup_code is None

    def test_update_sequence_to_list(self, job_def_factory):
        """A sequence of two blocks is stored as a two-element sequence."""
        job_def = job_def_factory(warmup_code=None)
        job_def.update_warmup_code(["a = 1", "b = 2"])
        warmup = job_def.warmup_code
        assert len(warmup) == 2
        assert warmup[0] == "a = 1"
        assert warmup[1] == "b = 2"


class TestJobDefChatStyle:
    """Tests for JobDef._chat_style field and chat_style property."""

    def test_chat_style_defaults_to_none(self, job_def_factory):
        """_chat_style is None by default."""
        job_def = job_def_factory()
        assert job_def._chat_style is None  # pylint: disable=protected-access

    def test_chat_style_property_returns_job_level_when_set(self, job_def_factory):
        """chat_style property returns the job-level value when explicitly set."""
        job_def = job_def_factory()
        job_def.set_chat_style(ChatStyle.CONSOLE)  # pylint: disable=no-member
        assert job_def.chat_style == ChatStyle.CONSOLE  # pylint: disable=no-member

    def test_chat_style_property_falls_back_to_settings(self, job_def_factory):
        """chat_style property returns StatekSettings.chat_style when _chat_style is None."""
        job_def = job_def_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings):
            assert job_def.chat_style == ChatStyle.MARKDOWN  # pylint: disable=no-member

    def test_chat_style_property_returns_none_when_both_unset(self, job_def_factory):
        """chat_style property returns None when neither job-level nor settings are set."""
        job_def = job_def_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = None
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings):
            assert job_def.chat_style is None


class TestJobDefLocale:
    """Tests for JobDef.locale field."""

    def test_locale_defaults_to_none(self, job_def_factory):
        """locale is None by default."""
        job_def = job_def_factory()
        assert job_def.locale is None

    def test_locale_set_at_construction(self, job_def_factory):
        """locale can be set at construction time."""

        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        assert job_def.locale is locale


# ---------------------------------------------------------------------------
# UserLogItem
# ---------------------------------------------------------------------------

class TestUserLogItem:
    """Tests for the UserLogItem dataclass."""

    def test_create_with_message(self, db0_fixture):  # pylint: disable=unused-argument
        """UserLogItem can be created with a message."""
        item = UserLogItem(message="hello")
        assert item.message == "hello"

    def test_timestamp_defaults_to_now(self, db0_fixture):  # pylint: disable=unused-argument
        """UserLogItem timestamp defaults to current time."""
        item = UserLogItem(message="test")
        assert item.timestamp is not None

    def test_empty_message(self, db0_fixture):  # pylint: disable=unused-argument
        """UserLogItem allows empty string for message."""
        item = UserLogItem(message="")
        assert item.message == ""


class TestJobCountLLMActions:
    """Tests for Job.count_llm_actions."""

    def test_empty_job_counts_zero_actions(self, job_factory):
        """An empty chat log has no LLM actions."""
        job = job_factory()

        assert job.count_llm_actions() == 0

    def test_direct_user_facing_response_counts_one_action(self, job_factory):
        """A visible DIRECT response counts as one LLM action."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="Final answer"))

        assert job.count_llm_actions() == 1

    def test_empty_direct_response_is_not_counted(self, job_factory):
        """An empty plain response is not a user-facing LLM action."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="  \n"))

        assert job.count_llm_actions() == 0

    def test_md_dialog_internal_response_is_not_counted(self, job_factory):
        """Internal MD_DIALOG text is not a user-facing response action."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="thinking out loud"))

        assert job.count_llm_actions() == 0

    def test_md_dialog_user_facing_response_counts_one_action(self, job_factory):
        """A visible MD_DIALOG response counts as one LLM action."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="# Answer\nDone"))

        assert job.count_llm_actions() == 1

    def test_code_block_counts_code_and_tool_calls(self, job_factory):
        """CodeBlock entries count non-empty code plus each tool call."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(
                code="x = 1",
                tool_calls=[CallSpec(id="call-1", func_name="python_cli")],
            ),
        ))

        assert job.count_llm_actions() == 2

    def test_code_block_without_code_counts_only_tool_calls(self, job_factory):
        """CodeBlock entries with blank code count only their tool calls."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(
                code="  \n",
                tool_calls=[
                    CallSpec(id="call-1", func_name="list_examples"),
                    CallSpec(id="call-2", func_name="show_example"),
                ],
            ),
        ))

        assert job.count_llm_actions() == 2

    def test_mixed_llm_items_sum_individual_actions(self, job_factory):
        """Multiple LLM entries are summed at action granularity."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="First answer"))
        job.chat_log.append(create_chat_log_item(
            console_pos=1,
            llm_resp=CodeBlock(
                code="x = 1",
                tool_calls=[
                    CallSpec(id="call-1", func_name="python_cli"),
                    CallSpec(id="call-2", func_name="render_chart"),
                ],
            ),
        ))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Second answer"))

        assert job.count_llm_actions() == 5

    def test_non_llm_log_items_are_not_counted(self, job_factory):
        """Warmup and framework/user messages do not count as LLM actions."""
        job = job_factory()
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")
        processor = CountActionsPostProcessor()
        job.chat_log.extend([
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            PostProcessedItem(console_pos=1, post_processor=processor, message="Review it."),
            ReminderLogItem(console_pos=2, reminder=RecurringReminder(text="Use the tool.")),
            SubTaskLogItem(console_pos=3, handler=handler),
            UserLogItem(message="user follow-up"),
        ])

        assert job.count_llm_actions() == 0


class TestJobHandleReminder:
    """Tests for Job.handle_reminder."""

    def test_recurring_reminder_without_min_dialog_len_is_ready(self, job_factory):
        """RecurringReminder is ready by default."""
        job = job_factory()
        reminder = RecurringReminder(text="Use report_outcome.")

        assert reminder.fire_ready(job) is True

    def test_recurring_reminder_waits_for_min_dialog_len(self, job_factory):
        """min_dialog_len gates reminders on the dialog length."""
        job = job_factory()
        job.chat_log.append("initial user message")
        reminder = RecurringReminder(text="Use report_outcome.", min_dialog_len=2)

        assert reminder.fire_ready(job) is False

        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(code="print('internal')"),
        ))

        assert reminder.fire_ready(job) is False

        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="response"))

        assert reminder.fire_ready(job) is True

    def test_recurring_reminder_is_processed(self, job_factory):
        """RecurringReminder appends console text and records a ReminderLogItem."""
        job = job_factory()
        job.py_env.console = ["existing"]
        reminder = RecurringReminder(text="Use report_outcome.")

        processed = job.handle_reminder(reminder)

        assert processed is True
        assert job.py_env.console == ["existing", "Use report_outcome."]
        assert len(job.chat_log) == 1
        assert isinstance(job.chat_log[0], ReminderLogItem)
        assert job.chat_log[0].console_pos == 1
        assert job.chat_log[0].reminder is reminder

    def test_unready_reminder_is_skipped(self, job_factory):
        """Unready reminders are skipped without side effects."""
        job = job_factory()
        reminder = RecurringReminder(text="Not yet supported.", min_dialog_len=1)

        processed = job.handle_reminder(reminder)

        assert processed is False
        assert job.py_env.console is None
        assert job.chat_log == []


class TestSubTaskNotifications:
    """Tests for subtask notification log integration."""

    def test_subtask_log_item_yields_system_message_and_fake_tool_result(self, job_factory):
        """Successful subtask notifications are visible as system + synthetic tool history."""
        job = job_factory()
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=handler, tool_log="done"))

        history = list(job.get_next_request()["chat_history"])

        assert history[0].role == ChatRole.SYSTEM
        assert history[0].content == "[Notification] sub-task id=child-1 completed successfully."
        assert history[0].content_src == ContentSource.SYSTEM
        assert history[1].role == ChatRole.ASSISTANT
        assert history[1].tool_calls[0].func_name == "python_cli"
        assert history[1].tool_calls[0].kwargs == {
            "code": "print(find_sub_task_handler(id='child-1'))"
        }
        assert history[2].role == ChatRole.TOOL
        assert history[2].content == "done"
        assert history[2].content_src == ContentSource.CONSOLE

    def test_subtask_log_item_formats_fake_tool_result_for_llm(self, job_factory):
        """Synthetic subtask tool results format with the generated tool call id."""
        job = job_factory()
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=handler, tool_log="done"))

        history = list(job.get_next_request()["chat_history"])
        messages = [format_chat_history_item(item, ChatStyle.DIRECT) for item in history]

        assert messages[1]["tool_calls"][0]["id"] == "STATEK-SUBTASK-000"
        assert messages[2] == {
            "role": "tool",
            "content": "done",
            "tool_call_id": "STATEK-SUBTASK-000",
        }

    def test_subtask_log_item_error_yields_system_message_only(self, job_factory):
        """Errored subtask notifications do not simulate a result lookup."""
        job = job_factory()
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", error="failed")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=handler))

        history = list(job.get_next_request()["chat_history"])

        assert len(history) == 1
        assert history[0].role == ChatRole.SYSTEM
        assert history[0].content == "[Error] sub-task id=child-1 failed with failed"

    def test_find_sub_task_handler_searches_reverse_chat_log(self, job_factory):
        """Most recent handlers are found first unless an id is requested."""
        job = job_factory()
        first = _completed_subtask_handler(job_factory(), subtask_id="first", result="one")
        second = _completed_subtask_handler(job_factory(), subtask_id="second", result="two")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=first))
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=second))

        assert job.find_sub_task_handler() is second
        assert job.find_sub_task_handler(id="first") is first

        with pytest.raises(RuntimeError, match="Sub-task id=missing has not completed"):
            job.find_sub_task_handler(id="missing")

    def test_find_sub_task_handler_without_id_returns_none_on_miss(self, job_factory):
        """Lookup without an id keeps the optional most-recent semantics."""
        job = job_factory()

        assert job.find_sub_task_handler() is None

    def test_find_sub_task_handler_prefers_pending_notifications(self, job_factory):
        """Pending notifications are searched before persisted chat log items."""
        job = job_factory()
        stored = _completed_subtask_handler(job_factory(), subtask_id="same", result="stored")
        pending = _completed_subtask_handler(job_factory(), subtask_id="same", result="pending")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=stored))
        job.set_status(JobStatus.STARTED)

        job.push_notification(pending)

        assert job.find_sub_task_handler(id="same") is pending
        assert job.chat_log[-1].handler is stored

    def test_tool_find_sub_task_handler_forwards_to_current_job(self, job_factory):
        """System tool lookup uses the current job context."""
        job = job_factory()
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")
        job.chat_log.append(SubTaskLogItem(console_pos=0, handler=handler))

        result = _run_with_current_job(
            job,
            lambda: find_sub_task_handler(id="child-1"),
        )

        assert result is handler

    def test_push_notification_requires_completed_handler(self, job_factory):
        """Uncompleted handlers cannot be pushed into parent notifications."""
        job = job_factory()
        handler = SubTaskHandler(job=job_factory(), id="child-1")

        with pytest.raises(RuntimeError, match="completed"):
            job.push_notification(handler)

    def test_push_notification_done_appends_and_restarts_parent(self, job_factory):
        """DONE parents receive completed notifications directly and restart."""
        job = job_factory()
        job.set_status(JobStatus.DONE)
        job.py_env.exit_status = "finished"
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")

        job.push_notification(handler)

        assert job.status == JobStatus.STARTED
        assert job.py_env.exit_status is None
        assert isinstance(job.chat_log[-1], SubTaskLogItem)
        assert job.chat_log[-1].handler is handler
        assert job.chat_log[-1].tool_log == "done"

    def test_push_notification_active_buffers_until_last_item_finalized(self, job_factory):
        """Active jobs keep notifications pending until tool-call log state is finalized."""
        job = job_factory()
        job.set_status(JobStatus.STARTED)
        job.chat_log.append(LLM_LogItem(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[CallSpec(id="c1", func_name="python_cli")]),
        ))
        handler = _completed_subtask_handler(job_factory(), subtask_id="child-1", result="done")

        job.push_notification(handler)

        assert len(job.chat_log) == 1
        assert job.find_sub_task_handler(id="child-1") is handler
        assert job._process_pending_notifications() is False  # pylint: disable=protected-access

        job.chat_log[-1].tool_log = [""]

        assert job._process_pending_notifications() is True  # pylint: disable=protected-access
        assert isinstance(job.chat_log[-1], SubTaskLogItem)
        assert job.chat_log[-1].handler is handler
        assert job._process_pending_notifications() is False  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# get_next_request with user messages (str / UserLogItem)
# ---------------------------------------------------------------------------

class TestGetNextRequestUserMessages:
    """Tests for chat_history surfacing user messages from chat_log."""

    def test_str_in_chat_log_yields_initial_user_item(self, job_factory):
        """A leading str entry in chat_log becomes the initial USER item."""
        job = job_factory()
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append("initial user message")
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))

        history = list(job.get_next_request()["chat_history"])

        # [USER(initial), ASSISTANT(resp), USER(console)]
        assert history[0].role == ChatRole.USER
        assert history[0].content == "initial user message"
        assert history[0].content_src == ContentSource.USER

    def test_str_in_chat_log_yields_initial_user_item_with_hint(self, job_def_factory):
        """Leading chat_log str in request history gets the language hint."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append("initial user message")
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))

        history = list(job.get_next_request()["chat_history"])

        assert history[0].role == ChatRole.USER
        assert "initial user message (PAMIĘTAJ:" in history[0].content

    def test_user_log_item_in_chat_log_yields_user_item(self, job_factory):
        """A UserLogItem in chat_log is yielded as a USER ChatHistoryItem."""
        job = job_factory()
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))
        job.chat_log.append(UserLogItem(message="user follow-up"))

        history = list(job.get_next_request()["chat_history"])

        assert history[-1].role == ChatRole.USER
        assert history[-1].content == "user follow-up"
        assert history[-1].content_src == ContentSource.USER

    def test_reminder_log_item_in_chat_log_yields_system_item(self, job_factory):
        """A ReminderLogItem is yielded as an injected SYSTEM ChatHistoryItem."""
        job = job_factory()
        reminder = RecurringReminder(text="Use report_outcome before finishing.")
        job.chat_log.append(ReminderLogItem(console_pos=0, reminder=reminder))

        history = list(job.get_next_request()["chat_history"])

        assert len(history) == 1
        assert history[0].role == ChatRole.SYSTEM
        assert history[0].content == "Use report_outcome before finishing."
        assert history[0].content_src == ContentSource.SYSTEM

    def test_reminder_log_item_formats_as_system_message(self, job_factory):
        """Reminder content is sent to OpenAI-compatible LLMs as role=system."""
        job = job_factory()
        reminder = RecurringReminder(text="Use report_outcome before finishing.")
        job.chat_log.append(ReminderLogItem(console_pos=0, reminder=reminder))
        history = list(job.get_next_request()["chat_history"])
        settings = types.SimpleNamespace(get_xml_box_tags=lambda: {"console": "out"})

        formatted = format_chat_history_item(history[0], ChatStyle.MARKDOWN, settings)

        assert formatted == {
            "role": "system",
            "content": "Use report_outcome before finishing.",
        }

    def test_user_log_item_in_chat_log_yields_user_item_with_hint(self, job_def_factory):
        """UserLogItem follow-ups get the language hint in chat history."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))
        job.chat_log.append(UserLogItem(message="user follow-up"))

        history = list(job.get_chat_history())

        assert history[-1].role == ChatRole.USER
        assert "user follow-up (PAMIĘTAJ:" in history[-1].content
        assert history[-1].content_src == ContentSource.USER

    def test_multiple_user_log_items_after_llm(self, job_factory):
        """Multiple UserLogItem follow-ups after LLM turns are all yielded."""
        job = job_factory()
        job.py_env.console = ["Out1"]
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="R1"))
        job.chat_log.append(UserLogItem(message="msg1"))
        job.chat_log.append(UserLogItem(message="msg2"))

        history = list(job.get_next_request()["chat_history"])

        user_items = [
            h for h in history
            if h.role == ChatRole.USER and h.content_src == ContentSource.USER
        ]
        assert len(user_items) == 2
        assert user_items[0].content == "msg1"
        assert user_items[1].content == "msg2"

    def test_user_messages_preserve_normal_history(self, job_factory):
        """Assistant history is preserved when user messages exist."""
        job = job_factory()
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))
        job.chat_log.append(UserLogItem(message="user msg"))

        history = list(job.get_next_request()["chat_history"])

        asst_items = [h for h in history if h.role == ChatRole.ASSISTANT]
        assert len(asst_items) == 1
        assert asst_items[0].content == "Response 1"

    def test_user_log_item_in_request_history_gets_hint(self, job_def_factory):
        """UserLogItem follow-ups get the language hint in request history."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        job_def = job_def_factory(locale=locale)
        job = Job(
            job_def=job_def, model_family="test",
            model="test-model", job_status=JobStatus.READY,  # pylint: disable=no-member
        )
        job.py_env.console = ["Out1", "Out2"]
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="Response 1"))
        job.chat_log.append(UserLogItem(message="user follow-up"))

        history = list(job.get_next_request()["chat_history"])
        user_items = [
            h for h in history
            if h.role == ChatRole.USER and h.content_src == ContentSource.USER
        ]

        assert any("user follow-up (PAMIĘTAJ:" in h.content for h in user_items)

    def test_empty_str_in_chat_log_skipped(self, job_factory):
        """An empty leading string in chat_log produces no USER item."""
        job = job_factory()
        job.chat_log.append("")

        history = list(job.get_next_request()["chat_history"])

        # Only the SYSTEM item — empty string contributes nothing.
        user_items = [
            h for h in history
            if h.role == ChatRole.USER and h.content_src == ContentSource.USER
        ]
        assert user_items == []

    def test_last_response_none_when_last_is_user_log_item(
            self, job_factory):
        """last_response returns None when last chat_log item is UserLogItem."""
        job = job_factory()
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="code"))
        job.chat_log.append(UserLogItem(message="follow-up"))
        assert job.last_response is None

    def test_last_response_none_when_last_is_str(self, job_factory):
        """last_response returns None when last chat_log item is str."""
        job = job_factory()
        job.chat_log.append(
            create_chat_log_item(console_pos=0, llm_resp="code"))
        job.chat_log.append("follow-up")
        assert job.last_response is None


class TestGetResponseTimes:
    """Tests for Job.get_response_times."""

    def test_initial_and_follow_up_messages_measure_to_first_llm_response(
        self, job_factory
    ):
        """Each user message maps to the first later LLM response."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=5)
        t2 = t1 + timedelta(seconds=7)
        t3 = t2 + timedelta(seconds=11)

        job.created_at = t0
        job.chat_log.append("initial msg")
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp1"))
        job.chat_log[-1].timestamp = t1
        job.chat_log.append(UserLogItem(message="follow-up", timestamp=t2))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="resp2"))
        job.chat_log[-1].timestamp = t3

        assert list(job.get_response_times()) == [
            (t0, 5.0),
            (t2, 11.0),
        ]

    def test_multiple_user_messages_share_same_first_response(self, job_factory):
        """Multiple queued user messages before one reply use that same first reply."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=7)

        job.chat_log.append(UserLogItem(message="msg1", timestamp=t0))
        job.chat_log.append(UserLogItem(message="msg2", timestamp=t1))
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp"))
        job.chat_log[-1].timestamp = t2

        assert list(job.get_response_times()) == [
            (t0, 10.0),
            (t1, 7.0),
        ]

    def test_unanswered_user_message_returns_none(self, job_factory):
        """User message without a later LLM response yields None."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        job.chat_log.append(UserLogItem(message="waiting", timestamp=t0))

        assert list(job.get_response_times()) == [(t0, None)]

    def test_warmup_items_do_not_count_as_responses(self, job_factory):
        """Warmup log items are ignored when computing response times."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=4)
        t2 = t1 + timedelta(seconds=6)

        job.chat_log.append(UserLogItem(message="msg", timestamp=t0))
        job.chat_log.append(WarmupLogItem(console_pos=0, warmup_block_num=0, timestamp=t1))
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp"))
        job.chat_log[-1].timestamp = t2

        assert list(job.get_response_times()) == [(t0, 10.0)]

    def test_direct_tool_call_turn_does_not_count_as_first_response(self, job_factory):
        """DIRECT latency should wait for the first plain dialog reply."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)  # pylint: disable=no-member
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=7)

        job.chat_log.append(UserLogItem(message="msg", timestamp=t0))
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[CallSpec(id="c1", func_name="python_cli")]),
        ))
        job.chat_log[-1].timestamp = t1
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="done"))
        job.chat_log[-1].timestamp = t2

        assert list(job.get_response_times()) == [(t0, 10.0)]

    def test_md_dialog_script_turn_does_not_count_as_first_response(self, job_factory):
        """MD_DIALOG latency should skip a pure script turn before dialog text."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)  # pylint: disable=no-member
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=4)
        t2 = t1 + timedelta(seconds=6)

        job.chat_log.append(UserLogItem(message="msg", timestamp=t0))
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 1"))
        job.chat_log[-1].timestamp = t1
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="# Done."))
        job.chat_log[-1].timestamp = t2

        assert list(job.get_response_times()) == [(t0, 10.0)]


class TestGetChatResponses:
    """Tests for Job.get_chat_responses."""

    def test_yields_llm_string_responses_in_order(self, job_factory):
        """Multiple LLM string responses are yielded oldest first."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="first"))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="second"))
        assert list(job.get_chat_responses()) == ["first", "second"]

    def test_direct_style_tool_call_response_is_not_yielded(self, job_factory):
        """DIRECT style: CodeBlock responses are not yielded, only plain text."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[CallSpec(id="c1", func_name="python_cli")]),
        ))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="done"))
        assert list(job.get_chat_responses()) == ["done"]

    def test_md_dialog_script_response_is_not_yielded(self, job_factory):
        """MD_DIALOG style: responses without markdown headers are not yielded."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 1"))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="# Answer"))
        assert list(job.get_chat_responses()) == ["# Answer"]

    def test_answer_tool_log_console_output_is_yielded(self, job_factory):
        """Console answer tool logs expose only their message body."""
        job = job_factory()
        job.py_env.console = [
            "log: answer(body='Brak dyzurow w przyszlym tygodniu.', media=None)",
        ]
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="answer(...)"))

        assert list(job.get_chat_responses()) == [
            "answer(...)",
            "Brak dyzurow w przyszlym tygodniu.",
        ]

    def test_answer_tool_log_is_ordered_between_llm_turns(self, job_factory):
        """Answer logs are yielded at their console position between LLM turns."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)
        job.py_env.console = [
            "log: answer(body='from first turn', media=None)",
            "ordinary console output",
            "log: answer(body='from second turn', media=None)",
        ]
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[CallSpec(id="c1", func_name="python_cli")]),
        ))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="plain reply"))

        assert list(job.get_chat_responses()) == [
            "from first turn",
            "plain reply",
            "from second turn",
        ]

    def test_non_answer_tool_log_console_output_is_not_yielded(self, job_factory):
        """Only answer tool logs are treated as chat responses."""
        job = job_factory()
        job.py_env.console = [
            "log: search('alice', limit=5)",
            "log: answer(body='done', media=None)",
        ]
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="internal"))

        assert list(job.get_chat_responses()) == ["internal", "done"]


class TestGetDialog:
    """Tests for Job.get_dialog."""

    def test_yields_user_and_assistant_messages_in_order(self, job_factory):
        """User messages and visible LLM responses are yielded chronologically."""
        job = job_factory()
        job.chat_log.append("hello")
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="# Hi"))
        job.chat_log.append(UserLogItem(message="follow-up"))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="# Done"))

        assert list(job.get_dialog()) == [
            DialogItem("user", "hello"),
            DialogItem("assistant", "# Hi"),
            DialogItem("user", "follow-up"),
            DialogItem("assistant", "# Done"),
        ]

    def test_filters_internal_turns_and_includes_answer_tool_output(self, job_factory):
        """Only exchanged dialog is yielded, including assistant answer tool bodies."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.DIRECT)
        job.chat_log.append("question")
        job.py_env.console = [
            "log: search('alice')",
            "log: answer(body='tool answer', media=None)",
        ]
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(tool_calls=[CallSpec(id="c1", func_name="python_cli")]),
        ))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="plain reply"))

        assert list(job.get_dialog()) == [
            DialogItem("user", "question"),
            DialogItem("assistant", "tool answer"),
            DialogItem("assistant", "plain reply"),
        ]

    def test_plain_code_block_code_is_not_dialog(self, job_factory):
        """Executable CodeBlock code is not yielded as assistant dialog."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(code="print('internal')"),
        ))

        assert not list(job.get_dialog())

    def test_md_dialog_code_block_extracts_only_dialog_text(self, job_factory):
        """Raw MD_DIALOG CodeBlock content yields only text outside fences."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(code="# Visible\n```python\nprint('internal')\n```"),
        ))

        assert list(job.get_dialog()) == [
            DialogItem("assistant", "# Visible"),
        ]

    def test_md_dialog_unfenced_code_block_text_is_dialog(self, job_factory):
        """Unfenced MD_DIALOG CodeBlock text is treated as assistant dialog."""
        job = job_factory()
        job.job_def.set_chat_style(ChatStyle.MD_DIALOG)
        job.chat_log.append(create_chat_log_item(
            console_pos=0,
            llm_resp=CodeBlock(code="# Visible"),
        ))

        assert list(job.get_dialog()) == [
            DialogItem("assistant", "# Visible"),
        ]


class TestGetLlmResponseTimes:
    """Tests for Job.get_llm_response_times."""

    def test_first_llm_response_uses_job_creation_time_as_reference(self, job_factory):
        """The first LLM response is measured from the job creation timestamp."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=9)

        job.created_at = t0
        first = create_chat_log_item(console_pos=0, llm_resp="resp1")
        first.timestamp = t1
        job.chat_log.append(first)

        assert list(job.get_llm_response_times()) == [(t0, 9.0)]

    def test_consecutive_llm_items_measure_inter_request_latency(self, job_factory):
        """Consecutive concrete LLM items define approximate request timings."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=9)
        t2 = t1 + timedelta(seconds=4)

        first = create_chat_log_item(console_pos=0, llm_resp="resp1")
        first.timestamp = t0
        second = create_chat_log_item(console_pos=1, llm_resp="resp2")
        second.timestamp = t1
        third = create_chat_log_item(console_pos=2, llm_resp="resp3")
        third.timestamp = t2
        job.chat_log.extend([first, second, third])

        assert list(job.get_llm_response_times()) == [
            (t0, 9.0),
            (t1, 4.0),
        ]

    def test_created_at_after_first_response_is_ignored(self, job_factory):
        """A later created_at timestamp must not yield negative latency."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=9)
        t_created = t1 + timedelta(seconds=1)

        job.created_at = t_created
        first = create_chat_log_item(console_pos=0, llm_resp="resp1")
        first.timestamp = t0
        second = create_chat_log_item(console_pos=1, llm_resp="resp2")
        second.timestamp = t1
        job.chat_log.extend([first, second])

        assert list(job.get_llm_response_times()) == [(t0, 9.0)]

    def test_pending_llm_marker_takes_precedence_over_previous_response(self, job_factory):
        """Awaiting-response markers provide the clean request timestamp."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=12)
        t2 = t1 + timedelta(seconds=5)

        first = create_chat_log_item(console_pos=0, llm_resp="resp1")
        first.timestamp = t0
        pending = create_chat_log_item(console_pos=1, llm_resp=None)
        pending.timestamp = t1
        second = create_chat_log_item(console_pos=1, llm_resp="resp2")
        second.timestamp = t2
        job.chat_log.extend([first, pending, second])

        assert list(job.get_llm_response_times()) == [(t1, 5.0)]

    def test_pending_llm_marker_without_response_returns_none(self, job_factory):
        """Outstanding LLM requests are returned with unknown duration."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        pending = create_chat_log_item(console_pos=0, llm_resp=None)
        pending.timestamp = t0
        job.chat_log.append(pending)

        assert list(job.get_llm_response_times()) == [(t0, None)]

    def test_non_llm_items_are_ignored(self, job_factory):
        """User and warmup entries do not affect the extracted LLM timings."""
        job = job_factory()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = t0 + timedelta(seconds=3)
        t2 = t1 + timedelta(seconds=7)
        t3 = t2 + timedelta(seconds=5)

        first = create_chat_log_item(console_pos=0, llm_resp="resp1")
        first.timestamp = t0
        pending = create_chat_log_item(console_pos=1, llm_resp=None)
        pending.timestamp = t2
        second = create_chat_log_item(console_pos=1, llm_resp="resp2")
        second.timestamp = t3
        job.chat_log.extend([
            first,
            UserLogItem(message="follow-up", timestamp=t1),
            WarmupLogItem(console_pos=0, warmup_block_num=0, timestamp=t2),
            pending,
            second,
        ])

        assert list(job.get_llm_response_times()) == [(t2, 5.0)]


class TestGetLlmStepSizes:
    """Tests for Job.get_llm_initial_step_size and Job.get_llm_step_sizes."""

    def test_initial_step_size_covers_system_prompt_and_warmup(self, job_factory):
        """get_llm_initial_step_size counts system prompt and warmup code tokens."""
        warmup = "x = 1"
        job = job_factory(warmup_code=warmup)
        system_prompt = job.system_prompt()
        input_tokens, output_tokens = job.get_llm_initial_step_size()
        assert output_tokens == 0
        assert input_tokens == (len(system_prompt) + len(warmup)) // 4

    def test_first_step_includes_initial_step_size(self, job_factory):
        """First step input tokens include the initial step size (prompt + warmup)."""
        warmup = "warmup_code = True"
        job = job_factory(warmup_code=warmup)
        item = create_chat_log_item(console_pos=0, llm_resp="resp")
        job.chat_log.append(item)
        initial_input, _ = job.get_llm_initial_step_size()
        input_tokens, _ = list(job.get_llm_step_sizes())[0]
        assert input_tokens == initial_input

    def test_console_and_user_message_counted_in_next_step_input(self, job_factory):
        """Console output and user messages between steps count in next step's input."""
        job = job_factory()
        job.py_env.console = ["console_line"]
        msg = "follow-up"
        item1 = create_chat_log_item(console_pos=0, llm_resp="resp1")
        user_item = UserLogItem(message=msg, timestamp=datetime.now())
        item2 = create_chat_log_item(console_pos=1, llm_resp="resp2")
        job.chat_log.extend([item1, user_item, item2])
        steps = list(job.get_llm_step_sizes())
        input2, _ = steps[1]
        assert input2 == (len("console_line") + len(msg)) // 4

    def test_code_block_tool_calls_counted_in_output_tokens(self, job_factory):
        """Tool call requests in a CodeBlock response contribute to output tokens."""
        job = job_factory()
        cs = CallSpec(id="c1", func_name="my_tool", args=None, kwargs={"x": 1})
        cb = CodeBlock(code="run = True", tool_calls=[cs])
        item = create_chat_log_item(console_pos=0, llm_resp=cb)
        job.chat_log.append(item)
        _, output_tokens = list(job.get_llm_step_sizes())[0]
        tool_json = json.dumps([{"name": "my_tool", "arguments": {"x": 1}}])
        assert output_tokens == (len("run = True") + len(tool_json)) // 4


class TestTokensPerSec:
    """Tests for Job.tokens_per_sec."""

    def test_divides_total_tokens_by_total_duration(self, job_factory):
        """tokens_per_sec = sum(input+output) / sum(durations) across all valid steps."""
        job = job_factory()
        with patch.object(Job, 'get_llm_response_times', return_value=[(None, 10.0)]):
            with patch.object(Job, 'get_llm_step_sizes', return_value=[(40, 20)]):
                assert job.tokens_per_sec() == (40 + 20) / 10.0

    def test_steps_with_none_duration_are_excluded(self, job_factory):
        """Steps whose duration is None do not contribute to either totals."""
        job = job_factory()
        with patch.object(Job, 'get_llm_response_times', return_value=[(None, 4.0), (None, None)]):
            with patch.object(Job, 'get_llm_step_sizes', return_value=[(100, 60), (200, 80)]):
                assert job.tokens_per_sec() == 160.0 / 4.0

    def test_no_valid_steps_returns_zero(self, job_factory):
        """Returns 0.0 when there are no completed steps to measure."""
        job = job_factory()
        with patch.object(Job, 'get_llm_response_times', return_value=[(None, None)]):
            with patch.object(Job, 'get_llm_step_sizes', return_value=[(50, 30)]):
                assert job.tokens_per_sec() == 0.0


class TestJobCreatedAt:
    """Tests for Job.created_at."""

    def test_created_at_defaults_on_job_creation(self, job_factory):
        """New jobs record a creation timestamp."""
        job = job_factory()

        assert job.created_at is not None
