"""Tests for Agent class."""

# pylint: disable=no-member

import os
from pathlib import Path
from unittest.mock import patch
import pytest
from statek.agents.agent import Agent, SupervisedAgent, WarmupDef, update_warmup_defs
from statek.locale import StatekLocale, StatekLangCode, StatekCountryCode
from statek.prompt_config import SystemPrompt, make_system_prompt, parse_system_prompt
from statek.system import tool
from statek.task_difficulty import TaskDifficulty
from statek.utils import CodeBlock
from tests.conftest import clock, docstr, exit_tool


def _internal_tool(**kwargs):  # pylint: disable=unused-argument
    """An internal tool that should not appear in the system prompt."""


@tool(hidden=True)
def hidden_agent_tool(value: str = "ok", **kwargs) -> str:  # pylint: disable=unused-argument
    """A hidden tool that should not appear in the system prompt.

    Args:
        value: Value to return.
    """
    return value


def _system_prompt(agent, task_difficulty=TaskDifficulty.medium, **kwargs):
    """Format an agent prompt with the default test difficulty."""
    return agent.system_prompt(task_difficulty=task_difficulty, **kwargs)


class TestAgent:  # pylint: disable=too-many-public-methods
    """Test cases for Agent class."""

    def test_system_prompt_formatting_single_tool(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with a single tool."""
        system_prompt = "Available tools:\n{tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt(system_prompt),

            _tools=tools
        )

        assert "clock()" in _system_prompt(agent)
        assert "Get the current time." in _system_prompt(agent)
        assert "Available tools:\n" in _system_prompt(agent)

    def test_system_prompt_formatting_multiple_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with multiple tools."""
        system_prompt = "You have access to these tools:\n{tools}"
        tools = [clock, docstr, exit_tool]

        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt(system_prompt),

            _tools=tools
        )

        assert "clock()" in _system_prompt(agent)
        assert "docstr(class_name, method_name" in _system_prompt(agent)
        assert "exit_tool(reason)" in _system_prompt(agent)
        assert "You have access to these tools:" in _system_prompt(agent)

    def test_system_prompt_formatting_no_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with empty explicit tools list"""
        system_prompt = "Available tools: {tools}"
        tools = []

        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt(system_prompt),

            _tools=tools
        )

        assert "Available tools:" in _system_prompt(agent)
        # System tools are in the registry, not on the agent
        assert "list_of_examples" not in _system_prompt(agent)
        assert "show_example" not in _system_prompt(agent)

    def test_system_prompt_with_block_comment(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt with block comment placeholder."""
        system_prompt = "# --- TOOLS ---\n# {tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt(system_prompt),

            _tools=tools
        )

        # Each line should be prefixed with #
        assert "# clock()" in _system_prompt(agent)
        assert "#     Get the current time." in _system_prompt(agent)

    def test_system_prompt_detailed_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt with detailed_tools placeholder."""
        system_prompt = "Tools:\n{detailed_tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt(system_prompt),

            _tools=tools
        )

        # detailed_tools uses py_syntax=True
        assert "def clock()" in _system_prompt(agent)
        assert '"""Get the current time.' in _system_prompt(agent)

    def test_format_tools_passes_agent_name(self, db0_fixture):  # pylint: disable=unused-argument
        """_format_tools passes the agent's type name to format_docstring."""
        agent = Agent(role="test", _system_prompt=make_system_prompt("{tools}"), _tools=[clock])

        with patch('statek.agents.agent.format_docstring', wraps=__import__(
                'statek.docstring', fromlist=['format_docstring']).format_docstring) as mock_fmt:
            agent._format_tools(brief=True, py_syntax=False)  # pylint: disable=protected-access

        assert mock_fmt.called
        _, call_kwargs = mock_fmt.call_args
        assert call_kwargs.get('agent') == 'Agent'

    def test_internal_tool_excluded_from_system_prompt(self, db0_fixture):  # pylint: disable=unused-argument
        """Tools with names starting with '_' are not reported in the system prompt."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{tools}"),
            _tools=[_internal_tool],
        )

        assert "_internal_tool" not in _system_prompt(agent)

    def test_internal_tool_included_in_all_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Tools with names starting with '_' are available in all_tools (execution context)."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{tools}"),
            _tools=[_internal_tool],
        )

        assert _internal_tool in agent.all_tools

    def test_hidden_tool_excluded_from_system_prompt(self, db0_fixture):  # pylint: disable=unused-argument
        """Hidden tools are not reported in the default tools placeholder."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{tools}"),
            _tools=[hidden_agent_tool],
        )

        assert "hidden_agent_tool" not in _system_prompt(agent)

    def test_hidden_tool_excluded_from_brief_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Hidden tools are not reported in the brief tools placeholder."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{brief_tools}"),
            _tools=[hidden_agent_tool],
        )

        assert "hidden_agent_tool" not in _system_prompt(agent)

    def test_hidden_tool_excluded_from_detailed_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Hidden tools are not reported in the detailed tools placeholder."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{detailed_tools}"),
            _tools=[hidden_agent_tool],
        )

        assert "hidden_agent_tool" not in _system_prompt(agent)

    def test_hidden_tool_included_in_all_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Hidden tools remain available for execution setup."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("{tools}"),
            _tools=[hidden_agent_tool],
        )

        assert hidden_agent_tool in agent.all_tools

    def test_python_cli_in_system_registry(self, db0_fixture):  # pylint: disable=unused-argument
        """python_cli is a global system tool available via find_tools."""
        from statek.system import find_tools  # pylint: disable=import-outside-toplevel
        tool_names = [t.__name__ for t in find_tools("SYSTEM")]
        assert "python_cli" in tool_names

    def test_description_default_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Agent description defaults to None."""
        agent = Agent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        assert agent.description is None

    def test_description_set_at_init(self, db0_fixture):  # pylint: disable=unused-argument
        """Agent description can be set at initialization."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
            _description="A test agent",
        )
        assert agent.description == "A test agent"

    def test_update_description_sets_value(self, db0_fixture):  # pylint: disable=unused-argument
        """update_description sets description when previously None."""
        agent = Agent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        result = agent.update_description("New description")
        assert result is True
        assert agent.description == "New description"

    def test_update_description_no_change(self, db0_fixture):  # pylint: disable=unused-argument
        """update_description returns False when value hasn't changed."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
            _description="Same",
        )
        result = agent.update_description("Same")
        assert result is False

    def test_update_description_changed_value(self, db0_fixture):  # pylint: disable=unused-argument
        """update_description returns True and updates when value changes."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
            _description="Old",
        )
        result = agent.update_description("New")
        assert result is True
        assert agent.description == "New"

    def test_update_metadata_no_change_keeps_existing_dict(self, db0_fixture):  # pylint: disable=unused-argument
        """update_metadata avoids replacing metadata when the content is unchanged."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
            _metadata={"MODEL": "test-model", "TEMPERATURE": "0.1"},
        )
        original_metadata = agent._metadata  # pylint: disable=protected-access

        result = agent.update_metadata({"MODEL": "test-model", "TEMPERATURE": "0.1"})

        assert result is False
        assert agent._metadata is original_metadata  # pylint: disable=protected-access

    def test_shared_var_names_resolved_from_job_params(self, db0_fixture):  # pylint: disable=unused-argument
        """shared_var_names placeholder is resolved to comma-delimited shared_vars keys."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("Available variables: {shared_var_names}"),
            _tools=[]
        )
        result = _system_prompt(agent, job_params={"shared_vars": ["user", "message"]})
        assert result == "Available variables: user, message"

    def test_shared_var_names_empty_when_no_shared_vars(self, db0_fixture):  # pylint: disable=unused-argument
        """shared_var_names resolves to empty string when shared_vars not in job_params."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("Variables: {shared_var_names}"),
            _tools=[]
        )
        result = _system_prompt(agent, job_params={"other": "value"})
        assert result == "Variables: "

    def test_shared_var_names_empty_when_no_job_params(self, db0_fixture):  # pylint: disable=unused-argument
        """shared_var_names resolves to empty string when job_params is None."""
        agent = Agent(
            role="test",
            _system_prompt=make_system_prompt("Variables: {shared_var_names}"),
            _tools=[]
        )
        result = _system_prompt(agent)
        assert result == "Variables: "


def test_system_prompt_stored_as_persistent_prompt(db0_fixture):  # pylint: disable=unused-argument
    """Raw prompt strings are parsed into persistent SystemPrompt objects."""
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("Intro\n\n--- high: Details ---\nHigh only."),
        _tools=[],
    )

    assert isinstance(agent._system_prompt, SystemPrompt)  # pylint: disable=protected-access
    assert agent._system_prompt.intro == "Intro"  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("context_key", "expected_found"),
    [
        ("message_adapter", True),
        ("_message_adapter", True),
        ("__message_adapter", True),
        ("other_adapter", False),
    ],
)
def test_get_adapter_resolves_context_key_variants(
        db0_fixture, context_key, expected_found):  # pylint: disable=unused-argument
    """get_adapter resolves exact, single-underscore, and double-underscore context keys."""
    adapter = object()
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("test"),
        _tools=[],
        _X__context={context_key: adapter},
    )

    result = agent.get_adapter("message_adapter")

    if expected_found:
        assert result is adapter
    else:
        assert result is None


def test_update_system_prompt_uses_structural_compare(db0_fixture):  # pylint: disable=unused-argument
    """Equivalent parsed prompts do not replace the stored SystemPrompt."""
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt("--- Identity ---\nSame."),
        _tools=[],
    )
    original_prompt = agent._system_prompt  # pylint: disable=protected-access

    result = agent.update_system_prompt(parse_system_prompt("## Identity\nSame."))

    assert result is False
    assert agent._system_prompt is original_prompt  # pylint: disable=protected-access


def test_system_prompt_filters_sections_by_task_difficulty(db0_fixture):  # pylint: disable=unused-argument
    """Prompt sections are formatted for the supplied task difficulty."""
    agent = Agent(
        role="test",
        _system_prompt=make_system_prompt(
            "Intro.\n\n"
            "--- low: Scope ---\nLow instructions.\n\n"
            "--- high: Scope ---\nHigh instructions."
        ),
        _tools=[],
    )

    result = agent.system_prompt(task_difficulty=TaskDifficulty.low)

    assert "Intro." in result
    assert "Low instructions." in result
    assert "High instructions." not in result


class TestSupervisedAgentCustomRole:
    """Test cases for SupervisedAgent with custom role support."""

    def test_supervised_agent_default_role(self, db0_fixture):  # pylint: disable=unused-argument
        """Test SupervisedAgent stores provided role."""
        agent = SupervisedAgent(
            role="test_role",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        assert agent.role == "test_role"

    def test_supervised_agent_role_in_job_def(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that custom role propagates to job definitions."""
        agent = SupervisedAgent(
            role="custom_worker",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "test-model"},
            _tools=[],
        )
        job_def = agent.create_job_def()
        assert job_def.agent.role == "custom_worker"


class TestWarmupDef:  # pylint: disable=too-many-public-methods
    """Test cases for WarmupDef and SupervisedAgent.update_warmup_def."""

    def test_warmup_def_creation_with_string(self, db0_fixture):  # pylint: disable=unused-argument
        """Test WarmupDef can be created with a simple string."""
        wd = WarmupDef(warmup_code="print('hello')")
        assert wd.warmup_code == "print('hello')"

    def test_warmup_def_creation_with_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Test WarmupDef can be created with None."""
        wd = WarmupDef(warmup_code=None)
        assert wd.warmup_code is None

    def test_warmup_def_creation_with_sequence(self, db0_fixture):  # pylint: disable=unused-argument
        """Test WarmupDef can be created with a sequence of strings."""
        wd = WarmupDef(warmup_code=["block1", "block2"])
        assert list(wd.warmup_code) == ["block1", "block2"]

    def test_warmup_def_metadata_defaults_to_none(self, db0_fixture):  # pylint: disable=unused-argument
        """WarmupDef metadata is optional."""
        wd = WarmupDef(warmup_code="print('hello')")
        assert wd.metadata is None

    def test_warmup_def_hidden_defaults_false(self, db0_fixture):  # pylint: disable=unused-argument
        """WarmupDef.hidden is False without explicit hidden metadata."""
        wd = WarmupDef(warmup_code="print('hello')")
        assert wd.hidden is False

    def test_warmup_def_hidden_true_from_metadata(self, db0_fixture):  # pylint: disable=unused-argument
        """WarmupDef.hidden is True only for hidden=True metadata."""
        wd = WarmupDef(warmup_code="print('hello')", metadata={"hidden": True})
        assert wd.hidden is True

    @pytest.mark.parametrize("metadata", [{"hidden": False}, {"hidden": "True"}, {"other": True}])
    def test_warmup_def_hidden_false_without_explicit_true(
            self, db0_fixture, metadata):  # pylint: disable=unused-argument
        """WarmupDef.hidden ignores non-True hidden metadata."""
        wd = WarmupDef(warmup_code="print('hello')", metadata=metadata)
        assert wd.hidden is False

    def test_supervised_agent_warmup_def_default_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Test SupervisedAgent has warmup_def defaulting to None."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        assert agent.warmup_def is None

    def test_update_warmup_def_sets_value(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def sets warmup_def when previously None."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        result = agent.update_warmup_def("print('hello')")
        assert result is True
        assert agent.warmup_def is not None
        assert agent.warmup_def.warmup_code == "print('hello')"
        assert agent.warmup_def.metadata is None

    def test_update_warmup_def_no_change(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def returns False when value hasn't changed."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("print('hello')")
        result = agent.update_warmup_def("print('hello')")
        assert result is False

    def test_update_warmup_def_changed_value(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def returns True and updates when value changes."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("print('hello')")
        result = agent.update_warmup_def("print('world')")
        assert result is True
        assert agent.warmup_def.warmup_code == "print('world')"

    def test_update_warmup_def_parses_code(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def parses warmup code through parse_warmup_code."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        # Multi-block warmup separated by dashes
        multi_block = "block1\n# ----------\nblock2"
        result = agent.update_warmup_def(multi_block)
        assert result is True
        assert agent.warmup_def is not None
        warmup = agent.warmup_def.warmup_code
        # Should be parsed into multiple blocks
        blocks = list(warmup)
        assert len(blocks) == 2

    def test_update_warmup_def_stores_metadata(self, db0_fixture):  # pylint: disable=unused-argument
        """update_warmup_def stores metadata parsed from warmup comments."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])

        result = agent.update_warmup_def(
            "#STATEK: hidden = True\n#STATEK: label = 'startup'\nprint('ready')"
        )

        assert result is True
        assert isinstance(agent.warmup_def.warmup_code, CodeBlock)
        assert agent.warmup_def.warmup_code.code == "print('ready')"
        assert agent.warmup_def.warmup_code.metadata == {"hidden": True, "label": "startup"}
        assert agent.warmup_def.metadata == {"hidden": True, "label": "startup"}
        assert agent.warmup_def.hidden is True

    def test_update_warmup_def_metadata_only_block_preserves_empty_code(
            self, db0_fixture):  # pylint: disable=unused-argument
        """A metadata-only warmup block stores metadata with empty parsed code."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])

        result = agent.update_warmup_def("#STATEK: hidden = True")

        assert result is True
        assert isinstance(agent.warmup_def.warmup_code, CodeBlock)
        assert agent.warmup_def.warmup_code.code == ""
        assert agent.warmup_def.warmup_code.metadata == {"hidden": True}
        assert agent.warmup_def.metadata == {"hidden": True}
        assert agent.warmup_def.hidden is True

    def test_update_warmup_def_aggregates_metadata_from_blocks(
            self, db0_fixture):  # pylint: disable=unused-argument
        """update_warmup_def aggregates metadata from each parsed block."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])

        agent.update_warmup_def(
            "#STATEK: hidden = True\n"
            "block1\n"
            "# ----------\n"
            "#STATEK: label = 'second'\n"
            "block2"
        )

        assert agent.warmup_def.metadata == {"hidden": True, "label": "second"}

    def test_update_warmup_def_changed_metadata_updates_value(
            self, db0_fixture):  # pylint: disable=unused-argument
        """Changing only metadata causes warmup_def to update."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("#STATEK: hidden = True\nprint('ready')")
        original_warmup_def = agent.warmup_def

        result = agent.update_warmup_def("#STATEK: hidden = False\nprint('ready')")

        assert result is True
        assert agent.warmup_def is not original_warmup_def
        assert isinstance(agent.warmup_def.warmup_code, CodeBlock)
        assert agent.warmup_def.warmup_code.code == "print('ready')"
        assert agent.warmup_def.warmup_code.metadata == {"hidden": False}
        assert agent.warmup_def.metadata == {"hidden": False}
        assert agent.warmup_def.hidden is False

    def test_update_warmup_def_to_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def can set warmup_def to None."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("print('hello')")
        result = agent.update_warmup_def(None)
        assert result is True
        assert agent.warmup_def is None

    def test_update_warmup_def_none_to_none_no_change(self, db0_fixture):  # pylint: disable=unused-argument
        """Test update_warmup_def returns False when None to None."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        result = agent.update_warmup_def(None)
        assert result is False

    def test_referenced_locals_empty_without_warmup_def(self, db0_fixture):  # pylint: disable=unused-argument
        """referenced_locals is empty when the agent has no warmup definition."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        assert agent.referenced_locals == []

    def test_referenced_locals_reads_agent_warmup_def(self, db0_fixture):  # pylint: disable=unused-argument
        """referenced_locals returns external names from the agent warmup definition."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("user = message.sender\nprint(user, locale)")

        assert agent.referenced_locals == ["message", "locale"]

    def test_referenced_locals_combines_multi_block_warmup(self, db0_fixture):  # pylint: disable=unused-argument
        """referenced_locals preserves first-use order across warmup blocks."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def(
            "user = message.sender\n"
            "# ----------\n"
            "result = format_message(user, locale, message)"
        )

        assert agent.referenced_locals == ["message", "locale"]

    def test_referenced_locals_handles_code_block_warmup(self, db0_fixture):  # pylint: disable=unused-argument
        """referenced_locals supports parsed CodeBlock warmup values."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.warmup_def = WarmupDef(
            warmup_code=CodeBlock(code="result = event.payload", tool_calls=[])
        )

        assert agent.referenced_locals == ["event"]

    def test_referenced_locals_is_cached(self, db0_fixture):  # pylint: disable=unused-argument
        """referenced_locals computes once until the warmup definition changes."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("user = message.sender")

        with patch("statek.task.get_referenced_locals", return_value=["message"]) as parser:
            assert agent.referenced_locals == ["message"]
            assert agent.referenced_locals == ["message"]

        assert parser.call_count == 1

    def test_referenced_locals_cache_cleared_on_warmup_update(self, db0_fixture):  # pylint: disable=unused-argument
        """update_warmup_def clears cached referenced locals when warmup changes."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("user = message.sender")
        assert agent.referenced_locals == ["message"]

        agent.update_warmup_def("payload = event.payload")

        assert agent.referenced_locals == ["event"]

    def test_referenced_locals_cache_cleared_when_warmup_removed(self, db0_fixture):  # pylint: disable=unused-argument
        """update_warmup_def clears cached referenced locals when warmup is removed."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        agent.update_warmup_def("user = message.sender")
        assert agent.referenced_locals == ["message"]

        agent.update_warmup_def(None)

        assert agent.referenced_locals == []


class TestCreateJobDefWarmup:
    """Test cases for SupervisedAgent.create_job_def warmup_def integration."""

    @staticmethod
    def _agent():
        return SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "test-model"},
            _tools=[],
        )

    def test_model_initialized_from_agent_metadata(self, db0_fixture):  # pylint: disable=unused-argument
        """JobDef freezes model metadata at creation time."""
        agent = SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "deepseek/deepseek-v3.2"},
            _tools=[],
        )
        job_def = agent.create_job_def()
        assert job_def.model == "deepseek/deepseek-v3.2"
        assert job_def.model_family == "deepseek"
        assert job_def.metadata == {"MODEL": "deepseek/deepseek-v3.2"}

    def test_metadata_is_frozen_by_replacing_agent_metadata_dict(self, db0_fixture):  # pylint: disable=unused-argument
        """Existing JobDefs keep their metadata object when the agent metadata changes."""
        agent = SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "deepseek/deepseek-v3.2", "TEMPERATURE": "0.3"},
            _tools=[],
        )

        job_def = agent.create_job_def()
        original_metadata = job_def.metadata

        agent.update_metadata({"MODEL": "openai/gpt-4o", "TEMPERATURE": "0.1"})

        assert job_def.metadata is original_metadata
        assert job_def.metadata == {"MODEL": "deepseek/deepseek-v3.2", "TEMPERATURE": "0.3"}
        assert agent._metadata == {"MODEL": "openai/gpt-4o", "TEMPERATURE": "0.1"}  # pylint: disable=protected-access

    def test_create_job_def_requires_model_metadata(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def fails fast when MODEL metadata is missing."""
        agent = SupervisedAgent(role="test", _system_prompt=make_system_prompt("test"), _tools=[])
        with pytest.raises(ValueError, match="MODEL"):
            agent.create_job_def()

    def test_no_warmup_code_no_warmup_def(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def with no warmup_code and no warmup_def produces None warmup."""
        agent = self._agent()
        job_def = agent.create_job_def()
        assert job_def.warmup_code is None

    def test_dynamic_warmup_only(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def with warmup_code but no warmup_def uses only dynamic code."""
        agent = self._agent()
        job_def = agent.create_job_def(warmup_code="dynamic_block")
        assert job_def.warmup_code == "dynamic_block"

    def test_warmup_def_only(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def with no warmup_code but with warmup_def uses only agent's warmup."""
        agent = self._agent()
        agent.update_warmup_def("agent_block")
        job_def = agent.create_job_def()
        assert job_def.warmup_code == "agent_block"

    def test_dynamic_and_warmup_def_combined(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def combines dynamic warmup_code followed by warmup_def blocks."""
        agent = self._agent()
        agent.update_warmup_def("agent_block")
        job_def = agent.create_job_def(warmup_code="dynamic_block")
        # Dynamic first, then agent's warmup_def
        blocks = list(job_def.warmup_code)
        assert blocks == ["dynamic_block", "agent_block"]

    def test_dynamic_list_and_warmup_def_combined(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def combines dynamic warmup_code list with warmup_def blocks."""
        agent = self._agent()
        agent.update_warmup_def("agent_block")
        job_def = agent.create_job_def(warmup_code=["d1", "d2"])
        blocks = list(job_def.warmup_code)
        assert blocks == ["d1", "d2", "agent_block"]

    def test_dynamic_and_multi_block_warmup_def(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def combines dynamic code with multi-block warmup_def."""
        agent = self._agent()
        agent.update_warmup_def("a1\n# ----------\na2")
        job_def = agent.create_job_def(warmup_code="dynamic_block")
        blocks = list(job_def.warmup_code)
        assert blocks == ["dynamic_block", "a1", "a2"]

    def test_job_params_still_work(self, db0_fixture):  # pylint: disable=unused-argument
        """create_job_def still passes kwargs as job_params when warmup_def is present."""
        agent = self._agent()
        agent.update_warmup_def("agent_block")
        job_def = agent.create_job_def(warmup_code="dynamic_block", user_question="hello")
        assert job_def.job_params == {"user_question": "hello"}


class TestCreateJobDefSharedVars:
    """Test cases for SupervisedAgent.create_job_def shared_vars integration."""

    @staticmethod
    def _agent():
        return SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "test-model"},
            _tools=[],
        )

    def test_no_shared_vars(self, db0_fixture):  # pylint: disable=unused-argument
        """Without shared_vars, job_params has no 'shared_vars' key."""
        agent = self._agent()
        job_def = agent.create_job_def()
        assert job_def.job_params is None

    def test_dict_shared_vars_populates_job_params(self, db0_fixture):  # pylint: disable=unused-argument
        """Dict shared_vars: keys are recorded as 'shared_vars' in job_params."""
        agent = self._agent()
        job_def = agent.create_job_def(shared_vars={"alpha": 1, "beta": 2})
        assert job_def.job_params == {"shared_vars": ["alpha", "beta"]}

    def test_empty_shared_vars(self, db0_fixture):  # pylint: disable=unused-argument
        """Empty shared_vars: 'shared_vars' is not added to job_params."""
        agent = self._agent()
        job_def = agent.create_job_def(shared_vars={})
        assert job_def.job_params is None

    def test_shared_vars_combined_with_kwargs(self, db0_fixture):  # pylint: disable=unused-argument
        """shared_vars coexist with other kwargs in job_params."""
        agent = self._agent()
        job_def = agent.create_job_def(
            shared_vars={"alpha": 1}, user_question="hi"
        )
        assert job_def.job_params == {
            "user_question": "hi",
            "shared_vars": ["alpha"],
        }

    def test_shared_vars_does_not_affect_warmup_code(self, db0_fixture):  # pylint: disable=unused-argument
        """shared_vars only reports names; it does not generate warmup_code."""
        agent = self._agent()
        job_def = agent.create_job_def(shared_vars={"alpha": 1})
        assert job_def.warmup_code is None


class TestCreateJobDefLocale:
    """Test cases for SupervisedAgent.create_job_def locale parameter."""

    @staticmethod
    def _agent():
        return SupervisedAgent(
            role="test",
            _system_prompt=make_system_prompt("test"),
            _metadata={"MODEL": "test-model"},
            _tools=[],
        )

    def test_no_locale_defaults_to_none(self, db0_fixture):  # pylint: disable=unused-argument
        """Without locale, JobDef.locale is None."""
        agent = self._agent()
        job_def = agent.create_job_def()
        assert job_def.locale is None

    def test_locale_forwarded_to_job_def(self, db0_fixture):  # pylint: disable=unused-argument
        """Explicit locale is stored on the resulting JobDef."""

        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        agent = self._agent()
        job_def = agent.create_job_def(locale=locale)
        assert job_def.locale is locale

    def test_locale_combined_with_other_params(self, db0_fixture):  # pylint: disable=unused-argument
        """locale coexists with warmup_code, shared_vars, and kwargs."""

        locale = StatekLocale(
            lang_code=StatekLangCode.DE,
            country_code=StatekCountryCode.DE,
        )
        agent = self._agent()
        job_def = agent.create_job_def(
            warmup_code="x = 1",
            shared_vars={"alpha": 1},
            locale=locale,
            user_question="hi",
        )
        assert job_def.locale is locale
        assert job_def.warmup_code == "x = 1"
        assert job_def.job_params["shared_vars"] == ["alpha"]
        assert job_def.job_params["user_question"] == "hi"


class TestUpdateWarmupDefs:
    """Test cases for the update_warmup_defs function."""

    def _create_warmup_file(self, directory, filename, content):
        """Helper to create a warmup definition file."""
        filepath = os.path.join(directory, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def test_updates_matching_agent(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs applies warmup from file matching agent role."""
        agent = SupervisedAgent(
            role="researcher",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        self._create_warmup_file(temp_dir, "researcher.py", "print('warmup')")

        update_warmup_defs(Path(temp_dir))

        assert agent.warmup_def is not None
        assert agent.warmup_def.warmup_code == "print('warmup')"

    def test_ignores_non_matching_files(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs ignores files that don't match any agent role."""
        agent = SupervisedAgent(
            role="researcher",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        self._create_warmup_file(temp_dir, "unknown_agent.py", "print('no match')")

        update_warmup_defs(Path(temp_dir))

        assert agent.warmup_def is None

    def test_updates_multiple_agents(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs updates multiple agents from their respective files."""
        agent_a = SupervisedAgent(
            role="planner",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        agent_b = SupervisedAgent(
            role="coder",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        self._create_warmup_file(temp_dir, "planner.py", "plan_code")
        self._create_warmup_file(temp_dir, "coder.py", "code_block")

        update_warmup_defs(Path(temp_dir))

        assert agent_a.warmup_def is not None
        assert agent_a.warmup_def.warmup_code == "plan_code"
        assert agent_b.warmup_def is not None
        assert agent_b.warmup_def.warmup_code == "code_block"

    def test_skips_non_py_files(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs only processes .py files."""
        agent = SupervisedAgent(
            role="researcher",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        self._create_warmup_file(temp_dir, "researcher.txt", "print('txt')")

        update_warmup_defs(Path(temp_dir))

        assert agent.warmup_def is None

    def test_no_update_when_unchanged(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs does not update when content hasn't changed."""
        agent = SupervisedAgent(
            role="researcher",
            _system_prompt=make_system_prompt("test"),
            _tools=[],
        )
        self._create_warmup_file(temp_dir, "researcher.py", "print('warmup')")

        update_warmup_defs(Path(temp_dir))
        assert agent.warmup_def is not None

        # Second call should not update (update_warmup_def returns False)
        result = agent.update_warmup_def("print('warmup')")
        assert result is False

    def test_skips_plain_agent_instances(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs only processes SupervisedAgent, not plain Agent."""
        Agent(role="researcher", _system_prompt=make_system_prompt("test"), _tools=[])
        self._create_warmup_file(temp_dir, "researcher.py", "print('warmup')")

        # Should not raise - plain Agent instances are not returned by db0.find(SupervisedAgent)
        update_warmup_defs(Path(temp_dir))

    def test_empty_directory(self, db0_fixture, temp_dir):  # pylint: disable=unused-argument
        """update_warmup_defs handles empty directory gracefully."""
        SupervisedAgent(role="researcher", _system_prompt=make_system_prompt("test"), _tools=[])

        # Should not raise
        update_warmup_defs(Path(temp_dir))
