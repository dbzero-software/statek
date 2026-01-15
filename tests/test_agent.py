"""Tests for Agent class."""

from statek.agent import Agent
from statek.utils import format_callable_decl


def clock():
    """clock mock."""
    return None


def docs(class_name: type, method_name: str = None) -> str:  # pylint: disable=unused-argument
    """docs mock."""
    return ""


def exit_tool(reason: str) -> None:  # pylint: disable=unused-argument
    """exit mock."""
    return None


class TestAgent:
    """Test cases for Agent class."""

    def test_system_prompt_formatting_single_tool(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with a single tool."""
        system_prompt = "Available tools: {tools}"
        tools = [clock]

        agent = Agent(_system_prompt=system_prompt, _tools=tools)

        assert format_callable_decl(clock) in agent.system_prompt
        assert "Available tools:" in agent.system_prompt

    def test_system_prompt_formatting_multiple_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with multiple tools."""
        system_prompt = "You have access to these tools:\n>{tools}"
        tools = [clock, docs, exit_tool]

        agent = Agent(_system_prompt=system_prompt, _tools=tools)

        for tool in tools:
            assert format_callable_decl(tool) in agent.system_prompt
        assert "You have access to these tools:" in agent.system_prompt

    def test_system_prompt_formatting_no_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with empty tools list."""
        system_prompt = "Available tools: {tools}"
        tools = []

        agent = Agent(_system_prompt=system_prompt, _tools=tools)

        assert "Available tools:" in agent.system_prompt
