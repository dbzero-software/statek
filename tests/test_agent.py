"""Tests for Agent class."""

from statek.agents.agent import Agent
from tests.conftest import clock, docs, exit_tool


class TestAgent:
    """Test cases for Agent class."""

    def test_system_prompt_formatting_single_tool(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with a single tool."""
        system_prompt = "Available tools:\n{tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=system_prompt,
            _prompt_template="Test",
            _tools=tools
        )

        assert "clock()" in agent.system_prompt
        assert "Get the current time." in agent.system_prompt
        assert "Available tools:\n" in agent.system_prompt

    def test_system_prompt_formatting_multiple_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with multiple tools."""
        system_prompt = "You have access to these tools:\n{tools}"
        tools = [clock, docs, exit_tool]

        agent = Agent(
            role="test",
            _system_prompt=system_prompt,
            _prompt_template="Test",
            _tools=tools
        )

        assert "clock()" in agent.system_prompt
        assert "docs(class_name, method_name" in agent.system_prompt
        assert "exit_tool(reason)" in agent.system_prompt
        assert "You have access to these tools:" in agent.system_prompt

    def test_system_prompt_formatting_no_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt property with empty tools list."""
        system_prompt = "Available tools: {tools}"
        tools = []

        agent = Agent(
            role="test",
            _system_prompt=system_prompt,
            _prompt_template="Test",
            _tools=tools
        )

        assert "Available tools:" in agent.system_prompt
        assert agent.system_prompt == "Available tools: "

    def test_system_prompt_with_block_comment(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt with block comment placeholder."""
        system_prompt = "# --- TOOLS ---\n# {tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=system_prompt,
            _prompt_template="Test",
            _tools=tools
        )

        # Each line should be prefixed with #
        assert "# clock()" in agent.system_prompt
        assert "#     Get the current time." in agent.system_prompt

    def test_system_prompt_detailed_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test system_prompt with detailed_tools placeholder."""
        system_prompt = "Tools:\n{detailed_tools}"
        tools = [clock]

        agent = Agent(
            role="test",
            _system_prompt=system_prompt,
            _prompt_template="Test",
            _tools=tools
        )

        # detailed_tools uses py_syntax=True
        assert "def clock()" in agent.system_prompt
        assert '"""Get the current time.' in agent.system_prompt
