"""Tests for PyEnv class."""

from statek.pyenv import PyEnv


class TestPyEnv:  # pylint: disable=too-few-public-methods
    """Test cases for PyEnv class."""

    def test_console_append(self, db0_fixture):  # pylint: disable=unused-argument
        """Test console_append creates a new list when console is None."""
        env = PyEnv()

        assert env.console is None
        env.console_append("First output")
        assert env.console == ["First output"]
        env.console_append("Second output")
        assert env.console == ["First output", "Second output"]
