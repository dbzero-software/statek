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

    def test_update_locals_adds_and_overwrites_values(self, db0_fixture):  # pylint: disable=unused-argument
        """update_locals merges new values into local_state."""
        env = PyEnv(local_state={"x": 1, "y": 2})

        env.update_locals(y=20, z=30)

        assert env.local_state == {"x": 1, "y": 20, "z": 30}

    def test_update_locals_initializes_local_state_when_missing(self, db0_fixture):  # pylint: disable=unused-argument
        """update_locals creates local_state when it is None."""
        env = PyEnv(local_state=None)

        env.update_locals(answer=42)

        assert env.local_state == {"answer": 42}
