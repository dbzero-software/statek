"""Tests for PyEnv class."""

import pytest

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


class TestPushToolResult:  # pylint: disable=unused-argument
    """Test cases for PyEnv.push_tool_result."""

    def test_push_first_result_initializes_tool_log(self, db0_fixture):
        """push_tool_result creates tool_log when it is None."""
        env = PyEnv()
        assert env.tool_log is None

        env.push_tool_result("result_a")

        assert env.tool_log is not None
        assert env.tool_log[0] == "result_a"

    def test_push_uses_console_length_as_key(self, db0_fixture):
        """Key is the current console length."""
        env = PyEnv()
        env.console = ["line1", "line2", "line3"]

        env.push_tool_result("result_b")

        assert env.tool_log[3] == "result_b"

    def test_push_second_result_at_same_key_creates_list(self, db0_fixture):
        """Two pushes at the same console position produce a list."""
        env = PyEnv()
        env.push_tool_result("first")
        env.push_tool_result("second")

        assert env.tool_log[0] == ["first", "second"]

    def test_push_third_result_at_same_key_appends(self, db0_fixture):
        """Third push appends to the existing list."""
        env = PyEnv()
        env.push_tool_result("a")
        env.push_tool_result("b")
        env.push_tool_result("c")

        assert env.tool_log[0] == ["a", "b", "c"]

    def test_push_with_console_pos_adjuster(self, db0_fixture):
        """console_pos_adjuster shifts the key."""
        env = PyEnv()
        env.console = ["line1"]  # len = 1

        env.push_tool_result("adjusted", console_pos_adjuster=2)

        assert 3 in env.tool_log  # 1 + 2
        assert env.tool_log[3] == "adjusted"

    def test_push_returns_key_and_index(self, db0_fixture):
        """push_tool_result returns a (key, index) tuple."""
        env = PyEnv()

        key0, idx0 = env.push_tool_result("first")
        key1, idx1 = env.push_tool_result("second")

        assert key0 == 0
        assert idx0 == 0
        assert key1 == 0
        assert idx1 == 1

    def test_push_at_different_console_positions(self, db0_fixture):
        """Pushes at different console positions use separate keys."""
        env = PyEnv()
        env.push_tool_result("at_zero")
        env.console = ["line1", "line2"]
        env.push_tool_result("at_two")

        assert env.tool_log[0] == "at_zero"
        assert env.tool_log[2] == "at_two"


class TestGetToolResult:  # pylint: disable=unused-argument
    """Test cases for PyEnv.get_tool_result."""

    def test_get_single_result(self, db0_fixture):
        """Retrieve a single string result at tool_call_id=0."""
        env = PyEnv()
        env.tool_log = {5: "only_result"}

        assert env.get_tool_result(console_pos=5, tool_call_id=0) == "only_result"

    def test_get_from_list_by_index(self, db0_fixture):
        """Retrieve results from a list by tool_call_id."""
        env = PyEnv()
        env.tool_log = {0: ["alpha", "beta", "gamma"]}

        assert env.get_tool_result(console_pos=0, tool_call_id=0) == "alpha"
        assert env.get_tool_result(console_pos=0, tool_call_id=1) == "beta"
        assert env.get_tool_result(console_pos=0, tool_call_id=2) == "gamma"

    def test_get_raises_when_tool_log_is_none(self, db0_fixture):
        """Raises KeyError when tool_log is None."""
        env = PyEnv()

        with pytest.raises(KeyError):
            env.get_tool_result(console_pos=0, tool_call_id=0)

    def test_get_raises_when_console_pos_missing(self, db0_fixture):
        """Raises KeyError when console_pos is not in tool_log."""
        env = PyEnv()
        env.tool_log = {0: "exists"}

        with pytest.raises(KeyError):
            env.get_tool_result(console_pos=99, tool_call_id=0)

    def test_get_raises_when_tool_call_id_out_of_range(self, db0_fixture):
        """Raises IndexError when tool_call_id exceeds list length."""
        env = PyEnv()
        env.tool_log = {0: ["a", "b"]}

        with pytest.raises(IndexError):
            env.get_tool_result(console_pos=0, tool_call_id=5)

    def test_get_raises_when_single_result_and_nonzero_id(self, db0_fixture):
        """Raises IndexError when a single result is stored but tool_call_id > 0."""
        env = PyEnv()
        env.tool_log = {0: "single"}

        with pytest.raises(IndexError):
            env.get_tool_result(console_pos=0, tool_call_id=1)
