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

    def test_perm_ctx_returns_existing_context(self, db0_fixture):  # pylint: disable=unused-argument
        """perm_ctx exposes _PERM_CTX from local_state."""
        ctx = {"last_example_id": 1}
        env = PyEnv(local_state={"_PERM_CTX": ctx})

        assert env.perm_ctx == ctx

    def test_perm_ctx_returns_none_when_missing(self, db0_fixture):  # pylint: disable=unused-argument
        """perm_ctx is read-only and does not create _PERM_CTX."""
        env = PyEnv(local_state={})

        assert env.perm_ctx is None
        assert "_PERM_CTX" not in env.local_state

    def test_resolve_locals_prefers_locals_and_returns_runtime_types(self, db0_fixture):  # pylint: disable=unused-argument
        """Local names take precedence over globals, including values of None."""
        env = PyEnv(
            local_state={"shared": None, "local": "hello"},
            global_state={"shared": "global", "global": 42},
        )

        assert env.resolve_locals(iter(["shared", "local", "global", "missing"])) == {
            "shared": type(None),
            "local": str,
            "global": int,
        }

    def test_resolve_locals_handles_absent_state_and_duplicate_names(self, db0_fixture):  # pylint: disable=unused-argument
        """Absent dictionaries are empty and duplicate handles share one result."""
        env = PyEnv(local_state=None, global_state=None)

        assert not env.resolve_locals(["missing", "missing"])
        assert not env.resolve_locals([])

        env.global_state = {"found": 1}
        assert env.resolve_locals(["found", "found"]) == {"found": int}

    def test_resolve_locals_raises_for_first_missing_handle(self, db0_fixture):  # pylint: disable=unused-argument
        """Strict lookup reports the first absent name in input order."""
        env = PyEnv(local_state={"local": 1}, global_state={"global": 2})

        with pytest.raises(KeyError) as exc_info:
            env.resolve_locals(["local", "first", "global", "second"], raise_for_missing=True)

        assert exc_info.value.args == ("first",)
