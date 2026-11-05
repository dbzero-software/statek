"""Tests for PyEnv class."""
# pylint: disable=no-member

import pytest
import dbzero as db0


from statek.pyenv import Error, ErrorKind, PyEnv
from statek.executors.chat_log_item import LLM_LogItem
from statek.executors.utils import exec_step
from tests.conftest import DB0_DIR, seed_pyenv


# History is intentionally private; these tests verify its persisted representation.
# pylint: disable=protected-access


def _legacy_env():
    """Construct the collection layout persisted before fields became private."""
    env = PyEnv(global_state={"global": 2}, local_state={"local": 1, "_PERM_CTX": {"x": 3}},
                console=["old output"], exceptions={0: Error(ErrorKind.HARNESS, "limit")},
                push_log={1: ["first", "second"]})
    for name in ("global_state", "local_state", "console", "exceptions", "push_log"):
        setattr(env, name, getattr(env, "_" + name))
        delattr(env, "_" + name)
    db0.tags(env).add("legacy-env-test")
    return env


def _reopen_env(identifier, read_only=False):
    db0.close()
    db0.init(DB0_DIR, read_write=not read_only)
    db0.open("test_prefix", "r" if read_only else "rw")
    return db0.fetch(identifier)


@pytest.mark.usefixtures("db0_fixture")
def test_legacy_collections_read_only_after_reopen():
    env = _reopen_env(db0.uuid(_legacy_env()), read_only=True)
    assert env.console_len == 1
    assert list(env.iter_console()) == ["old output"]
    assert env.has_global_state
    assert env.get_local("local") == 1
    assert env.get_global("global") == 2
    assert env.get_exception(0).kind == ErrorKind.HARNESS
    assert [(pos, error.message) for pos, error in env.iter_exceptions()] == [(0, "limit")]
    assert dict(env.iter_push_log()) == {1: ["first", "second"]}
    assert dict(env.iter_locals()) == {"local": 1, "_PERM_CTX": {"x": 3}}
    assert dict(env.iter_globals()) == {"global": 2}
    assert env.perm_ctx == {"x": 3}
    assert list(env.fetch_state(["local", "global"], 0)) == [("local", 1), ("global", 2)]
    for name in ("_global_state", "_local_state", "_console", "_exceptions", "_push_log"):
        assert not hasattr(env, name)


@pytest.mark.usefixtures("db0_fixture")
def test_legacy_writes_preserve_contents_and_references():
    env = _reopen_env(db0.uuid(_legacy_env()))
    context = env.perm_ctx
    error = Error(ErrorKind.EXECUTION, "failure")
    env.console_append("new output", error=error)
    env.push_append("third")
    env.update_globals(added=4)
    env.update_state(local=10)
    assert env.perm_ctx is context
    env = _reopen_env(db0.uuid(env))
    assert list(env.iter_console()) == ["old output", "new output"]
    assert env.get_exception(0).message == "limit"
    assert env.get_exception(1).message == "failure"
    assert dict(env.iter_push_log()) == {1: ["first", "second"], 2: "third"}
    assert dict(env.iter_globals()) == {"global": 2, "added": 4}
    assert env.get_local("local") == 10
    assert env.fetch_state_var("local", 0) == 1
    assert env._generation == 1


@pytest.mark.usefixtures("db0_fixture")
@pytest.mark.parametrize("empty", [None, {}])
def test_private_collections_override_legacy_even_when_empty(empty):
    env = _legacy_env()
    env._global_state = empty
    env._local_state = empty
    env._console = None
    env._exceptions = empty
    env._push_log = empty
    assert env.has_global_state is (empty is not None)
    assert not list(env.iter_locals())
    assert not list(env.iter_globals())
    assert env.get_local("local") is None
    assert env.get_global("global") is None
    assert env.console_len == 0
    assert not list(env.iter_console())
    assert env.get_exception(0) is None
    assert not list(env.iter_exceptions())
    assert not list(env.iter_push_log())
    env.console_append("fresh")
    assert list(env.iter_console()) == ["fresh"]


@pytest.mark.usefixtures("db0_fixture")
def test_update_state_new_bindings_do_not_advance_generation():
    env = PyEnv(local_state=None, global_state=None)
    env.update_state()
    env.update_state(answer=42, empty=None)
    assert dict(env.iter_locals()) == {"answer": 42, "empty": None}
    assert env._update_log is None
    assert env._generation is None


@pytest.mark.usefixtures("db0_fixture")
def test_update_state_records_overwrites_once_per_batch():
    env = PyEnv(local_state={"x": 1, "y": None}, global_state={"x": "global", "z": 3})
    env.update_state(x=2, y=20, z=30, new="first")
    assert env._generation == 1
    assert env._update_log == {"x": (None, 1), "y": (None, None), "z": (None, 3)}
    assert dict(env.iter_globals()) == {"x": "global", "z": 3}
    assert dict(env.iter_locals()) == {"x": 2, "y": 20, "z": 30, "new": "first"}

    env.update_state(x=4, y=40)
    assert env._generation == 2
    assert env._update_log["x"] == [(None, 1), (1, 2)]
    assert env._update_log["y"] == [(None, None), (1, 20)]
    env.update_state(x=5)
    assert env._generation == 3
    assert env._update_log["x"] == [(None, 1), (1, 2), (2, 4)]
    env.update_state(another="new")
    env.update_state()
    assert env._generation == 3


@pytest.mark.usefixtures("db0_fixture")
def test_update_state_records_equal_and_identical_overwrites():
    env = PyEnv(local_state={"value": [1], "equal": 1000})
    old_value = env.get_local("value")
    env.update_state(value=old_value, equal=int("1000"))
    assert env._generation == 1
    assert env._update_log["equal"] == (None, 1000)
    old_value.append(2)
    assert env._update_log["value"][1] == [1, 2]


@pytest.mark.usefixtures("db0_fixture")
def test_update_state_history_survives_reopen():
    env = PyEnv(local_state={"value": 1})
    env.update_state(value=2)
    env.update_state(value=3)
    db0.tags(env).add("history-reopen-test")
    identifier = db0.uuid(env)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    restored = db0.fetch(identifier)
    assert restored._generation == 2
    assert restored._update_log["value"] == [(None, 1), (1, 2)]
    restored.update_state(value=4)
    assert restored._generation == 3
    assert restored._update_log["value"] == [(None, 1), (1, 2), (2, 3)]


@pytest.mark.asyncio
async def test_executor_batches_overwrites(job_factory):
    job = job_factory()
    await exec_step("x = 1\ny = 2\nunchanged = 3", job)
    env = job.py_env
    assert env._generation is None
    await exec_step("x = 10\ny = 20\nnew = 4", job)
    assert env._generation == 1
    assert env._update_log == {"x": (None, 1), "y": (None, 2)}
    assert env.get_local("new") == 4
    await exec_step("unchanged = 3", job)
    assert env._generation == 1
    assert "_STATEK_CTX" not in dict(env.iter_locals())


@pytest.mark.usefixtures("db0_fixture")
@pytest.mark.parametrize("initial_generation", [None, 0])
def test_fetch_state_var_overwrite_boundaries(initial_generation):
    env = PyEnv(local_state={"value": "initial"})
    env._generation = initial_generation
    assert env.fetch_state_var("value", None) == "initial"
    env.update_state(value="second")
    for generation in (None, 0):
        assert env.fetch_state_var("value", generation) == "initial"
    assert env.fetch_state_var("value", 1) == "initial"
    env.update_state(value="third")
    env.update_state(value="current")
    for generation, expected in [(None, "initial"), (0, "initial"),
                                 (1, "second"), (2, "third"),
                                 (3, "third"), (100, "third")]:
        assert env.fetch_state_var("value", generation) == expected


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_var_sparse_history():
    env = PyEnv(local_state={"value": "initial", "other": 0})
    env.update_state(other=1)
    env.update_state(other=2)
    env.update_state(value="middle")
    env.update_state(other=3)
    env.update_state(other=4)
    env.update_state(value="current")
    for generation, expected in [(None, "current"), (0, "current"),
                                 (1, "current"), (2, "initial"),
                                 (3, "initial"), (4, "initial"), (5, "middle"),
                                 (6, "middle"), (100, "middle")]:
        assert env.fetch_state_var("value", generation) == expected


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_var_current_bindings_and_none():
    env = PyEnv(local_state={"shared": None},
                global_state={"shared": "global", "global": None})
    for generation in (None, 0, 100):
        assert env.fetch_state_var("shared", generation) is None
        assert env.fetch_state_var("global", generation) is None
        with pytest.raises(KeyError) as exc_info:
            env.fetch_state_var("missing", generation)
        assert exc_info.value.args == ("missing",)
    env.update_state(**{"shared": "local", "global": "overridden"})
    assert env.fetch_state_var("shared", 0) is None
    assert env.fetch_state_var("global", 0) is None
    assert env.fetch_state_var("shared", 1) is None
    assert env.fetch_state_var("global", 1) is None
    empty = PyEnv(local_state=None, global_state=None)
    with pytest.raises(KeyError):
        empty.fetch_state_var("missing", None)
    empty.update_state(created="later")
    assert empty.fetch_state_var("created", 0) == "later"
    assert empty._update_log is None
    assert empty._generation is None


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_preserves_order_duplicates_and_generator_inputs():
    env = PyEnv(local_state={"x": 1, "y": 2})
    env.update_state(x=3, y=4)
    assert not list(env.fetch_state([], None))
    assert not list(env.fetch_state(iter(()), 1))
    assert list(env.fetch_state((key for key in ["y", "x", "y"]), 0)) == [
        ("y", 2), ("x", 1), ("y", 2)]
    assert list(env.fetch_state(["x", "y"], 1)) == [("x", 1), ("y", 2)]


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_missing_key_fails_before_first_result():
    env = PyEnv(local_state={"present": 1})
    with pytest.raises(KeyError) as exc_info:
        results = iter(env.fetch_state(iter(["present", "first", "second"]), None))
        next(results)
    assert exc_info.value.args == ("first",)
    assert env._update_log is None
    assert env._generation is None


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_retains_value_references():
    env = PyEnv(local_state={"value": [1]})
    previous = env.get_local("value")
    env.update_state(value=[2])
    assert env.fetch_state_var("value", 0) is previous
    assert list(env.fetch_state(["value"], 0))[0][1] is previous
    assert env.fetch_state_var("value", 1) is previous
    previous.append(3)
    assert env.fetch_state_var("value", 0) == [1, 3]


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_history_survives_reopen_without_mutation():
    env = PyEnv(local_state={"single": 1, "multiple": 10})
    env.update_state(single=2, multiple=20)
    env.update_state(multiple=30)
    db0.tags(env).add("fetch-state-reopen-test")
    identifier = db0.uuid(env)
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    restored = db0.fetch(identifier)
    for generation, expected in [(None, [("single", 1), ("multiple", 10)]),
                                 (0, [("single", 1), ("multiple", 10)]),
                                 (1, [("single", 1), ("multiple", 20)]),
                                 (2, [("single", 1), ("multiple", 20)]),
                                 (100, [("single", 1), ("multiple", 20)])]:
        assert list(restored.fetch_state(["single", "multiple"], generation)) == expected
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open("test_prefix", "rw")
    restored = db0.fetch(identifier)
    assert restored._generation == 2
    assert restored._update_log == {"single": (None, 1),
                                    "multiple": [(None, 10), (1, 20)]}
    assert dict(restored.iter_locals()) == {"single": 2, "multiple": 30}


class TestPyEnv:  # pylint: disable=too-few-public-methods
    """Test cases for PyEnv class."""

    def test_console_append(self, db0_fixture):  # pylint: disable=unused-argument
        """Test console_append creates a new list when console is None."""
        env = PyEnv()

        assert not list(env.iter_console())
        env.console_append("First output")
        assert list(env.iter_console()) == ["First output"]
        env.console_append("Second output")
        assert list(env.iter_console()) == ["First output", "Second output"]

    def test_update_state_adds_and_overwrites_values(self, db0_fixture):  # pylint: disable=unused-argument
        """update_state merges new values into local_state."""
        env = PyEnv(local_state={"x": 1, "y": 2})

        env.update_state(y=20, z=30)

        assert dict(env.iter_locals()) == {"x": 1, "y": 20, "z": 30}

    def test_update_state_initializes_local_state_when_missing(self, db0_fixture):  # pylint: disable=unused-argument
        """update_state creates local_state when it is None."""
        env = PyEnv(local_state=None)

        env.update_state(answer=42)

        assert dict(env.iter_locals()) == {"answer": 42}

    def test_perm_ctx_returns_existing_context(self, db0_fixture):  # pylint: disable=unused-argument
        """perm_ctx exposes _PERM_CTX from local_state."""
        ctx = {"last_example_id": 1}
        env = PyEnv(local_state={"_PERM_CTX": ctx})

        assert env.perm_ctx == ctx

    def test_perm_ctx_returns_none_when_missing(self, db0_fixture):  # pylint: disable=unused-argument
        """perm_ctx is read-only and does not create _PERM_CTX."""
        env = PyEnv(local_state={})

        assert env.perm_ctx is None
        assert "_PERM_CTX" not in dict(env.iter_locals())

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

        env = seed_pyenv(env, global_state={"found": 1})
        assert env.resolve_locals(["found", "found"]) == {"found": int}

    def test_resolve_locals_raises_for_first_missing_handle(self, db0_fixture):  # pylint: disable=unused-argument
        """Strict lookup reports the first absent name in input order."""
        env = PyEnv(local_state={"local": 1}, global_state={"global": 2})

        with pytest.raises(KeyError) as exc_info:
            env.resolve_locals(["local", "first", "global", "second"], raise_for_missing=True)

        assert exc_info.value.args == ("first",)


@pytest.mark.usefixtures("db0_fixture")
def test_collection_interface():
    """Collections are accessed through scalar reads and isolated iterators."""
    env = PyEnv(local_state=None, global_state=None)
    assert env.console_len == 0
    assert not env.has_global_state
    assert env.get_local("missing", 3) == 3
    assert env.get_global("missing") is None
    assert env.get_exception(0) is None
    assert not list(env.iter_locals())
    assert not list(env.iter_globals())
    assert not list(env.iter_console())
    assert not list(env.iter_exceptions())
    assert not list(env.iter_push_log())
    env.update_globals(shared="global", count=1)
    env.update_state(shared=None)
    assert env.has_global_state
    assert env.get_global("count") == 1
    assert env.get_local("shared", 3) is None
    assert dict(env.iter_globals()) == {"shared": "global", "count": 1}
    assert env.resolve_locals(["shared"]) == {"shared": type(None)}
    assert PyEnv(global_state={}).has_global_state
    for name in ("local_state", "global_state", "console", "exceptions", "push_log",
                 "update_locals"):
        assert not hasattr(env, name)


@pytest.mark.usefixtures("db0_fixture")
def test_console_ranges_errors_and_pushes():
    """Console bounds follow slices and diagnostics retain their classification."""
    errors = {2: Error(ErrorKind.HARNESS, "limit"), 0: Error(ErrorKind.EXECUTION, "bad")}
    env = PyEnv(console=["a", "b", "c"], exceptions=errors,
                push_log={3: ["one", "two"], 0: "initial"})
    assert [pos for pos, _ in env.iter_exceptions()] == [0, 2]
    assert env.get_exception(0).kind == ErrorKind.EXECUTION
    for start, stop in [(0, None), (1, 2), (-2, None), (-99, 99), (2, 1), (0, -1)]:
        assert list(env.iter_console(start, stop)) == ["a", "b", "c"][start:stop]
    env.push_append("three")
    pushes = dict(env.iter_push_log())
    assert list(pushes) == [0, 3]
    assert pushes[3] == ["one", "two", "three"]
    pushes[3].append("detached")
    assert dict(env.iter_push_log())[3] == ["one", "two", "three"]
    error = Error(ErrorKind.EXECUTION, "failure")
    env.console_append("failure", error=error)
    assert env.console_len == 4
    assert env.get_exception(3) is error
    assert list(env.iter_console(3)) == ["failure"]


@db0.memo
class DerivedEnv(PyEnv):
    """Expose bindings without using PyEnv storage."""

    def get_local(self, name, default=None):
        return {"_PERM_CTX": {"x": 1}, "shared": None}.get(name, default)

    def get_global(self, name, default=None):
        return {"shared": 3, "global": "value"}.get(name, default)


@pytest.mark.usefixtures("db0_fixture")
def test_overridden_reads():
    """Derived reads dispatch through the public accessors."""
    env = DerivedEnv()
    assert env.perm_ctx == {"x": 1}
    assert env.resolve_locals(["shared", "global"]) == {"shared": type(None), "global": str}


@pytest.mark.usefixtures("db0_fixture")
def test_fetch_state_uses_overridden_accessors():
    env = DerivedEnv()
    assert env.fetch_state_var("shared", None) is None
    assert env.fetch_state_var("global", 100) == "value"
    assert list(env.fetch_state(["global", "shared"], 0)) == [
        ("global", "value"), ("shared", None)]
    env.update_state(shared="overwritten")
    assert env.fetch_state_var("shared", 0) is None


@db0.memo
class InterfaceEnv(PyEnv):
    """Supply virtual bindings and history through the public access interface."""

    def get_local(self, name, default=None):
        if name == "seed":
            return 40
        return super().get_local(name, default)

    def iter_locals(self):
        yield from super().iter_locals()
        yield "seed", self.get_local("seed")

    def iter_globals(self):
        yield "offset", 2

    def iter_console(self, from_pos=0, to_pos=None):
        yield from ["public output"][from_pos:to_pos]

    def iter_push_log(self):
        yield 0, "public push"

    def iter_exceptions(self):
        yield 0, Error(ErrorKind.EXECUTION, "public diagnostic")


@pytest.mark.asyncio
async def test_consumers_use_overridden_interface(job_factory):
    """Execution, discovery, history, and counters honor overridden public reads."""
    job = job_factory()
    job.py_env = InterfaceEnv()
    job.chat_log.append(LLM_LogItem(console_pos=0, llm_resp="print(result)"))

    await exec_step("result = seed + offset\nprint(result)", job)

    assert job.py_env.get_local("result") == 42
    assert list(job.find_locals(var_name="seed")) == [40]
    assert job.exception_count == 1
    assert job.max_consecutive_exceptions == 1
    history = "\n".join(item.content or "" for item in job.get_chat_history())
    assert "public output" in history
    assert "public push" in history
