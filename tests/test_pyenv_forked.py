"""Persistent environment views and independently owned continuations."""
# pylint: disable=protected-access,no-member
import dbzero as db0
import pytest

from statek.pyenv import Error, ErrorKind, PyEnv, PyEnvForked, fork_env
from tests.test_pyenv import _legacy_env, _reopen_env


pytestmark = pytest.mark.usefixtures("db0_fixture")


def test_bindings_are_views_with_private_overrides():
    source = PyEnv(local_state={"x": 1, "shared": None, "_PERM_CTX": {"x": 2}},
                   global_state={"shared": "global", "g": 3})
    fork = fork_env(source)
    assert isinstance(fork, PyEnvForked)
    assert not fork._local_state
    assert not fork._global_state
    assert dict(fork.iter_locals()) == dict(source.iter_locals())
    assert dict(fork.iter_globals()) == dict(source.iter_globals())
    assert fork.perm_ctx is source.perm_ctx
    assert fork.resolve_locals(["shared", "g"]) == {"shared": type(None), "g": int}
    assert fork.get_local("absent", 9) == 9
    fork.update_state(x=4, shared=5, new=None)
    fork.update_globals(g=6)
    assert dict(fork.iter_locals()) == {"x": 4, "shared": 5, "_PERM_CTX": {"x": 2},
                                        "new": None}
    assert fork.get_global("g") == 6
    assert source.get_local("x") == 1
    assert source.get_global("g") == 3
    assert fork_env(source).get_local("x") == 1


def test_captured_generation_and_fork_history():
    source = PyEnv(local_state={"x": 1})
    source.update_state(x=2)
    source.update_state(x=3)
    fork = fork_env(source)
    assert fork.get_local("x") == 2
    assert dict(fork.iter_locals()) == {"x": 2}
    assert fork.fetch_state_var("x", 0) == 1
    assert fork.fetch_state_var("x", 100) == 2
    fork.update_state(x=4, new=8)
    assert fork._generation == 3
    assert fork.get_local("x") == 4
    assert fork.fetch_state_var("x", 0) == 1
    assert fork.fetch_state_var("x", 100) == 2
    nested = fork_env(fork)
    assert nested.get_local("x") == 2
    assert nested.get_local("new") == 8
    with pytest.raises(KeyError):
        next(iter(nested.fetch_state(["x", "absent"], None)))


def test_console_errors_pushes_and_boundaries():
    error = Error(ErrorKind.EXECUTION, "old")
    source = PyEnv(console=["a", "b"], exceptions={0: error},
                   push_log={0: "initial", 2: ["one", "two"]})
    fork = fork_env(source)
    fork.push_append("three")
    assert dict(fork.iter_push_log())[2] == ["one", "two", "three"]
    assert dict(source.iter_push_log())[2] == ["one", "two"]
    messages = dict(fork.iter_push_log())[2]
    messages.append("detached")
    assert dict(fork.iter_push_log())[2] == ["one", "two", "three"]
    new_error = Error(ErrorKind.HARNESS, "new")
    fork.console_append("c", error=new_error)
    assert fork.console_len == 3
    for start, stop in [(0, None), (-2, None), (-99, 99), (2, 1), (0, -1)]:
        assert list(fork.iter_console(start, stop)) == ["a", "b", "c"][start:stop]
    assert list(fork.iter_exceptions()) == [(0, error), (2, new_error)]
    assert fork.get_exception(1) is None
    source.console_append("excluded")
    assert list(fork.iter_console()) == ["a", "b", "c"]
    assert source.get_exception(2) is None


@pytest.mark.parametrize("globals_state", [None, {}])
def test_empty_state_and_independent_continuation(globals_state):
    source = PyEnv(local_state=None, global_state=globals_state, next_instr_id=3,
                   warmup_block_num=2, exit_status="done", future_result=PyEnv())
    fork = fork_env(source)
    assert fork.has_global_state is (globals_state is not None)
    assert not list(fork.iter_console())
    assert not list(fork.iter_exceptions())
    assert not list(fork.iter_push_log())
    for name in ("next_instr_id", "warmup_block_num", "exit_status", "future_result"):
        assert getattr(fork, name) is None
    fork.update_globals()
    assert fork.has_global_state
    fork.update_state(created=None)
    assert fork._generation is None
    assert fork.get_local("created", 9) is None


@pytest.mark.parametrize("read_only", [False, True])
def test_nested_legacy_views_survive_reopen(read_only):
    source = _legacy_env()
    fork = fork_env(source)
    fork.update_state(local=10)
    fork.console_append("fork output")
    nested = fork_env(fork)
    nested.push_append("nested push")
    db0.tags(nested).add("nested-fork-test")
    restored = _reopen_env(db0.uuid(nested), read_only=read_only)
    assert restored.get_local("local") == 1
    assert list(restored.iter_console()) == ["old output", "fork output"]
    assert dict(restored.iter_push_log())[2] == "nested push"
    assert restored.get_exception(0).message == "limit"


def test_factory_rejects_non_environment():
    with pytest.raises(TypeError):
        fork_env({})


@db0.memo
class LazySource(PyEnv):
    """A virtual source whose bindings cannot be eagerly enumerated."""

    def iter_locals(self):
        raise AssertionError('fork construction must not enumerate locals')

    def iter_globals(self):
        raise AssertionError('fork construction must not enumerate globals')

    def get_local(self, name, default=None):
        return 99 if name == 'virtual' else super().get_local(name, default)

    def fetch_state_var(self, key, generation):
        return 42 if key == 'virtual' else super().fetch_state_var(key, generation)


def test_factory_is_lazy_and_uses_public_source_reads():
    fork = fork_env(LazySource())
    assert fork.get_local('virtual') == 42
    assert fork.fetch_state_var('virtual', None) == 42
    assert fork.perm_ctx is None


def test_fetch_keeps_inherited_local_precedence_over_global_override():
    fork = fork_env(PyEnv(local_state={'shared': None}, global_state={'shared': 1}))
    fork.update_globals(shared=2)
    assert fork.get_global('shared') == 2
    assert fork.fetch_state_var('shared', None) is None
    fork.update_state(shared=3)
    assert fork.fetch_state_var('shared', None) is None


def test_later_log_entries_outside_captured_maximum_are_excluded():
    source = PyEnv(console=['a'], push_log={0: 'initial'})
    fork = fork_env(source)
    source.push_append('later')
    source.console_append('excluded', error=Error(ErrorKind.EXECUTION, 'later error'))
    assert dict(fork.iter_push_log()) == {0: 'initial'}
    assert not list(fork.iter_exceptions())
    assert fork.get_exception(1) is None
    assert list(fork.iter_console()) == ['a']


def test_fork_only_binding_with_sparse_history_uses_current_fallback():
    source = PyEnv(local_state={'source': 1})
    source.update_state(source=2)
    source.update_state(source=3)
    fork = fork_env(source)
    fork.update_state(created='first')
    fork.update_state(created='current')
    assert fork.fetch_state_var('created', 0) == 'current'
    assert fork.fetch_state_var('created', 2) == 'first'
    assert fork_env(fork).get_local('created') == 'first'
