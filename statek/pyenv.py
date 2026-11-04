# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, List, Dict, Iterable, Optional, Tuple, Type, Union
import dbzero as db0

from statek.future import FutureResult


@db0.enum(values=["EXECUTION", "HARNESS"])
class ErrorKind:
    """Origin of a recorded console error."""


@db0.memo(no_default_tags=True)
@dataclass
class Error:
    """Persistent error message classified at its point of origin."""

    kind: ErrorKind
    message: str


_DEFAULT_STATE = object()


@db0.memo
@dataclass(init=False)
class PyEnv:
    """Persistent execution bindings, console output, and continuation state."""

    _global_state: Optional[Dict]
    _local_state: Optional[Dict]
    _console: Optional[List[str]]
    _exceptions: Optional[Dict[int, Error]]
    _push_log: Optional[Dict[int, Union[str, List[str]]]]
    _update_log: Optional[Dict[str, Union[
        Tuple[Optional[int], Any], List[Tuple[Optional[int], Any]]
    ]]] = None
    _generation: Optional[int] = None
    next_instr_id: int = None
    warmup_block_num: int = None
    exit_status: str = None
    future_result: FutureResult = None

    # Retain the constructor keywords while keeping collection storage private.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, global_state=_DEFAULT_STATE, local_state=_DEFAULT_STATE,
                 console=None, exceptions=None, push_log=None, next_instr_id=None,
                 warmup_block_num=None, exit_status=None, future_result=None):
        self._global_state = {} if global_state is _DEFAULT_STATE else global_state
        self._local_state = {} if local_state is _DEFAULT_STATE else local_state
        self._console = console
        self._exceptions = exceptions
        self._push_log = push_log
        self.next_instr_id = next_instr_id
        self.warmup_block_num = warmup_block_num
        self.exit_status = exit_status
        self.future_result = future_result

    def _read_collection(self, name):
        """Read renamed storage without migrating objects in read-only sessions."""
        value = getattr(self, "_" + name, _DEFAULT_STATE)
        return getattr(self, name, None) if value is _DEFAULT_STATE else value

    def _writable_collection(self, name, factory):
        """Adopt legacy storage by reference on the first write."""
        private_name = "_" + name
        if getattr(self, private_name, None) is None:
            value = self._read_collection(name)
            setattr(self, private_name, factory() if value is None else value)
        return getattr(self, private_name)

    @property
    def console_len(self):
        """Number of recorded console entries."""
        console = self._read_collection("console")
        return len(console) if console is not None else 0

    @property
    def has_global_state(self):
        """Whether globals were supplied, including an empty dictionary."""
        return self._read_collection("global_state") is not None

    @property
    def perm_ctx(self):
        """Return the persistent context with its existing reference semantics."""
        return self.get_local("_PERM_CTX")

    def get_local(self, name, default=None):
        """Read one local binding."""
        state = self._read_collection("local_state")
        return state.get(name, default) if state is not None else default

    def get_global(self, name, default=None):
        """Read one global binding."""
        state = self._read_collection("global_state")
        return state.get(name, default) if state is not None else default

    def get_exception(self, position):
        """Read a classified diagnostic, or None for an unrecorded position."""
        exceptions = self._read_collection("exceptions")
        return exceptions.get(position) if exceptions is not None else None

    def iter_locals(self):
        """Yield local name/value pairs without exposing their container."""
        state = self._read_collection("local_state")
        if state is not None:
            yield from state.items()

    def iter_globals(self):
        """Yield global name/value pairs without exposing their container."""
        state = self._read_collection("global_state")
        if state is not None:
            yield from state.items()

    def iter_exceptions(self):
        """Yield classified diagnostics in console order."""
        exceptions = self._read_collection("exceptions")
        if exceptions is not None:
            for position in sorted(exceptions):
                yield position, self.get_exception(position)

    def iter_push_log(self):
        """Yield pushed messages in console order, copying repeated-message lists."""
        push_log = self._read_collection("push_log")
        if push_log is not None:
            for position in sorted(push_log):
                messages = push_log[position]
                yield position, messages if isinstance(messages, str) else list(messages)

    def iter_console(self, from_pos=0, to_pos=None):
        """Yield a console slice with an exclusive upper bound."""
        start, stop, step = slice(from_pos, to_pos).indices(self.console_len)
        console = self._read_collection("console")
        for position in range(start, stop, step):
            yield console[position]

    def console_append(self, out: str, error: Optional[Error] = None):
        """Record output and its optional classified diagnostic together."""
        position = self.console_len
        self._writable_collection("console", list).append(out)
        if error is not None:
            self._writable_collection("exceptions", dict)[position] = error

    def push_append(self, message: str):
        """Record a message at the current console position, preserving order."""
        push_log = self._writable_collection("push_log", dict)
        position = self.console_len
        existing = push_log.get(position)
        if existing is None:
            push_log[position] = message
        elif isinstance(existing, str):
            push_log[position] = [existing, message]
        else:
            existing.append(message)

    def update_state(self, **kwargs):
        """Retain overwritten bindings and advance the generation once per batch."""
        missing = object()
        copied = False
        for name in kwargs:
            previous = self.get_local(name, missing)
            if previous is missing:
                previous = self.get_global(name, missing)
            if previous is missing:
                continue
            if self._update_log is None:
                self._update_log = {}
            entry = (self._generation, previous)
            history = self._update_log.get(name)
            if history is None:
                self._update_log[name] = entry
            elif hasattr(history, "append"):
                history.append(entry)
            else:
                self._update_log[name] = [history, entry]
            copied = True
        if copied:
            self._generation = 1 if self._generation is None else self._generation + 1
        self._writable_collection("local_state", dict).update(kwargs)

    def update_globals(self, **kwargs):
        """Merge global bindings, creating storage if needed."""
        self._writable_collection("global_state", dict).update(kwargs)

    def fetch_state(self, keys: Iterable[str],
                    generation: int | None) -> Iterable[Tuple[str, Any]]:
        """Resolve every requested binding before yielding, preserving input order."""
        resolved = [(key, self.fetch_state_var(key, generation)) for key in keys]
        yield from resolved

    def fetch_state_var(self, key: str, generation: int | None) -> Any:
        """Read the value active at a generation, with None equivalent to zero."""
        generation = 0 if generation is None else generation
        history = self._update_log.get(key) if self._update_log is not None else None
        if history is not None:
            entries = history if hasattr(history, "append") else (history,)
            # Entries retain the value immediately before an overwrite.
            for recorded_generation, value in entries:
                recorded_generation = 0 if recorded_generation is None else recorded_generation
                if recorded_generation >= generation:
                    return value
        missing = object()
        value = self.get_local(key, missing)
        if value is missing:
            value = self.get_global(key, missing)
        if value is missing:
            raise KeyError(key)
        return value

    def resolve_locals(self, handles: Iterable[str],
                       raise_for_missing: bool = False) -> Dict[str, Type]:
        """Resolve runtime types with local bindings taking precedence."""
        missing = object()
        resolved = {}
        for handle in handles:
            value = self.get_local(handle, missing)
            if value is missing:
                value = self.get_global(handle, missing)
            if value is not missing:
                resolved[handle] = type(value)
            elif raise_for_missing:
                raise KeyError(handle)
        return resolved
