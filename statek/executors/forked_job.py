"""Persistent continuations viewing source history with independent execution."""
# pylint: disable=protected-access,no-member
from dataclasses import dataclass
from itertools import islice
from operator import index as integer_index

import dbzero as db0

from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, WarmupLogItem,
)
from statek.executors.job import (
    Job, JobStatus, JobCompactionResult, _get_static_task_difficulty,
)
from statek.pyenv import fork_env
from statek.utils import CodeBlock


@db0.memo
class _ForkedList:
    """A bounded source prefix followed by a persistent, independently owned tail."""

    def __init__(self, owner, attribute, boundary):
        self.owner = owner
        self.attribute = attribute
        self.boundary = boundary
        self.tail = []

    def __len__(self):
        return self.boundary + len(self.tail)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[position] for position in range(*key.indices(len(self)))]
        position = integer_index(key)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError('forked list index out of range')
        if position < self.boundary:
            return getattr(self.owner, self.attribute)[position]
        return self.tail[position - self.boundary]

    def __setitem__(self, key, value):
        position = integer_index(key)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError('forked list index out of range')
        if position < self.boundary:
            raise TypeError('Cannot replace inherited history')
        self.tail[position - self.boundary] = value

    def __iter__(self):
        return (self[position] for position in range(len(self)))

    def append(self, value):
        self.tail.append(value)

    def extend(self, values):
        for value in values:
            self.append(value)


@db0.memo
@dataclass(init=False, eq=False, repr=False)
class ForkedJob(Job):
    """Continue a bounded conversation using generation-resolved Python bindings.

    Variable bindings are source views with private overrides; their values remain
    shared. Historical forks do not rewind application state. Inherited history
    is a live, bounded view; only the continuation is owned by this job.
    ``forked_at`` is the first excluded raw chat-log index, including user
    messages and warmup entries. The inherited range is [0, forked_at).
    """
    __src_job: Job
    __forked_at: int

    def __init__(self, src_job: Job, forked_at: int):
        if not isinstance(src_job, Job):
            raise TypeError('src_job must be a Job')
        if not isinstance(forked_at, int) or isinstance(forked_at, bool):
            raise TypeError('forked_at must be an integer')
        if not 0 <= forked_at <= len(src_job.chat_log):
            raise ValueError('forked_at is outside the source history')
        end = forked_at
        inherited_log = islice(src_job.chat_log, end)
        for item in inherited_log:
            if not isinstance(item, LLM_LogItem):
                continue
            if item.llm_resp is None:
                raise ValueError('Cannot inherit a pending LLM response')
            calls = item.llm_resp.tool_calls if isinstance(item.llm_resp, CodeBlock) else None
            if calls:
                for index in range(len(calls)):
                    try:
                        item.get_tool_result(index)
                    except (KeyError, IndexError) as error:
                        raise ValueError('Cannot inherit incomplete tool results') from error

        source_console_len = src_job.py_env.console_len
        console_end = next((item.console_pos for item in islice(src_job.chat_log, end, None)
                            if isinstance(item, ChatLogItem)
                            and not isinstance(item, DifficultyChangeLogItem)), source_console_len)
        if end == 0:
            console_end = 0
        # A push at the cutoff console position belongs to the retained tail.
        # A full-history fork also inherits pushes awaiting the next LLM turn.
        full_history = end == len(src_job.chat_log)
        push_end = (console_end + 1 if full_history else max(1, console_end)) if end else 0
        summaries = src_job._Job__job_summaries or []
        if isinstance(summaries, JobCompactionResult):
            summaries = [summaries]
        selected_summaries = [result for result in summaries
                              if result.chat_log_pos + result.chat_log_len <= end]
        self.__src_job = src_job
        self.__forked_at = forked_at
        self.__inherited_console_len = console_end
        # Establish the model baseline before Job initializes usage pricing.
        self._Job__last_difficulty = (src_job._Job__last_difficulty
                                     or _get_static_task_difficulty(src_job.job_def.metadata))
        environment = fork_env(src_job.py_env)
        environment._bound_history(console_end, push_end)
        super().__init__(job_def=src_job.job_def, job_status=JobStatus.STARTED,
                         py_env=environment,
                         chat_log=_ForkedList(self.__src_job, 'chat_log', end))
        self._sync_usage_pricing()
        self._Job__job_summaries = selected_summaries
        if not any(isinstance(item, WarmupLogItem) for item in islice(src_job.chat_log, end, None)):
            self._Job__warmup_summary = src_job._Job__warmup_summary

    @property
    def forked_at(self) -> int:
        """First raw source chat-log index excluded from the inherited history."""
        return self.__forked_at

    def _execution_chat_log(self):
        return self.chat_log[self.__forked_at:]

    @property
    def num_execution_turns(self) -> int:
        """Count only responses recorded by this fork."""
        return sum(isinstance(item, LLM_LogItem) for item in self._execution_chat_log())

    @property
    def last_chat_log_item(self):
        """Return only a local item that the execution engine may modify."""
        return next((item for item in reversed(self._execution_chat_log())
                     if not isinstance(item, DifficultyChangeLogItem)), None)

    def get_next_code_block(self):
        """Never replay inherited warmup or source responses."""
        return None if self.status == JobStatus.DONE else self.last_response

    def _log_console_batch(self, from_pos, to_pos):
        """Log only console output produced by this fork."""
        super()._log_console_batch(max(from_pos, self.__inherited_console_len), to_pos)

    def _exception_messages_by_turn(self):
        remaining = sum(isinstance(item, LLM_LogItem)
                        for item in self.chat_log[:self.__forked_at])
        for item, messages in super()._exception_messages_by_turn():
            if isinstance(item, WarmupLogItem):
                continue
            if remaining:
                remaining -= 1
                continue
            yield item, messages
