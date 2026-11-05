"""Persistent continuations with independent history and execution accounting."""
# pylint: disable=protected-access,no-member
from dataclasses import dataclass, fields

import dbzero as db0

from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, ToolError, UserLogItem,
    WarmupLogItem,
)
from statek.executors.job import (
    Job, JobStatus, JobCompactionResult, JobSummary, StateSummary, PyHandleDesc,
    _get_static_task_difficulty,
)
from statek.pyenv import Error, PyEnv
from statek.utils import CodeBlock, CallSpec


def _copy_history(value):
    """Copy owned history data while preserving external application handles."""
    if isinstance(value, (ChatLogItem, UserLogItem, CodeBlock, CallSpec, ToolError,
                          Error, JobCompactionResult, JobSummary, StateSummary, PyHandleDesc)):
        return type(value)(**{
            field.name: _copy_history(
                getattr(value, field.name, field.default)
                if field.name in ('raw_response_text', 'est_token_usage', '_token_estimation_card')
                else getattr(value, field.name))
            for field in fields(value)
        })
    if isinstance(value, (list, db0.types.List)):
        return [_copy_history(item) for item in value]
    if isinstance(value, (dict, db0.types.Dict)):
        return {key: _copy_history(item) for key, item in value.items()}
    if isinstance(value, (set, db0.types.Set)):
        return {_copy_history(item) for item in value}
    if isinstance(value, (tuple, db0.types.Tuple)):
        return tuple(_copy_history(item) for item in value)
    return value


@db0.memo
@dataclass(init=False, eq=False, repr=False)
class ForkedJob(Job):
    """Continue a bounded conversation using current Python variable bindings.

    Variable maps are private, but objects referenced by their values remain
    shared. Historical forks do not rewind application state.
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
        inherited_log = src_job.chat_log[:end]
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

        source_console = src_job.py_env.console or []
        console_end = next((item.console_pos for item in src_job.chat_log[end:]
                            if isinstance(item, ChatLogItem)
                            and not isinstance(item, DifficultyChangeLogItem)), len(source_console))
        if end == 0:
            console_end = 0
        # A push at the cutoff console position belongs to the retained tail.
        # A full-history fork also inherits pushes awaiting the next LLM turn.
        full_history = end == len(src_job.chat_log)
        push_log = {
            key: _copy_history(value)
            for key, value in (src_job.py_env.push_log or {}).items()
            if (full_history or end > 0)
            and (key == 0 or key < console_end or (full_history and key <= console_end))
        }
        summaries = getattr(src_job, '_Job__job_summaries', None) or []
        if isinstance(summaries, JobCompactionResult):
            summaries = [summaries]
        selected_summaries = [result for result in summaries
                              if result.chat_log_pos + result.chat_log_len <= end]
        self.__src_job = src_job  # pylint: disable=unused-private-member
        self.__forked_at = forked_at
        self.__inherited_log_len = end
        self.__inherited_console_len = console_end
        # Establish the model baseline before Job initializes usage pricing.
        self.__initial_difficulty = (getattr(src_job, '_Job__last_difficulty', None)
                                     or _get_static_task_difficulty(src_job.job_def.metadata))
        self.__initializing = True
        environment = PyEnv(
            local_state=dict(src_job.py_env.local_state or {}),
            global_state=dict(src_job.py_env.global_state or {}),
            console=list(source_console[:console_end]),
            exceptions={key: _copy_history(value)
                        for key, value in (src_job.py_env.exceptions or {}).items()
                        if key < console_end},
            push_log=push_log,
        )
        super().__init__(job_def=src_job.job_def, job_status=JobStatus.STARTED,
                         py_env=environment,
                         chat_log=[_copy_history(item) for item in inherited_log])
        self._Job__last_difficulty = self.__initial_difficulty
        self.__initializing = False
        self._sync_usage_pricing()
        self._Job__job_summaries = [_copy_history(result) for result in selected_summaries]
        if not any(isinstance(item, WarmupLogItem) for item in src_job.chat_log[end:]):
            self._Job__warmup_summary = _copy_history(
                getattr(src_job, '_Job__warmup_summary', None)
            )

    def get_current_difficulty(self):
        """Keep initialization on the source model without replaying escalation."""
        if self.__initializing:
            return self.__initial_difficulty
        return super().get_current_difficulty()

    @property
    def forked_at(self) -> int:
        """First raw source chat-log index excluded from the inherited history."""
        # Older persisted forks stored an LLM count in __forked_at; their
        # inherited-log boundary already records the corresponding raw index.
        return getattr(self, '_ForkedJob__inherited_log_len', self.__forked_at)

    def _execution_chat_log(self):
        return self.chat_log[self.__inherited_log_len:]

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
                        for item in self.chat_log[:self.__inherited_log_len])
        for item, messages in super()._exception_messages_by_turn():
            if isinstance(item, WarmupLogItem):
                continue
            if remaining:
                remaining -= 1
                continue
            yield item, messages
