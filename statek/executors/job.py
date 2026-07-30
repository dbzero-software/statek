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
from datetime import datetime
from collections import namedtuple
import hashlib
import json
import re
import traceback as _traceback_module
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Iterable,
    Dict,
    Any,
    Sequence,
    Tuple,
    Type,
    Union,
)
import dbzero as db0
from dbzero import memo, enum
from statek.pyenv import PyEnv
from statek.executors.llm_usage import LLM_Usage
from statek.executors.chat_log_item import (
    ChatLogItem,
    LLM_LogItem,
    PostProcessedItem,
    ReminderLogItem,
    SubTaskLogItem,
    ToolError,
    WarmupLogItem,
    UserLogItem,
)
from statek.llm_api import LLM_API, LLM_Response, LLM_StepData
from statek.provider_config import ProviderConfig, provider_config_identity
from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.utils import (
    _STATEK_TOOL_MARKER,
    CallSpec,
    CallSpecWrapper,
    CodeBlock,
    ParsedWarmupBlock,
    _is_empty_code,
    _is_hidden_warmup_block,
    _find_locals_in_context,
    build_warmup_code,
    extract_dialog,
    get_current_job,
    parse_tool_log,
    parse_warmup_block,
    perm_ctx_get,
    prompt_append_console,
    strip_markup,
)
from statek.future import FutureResult
from statek.locale import get_language_rule, get_language_hint
from statek.model_name import ensure_model_name, format_model_for_provider, select_model_provider
from statek.model_pricing import get_model_pricing
from statek.settings import get_statek_settings, ChatStyle
from statek.executors.post_processor import (
    PostProcessor,
    PostProcessingInput,
    post_processing_identity,
    stored_post_processing,
)
from statek.task_difficulty import (
    TaskDifficulty,
    max_task_difficulty,
    parse_task_difficulty,
)

if TYPE_CHECKING:
    from statek.agents.dialog_agent import Reminder
    from statek.task import SubTaskHandler

DialogItem = namedtuple("DialogItem", ["role", "message"])
WarmupBlockInput = Union[str, "CodeBlock"]
WarmupCodeInput = Optional[Union[WarmupBlockInput, Sequence[WarmupBlockInput]]]
ParsedWarmupCode = Optional[Union[str, "CodeBlock", List[Union[str, "CodeBlock"]]]]

"""
READY: a fresh job instance ready for execution
WARMING_UP: executing startup code before first LLM interaction
STARTED: job execution in progress
SUSPENDED: job execution suspended - waiting on external events
DONE: execution has been completed (with either success or failure)
"""
@enum(values=["READY", "WARMING_UP", "STARTED", "SUSPENDED", "DONE"])
class JobStatus:
    pass

def _parsed_code_block(block: CodeBlock) -> Optional[ParsedWarmupBlock]:
    """Return a parsed warmup representation for an already-parsed block."""
    if not block.code and not block.tool_calls and not block.metadata:
        return None
    return ParsedWarmupBlock(
        code=block.code,
        tool_calls=list(block.tool_calls) if block.tool_calls else [],
        metadata=dict(block.metadata) if block.metadata else {},
    )


def _parse_warmup_blocks(warmup_code: WarmupCodeInput) -> Optional[List[ParsedWarmupBlock]]:
    """Parse raw warmup input into per-block parsed warmup definitions.

    Args:
        warmup_code: Single raw warmup string, sequence of blocks, or None.

    Returns:
        Parsed warmup blocks, or None when no non-empty blocks are present.
    """
    if warmup_code is None:
        return None

    if isinstance(warmup_code, str):
        raw_blocks = re.split(r'\n\s*#\s*-{10,}\s*\n', warmup_code)
    elif isinstance(warmup_code, CodeBlock):
        raw_blocks = [warmup_code]
    else:
        raw_blocks = list(warmup_code)

    parsed_blocks = []
    for block in raw_blocks:
        if isinstance(block, CodeBlock):
            parsed_block = _parsed_code_block(block)
            if parsed_block is not None:
                parsed_blocks.append(parsed_block)
            continue
        stripped_block = block.strip()
        if stripped_block:
            parsed_blocks.append(parse_warmup_block(stripped_block))

    if not parsed_blocks:
        return None

    return parsed_blocks


def parse_warmup_code_with_metadata(
    warmup_code: WarmupCodeInput,
) -> Tuple[ParsedWarmupCode, Optional[Dict[str, Any]]]:
    """Parse warmup code and return aggregate metadata from parsed blocks.

    Args:
        warmup_code: Single raw warmup string, sequence of blocks, or None.

    Returns:
        Tuple containing built warmup code and aggregate metadata, each optional.
    """
    parsed_blocks = _parse_warmup_blocks(warmup_code)
    if parsed_blocks is None:
        return None, None

    metadata = {}
    for parsed_block in parsed_blocks:
        if parsed_block.metadata:
            metadata.update(parsed_block.metadata)

    return build_warmup_code(parsed_blocks), metadata or None


def parse_warmup_code(warmup_code: WarmupCodeInput) -> ParsedWarmupCode:
    """Parse warmup_code through parse_warmup_block and build_warmup_code.

    If warmup_code is a string containing comment lines with 10+ dashes
    (e.g. # ----------), it will be split into multiple blocks first.
    Each block is then parsed for tool-call annotations and converted into
    a plain string (no tool calls) or a CodeBlock (tool calls present).

    Args:
        warmup_code: Single string/CodeBlock, sequence of strings/CodeBlocks,
            or None

    Returns:
        None if input is None or results in no blocks
        Single str or CodeBlock if only one block
        List of str and/or CodeBlock if multiple blocks
    """
    parsed_code, _metadata = parse_warmup_code_with_metadata(warmup_code)
    return parsed_code


def parse_model_metadata(input: str) -> Union[str, Dict[TaskDifficulty, str]]:
    """Parse MODEL metadata into a plain model or a difficulty mapping.

    Supported forms:
      - ``gpt-5.4-mini``
      - ``L:gpt-5.4-nano,MH:gpt-5.4-mini``
      - ``LM:gpt-5.4-mini,H:gpt-5.4``

    If any difficulty labels are used, all three levels must be specified
    exactly once.
    """
    if input is None:
        raise ValueError("MODEL metadata must be provided")

    value = str(input).strip()
    if not value:
        raise ValueError("MODEL metadata must not be empty")

    if ":" not in value:
        return value

    result: Dict[TaskDifficulty, str] = {}
    for part in value.split(","):
        labels, sep, model = part.partition(":")
        if not sep:
            raise ValueError(
                "MODEL metadata must use either a single model or difficulty labels for all entries"
            )

        labels = labels.strip().upper()
        model = model.strip()
        if not labels or not model:
            raise ValueError("MODEL metadata difficulty labels and model names must be non-empty")

        for label in labels:
            difficulty = parse_task_difficulty(label)
            if difficulty in result:
                raise ValueError(f"Duplicate MODEL metadata for difficulty {difficulty}")
            result[difficulty] = model

    missing = [difficulty for difficulty in TaskDifficulty.values() if difficulty not in result]
    if missing:
        raise ValueError(
            "MODEL metadata must specify all difficulty levels when using difficulty labels"
        )
    return result


def _is_model_mapping(value) -> bool:
    """Return whether value behaves like a difficulty-to-model mapping."""
    return not isinstance(value, str) and hasattr(value, "items") and hasattr(value, "__getitem__")


def _format_task_difficulty_short(difficulty: TaskDifficulty) -> str:
    """Return the short label used in MODEL metadata."""
    return str(difficulty)[0].upper()


def _get_static_task_difficulty(metadata: Optional[Dict[str, Any]]) -> TaskDifficulty:
    """Return the configured static difficulty, falling back to StatekSettings."""
    metadata = metadata or {}
    difficulty = parse_task_difficulty(metadata.get("DEFAULT_DIFFICULTY"))
    if difficulty is not None:
        return difficulty
    return parse_task_difficulty(get_statek_settings().statek_default_difficulty)


def _get_example_difficulty_for_job(
    agent_name: Optional[str],
    example_id: int,
) -> Optional[TaskDifficulty]:
    """Return the configured difficulty for an agent example."""
    if agent_name is None:
        return None

    from statek.agents.list_of_examples import (  # pylint: disable=import-outside-toplevel
        get_example_difficulty,
    )

    return get_example_difficulty(agent_name, example_id)


def _next_task_difficulty(difficulty: TaskDifficulty) -> TaskDifficulty:
    """Return the next higher task difficulty level."""
    values = tuple(TaskDifficulty.values())  # pylint: disable=no-member
    index = values.index(difficulty)
    if index >= len(values) - 1:
        raise RuntimeError(f"Job is already at high difficulty: {difficulty}")
    return values[index + 1]


def _warmup_code_text(warmup) -> str:
    """Return concatenated LLM-facing text of non-hidden warmup code blocks."""
    if warmup is None:
        return ""
    if _is_hidden_warmup_block(warmup):
        return ""
    if isinstance(warmup, str):
        return warmup
    if isinstance(warmup, CodeBlock):
        return warmup.code or ""
    return "".join(
        w if isinstance(w, str) else (w.code or "")
        for w in warmup
        if not _is_hidden_warmup_block(w)
    )


def _tool_log_text(item: Optional["LLM_LogItem"]) -> str:
    """Return all tool log text from an LLM_LogItem as a single string."""
    if item is None or item.tool_log is None:
        return ""
    tool_log = item.tool_log
    if isinstance(tool_log, str):
        return tool_log
    if isinstance(tool_log, ToolError):
        return tool_log.err_message
    return "\n".join(tl if isinstance(tl, str) else tl.err_message for tl in tool_log)


def _llm_resp_output_tokens(llm_resp) -> int:
    """Approximate output token count for an LLM response."""
    if isinstance(llm_resp, str):
        return len(llm_resp) // 4
    code = (llm_resp.code or "") if llm_resp else ""
    tool_calls = list(llm_resp.tool_calls) if llm_resp and llm_resp.tool_calls else []
    if not tool_calls:
        return len(code) // 4
    tool_json = json.dumps([{"name": cs.func_name, "arguments": dict(cs.kwargs or {})} for cs in tool_calls])
    return (len(code) + len(tool_json)) // 4


@memo
class JobDefError:
    """Container for storing error information from a job failure."""
    error_message: str
    """Populated when the traceback collection is turned on"""
    traceback: Optional[List[str]] = None

    def __init__(self, error: Exception, collect_traceback: bool = True):
        """Create JobDefError initialized with information from exception."""
        self.error_message = str(error)
        if collect_traceback and error.__traceback__ is not None:
            self.traceback = _traceback_module.format_tb(error.__traceback__)
        else:
            self.traceback = None


_JOBDEF_HASH_TAG_PREFIX = "STATEK_JOBDEF:H:"
_JOB_LIFETIME_TAG = "STATEK_JOB"


def _job_def_identity_hash(
    warmup_code,
    model_family,
    model,
    job_params,
    locale,
    chat_style,
    post_processing,
    provider_config=None,
) -> str:
    payload = (
        warmup_code,
        model_family,
        model,
        job_params,
        locale,
        chat_style,
        post_processing_identity(post_processing),
        provider_config_identity(provider_config),
    )
    encoded = str(payload).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:4]


def _job_def_identity_tag(
    warmup_code,
    model_family,
    model,
    job_params,
    locale,
    chat_style,
    post_processing=None,
    provider_config=None,
) -> str:
    return (
        f"{_JOBDEF_HASH_TAG_PREFIX}"
        f"{_job_def_identity_hash(warmup_code, model_family, model, job_params, locale, chat_style, post_processing, provider_config)}"
    )


def job_def_identity_tag_for_job_def(job_def: "JobDef") -> str:
    """Return the single derived lookup tag for a JobDef."""
    return _job_def_identity_tag(
        job_def.warmup_code,
        job_def.model_family,
        job_def.model,
        job_def.job_params,
        job_def.locale,
        getattr(job_def, "_chat_style", None),
        job_def.post_processing,
        job_def.provider_config,
    )


@memo
@db0.tag_fields("agent")
@dataclass
class JobDef:
    """
    The `JobDescr` instances, as the name suggests - hold job descriptions / definitions.
    """
    # An agent assigned to this job
    agent: "Agent"
    # Frozen job configuration such as model, temperature, reasoning, etc.
    metadata: Optional[Dict[str, str]] = None
    # Job params to be fed into the agent's prompt template
    job_params: Optional[Dict[str, Any]] = None
    # Optional warmup code (single block or sequence of blocks) executed before the first prompt
    warmup_code: Optional[Union[str, CodeBlock, Sequence[Union[str, CodeBlock]]]] = None
    # Optional job-level chat style override (falls back to StatekSettings.chat_style)
    _chat_style: Optional[ChatStyle] = None
    # Optional locale for language-specific behaviour
    locale: Optional["StatekLocale"] = None
    # Optional post-processing sequence applied after LLM responses
    post_processing: PostProcessingInput = None
    # Durable provider configuration snapshot used to create this job definition.
    provider_config: Optional[ProviderConfig] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = (
                self.agent._metadata
                if self.agent is not None and self.agent._metadata
                else {}
            )

        metadata_model = self.metadata.get('MODEL') if self.metadata else None
        if metadata_model is None:
            role = self.agent.role if self.agent is not None else '<unknown>'
            raise ValueError(
                f"JobDef for agent '{role}' requires metadata field 'MODEL'"
            )
        if not _is_model_mapping(metadata_model):
            self.metadata["MODEL"] = parse_model_metadata(metadata_model)
        self.post_processing = stored_post_processing(self.post_processing)
        self._sync_identity_hash_tag()

    def _sync_identity_hash_tag(self, old_tag: Optional[str] = None) -> None:
        """Ensure this JobDef has exactly one current identity hash tag."""
        new_tag = job_def_identity_tag_for_job_def(self)
        if old_tag is not None and old_tag != new_tag:
            db0.tags(self).remove(old_tag)
        db0.tags(self).add(new_tag)

    @property
    def model(self) -> Optional[str]:
        """Return the frozen model configured for this job definition."""
        metadata = self.metadata or {}
        model = metadata.get('MODEL')
        if _is_model_mapping(model):
            models = {value for _, value in model.items()}
            if len(models) == 1:
                return next(iter(models))
            return ",".join(
                f"{_format_task_difficulty_short(difficulty)}:{model[difficulty]}"
                for difficulty in TaskDifficulty.values()
                if difficulty in model
            )
        return str(model) if model is not None else None

    @property
    def model_family(self) -> Optional[str]:
        """Return the frozen model family configured for this job definition."""
        model = self.model
        if model is None:
            return None
        return ensure_model_name(model).model_family

    @property
    def provider(self) -> Optional[str]:
        """Return the effective LLM provider for this job definition."""
        metadata = self.metadata or {}
        return select_model_provider(
            self.model or None,
            default_provider=(
                metadata.get("PROVIDER")
                or get_statek_settings().default_llm_api_provider
            ),
        )

    def set_error(self, error: Exception, collect_traceback: bool = True) -> None:
        """Create a JobDefError from the given exception and associate it with this JobDef."""
        jde = JobDefError(error, collect_traceback=collect_traceback)
        db0.tags(jde).add(db0.as_tag(self))

    def get_errors(self) -> Iterable["JobDefError"]:
        """Yield all JobDefError instances associated with this JobDef."""
        yield from db0.find(JobDefError, db0.as_tag(self))

    def has_errors(self) -> bool:
        """Return True if any errors are associated with this JobDef."""
        return any(True for _ in self.get_errors())

    def clear_errors(self):
        """Remove all JobDefError instances associated with this JobDef."""
        tag = db0.as_tag(self)
        for err in list(self.get_errors()):
            db0.tags(err).remove(tag)
        
    def update_warmup_code(
        self,
        warmup_code: Optional[Union[str, Sequence[str]]],
    ) -> None:
        """Update warmup_code only when the parsed new value differs from the current one.

        Parses the input through parse_warmup_code before comparing. A debug
        message is logged only when an actual change is applied.

        Args:
            warmup_code: New warmup code (raw string, sequence of strings, or None).
        """
        new_value = parse_warmup_code(warmup_code)
        if new_value != self.warmup_code:
            old_tag = job_def_identity_tag_for_job_def(self)
            self.warmup_code = new_value
            self._sync_identity_hash_tag(old_tag)

    def set_chat_style(self, chat_style: Optional[ChatStyle]) -> None:
        """Update _chat_style only when the new value differs from the current one.

        Args:
            chat_style: New chat style value, or None to clear the override.
        """
        if self._chat_style != chat_style:
            old_tag = job_def_identity_tag_for_job_def(self)
            self._chat_style = chat_style
            self._sync_identity_hash_tag(old_tag)

    def set_post_processing(self, post_processing: PostProcessingInput) -> None:
        """Update post_processing only when the effective processor identity changes."""
        if post_processing_identity(self.post_processing) != post_processing_identity(post_processing):
            old_tag = job_def_identity_tag_for_job_def(self)
            self.post_processing = stored_post_processing(post_processing)
            self._sync_identity_hash_tag(old_tag)

    @property
    def chat_style(self) -> Optional[ChatStyle]:
        """Return the job-level chat style, or fall back to the system default from StatekSettings."""
        if self._chat_style is not None:
            return self._chat_style
        return get_statek_settings().chat_style

@memo
@dataclass
class ErrorHandler:
    """Holds an error-handling callable together with the context it should be called with.

    The ``__call__`` method forwards the underlying handler with ``context`` and an
    optional exception, suppressing any exception the handler itself might raise.
    """
    error_handler: Callable
    context: Any

    def __call__(self, error: Optional[Exception] = None):
        try:
            self.error_handler(self.context, error=error)
        except Exception:  # pylint: disable=broad-except
            pass


@memo
@db0.tag_fields("agent", "job_def", "job_status")
class Job:
    """
    A single "job" is a stateful class representing the current state of a single unit-of-work, being performed end-to-end by a single agent. By a "job" we might mean either a very simple operation such as answering a basic question ("Hey, what day of week is today") or a complex task involving retrieving information from external systems, communicating with external actors, waiting for approvals etc. - before the final response is generated.
    """

    def __init__(
        self,
        job_def: JobDef,
        model_family: Optional[str] = None,
        model: Optional[str] = None,
        job_status: JobStatus = JobStatus.READY,
        py_env: PyEnv = None,
        chat_log: List[Union[str, ChatLogItem, UserLogItem]] = None,
        awaited_result: Optional[FutureResult] = None,
        next_instr_num: Optional[int] = None,
        warmup_block_num: Optional[int] = None,
        error: Optional[JobDefError] = None,
        created_at: Optional[datetime] = None,
        parent_job: Optional["Job"] = None,
    ):
        del model_family, model  # backward-compatible init args; source of truth is JobDef
        self.job_def = job_def
        self.agent = self.job_def.agent
        self.parent_job = parent_job
        self.job_status = job_status
        db0.tags(self).add(_JOB_LIFETIME_TAG)
        # LLM program's execution environment
        self.py_env = py_env if py_env is not None else PyEnv()
        # Current chat state
        self.chat_log = chat_log if chat_log is not None else []
        # Suspended job's awaited result
        self.awaited_result = awaited_result
        # Continuation instruction number
        self.next_instr_num = next_instr_num
        # Continuation warmup block number (for multi-block warmup_code)
        self.warmup_block_num = warmup_block_num
        # Error information if the job failed
        self.error: Optional[JobDefError] = error
        # Timestamp when the job object was created
        self.created_at = created_at or datetime.now()
        # Registered error handlers (ErrorHandler instances)
        self.error_handlers: List[ErrorHandler] = []
        # The last dynamically-resolved difficulty level in this task.
        self.__last_difficulty: Optional[TaskDifficulty] = None
        self.usage = LLM_Usage(pricing=self._current_model_pricing())
        self.error = None
        # Number of completed DONE transitions (None until first completion)
        self.num_completions: Optional[int] = None
        # Application-specific external memo references, created lazily.
        self.__ext_ref = None
        # Subtask notifications received while the current chat item is active.
        self.__pending_chat_log: List[SubTaskLogItem] = []
        # Shared context is bound explicitly by the init_shared_context tool.
        self.__shared_context = None
        # Locale selected by the queued user message awaiting its response.
        self.__next_response_locale: Optional["StatekLocale"] = None

        # Log system prompt on job creation if logging is enabled
        if self.logs_path and self.job_def.agent is not None:
            self._log(self.system_prompt())

    @property
    def model_family(self) -> Optional[str]:
        """Return the frozen model family stored on the job definition."""
        return self.job_def.model_family if self.job_def is not None else None

    @property
    def model(self) -> Optional[str]:
        """Return the frozen model stored on the job definition."""
        return self.job_def.model if self.job_def is not None else None

    def _set_shared_context(self, context: Any) -> None:
        """Bind an active shared context to this job."""
        self.__shared_context = context

    def _get_shared_context(self) -> Any:
        """Return this job's active shared context, if initialized."""
        return self.__shared_context

    def find_locals(self, var_type: Optional[Type] = None,
                    var_name: Optional[str] = None,
                    ext_scan: bool = True) -> Iterable[Any]:
        """Search this job's Python locals and optional persistent context."""
        def get_perm_ctx():
            return getattr(self.py_env, 'perm_ctx', None)

        yield from _find_locals_in_context(
            self.py_env.local_state,
            var_type=var_type,
            var_name=var_name,
            ext_scan=ext_scan,
            perm_ctx_getter=get_perm_ctx,
        )

    def add_locals(self, **kwargs) -> None:
        """Inject additional values into this job's Python local state."""
        self.py_env.update_locals(**kwargs)

    def _pending_notifications(self) -> List[SubTaskLogItem]:
        """Return pending subtask notifications, initializing old jobs lazily."""
        if not hasattr(self, "_Job__pending_chat_log") or self.__pending_chat_log is None:
            self.__pending_chat_log = []
        return self.__pending_chat_log

    def find_sub_task_handler(self, id: Optional[Any] = None) -> Optional["SubTaskHandler"]:  # pylint: disable=redefined-builtin
        """Find a pending or logged subtask handler by id, or the most recent one."""
        for item in reversed(self._pending_notifications()):
            if id is None or item.handler.id == id:
                return item.handler
        for item in reversed(self.chat_log):
            if isinstance(item, SubTaskLogItem) and (id is None or item.handler.id == id):
                return item.handler
        if id is not None:
            raise RuntimeError(f"Sub-task id={id} has not completed; wait for the callback notification.")
        return None

    def _chat_log_finalized_for_notifications(self) -> bool:
        """Return whether pending notifications can be safely appended now."""
        if not self.chat_log:
            return True
        last = self.chat_log[-1]
        if not isinstance(last, LLM_LogItem):
            return True
        if last.llm_resp is None:
            return False
        if isinstance(last.llm_resp, CodeBlock) and last.llm_resp.tool_calls:
            tool_log = last.tool_log
            if tool_log is None:
                return False
            if isinstance(tool_log, list):
                return len(tool_log) >= len(last.llm_resp.tool_calls)
            return len(last.llm_resp.tool_calls) == 1
        return True

    def _process_pending_notifications(self) -> bool:
        """Move pending subtask notifications to chat_log at safe boundaries."""
        pending = self._pending_notifications()
        if not pending or not self._chat_log_finalized_for_notifications():
            return False
        self.chat_log.extend(pending)
        self.__pending_chat_log = []
        return True

    def push_notification(self, handler: "SubTaskHandler") -> None:
        """Push a completed subtask notification into this job."""
        if not handler.is_completed:
            raise RuntimeError("Subtask handler must be completed before notification")

        item = SubTaskLogItem(
            console_pos=len(self.py_env.console) if self.py_env.console else 0,
            handler=handler,
        )
        if handler.error is None:
            item.tool_log = str(handler)

        if self.status == JobStatus.DONE:  # pylint: disable=no-member
            self.chat_log.append(item)
            self.set_status(JobStatus.STARTED)  # pylint: disable=no-member
            self.py_env.exit_status = None
            return
        self._pending_notifications().append(item)

    def _current_model_pricing(self):
        """Return pricing for the concrete model used by the current LLM request."""
        metadata = self.job_def.metadata or {}
        raw_model = self.get_current_model()
        provider = select_model_provider(
            raw_model,
            default_provider=(
                metadata.get("PROVIDER")
                or get_statek_settings().default_llm_api_provider
            ),
        )
        model_name = ensure_model_name(raw_model)
        model = format_model_for_provider(raw_model, provider) if provider else model_name.model
        provider_key = provider.upper() if provider else None
        model_family = model_name.model_family if provider_key == "OPENROUTER" else None
        return get_model_pricing(provider or "UNKNOWN", model or "", model_family)

    def _sync_usage_pricing(self) -> None:
        """Keep persisted usage objects priced against the current concrete model."""
        self.usage.pricing = self._current_model_pricing()

    def system_prompt(self, difficulty: Optional[TaskDifficulty] = None) -> str:
        """Return the agent system prompt formatted for this job's current difficulty."""
        if self.job_def is None or self.job_def.agent is None:
            return ""
        return self.job_def.agent.system_prompt(
            task_difficulty=difficulty or self.get_current_difficulty(),
            job_params=self.job_def.job_params,
        )

    def _ensure_ext_ref_storage(self):
        """Return external-reference storage, creating it for older persisted jobs."""
        if not hasattr(self, "_Job__ext_ref") or self.__ext_ref is None:
            self.__ext_ref = db0.weak_set()
        return self.__ext_ref

    def add_ext_ref(self, obj: Any) -> None:
        """Register an application-specific weak reference to an external memo object."""
        if not db0.is_memo(obj):
            return

        storage = self._ensure_ext_ref_storage()
        storage.add(obj)

    def contains_ext_ref(self, obj: Any) -> bool:
        """Return whether *obj* was previously registered as an external reference."""
        if not db0.is_memo(obj):
            return False

        storage = getattr(self, "_Job__ext_ref", None)
        if storage is None:
            return False
        return obj in storage

    def add_error_handler(self, error_handler: Callable, context: Any) -> None:
        """Register an error handler with this job.

        Args:
            error_handler: The error-handling callable (must satisfy the
                error-handler protocol, i.e. decorated with
                ``@statek.error_handler``).
            context: The context value passed as the first argument when the
                handler is invoked.
        """
        self.error_handlers.append(ErrorHandler(error_handler=error_handler, context=context))

    def add_error_handlers_from(self, parent_job: 'Job') -> None:
        """Copy all registered error handlers from *parent_job* into this job."""
        self.error_handlers.extend(parent_job.error_handlers)

    def notify_handlers(self, error: Optional[Exception] = None) -> None:
        """Invoke all registered error handlers and clear the handler list.

        Args:
            error: Optional exception that caused the job to terminate.
        """
        handlers, self.error_handlers = self.error_handlers, []
        for handler in handlers:
            handler(error=error)

    @property
    def logs_path(self) -> Optional[str]:
        """Get the logs path from StatekSettings."""
        return get_statek_settings().logs_path

    def _log(self, content: str):
        """
        Write content to the log file and to the console logger.
        
        Args:
            content: The content to log
        """
        if not self.logs_path:
            return

        from statek.settings import get_statek_logger  # pylint: disable=import-outside-toplevel
        import os

        # Write to log file
        agent_name = self.job_def.agent.role if self.job_def.agent else "unknown"
        job_uuid = db0.uuid(db0.materialized(self))
        log_filename = f"{agent_name}_{job_uuid}.log"
        log_filepath = os.path.join(self.logs_path, log_filename)
        os.makedirs(self.logs_path, exist_ok=True)
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n\n")

        # Write to console logger
        logger = get_statek_logger()
        logger.info("%s", content)

    def _log_console_batch(self, from_pos: int, to_pos: int):
        """Log a slice of the console as a single batch with XML tags if configured.

        Called at block/turn boundaries so the log matches the format sent to the LLM.

        Args:
            from_pos: First console element to include
            to_pos: One-past-the-last console element to include
        """
        if to_pos <= from_pos:
            return
        settings = get_statek_settings()
        text = prompt_append_console(
            self.py_env.console, self.job_def.chat_style,
            from_pos=from_pos, limit=to_pos - from_pos,
            xml_tags=settings.get_xml_box_tags()
        )
        if text:
            self._log(text)

    def _log_tool_call_result(self, call_spec, result: str):
        """Log a tool call and its result.

        The call line is annotated with the STATEK tool marker.
        The result is formatted like console output (``> ``-prefixed in CONSOLE
        style, plain in MARKDOWN style) and wrapped in XML console tags when
        configured.

        Args:
            call_spec: The CallSpec describing the tool call
            result: The tool execution result string
        """
        settings = get_statek_settings()
        chat_style = self.job_def.chat_style
        raw_call = f"{call_spec.format()}  {_STATEK_TOOL_MARKER}"
        if chat_style in (ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
            call_line = f"```python\n{raw_call}\n```"
        else:
            call_line = raw_call
        if result:
            result_lines = result.split('\n')
            if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
                result_text = '<CONSOLE>\n' + '\n'.join(result_lines) + '\n</CONSOLE>'
            elif chat_style in (ChatStyle.MARKDOWN, ChatStyle.DIRECT):  # pylint: disable=no-member
                result_text = '\n'.join(result_lines)
            else:
                result_text = '\n'.join(f"> {line}" for line in result_lines)
            xml_tags = settings.get_xml_box_tags()
            tag = xml_tags and xml_tags.get("console") if chat_style != ChatStyle.DIRECT else None  # pylint: disable=no-member
            if tag:
                result_text = f"<{tag}>\n{result_text}\n</{tag}>"
            self._log(f"{call_line}\n{result_text}")
        else:
            self._log(call_line)

    def console_append(self, output: str, error_message: str = None):
        """
        Append output to the console and optionally log it.

        Args:
            output: The output string to append
            error_message: optional error message (if execution resulted in an exception)
        """
        self.py_env.console_append(output)
        if error_message is not None:
            console_pos = self.chat_log[-1].console_pos if self.chat_log else 0
            if self.py_env.exceptions is None:
                self.py_env.exceptions = {}
            self.py_env.exceptions[console_pos] = error_message
        # Console is logged in batches at block/turn boundaries (see _log_console_batch)

    @property
    def status(self) -> JobStatus:
        """
        Returns the current job status.

        Returns:
            JobStatus: The current status of the job
        """
        return self.job_status

    def set_status(self, new_status: JobStatus):
        """
        Sets or updates job status. The job_status tag field keeps lookup tags
        synchronized with the persisted status field.

        Args:
            new_status: The status/tag to be assigned or updated
        """
        self.job_status = new_status

    def _language_hint_suffix(self) -> str:
        """Return the configured locale hint suffix for user messages, if enabled.

        A queued message locale applies before the JobDef locale. The result is
        empty when neither locale exists or AUTO_LANG_HINT is disabled.
        """
        locale = self._effective_locale()
        if locale is None:
            return ""
        metadata = self.job_def.metadata or {}
        if metadata.get("AUTO_LANG_HINT", "").upper() == "FALSE":
            return ""
        hint = get_language_hint(locale.lang_code)
        if hint:
            return f" ({hint})"
        return ""

    def _effective_locale(self) -> Optional["StatekLocale"]:
        """Return the queued-message locale when present, otherwise JobDef locale."""
        if not hasattr(self, "_Job__next_response_locale"):
            self.__next_response_locale = None
        return self.__next_response_locale or self.job_def.locale

    def _with_language_hint(self, message: str) -> str:
        """Append the locale language hint to a real user message when enabled."""
        if not message:
            return message
        return f"{message}{self._language_hint_suffix()}"

    def _collect_push_log(self, from_pos: int) -> str:
        """Return push_log messages with key >= from_pos as a newline-joined string.

        Args:
            from_pos: Include only entries whose console-position key is >= this value.

        Returns:
            Newline-joined string of all matching messages, or empty string if none.
        """
        if not self.py_env.push_log:
            return ""
        hint_suffix = self._language_hint_suffix()
        parts = []
        for key in sorted(self.py_env.push_log.keys()):
            if key >= from_pos:
                value = self.py_env.push_log[key]
                if not isinstance(value, str):
                    parts.extend(f"{v}{hint_suffix}" for v in value)
                else:
                    parts.append(f"{value}{hint_suffix}")
        return "\n".join(parts)

    @staticmethod
    def _subtask_lookup_code(handler: "SubTaskHandler") -> str:
        """Return the synthetic lookup code shown for a completed subtask."""
        if handler.id is None:
            return "print(find_sub_task_handler())"
        return f"print(find_sub_task_handler(id={handler.id!r}))"

    @staticmethod
    def _subtask_tool_call(idx: int, handler: "SubTaskHandler") -> CallSpecWrapper:
        """Return the synthetic python_cli call representing subtask result lookup."""
        return CallSpecWrapper(
            id=f"STATEK-SUBTASK-{idx:03d}",
            func_name="python_cli",
            args=None,
            kwargs={"code": Job._subtask_lookup_code(handler)},
        )

    def get_next_prompt(self) -> str:
        """
        Generate the next prompt to be included in the LLM chat.

        If this is the first prompt (chat_log is empty), use job_def.prompt
        and append the entire py_env console starting from position 0.
        Otherwise, format the console starting from the last chat element's
        console position to provide the console result for LLM analysis.

        If push_log is available, push_log entries whose console-position key falls
        within the relevant range are appended after the console output.

        Returns:
            The formatted prompt string ready to be sent to the LLM
        """
        chat_style = self.job_def.chat_style
        xml_tags = get_statek_settings().get_xml_box_tags()
        # Filter for LLM items only (WarmupLogItems don't count as LLM turns)
        llm_items = [item for item in self.chat_log if isinstance(item, LLM_LogItem)]
        if not llm_items:
            warmup = self.job_def.warmup_code
            if warmup:
                # Warmup exists: template and warmup turns go in chat_history as
                # alternating assistant/user pairs.  The prompt (current user message)
                # is the console output OF the last warmup block, spanning from the
                # end of the previous block to the end of the last block.
                warmup_blocks = [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
                visible_indices = [
                    idx for idx, block in enumerate(warmup_blocks)
                    if not _is_hidden_warmup_block(block)
                ]
                if not visible_indices:
                    return ""
                block_idx = visible_indices[-1]
                positions = self._warmup_end_positions()
                last_end = (
                    positions[block_idx]
                    if block_idx < len(positions)
                    else len(self.py_env.console) if self.py_env.console else 0
                )
                prev_end = (
                    positions[block_idx - 1]
                    if block_idx >= 1 and block_idx - 1 < len(positions)
                    else 0
                )
                limit = last_end - prev_end
                console_part = prompt_append_console(
                    self.py_env.console, chat_style,
                    from_pos=prev_end, limit=limit if limit > 0 else 0,
                    xml_tags=xml_tags
                ) if limit > 0 else ""
                push_part = self._collect_push_log(from_pos=prev_end)
                parts = [p for p in [console_part, push_part] if p]
                return "\n".join(parts)
            else:
                # No warmup: system_prompt + all console is the first user prompt
                sys_prompt = self.system_prompt() or None
                console_part = prompt_append_console(
                    self.py_env.console, chat_style, from_pos=0, xml_tags=xml_tags
                )
                push_part = self._collect_push_log(from_pos=0)
                parts = [p for p in [sys_prompt, console_part, push_part] if p]
                return "\n".join(parts)
        else:
            # Not first prompt: format console from last LLM element's console position
            from_pos = llm_items[-1].console_pos
            console_part = prompt_append_console(
                self.py_env.console,
                chat_style,
                from_pos=from_pos,
                xml_tags=xml_tags
            )
            push_part = self._collect_push_log(from_pos=from_pos)
            parts = [p for p in [console_part, push_part] if p]
            return "\n".join(parts)

    def _iter_chat_history(
        self,
        chat_log: Optional[Sequence[Union[str, ChatLogItem, UserLogItem]]] = None,
        console: Optional[Sequence[str]] = None,
        push_log: Optional[Dict[int, Union[str, List[str]]]] = None,
    ) -> Iterable[ChatHistoryItem]:
        """Yield LLM-facing chat history for the provided append-only state."""
        chat_log = self.chat_log if chat_log is None else chat_log
        push_log = self.py_env.push_log if push_log is None else push_log
        console = self.py_env.console if console is None else console

        yield from self._build_chat_history(
            chat_log=chat_log,
            console=console,
            push_log=push_log,
        )

    def _build_chat_history(
        self,
        chat_log: Sequence[Union[str, ChatLogItem, UserLogItem]],
        console: Optional[Sequence[str]],
        push_log: Optional[Dict[int, Union[str, List[str]]]],
    ) -> Iterable[ChatHistoryItem]:
        """Yield the full LLM-facing chat history as ``ChatHistoryItem`` objects.

        Lazy generator producing a normalised, provider-agnostic timeline:

          1. Optional USER item with the initial user message — sourced from
             ``chat_log[0]`` (str, set by ``push_user_message`` in
             MD_DIALOG/DIRECT) and/or ``py_env.push_log[0]``.  When neither
             exists this slot is omitted.
          2. Warmup blocks as alternating ASSISTANT (code or tool-call request)
             + USER/TOOL (console output / tool result) pairs, in
             ``warmup_block_num`` order.
          3. The same alternating pattern for LLM turns: ASSISTANT (LLM
             response) + USER/TOOL (console / tool result), woven together
             with any USER messages from ``UserLogItem`` entries, SYSTEM
             post-processor/reminder messages, and USER messages from
             ``py_env.push_log`` keyed by console position.

        ``CodeBlock`` responses with ``tool_calls`` produce an ASSISTANT item
        with ``tool_calls`` set, followed by one TOOL item per call carrying
        its captured result.  Plain code (no tool calls) becomes an
        ASSISTANT item with ``content`` set.  Pending LLM turns (e.g.
        ``llm_resp=None`` after a DONE→STARTED transition) emit no assistant
        item but still surface any preceding push_log user messages.

        Yields:
            ChatHistoryItem instances ready for ``format_chat_history_item``.
        """
        push_log = push_log or {}
        console = console or []
        console_len = len(console)
        is_direct = self.job_def.chat_style == ChatStyle.DIRECT  # pylint: disable=no-member
        warmup = self.job_def.warmup_code
        warmup_blocks: List[Union[str, CodeBlock]] = (
            [warmup] if isinstance(warmup, (str, CodeBlock)) else list(warmup)
        ) if warmup else []

        # 1) Initial user message — chat_log[0] (str) and/or push_log[0].
        initial_parts: List[str] = []
        if (
            chat_log
            and isinstance(chat_log[0], str)
            and chat_log[0]
        ):
            initial_parts.append(self._with_language_hint(chat_log[0]))
        if 0 in push_log:
            v = push_log[0]
            if isinstance(v, list):
                initial_parts.extend(
                    self._with_language_hint(s) for s in v if s
                )
            elif v:
                initial_parts.append(self._with_language_hint(v))
        if initial_parts:
            yield ChatHistoryItem(
                role=ChatRole.USER,
                content="\n".join(initial_parts),
                content_src=ContentSource.USER,
            )

        # 2) Walk chat_log items in chronological order.
        items = [it for it in chat_log if not isinstance(it, str)]

        # Pre-compute the console end position for each ChatLogItem (the
        # console_pos of the next ChatLogItem, or console_len for the last).
        end_positions: Dict[int, int] = {}
        for i, item in enumerate(items):
            if isinstance(item, UserLogItem):
                continue
            next_pos = console_len
            for j in range(i + 1, len(items)):
                if isinstance(items[j], (
                    WarmupLogItem,
                    LLM_LogItem,
                    PostProcessedItem,
                    ReminderLogItem,
                    SubTaskLogItem,
                )):
                    next_pos = items[j].console_pos
                    break
            end_positions[i] = next_pos

        # Push log keys (excluding 0, already consumed as the initial user msg).
        sorted_push_keys = sorted(k for k in push_log.keys() if k != 0)

        def _yield_pushes(lo: int, hi: int) -> Iterable[ChatHistoryItem]:
            """Yield USER items for push_log entries in (lo, hi]."""
            for k in sorted_push_keys:
                if lo < k <= hi:
                    v = push_log[k]
                    msgs = v if isinstance(v, list) else [v]
                    for msg in msgs:
                        yield ChatHistoryItem(
                            role=ChatRole.USER,
                            content=self._with_language_hint(msg),
                            content_src=ContentSource.USER,
                        )

        for idx, item in enumerate(items):
            if isinstance(item, UserLogItem):
                if item.message:
                    yield ChatHistoryItem(
                        role=ChatRole.USER,
                        content=self._with_language_hint(item.message),
                        content_src=ContentSource.USER,
                    )
                continue

            if isinstance(item, PostProcessedItem):
                if item.message:
                    yield ChatHistoryItem(
                        role=ChatRole.SYSTEM,
                        content=item.message,
                        content_src=ContentSource.SYSTEM,
                    )
                continue

            if isinstance(item, ReminderLogItem):
                if item.reminder.text:
                    yield ChatHistoryItem(
                        role=ChatRole.SYSTEM,
                        content=item.reminder.text,
                        content_src=ContentSource.SYSTEM,
                    )
                continue

            if isinstance(item, SubTaskLogItem):
                yield ChatHistoryItem(
                    role=ChatRole.SYSTEM,
                    content=item.handler.get_log_message(),
                    content_src=ContentSource.SYSTEM,
                )
                if item.tool_log is not None:
                    call = self._subtask_tool_call(idx, item.handler)
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=None,
                        content_src=ContentSource.SYSTEM,
                        tool_calls=[call],
                    )
                    result = item.get_tool_result(0)
                    if isinstance(result, ToolError):
                        result = result.err_message
                    yield ChatHistoryItem(
                        role=ChatRole.TOOL,
                        content=result,
                        content_src=ContentSource.CONSOLE,
                        tool_calls=[call],
                    )
                continue

            # Determine the source content of the assistant turn.
            if isinstance(item, WarmupLogItem):
                block_num = item.warmup_block_num
                block = (
                    warmup_blocks[block_num]
                    if block_num < len(warmup_blocks) else None
                )
                if _is_hidden_warmup_block(block):
                    continue
                asst_src = ContentSource.SYSTEM
            elif isinstance(item, LLM_LogItem):
                block = item.llm_resp
                asst_src = ContentSource.ASSISTANT
            else:
                continue

            # Yield the assistant turn (and any tool results).
            if block is not None:
                tool_calls = (
                    list(block.tool_calls)
                    if isinstance(block, CodeBlock) and block.tool_calls
                    else None
                )
                block_code = block.code if isinstance(block, CodeBlock) else block

                # DIRECT style: wrap plain warmup code as python_cli tool calls
                # so the LLM sees them as tool use, not confusing text/console.
                if is_direct and isinstance(item, WarmupLogItem) and not tool_calls and block_code:
                    warmup_call = CallSpecWrapper(
                        id=f"STATEK-WARMUP-{item.warmup_block_num:03d}",
                        func_name="python_cli",
                        args=None,
                        kwargs={"code": block_code},
                    )
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=None,
                        content_src=asst_src,
                        tool_calls=[warmup_call],
                    )
                    # Console output as TOOL result instead of USER message
                    from_pos = item.console_pos
                    to_pos = end_positions[idx]
                    console_text = "\n".join(console[from_pos:to_pos]) if to_pos > from_pos and console else ""
                    yield ChatHistoryItem(
                        role=ChatRole.TOOL,
                        content=console_text,
                        content_src=ContentSource.CONSOLE,
                        tool_calls=[warmup_call],
                    )
                    yield from _yield_pushes(from_pos, to_pos)
                    continue

                if tool_calls:
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=block_code if block_code else None,
                        content_src=asst_src,
                        tool_calls=tool_calls,
                        provider_reasoning_payload=(
                            getattr(item, "llm_reasoning_payload", None)
                            if isinstance(item, LLM_LogItem) else None
                        ),
                    )
                    for j, cs in enumerate(tool_calls):
                        try:
                            result = item.get_tool_result(j)
                        except (KeyError, IndexError):
                            result = ""
                        if isinstance(result, ToolError):
                            result = result.err_message
                        yield ChatHistoryItem(
                            role=ChatRole.TOOL,
                            content=result,
                            content_src=ContentSource.CONSOLE,
                            tool_calls=cs,
                        )
                    # python_cli output already travelled via tool_log /
                    # TOOL items above — the console slice that also
                    # received it (for console_pos advancement) would only
                    # duplicate it as a USER message.  Skip that dump and
                    # any push_log yielding for this block.
                    from_pos = item.console_pos
                    to_pos = end_positions[idx]
                    yield from _yield_pushes(from_pos, to_pos)
                    continue
                elif block_code:
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        content=block_code,
                        content_src=asst_src,
                        provider_reasoning_payload=(
                            getattr(item, "llm_reasoning_payload", None)
                            if isinstance(item, LLM_LogItem) else None
                        ),
                    )
                elif isinstance(item, LLM_LogItem) and item.llm_reasoning_payload is not None:
                    yield ChatHistoryItem(
                        role=ChatRole.ASSISTANT,
                        provider_reasoning_payload=item.llm_reasoning_payload,
                    )

            # Console output produced by this block (USER + CONSOLE).
            from_pos = item.console_pos
            to_pos = end_positions[idx]
            if to_pos > from_pos and console:
                console_text = "\n".join(console[from_pos:to_pos])
                yield ChatHistoryItem(
                    role=ChatRole.USER,
                    content=console_text,
                    content_src=ContentSource.CONSOLE,
                )

            # Any push_log messages emitted while this block was running.
            yield from _yield_pushes(from_pos, to_pos)

    def get_chat_history(self) -> Iterable[ChatHistoryItem]:
        """Yield the full LLM-facing chat history as ``ChatHistoryItem`` objects."""
        yield from self._iter_chat_history()

    def _build_request_data(
        self,
        chat_log: Optional[Sequence[Union[str, ChatLogItem, UserLogItem]]] = None,
        console: Optional[Sequence[str]] = None,
        push_log: Optional[Dict[int, Union[str, List[str]]]] = None,
    ) -> Dict[str, Any]:
        """Build request params for the provided append-only job state."""
        metadata = dict(self.job_def.metadata or {})
        model = LLM_API.require_model(self.get_current_model())
        temperature = LLM_API.parse_temperature(metadata.get("TEMPERATURE"))
        system_prompt = self.system_prompt()

        if (
            self._effective_locale() is not None
            and metadata.get("AUTO_LANG_RULE", "").upper() != "FALSE"
        ):
            lang_rule = get_language_rule(self._effective_locale().lang_code)
            if lang_rule:
                system_prompt = f"{system_prompt}\n\n{lang_rule}"

        request_params: Dict[str, Any] = {
            "chat_history": self._iter_chat_history(
                chat_log=chat_log,
                console=console,
                push_log=push_log,
            ),
            "system_prompt": system_prompt,
            "model": model,
            "metadata": metadata,
            "available_tools": self.job_def.agent.all_tools,
            "provider_config": self.job_def.provider_config,
        }
        if temperature is not None:
            request_params["temperature"] = temperature
        chat_style = self.job_def.chat_style
        if chat_style is not None:
            request_params["chat_style"] = chat_style

        return request_params

    def get_next_request(self) -> Dict[str, Any]:
        """Build the parameter dict consumed by ``LLM_API.process_request``.

        ``chat_history`` is the lazy generator returned by
        :meth:`get_chat_history` and contains only conversational turns.
        The agent system prompt is passed separately via ``system_prompt``.

        Returns:
            Dict[str, Any]: With keys ``chat_history`` (Iterable[ChatHistoryItem]),
            ``system_prompt`` (str), ``model`` (str), ``metadata`` (dict),
            ``available_tools`` (list), and optionally ``chat_style``,
            ``temperature``, and ``provider_config``.
        """
        return self._build_request_data()

    def get_request_data(self, turn_num: int) -> Dict[str, Any]:
        """Reconstruct the request params used for a historical LLM turn."""
        if turn_num < 0:
            raise IndexError(f"turn_num out of range: {turn_num}")

        target_idx = None
        console_limit = 0
        llm_turn_num = 0
        for idx, item in enumerate(self.chat_log):
            if not isinstance(item, LLM_LogItem) or item.llm_resp is None:
                continue
            if llm_turn_num == turn_num:
                target_idx = idx
                console_limit = item.console_pos
                break
            llm_turn_num += 1

        if target_idx is None:
            raise IndexError(f"turn_num out of range: {turn_num}")

        console = self.py_env.console or []
        push_log = self.py_env.push_log or {}
        history_chat_log = self.chat_log[:target_idx]
        history_console = console[:console_limit]
        history_push_log = {
            key: value for key, value in push_log.items()
            if key <= console_limit
        }
        return self._build_request_data(
            chat_log=history_chat_log,
            console=history_console,
            push_log=history_push_log,
        )

    def get_current_model(self) -> str:
        """Return the concrete model configured for the job's current difficulty."""
        metadata = self.job_def.metadata or {}
        model_config = metadata.get("MODEL")
        if not _is_model_mapping(model_config):
            return str(model_config) if model_config is not None else None

        difficulty = self.get_current_difficulty()
        return model_config[difficulty]

    def _get_last_example_id(self) -> Optional[int]:
        perm_ctx = self.py_env.perm_ctx or {}
        if "last_example_id" in perm_ctx:
            # return from the permanent context if available
            return perm_ctx["last_example_id"]
        
        # Return from the non-persistent current context only when this job is
        # the registered execution job.
        if get_current_job() is self:
            result = perm_ctx_get("last_example_id", None)
            if result is not None:
                return result
        
        # finally, try resolving from the default_example_id in the local state
        local_state = self.py_env.local_state or {}
        if "default_example_id" in local_state:
            return local_state["default_example_id"]        
        return None
    
    def get_current_difficulty(self) -> TaskDifficulty:
        """Return the current task difficulty for this job.

        Resolution checks the last shown example first, then static job
        metadata, then settings. Example difficulty is dynamic: it updates the
        stored value only when it would keep or raise the current difficulty.
        """        
        last_example_id = self._get_last_example_id()

        if last_example_id is not None:            
            agent = self.job_def.agent if self.job_def is not None else None
            agent_name = agent.role if agent is not None else None
            difficulty = _get_example_difficulty_for_job(agent_name, last_example_id)
            # update the dynamically resolved difficulty                 
            if difficulty is not None:
                if self.__last_difficulty is None:
                    self.__last_difficulty = difficulty
                else:
                    max_difficulty = max_task_difficulty(self.__last_difficulty, difficulty)
                    # only update if changed
                    if self.__last_difficulty != max_difficulty:
                        self.__last_difficulty = max_difficulty                    

        if self.__last_difficulty is not None:
            return self.__last_difficulty
        
        return _get_static_task_difficulty(self.job_def.metadata)

    def panic(self):
        """Escalate the current job difficulty to the next higher level."""
        current_difficulty = self.get_current_difficulty()
        self.__last_difficulty = _next_task_difficulty(current_difficulty)

    def append_chat_log(self, request: Dict, llm_resp: LLM_Response):
        """
        Register the LLM response in the Job's chat_log container.

        This method appends a new ChatLogItem to the chat_log, recording the LLM's
        response and the current console position. When the LLM_Response contains
        tool call requests, the response is stored as a CodeBlock; otherwise it is
        stored as a plain string.

        Args:
            request: The original request parameters (compatible with LLM_API.process_request)
            llm_resp: The LLM's response as an LLM_Response object

        The console_pos is set to len(console), marking the position past the end
        of the current console output.
        """
        chat_style = self.job_def.chat_style
        is_md_style = chat_style in (  # pylint: disable=no-member
            ChatStyle.MARKDOWN, ChatStyle.MD_DIALOG)
        is_direct = chat_style == ChatStyle.DIRECT  # pylint: disable=no-member
        step_data = llm_resp.step_data

        if step_data.call_requests:
            # When tool calls are present we build a CodeBlock and need to
            # decide what (if anything) goes into its `code` field.
            #   - DIRECT: dialog text (not Python) — stored for chat history,
            #     never executed (exec_all_steps skips code in DIRECT mode).
            #   - MARKDOWN/MD_DIALOG: code MUST be inside ```python fences,
            #     so extract only those; never feed surrounding prose to
            #     ast.parse.
            #   - other styles: pass the response through as-is.
            if is_direct:
                response_code = extract_dialog(step_data.text or "")
            elif is_md_style:
                response_code = strip_markup(step_data.text, strict=True)
            else:
                response_code = step_data.text
            tool_calls = [
                CallSpec(id=cp.id, func_name=cp.name, args=cp.args or [], kwargs=cp.kwargs or {})
                for cp in step_data.call_requests
            ]
            stored_resp = CodeBlock(code=response_code or None, tool_calls=tool_calls)
        else:
            # No tool calls — store the response so it appears in chat
            # history. DIRECT keeps only the dialog text; markdown styles
            # keep the cleaned-up form; other styles store as-is.
            if is_direct:
                stored_resp = extract_dialog(step_data.text or "")
            else:
                stored_resp = strip_markup(step_data.text, strict=is_md_style)

        chat_item = LLM_LogItem(
            console_pos=len(self.py_env.console) if self.py_env.console else 0,
            llm_resp=stored_resp,
            llm_reasoning_payload=step_data.reasoning_payload,
        )
        self.chat_log.append(chat_item)
        self.__next_response_locale = None

        # Log the LLM response in the same format sent back as history
        resp_code = stored_resp.code if isinstance(stored_resp, CodeBlock) else stored_resp
        if resp_code is None:
            resp_code = ""
        log_resp = f"```python\n{resp_code}\n```" if is_md_style else resp_code
        self._log(log_resp)

    def handle_reminder(self, reminder: "Reminder") -> bool:
        """Process a reminder if its conditions are currently met.

        Reminder implementations own their activation conditions.
        """
        if not reminder.fire_ready(self):
            return False

        console_pos = len(self.py_env.console) if self.py_env.console else 0
        self.chat_log.append(ReminderLogItem(
            console_pos=console_pos,
            reminder=reminder,
        ))
        self.py_env.console_append(reminder.text)
        return True

    def handle_post_processor(
        self,
        post_processor: PostProcessor,
        llm_step: LLM_StepData,
    ) -> Tuple[Optional[LLM_StepData], bool]:
        """Run one post-processor and normalize its return value.

        Args:
            post_processor: Processor to run against the uncommitted LLM step.
            llm_step: LLM response that has not yet been appended to chat_log.

        Returns:
            A tuple containing the next LLM step, or None when the step is
            suppressed, plus whether the processor emitted a system message or
            replaced the input LLM step.

        Raises:
            TypeError: If the processor returns an unsupported value.
            ValueError: If a tuple return contains more than one LLM step.
        """
        result = post_processor.process(llm_step, self)
        if isinstance(result, LLM_StepData):
            return result, result is not llm_step

        if isinstance(result, str):
            self.chat_log.append(PostProcessedItem(
                console_pos=len(self.py_env.console) if self.py_env.console else 0,
                post_processor=post_processor,
                message=result,
            ))
            return None, True

        if not isinstance(result, tuple):
            raise TypeError("Unsupported post-processor return value")

        processed_step: Optional[LLM_StepData] = None
        activated = False
        for item in result:
            if isinstance(item, LLM_StepData):
                if processed_step is not None:
                    raise ValueError("Post-processor tuple may contain at most one LLM_StepData")
                processed_step = item
                activated = activated or item is not llm_step
                continue

            if isinstance(item, str):
                self.chat_log.append(PostProcessedItem(
                    console_pos=len(self.py_env.console) if self.py_env.console else 0,
                    post_processor=post_processor,
                    message=item,
                ))
                activated = True
                continue

            raise TypeError("Unsupported post-processor return value")

        return processed_step, activated

    @property
    def last_response(self) -> Union[str, CodeBlock, None]:
        """
        Retrieves the last response received from the LLM (i.e. the Python code to be executed,
        or a CodeBlock when the LLM requested tool calls) or None if chat_log is empty.
        """
        if not self.chat_log:
            return None
        last = self.chat_log[-1]
        if isinstance(last, LLM_LogItem):
            return last.llm_resp
        return None

    def _is_user_facing_llm_response(self, item: LLM_LogItem) -> bool:
        """Return True when the LLM log item corresponds to a visible reply.

        ``get_response_times()`` tracks user-visible latency, so internal
        execution turns in dialog styles should not count as the first reply.
        """
        response = item.llm_resp
        if response is None:
            return False

        chat_style = self.job_def.chat_style
        if chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
            return isinstance(response, str) and bool(response.strip())

        if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
            if isinstance(response, CodeBlock):
                return False
            if not isinstance(response, str):
                return False
            return any(line.startswith("# ") for line in response.splitlines())

        return True

    @staticmethod
    def _chat_response_from_tool_log(line: str) -> Optional[str]:
        """Extract a user-facing answer body from a console tool-log line."""
        call_params = parse_tool_log(line)
        if call_params is None or call_params.name != "answer":
            return None

        body = None
        if call_params.args:
            body = call_params.args[0]
        if call_params.kwargs:
            body = call_params.kwargs.get("body", body)
            body = call_params.kwargs.get("content", body)
        if not isinstance(body, str) or not body:
            return None
        return body

    def _iter_chat_response_tool_logs(self, from_pos: int, to_pos: int) -> Iterable[str]:
        """Yield answer bodies from console output in ``[from_pos, to_pos)``."""
        console = self.py_env.console or []
        for output in console[from_pos:to_pos]:
            for line in output.splitlines():
                body = self._chat_response_from_tool_log(line)
                if body is not None:
                    yield body

    @staticmethod
    def _dialog_from_code_block(resp: CodeBlock) -> Optional[str]:
        """Extract dialog from a raw MD_DIALOG response stored as a CodeBlock."""
        return extract_dialog(resp.code or "")

    def get_chat_responses(self) -> Iterable[str]:
        """Yield user-facing chat responses from the chat log, oldest first."""
        console_len = len(self.py_env.console) if self.py_env.console else 0
        items = [
            item for item in self.chat_log
            if isinstance(item, (WarmupLogItem, LLM_LogItem))
        ]
        end_positions = [
            items[idx + 1].console_pos if idx + 1 < len(items) else console_len
            for idx in range(len(items))
        ]

        for idx, item in enumerate(items):
            if isinstance(item, LLM_LogItem) and self._is_user_facing_llm_response(item):
                resp = item.llm_resp
                if isinstance(resp, str):
                    yield resp
                elif isinstance(resp, CodeBlock) and resp.code:
                    yield resp.code
            yield from self._iter_chat_response_tool_logs(
                item.console_pos,
                end_positions[idx],
            )

    def get_dialog(self) -> Iterable[DialogItem]:
        """Yield user/assistant dialog messages in chronological order."""
        console_len = len(self.py_env.console) if self.py_env.console else 0
        turn_items = [
            item for item in self.chat_log
            if isinstance(item, (WarmupLogItem, LLM_LogItem))
        ]
        end_positions = {
            id(item): (
                turn_items[idx + 1].console_pos
                if idx + 1 < len(turn_items) else console_len
            )
            for idx, item in enumerate(turn_items)
        }

        for item in self.chat_log:
            if isinstance(item, str):
                if item:
                    yield DialogItem("user", item)
                continue

            if isinstance(item, UserLogItem):
                if item.message:
                    yield DialogItem("user", item.message)
                continue

            if not isinstance(item, (WarmupLogItem, LLM_LogItem)):
                continue

            if isinstance(item, LLM_LogItem) and self._is_user_facing_llm_response(item):
                resp = item.llm_resp
                if isinstance(resp, str):
                    yield DialogItem("assistant", resp)
            elif (
                isinstance(item, LLM_LogItem)
                and self.job_def.chat_style == ChatStyle.MD_DIALOG  # pylint: disable=no-member
                and isinstance(item.llm_resp, CodeBlock)
            ):
                dialog = self._dialog_from_code_block(item.llm_resp)
                if dialog:
                    yield DialogItem("assistant", dialog)

            for body in self._iter_chat_response_tool_logs(
                item.console_pos,
                end_positions[id(item)],
            ):
                yield DialogItem("assistant", body)

    def get_response_times(self) -> Iterable[tuple[datetime, Optional[float]]]:
        """Return user-message timestamps paired with time-to-first user reply.

        Only timestamped user messages stored in ``chat_log`` are included:
        the initial dialog-style message (stored as a plain ``str`` and
        measured from ``created_at``) and any subsequent ``UserLogItem``
        entries. Warmup items are ignored. Internal dialog-mode execution
        turns such as tool-call requests or script-only responses are skipped.
        If no later user-facing ``LLM_LogItem`` responds to a message, its
        duration is ``None``.
        """
        response_times: List[tuple[datetime, Optional[float]]] = []
        pending_indices: List[int] = []

        for item in self.chat_log:
            if isinstance(item, str):
                if item and self.created_at is not None:
                    pending_indices.append(len(response_times))
                    response_times.append((self.created_at, None))
                continue

            if isinstance(item, UserLogItem):
                if item.message:
                    pending_indices.append(len(response_times))
                    response_times.append((item.timestamp, None))
                continue

            if isinstance(item, LLM_LogItem):
                if not pending_indices or not self._is_user_facing_llm_response(item):
                    continue
                for idx in pending_indices:
                    message_time, _ = response_times[idx]
                    response_times[idx] = (
                        message_time,
                        (item.timestamp - message_time).total_seconds(),
                    )
                pending_indices = []

        return response_times

    def get_llm_response_times(self) -> Iterable[tuple[datetime, Optional[float]]]:
        """Return per-request LLM timings from the chat log.

        Each tuple contains the request timestamp and time-to-response in
        seconds. Explicit pending ``LLM_LogItem`` markers (``llm_resp=None``)
        are treated as the cleanest request boundary and paired with the next
        concrete ``LLM_LogItem``. When no pending marker exists, the first
        concrete response is measured from ``created_at`` and later concrete
        ``LLM_LogItem`` entries use the previous response timestamp as an
        approximate boundary.

        Non-LLM entries are ignored. A trailing pending marker with no later
        response is returned with ``None`` as its duration.
        """
        response_times: List[tuple[datetime, Optional[float]]] = []
        pending_request_time: Optional[datetime] = None
        previous_response_time: Optional[datetime] = None

        for item in self.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue

            if item.llm_resp is None:
                pending_request_time = item.timestamp
                continue

            if pending_request_time is not None:
                response_times.append((
                    pending_request_time,
                    (item.timestamp - pending_request_time).total_seconds(),
                ))
                pending_request_time = None
            elif previous_response_time is not None:
                response_times.append((
                    previous_response_time,
                    (item.timestamp - previous_response_time).total_seconds(),
                ))
            elif self.created_at is not None and self.created_at <= item.timestamp:
                response_times.append((
                    self.created_at,
                    (item.timestamp - self.created_at).total_seconds(),
                ))

            previous_response_time = item.timestamp

        if pending_request_time is not None:
            response_times.append((pending_request_time, None))

        return response_times

    def get_llm_initial_step_size(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) for the constant initial portion.

        Covers the system prompt and all warmup code blocks — the fixed overhead
        sent with every first LLM request. Token counts use the ``len // 4``
        approximation. Output tokens are always zero (no LLM output at this stage).
        """
        system_prompt = self.system_prompt()
        warmup_text = _warmup_code_text(self.job_def.warmup_code if self.job_def else None)
        return (len(system_prompt) + len(warmup_text)) // 4, 0

    def get_llm_step_sizes(self) -> Iterable[tuple[int, int]]:
        """Return (input_tokens, output_tokens) per LLM step, ordered as get_llm_response_times.

        One entry is yielded per concrete ``LLM_LogItem`` (``llm_resp`` is not
        ``None``) plus one entry for a trailing pending marker if present.
        Token counts use the ``len // 4`` approximation.

        Input tokens for each step cover: console output produced since the
        previous step, tool log from the previous step, and any user messages
        accumulated since then. The first step additionally includes the initial
        step size (system prompt + warmup code).

        Output tokens cover the LLM response text plus the JSON encoding of any
        tool call requests.
        """
        console = self.py_env.console or []
        is_first = True
        prev_console_pos = 0
        prev_item = None
        accumulated_text = ""
        trailing_pending = None

        for item in self.chat_log:
            if isinstance(item, str):
                accumulated_text += item
                continue
            if isinstance(item, UserLogItem):
                accumulated_text += item.message or ""
                continue
            if not isinstance(item, LLM_LogItem):
                continue
            console_text = "\n".join(console[prev_console_pos:item.console_pos])
            tool_text = _tool_log_text(prev_item)
            input_tokens = (len(console_text) + len(tool_text) + len(accumulated_text)) // 4
            if is_first:
                input_tokens += self.get_llm_initial_step_size()[0]
            if item.llm_resp is None:
                trailing_pending = input_tokens
                continue
            is_first = False
            trailing_pending = None
            yield (input_tokens, _llm_resp_output_tokens(item.llm_resp))
            accumulated_text = ""
            prev_console_pos = item.console_pos
            prev_item = item

        if trailing_pending is not None:
            yield (trailing_pending, 0)

    def tokens_per_sec(self) -> float:
        """Return total tokens divided by total LLM processing time in seconds.

        Combines ``get_llm_response_times`` and ``get_llm_step_sizes`` to
        compute the overall throughput metric. Steps with unknown duration
        (``None``) are excluded from both totals. Returns ``0.0`` when no
        step has a known duration.
        """
        total_tokens = 0
        total_seconds = 0.0
        for (_, duration), (input_tok, output_tok) in zip(
            self.get_llm_response_times(), self.get_llm_step_sizes()
        ):
            if duration is None:
                continue
            total_tokens += input_tok + output_tok
            total_seconds += duration
        if total_seconds == 0.0:
            return 0.0
        return total_tokens / total_seconds

    def _get_warmup_block_count(self) -> int:
        """
        Returns the total number of warmup blocks.

        Returns:
            0 if no warmup_code, 1 if single string, len(warmup_code) if sequence
        """
        warmup = self.job_def.warmup_code
        if warmup is None:
            return 0
        if isinstance(warmup, (str, CodeBlock)):
            return 1
        return len(warmup)

    def has_more_warmup_blocks(self) -> bool:
        """
        Check if there are more warmup blocks to execute.

        Returns:
            True if there are more warmup blocks pending, False otherwise
        """
        block_count = self._get_warmup_block_count()
        if block_count == 0:
            return False
        current_block = self.warmup_block_num if self.warmup_block_num is not None else 0
        return current_block < block_count - 1

    def advance_warmup_block(self):
        """
        Advance to the next warmup block.
        Sets warmup_block_num to the next block index.
        """
        if self.warmup_block_num is None:
            self.warmup_block_num = 1
        else:
            self.warmup_block_num += 1

    def _warmup_end_positions(self) -> List[int]:
        """Return end positions for each warmup block, ordered by warmup_block_num.

        The end position is derived from the next element's console_pos in chat_log.
        For the last warmup item (no next element), the current console length is used.
        """
        console_len = len(self.py_env.console) if self.py_env.console else 0
        # Collect (warmup_block_num, end_pos) pairs
        pairs = []
        for i, item in enumerate(self.chat_log):
            if isinstance(item, WarmupLogItem):
                end_pos = (
                    self.chat_log[i + 1].console_pos
                    if i + 1 < len(self.chat_log)
                    else console_len
                )
                pairs.append((item.warmup_block_num, end_pos))
        pairs.sort(key=lambda x: x[0])
        return [end_pos for _, end_pos in pairs]

    def get_next_code_block(self) -> Union[str, CodeBlock, None]:
        """
        Retrieves the Python code block pending execution (or execution continuation).

        Returns:
            - None if status is DONE
            - Current warmup block (str or CodeBlock) if status is READY or WARMING_UP
            - last_response (str or CodeBlock) in all other cases
        """
        if self.status == JobStatus.DONE:
            return None
        if self.status == JobStatus.READY or self.status == JobStatus.WARMING_UP:
            warmup = self.job_def.warmup_code
            if warmup is None:
                return None
            if isinstance(warmup, (str, CodeBlock)):
                return warmup
            # warmup is a sequence of blocks
            block_num = self.warmup_block_num if self.warmup_block_num is not None else 0
            if block_num < len(warmup):
                return warmup[block_num]
            # All blocks completed
            return None
        return self.last_response

    @property
    def num_turns(self) -> int:
        """Returns the number of LLM turns (excludes warmup code blocks)."""
        return sum(1 for item in self.chat_log if isinstance(item, LLM_LogItem))

    def count_llm_actions(self) -> int:
        """Return the number of LLM actions recorded in chat_log.

        This differs from ``num_turns``: ``num_turns`` counts LLM log items,
        while this method counts finer-grained work inside those items. One
        non-empty executable code block, one requested tool call, and one
        user-facing response each count as one action. Framework-generated
        messages such as warmup, reminder, subtask, user, and post-processor
        items are ignored.

        Returns:
            The total number of recorded LLM actions.
        """
        action_count = 0
        for item in self.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue

            response = item.llm_resp
            if isinstance(response, CodeBlock):
                if response.code and response.code.strip():
                    action_count += 1
                if response.tool_calls:
                    action_count += len(response.tool_calls)
                continue

            if isinstance(response, str) and self._is_user_facing_llm_response(item):
                action_count += 1

        return action_count

    def is_step_final(self, llm_step: LLM_StepData) -> bool:
        """Return whether accepting an LLM step would naturally finish the job.

        Args:
            llm_step: Uncommitted LLM step to classify.

        Returns:
            True for DIRECT text-only steps and MD_DIALOG steps with no tool
            calls or executable fenced Python code. Other chat styles do not
            have auto-finalization semantics here.
        """
        chat_style = self.job_def.chat_style
        if chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
            return not llm_step.call_requests

        if chat_style == ChatStyle.MD_DIALOG:  # pylint: disable=no-member
            return not llm_step.call_requests and _is_empty_code(
                strip_markup(llm_step.text, strict=True)
            )

        return False

    @property
    def _warmup_console_positions(self) -> set:
        """Returns console_pos values that correspond to warmup blocks."""
        return {item.console_pos for item in self.chat_log if isinstance(item, WarmupLogItem)}

    @staticmethod
    def _tool_error_count(item) -> int:
        """Return the number of ToolError entries in item.tool_log."""
        if item.tool_log is None:
            return 0
        if isinstance(item.tool_log, ToolError):
            return 1
        if isinstance(item.tool_log, str):
            return 0
        return sum(1 for entry in item.tool_log if isinstance(entry, ToolError))

    def _has_exception(self, item) -> bool:
        """Return True if *item* has a code exception or a tool exception."""
        if self.py_env.exceptions and item.console_pos in self.py_env.exceptions:
            return True
        return self._tool_error_count(item) > 0

    def _exception_count_for_item(self, item) -> int:
        """Return the number of exceptions for a single chat log item."""
        count = 0
        if self.py_env.exceptions and item.console_pos in self.py_env.exceptions:
            count += 1
        count += self._tool_error_count(item)
        return count

    @property
    def exception_count(self) -> int:
        """Returns the total number of exceptions from LLM turns (excludes warmup blocks).

        Counts both code-execution exceptions (py_env.exceptions) and
        tool-call exceptions (ToolError entries in tool_log).
        """
        if not self.chat_log:
            return 0
        return sum(self._exception_count_for_item(item) for item in self.chat_log
                   if isinstance(item, LLM_LogItem))

    @property
    def max_consecutive_exceptions(self) -> int:
        """Returns the maximum number of consecutive exceptions among LLM turns only.

        Considers both code-execution exceptions and tool-call exceptions.
        """
        if not self.chat_log:
            return 0
        max_streak = 0
        streak = 0
        for item in self.chat_log:
            if not isinstance(item, LLM_LogItem):
                continue
            if self._has_exception(item):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @property
    def approx_token_usage(self) -> int:
        """Calculates approximate token usage based on total bytes sent and received."""
        return (self.usage.total_bytes_sent + self.usage.total_bytes_received) // 4

    def _resolve_user_message(self, message: Any) -> str:
        """Resolve a pushed message object to the string stored in job logs."""
        if isinstance(message, str):
            return message
        agent = self.job_def.agent if self.job_def is not None else None
        adapter = agent.get_adapter("message_adapter") if agent is not None else None
        if adapter is not None:
            return str(adapter(message))
        return str(message)

    def push_user_message(self, message: Any, locale: Optional["StatekLocale"] = None) -> bool:
        """Append a user message and re-activate the job if DONE.

        Intended for processing push notifications (e.g. a user writing a
        message into a completed job).

        In MD_DIALOG chat style the message is stored directly in
        ``chat_log`` — as a plain ``str`` when ``chat_log`` is empty (first
        message) or as a ``UserLogItem`` for subsequent messages.

        For all other chat styles the message is appended to
        ``py_env.push_log`` keyed by the current console length.

        When the job transitions DONE → STARTED a new ``LLM_LogItem`` with
        ``llm_resp=None`` is appended to ``chat_log`` to indicate that an
        LLM response is now awaited.

        Args:
            message: The message to push. Non-string values are converted with
                the agent's ``message_adapter`` when registered, otherwise with
                ``str(message)``.
            locale: Optional locale used only for the response to this message.

        Returns:
            True if the job was transitioned DONE → STARTED, False otherwise.
        """
        message = self._resolve_user_message(message)
        if locale is not None:
            self.__next_response_locale = locale

        if self.job_def.chat_style in (ChatStyle.MD_DIALOG, ChatStyle.DIRECT):  # pylint: disable=no-member
            if not self.chat_log:
                self.chat_log.append(message)
            else:
                self.chat_log.append(UserLogItem(message=message))
        else:
            if self.py_env.push_log is None:
                self.py_env.push_log = {}
            key = len(self.py_env.console) if self.py_env.console else 0
            existing = self.py_env.push_log.get(key)
            if existing is None:
                self.py_env.push_log[key] = message
            elif isinstance(existing, list):
                existing.append(message)
            else:
                self.py_env.push_log[key] = [existing, message]

        if self.status == JobStatus.DONE:  # pylint: disable=no-member
            self.num_completions = 1 if self.num_completions is None else self.num_completions + 1
            self.set_status(JobStatus.STARTED)  # pylint: disable=no-member
            self.py_env.exit_status = None
            self.chat_log.append(LLM_LogItem(
                console_pos=len(self.py_env.console) if self.py_env.console else 0,
                llm_resp=None
            ))
            return True
        return False  # pylint: disable=no-member
