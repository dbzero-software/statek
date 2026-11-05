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

"""Persistent model properties, initialized once per field."""

# pylint: disable=no-member

from dataclasses import dataclass, fields
from functools import singledispatch
import json
import math
from typing import Optional

import dbzero as db0

from ._model_tags import model_tags
from ._model_info import iter_model_info_rows
from .executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, NotificationLogItem,
    PostProcessedItem, ReminderLogItem, SubTaskLogItem, SystemLogItem,
    ToolError, UserLogItem, WarmupLogItem,
)
from .utils import CodeBlock


def _json_container(value):
    """Expose persistent JSON containers without serializing arbitrary objects."""
    if isinstance(value, db0.types.Dict):
        return dict(value)
    if isinstance(value, (db0.types.List, db0.types.Tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=_json_container).encode("utf-8"))


def _tool_log_bytes(item: ChatLogItem) -> int:
    results = item.tool_log
    if results is None:
        return 0
    if isinstance(results, (str, ToolError)):
        results = [results]
    return sum(len((result.err_message if isinstance(result, ToolError) else result)
                   .encode("utf-8")) for result in results)


@singledispatch
def _payload_bytes(payload) -> int:
    raise TypeError(f"Unsupported token estimation payload: {type(payload).__name__}")


@_payload_bytes.register(str)
def _text_bytes(payload: str) -> int:
    return len(payload.encode("utf-8"))


@_payload_bytes.register(dict)
@_payload_bytes.register(db0.types.Dict)
def _dict_bytes(payload: dict) -> int:
    return _json_bytes(payload)


@_payload_bytes.register(ChatLogItem)
def _chat_log_bytes(payload: ChatLogItem) -> int:
    return _tool_log_bytes(payload)


@_payload_bytes.register(LLM_LogItem)
def _llm_log_bytes(payload: LLM_LogItem) -> int:
    size = _tool_log_bytes(payload)
    response = payload.llm_resp
    if isinstance(response, CodeBlock):
        size += _text_bytes(response.code or "")
        if response.tool_calls:
            size += _json_bytes([
                {"id": call.id, "name": call.func_name,
                 "args": list(call.args or []), "kwargs": dict(call.kwargs or {})}
                for call in response.tool_calls
            ])
    elif response is not None:
        size += _text_bytes(response)
    reasoning = payload.llm_reasoning_payload
    if isinstance(reasoning, bytes):
        size += len(reasoning)
    elif isinstance(reasoning, str):
        size += _text_bytes(reasoning)
    elif reasoning is not None:
        size += _json_bytes(reasoning)
    return size


@_payload_bytes.register(WarmupLogItem)
def _warmup_log_bytes(payload: WarmupLogItem) -> int:
    # The source code requires the owning JobDef, unavailable on a log item.
    return _tool_log_bytes(payload)


@_payload_bytes.register(SystemLogItem)
@_payload_bytes.register(PostProcessedItem)
@_payload_bytes.register(UserLogItem)
def _message_log_bytes(payload) -> int:
    return _text_bytes(payload.message)


@_payload_bytes.register(ReminderLogItem)
def _reminder_log_bytes(payload: ReminderLogItem) -> int:
    return _text_bytes(payload.reminder.text)


@_payload_bytes.register(SubTaskLogItem)
def _subtask_log_bytes(payload: SubTaskLogItem) -> int:
    return _text_bytes(payload.handler.get_log_message()) + _tool_log_bytes(payload)


@_payload_bytes.register(NotificationLogItem)
def _notification_log_bytes(payload: NotificationLogItem) -> int:
    if payload.tool_log is not None:
        return _tool_log_bytes(payload)
    return _text_bytes(payload.message)


@_payload_bytes.register(DifficultyChangeLogItem)
def _difficulty_log_bytes(payload: DifficultyChangeLogItem) -> int:
    # Difficulty transitions are bookkeeping, excluded from conversation history.
    return 0


@db0.memo
@dataclass
class ModelCard:
    """Inherent model properties used by the framework."""

    max_context_length: int = 0
    token_to_bytes: float = 0.0
    total_parameter_count: Optional[int] = None
    active_parameter_count: Optional[int] = None
    token_estimation_margin: float = 0.05

    def est_token_count(self, payload: str | dict | ChatLogItem | UserLogItem) -> tuple[int, int]:
        """Estimate stored content tokens using UTF-8 bytes and the model's margin.

        These bounds are approximate, not guaranteed tokenizer limits. Dictionaries
        use JSON with unescaped Unicode. Log items include their stored LLM-facing
        content, excluding bookkeeping and provider/job-specific formatting.
        Warmup items include tool results only, as their source needs a JobDef.
        No tokenizer is used and the payload is never modified.

        Raises ValueError for an uninitialized or invalid token_to_bytes ratio,
        and TypeError for unsupported payloads or non-JSON-serializable values.
        """
        if not math.isfinite(self.token_to_bytes) or self.token_to_bytes <= 0:
            raise ValueError("token_to_bytes must be finite and positive")
        margin = getattr(self, "token_estimation_margin", 0.05)
        if not math.isfinite(margin) or not 0 <= margin < 1:
            raise ValueError("token_estimation_margin must be finite and in [0, 1)")
        estimate = _payload_bytes(payload) / self.token_to_bytes
        return math.floor((1 - margin) * estimate), math.ceil((1 + margin) * estimate)


def get_model_card(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    no_create: bool = False,
    **kwargs,
) -> Optional[ModelCard]:
    """Find a tagged card, creating one when allowed and absent."""
    tags = model_tags(provider, model, model_family)
    existing = next(iter(db0.find(ModelCard, *tags)), None)
    if existing is not None or no_create:
        return existing
    card = ModelCard(**kwargs)
    db0.tags(card).add(tags)
    return card


def set_model_card(
    provider: str,
    model: str,
    model_family: Optional[str] = None,
    **kwargs,
) -> ModelCard:
    """Set only fields that still have their default value or None."""
    existing = get_model_card(provider, model, model_family, no_create=True)
    if existing is None:
        card = get_model_card(provider, model, model_family, **kwargs)
        assert card is not None
        return card

    defaults = {field.name: field.default for field in fields(ModelCard)}
    for name, value in kwargs.items():
        if name not in defaults:
            raise TypeError(f"Unexpected ModelCard field: {name}")
        current = getattr(existing, name, defaults[name])
        unset_values = (None, defaults[name]) if name == "token_estimation_margin" else (None, 0)
        if current not in (*unset_values, value):
            raise ValueError(f"ModelCard.{name} is already initialized")

    for name, value in kwargs.items():
        if getattr(existing, name, defaults[name]) != value:
            setattr(existing, name, value)
    return existing


def _load_card_row(row: dict, provider: str, model: str,
                   model_family: Optional[str]) -> None:
    if "MAX_CONTEXT_LENGTH" not in row or "TOKEN_TO_BYTES" not in row:
        return
    try:
        max_context_length = int(row["MAX_CONTEXT_LENGTH"])
        token_to_bytes = float(row["TOKEN_TO_BYTES"])
        total_count = row.get("TOTAL_PARAMETER_COUNT")
        active_count = row.get("ACTIVE_PARAMETER_COUNT")
        total_count = int(total_count) if total_count and total_count.strip() else None
        active_count = int(active_count) if active_count and active_count.strip() else None
        raw_margin = row.get("TOKEN_ESTIMATION_MARGIN")
        margin = float(raw_margin) if raw_margin and raw_margin.strip() else None
    except (ValueError, TypeError):
        return
    if max_context_length <= 0 or not math.isfinite(token_to_bytes) or token_to_bytes <= 0:
        return
    if any(count is not None and count <= 0 for count in (total_count, active_count)):
        return
    if margin is not None and (not math.isfinite(margin) or not 0 <= margin < 1):
        return
    card_values = {
        "max_context_length": max_context_length,
        "token_to_bytes": token_to_bytes,
    }
    if total_count is not None:
        card_values["total_parameter_count"] = total_count
    if active_count is not None:
        card_values["active_parameter_count"] = active_count
    if margin is not None:
        card_values["token_estimation_margin"] = margin
    set_model_card(provider, model, model_family=model_family, **card_values)


def init_model_cards(model_info_dir: str) -> None:
    """Load model cards from CSV or TXT files in a directory tree."""
    for provider, model, model_family, row in iter_model_info_rows(model_info_dir):
        _load_card_row(row, provider, model, model_family)
