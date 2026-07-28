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

"""LLM API abstraction layer for Statek."""

# pylint: disable=no-member

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Iterable, Sequence, List, Dict, Callable, Tuple, Any, Type
import json
import httpx

from .settings import LLM_API_Settings, get_provider_settings
from .exceptions import InvalidFormat
from .chat_history import (
    ChatHistoryItem, ChatRole, format_chat_history_item,
)
from .llm_tools_scope import LLM_ToolsScope, parse_llm_tools_scope
from .provider_config import ProviderConfig
from .model_name import ensure_model_name, format_model_for_provider, select_model_provider

_CUSTOM_LLM_API_PROVIDERS: Dict[str, Dict[str, Any]] = {}
_SETTINGS_KWARGS = {
    "api_url",
    "api_key",
    "response_format_file",
    "use_prompt_caching",
}
_INTERNAL_LLM_API_PROVIDER_ALIASES = {
    "VERTEXAI", "VERTEX_AI", "GOOGLE_VERTEXAI", "GOOGLE",
    "CLAUDEAI", "CLAUDE_AI", "CLAUDE", "ANTHROPIC",
}
def _func_name_from_tool_calls(tool_calls) -> str:
    """Return the function name from a single CallSpec or the first item of a list."""
    if tool_calls is None:
        return ""
    cs = tool_calls if not isinstance(tool_calls, list) else (tool_calls[0] if tool_calls else None)
    return cs.func_name if cs is not None else ""


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Return a recursive merge where provider configuration wins conflicts."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _matching_reasoning_payload(
    payload: Any,
    provider: Optional[str],
    payload_format: str,
) -> Optional[Any]:
    """Return a built-in continuation envelope only for its originating provider."""
    if not callable(getattr(payload, "get", None)) or not isinstance(provider, str):
        return None
    payload_provider = payload.get("provider")
    if (
        not isinstance(payload_provider, str)
        or payload_provider.casefold() != provider.casefold()
        or payload.get("format") != payload_format
    ):
        return None
    return payload


def _request_ready_reasoning_value(value: Any) -> Any:
    """Recursively copy durable provider continuation collections for JSON requests."""
    if callable(getattr(value, "items", None)):
        return {
            key: _request_ready_reasoning_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (str, bytes)) or not callable(getattr(value, "__iter__", None)):
        return value
    return [_request_ready_reasoning_value(item) for item in value]


def _reasoning_mapping(value: Any) -> Optional[Dict]:
    """Copy a regular or durable mapping into plain nested request-ready values."""
    if not callable(getattr(value, "items", None)):
        return None
    return _request_ready_reasoning_value(value)


def _reasoning_sequence(value: Any) -> Optional[List]:
    """Copy a regular or durable provider block sequence with nested plain values."""
    if isinstance(value, (str, bytes)) or not callable(getattr(value, "__iter__", None)):
        return None
    return _request_ready_reasoning_value(value)


def resolve_reasoning_payload(
    model: str,
    provider_config: Optional[ProviderConfig],
    default_provider: Optional[str],
) -> Tuple[str, Optional[Dict]]:
    """Resolve model parameters into an upstream model ID and provider payload."""
    model_name = ensure_model_name(model)
    provider = select_model_provider(model_name, default_provider=default_provider)
    aliases = [
        model_name.params[key]
        for key in ("rl", "reasoning_level")
        if key in model_name.params
    ]
    payload = None
    if aliases:
        if len(aliases) == 2 and aliases[0] != aliases[1]:
            raise ValueError("rl and reasoning_level specify conflicting values")
        try:
            level = int(aliases[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("reasoning level must be an integer from 0 through 100") from exc
        if not 0 <= level <= 100 or str(level) != aliases[0].strip():
            raise ValueError("reasoning level must be an integer from 0 through 100")
        if level > 0:
            if provider_config is None:
                raise ValueError("positive reasoning level requires a provider configuration")
            payload = provider_config.find_payload(
                provider, model_name.model_family, model_name.model, reasoning_level=level)
            if payload is None:
                raise ValueError(
                    "positive reasoning level has no matching provider configuration payload")
    upstream_model = (
        format_model_for_provider(model_name, provider)
        if provider is not None else (
            f"{model_name.model_family}/{model_name.model}"
            if model_name.model_family else model_name.model
        )
    )
    return upstream_model, payload

LLM_Stats = namedtuple(
    "LLM_Stats",
    ["total_bytes_sent", "total_bytes_received", "cost",
     "input_tokens", "output_tokens", "cached_tokens"],
    defaults=[0, 0, 0],
)
LLM_StepData = namedtuple(
    "LLM_StepData",
    ["text", "call_requests", "reasoning_payload"],
    defaults=[None],
)
LLM_Response = namedtuple("LLM_Response", ["step_data", "stats"])

class CallParams:
    """Parameters for a single function/tool call requested by the LLM.

    id     - call identifier assigned by the LLM provider
    name   - function / tool name to invoke
    args   - positional arguments (always [] for OpenAI-format calls)
    kwargs - keyword arguments parsed from the JSON arguments string

    Instances are hashable and compare equal when their id matches, so a
    dict of results can be keyed directly by CallParams objects.
    """

    __slots__ = ("id", "name", "args", "kwargs")

    def __init__(self, call_id: str, name: str, args: list, kwargs: dict):
        self.id = call_id
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CallParams):
            return self.id == other.id
        return NotImplemented

    def __repr__(self):
        return (
            f"CallParams(id={self.id!r}, name={self.name!r},"
            f" args={self.args!r}, kwargs={self.kwargs!r})"
        )


def extract_call_params(tool_call_req: Dict) -> CallParams:
    """Extract function call parameters from an OpenAI- or Anthropic-format tool call.

    Supports two wire formats:

    **OpenAI / OpenRouter** (``type`` is ``"function"`` or absent)::

        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Boston"}'
            }
        }

    **Anthropic / Claude** (``type`` is ``"tool_use"``)::

        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_weather",
            "input": {"city": "Boston"}
        }

    ``args`` is always ``[]`` — both protocols pass only keyword arguments.

    Returns:
        CallParams with ``id``, ``name``, ``args=[]``, and ``kwargs``.

    Raises:
        InvalidFormat: If required keys are missing, the type is unrecognised,
            ``arguments`` is not valid JSON, or the decoded value is not a dict.
    """
    try:
        call_id = tool_call_req.get("id", "")
        call_type = tool_call_req.get("type")

        # ---- Anthropic / Claude format ----------------------------------------
        if call_type == "tool_use":
            if "name" not in tool_call_req:
                raise InvalidFormat(
                    "Missing 'name' in Anthropic tool_use block."
                )
            name = tool_call_req["name"]
            if not isinstance(name, str):
                raise InvalidFormat(
                    f"'name' must be a string, got {type(name).__name__}."
                )
            kwargs = tool_call_req.get("input", {})
            if not isinstance(kwargs, dict):
                raise InvalidFormat(
                    f"'input' must be a dict, got {type(kwargs).__name__}."
                )
            return CallParams(call_id=call_id, name=name, args=[], kwargs=kwargs)

        # ---- OpenAI / OpenRouter format ---------------------------------------
        if call_type is not None and call_type != "function":
            raise InvalidFormat(
                f"Unsupported tool call type: '{call_type}'. Expected 'function' or 'tool_use'."
            )

        if "function" not in tool_call_req:
            raise InvalidFormat("Missing 'function' key in tool call request.")

        fn = tool_call_req["function"]

        if "name" not in fn:
            raise InvalidFormat("Missing 'name' in tool call function.")

        name = fn["name"]
        if not isinstance(name, str):
            raise InvalidFormat(
                f"'name' must be a string, got {type(name).__name__}."
            )

        arguments_str = fn.get("arguments", "{}")
        try:
            kwargs = json.loads(arguments_str)
        except json.JSONDecodeError as exc:
            raise InvalidFormat(
                f"Invalid JSON in 'arguments': {exc}"
            ) from exc

        if not isinstance(kwargs, dict):
            raise InvalidFormat(
                f"'arguments' must decode to a JSON object, "
                f"got {type(kwargs).__name__}."
            )

        return CallParams(call_id=call_id, name=name, args=[], kwargs=kwargs)

    except InvalidFormat:
        raise
    except Exception as exc:
        raise InvalidFormat(
            f"Unable to parse tool call request: {exc}"
        ) from exc


def _tool_name(tool_func: Callable) -> str:
    """Return the LLM-facing function name for a tool callable."""
    return getattr(tool_func, "__name__", "")


def _tool_is_hidden(tool_func: Callable) -> bool:
    """Return whether a tool is hidden from LLM-facing request selection."""
    return bool(getattr(tool_func, "tool_hidden", False))


def _dedupe_tools_by_name(tools: Iterable[Callable]) -> List[Callable]:
    """Deduplicate tools by name while preserving first occurrence."""
    result = []
    seen: set[str] = set()
    for tool_func in tools:
        name = _tool_name(tool_func)
        if name in seen:
            continue
        seen.add(name)
        result.append(tool_func)
    return result


def _find_tool_by_name(tools: Iterable[Callable], name: str) -> Optional[Callable]:
    """Return the first tool with ``name`` from ``tools``."""
    for tool_func in tools:
        if _tool_name(tool_func) == name:
            return tool_func
    return None


def _scope_is_empty(scope: LLM_ToolsScope) -> bool:
    """Return whether a parsed scope contains no category or explicit lists."""
    return (
        scope.category is None
        and scope.additional_tools is None
        and scope.removed_tools is None
    )


def _tool_matches_chat_style(tool_func: Callable, chat_style) -> bool:
    """Return whether a single tool is compatible with the active chat style."""
    if chat_style is None:
        return True
    from .system import select_tools  # pylint: disable=import-outside-toplevel
    return bool(list(select_tools([tool_func], "ALL", chat_style=chat_style)))


def _resolve_explicit_tool(
    name: str,
    available_tools: Sequence[Callable],
    registered_system_tools: Sequence[Callable],
    chat_style=None,
) -> Callable:
    """Resolve one explicit LLM_TOOLS_SCOPE tool name."""
    tool_func = _find_tool_by_name(available_tools, name)
    if tool_func is None:
        tool_func = _find_tool_by_name(registered_system_tools, name)
    if tool_func is None:
        raise ValueError(f"LLM_TOOLS_SCOPE references unknown tool: {name!r}")
    if not _tool_matches_chat_style(tool_func, chat_style):
        raise ValueError(
            f"LLM_TOOLS_SCOPE tool {name!r} is not compatible with chat_style "
            f"{chat_style!r}"
        )
    return tool_func


def _merge_registered_system_tools(
    tools: Iterable[Callable],
    registered_system_tools: Iterable[Callable],
) -> List[Callable]:
    """Merge category tools with registered system tools, keeping first names."""
    return _dedupe_tools_by_name([*tools, *registered_system_tools])


def _select_scope_category_tools(
    scope: LLM_ToolsScope,
    available_tools: Sequence[Callable],
    chat_style=None,
) -> List[Callable]:
    """Select the category-driven base tools for a parsed scope."""
    from .system import find_tools, select_tools  # pylint: disable=import-outside-toplevel

    if scope.category is None:
        return []

    tools = list(select_tools(available_tools, scope.category, chat_style=chat_style))
    if scope.category in ("SYSTEM", "ALL"):
        return _merge_registered_system_tools(
            tools,
            find_tools("SYSTEM", chat_style=chat_style),
        )
    return tools


def _apply_explicit_scope_lists(
    scope: LLM_ToolsScope,
    selected_tools: Iterable[Callable],
    available_tools: Sequence[Callable],
    chat_style=None,
) -> List[Callable]:
    """Apply explicit additions and removals to category-selected tools."""
    if scope.additional_tools is None and scope.removed_tools is None:
        return _dedupe_tools_by_name(selected_tools)

    from .system import find_tools  # pylint: disable=import-outside-toplevel

    registered_system_tools = list(find_tools("SYSTEM"))
    tools = list(selected_tools)

    if scope.additional_tools is not None:
        for name in scope.additional_tools:
            tools.append(_resolve_explicit_tool(
                name,
                available_tools,
                registered_system_tools,
                chat_style=chat_style,
            ))

    tools = _dedupe_tools_by_name(tools)
    if scope.removed_tools is None:
        return tools

    candidate_names = {_tool_name(tool_func) for tool_func in tools}
    removed_names: set[str] = set()
    for name in scope.removed_tools:
        tool_func = _resolve_explicit_tool(
            name,
            available_tools,
            registered_system_tools,
            chat_style=chat_style,
        )
        tool_name = _tool_name(tool_func)
        if tool_name not in candidate_names:
            raise ValueError(
                f"LLM_TOOLS_SCOPE removal references unselected tool: {name!r}"
            )
        removed_names.add(tool_name)

    return [tool_func for tool_func in tools if _tool_name(tool_func) not in removed_names]


def select_request_tools(
    metadata: Optional[Dict[str, str]] = None,
    available_tools: Optional[Sequence[Callable]] = None,
    chat_style=None,
) -> Optional[List[Callable]]:
    """Return the tool callables that would be forwarded in an LLM request."""
    tools_scope = metadata.get("LLM_TOOLS_SCOPE") if metadata else None
    if not tools_scope or available_tools is None:
        return None

    scope = parse_llm_tools_scope(tools_scope)
    if _scope_is_empty(scope):
        return None

    materialized_tools = [
        tool_func for tool_func in available_tools
        if not _tool_is_hidden(tool_func)
    ]
    selected_tools = _select_scope_category_tools(
        scope,
        materialized_tools,
        chat_style=chat_style,
    )
    return _apply_explicit_scope_lists(
        scope,
        selected_tools,
        materialized_tools,
        chat_style=chat_style,
    )


class LLM_API(ABC):
    """Abstract base class for LLM API wrappers.

    This class provides a stateless interface to abstract away individual
    provider specifics and match them with Statek's requirements.
    A single LLM_API instance is intended for a single session with the LLM agent.
    """
    def _prepare_request_kwargs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        available_tools: Optional[Sequence[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict[str, Any]:
        """Resolve shared request parameters used by preview and execution."""
        if metadata is not None:
            metadata = dict(metadata)
        if available_tools is not None and not isinstance(available_tools, list):
            available_tools = list(available_tools)
        if chat_history is not None and not isinstance(chat_history, list):
            chat_history = list(chat_history)
        model = self.require_model(model)
        tools = select_request_tools(
            metadata=metadata,
            available_tools=available_tools,
            chat_style=chat_style,
        )
        result = {
            "system_prompt": system_prompt,
            "model": model,
            "metadata": metadata,
            "tools": tools,
            "chat_history": chat_history,
            "chat_style": chat_style,
            "temperature": temperature,
            "provider_config": provider_config,
        }
        return result

    def preview_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        available_tools: Optional[Sequence[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict:
        """Return the provider JSON payload that would be sent for a request."""
        return self._build_request_payload(**self._prepare_request_kwargs(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            available_tools=available_tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            provider_config=provider_config,
        ))

    async def process_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        available_tools: Optional[Sequence[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> LLM_Response:
        """Process a request to the LLM API.

        ``chat_history`` is the canonical conversation input — a stream of
        :class:`ChatHistoryItem` objects representing the user/assistant/tool
        turns. The system prompt is supplied separately via ``system_prompt``.
        The provider-specific ``_process_request`` is responsible for
        converting those inputs into the wire format expected by the API.

        When metadata contains ``"LLM_TOOLS_SCOPE"``, the tools from
        ``available_tools`` are filtered by that scope (via ``select_tools``)
        and forwarded to the provider as a formal tools parameter.

        Args:
            system_prompt: Optional system prompt to pass separately from the
                conversational history.
            model: Required provider model for this request.
            metadata: Optional metadata key/value pairs. ``"LLM_TOOLS_SCOPE"``
                selects tools from ``available_tools`` to send to the provider.
            available_tools: All tools available in the agent's local context.
                Only used when ``"LLM_TOOLS_SCOPE"`` is set in metadata.
            chat_history: Conversation history as ``ChatHistoryItem`` objects.
            chat_style: Optional ChatStyle. Threaded through for tool selection
                and message formatting; in DIRECT mode the history is rewritten
                so assistant code becomes ``python_cli`` tool calls.
            temperature: Optional provider temperature.
            provider_config: Optional durable provider-configuration snapshot.

        Returns:
            LLM_Response containing step data and provider usage stats.
        """
        request_kwargs = self._prepare_request_kwargs(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            available_tools=available_tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            provider_config=provider_config,
        )

        response = await self._process_request(**request_kwargs)
        return response

    @abstractmethod
    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict:
        """Provider-specific payload builder shared by preview and execution."""

    @abstractmethod
    async def _process_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> LLM_Response:
        """Provider-specific request implementation. Called by process_request."""

    @staticmethod
    def parse_temperature(value: Optional[str]) -> Optional[float]:
        """Validate a temperature value that may have come from metadata."""
        if value is None:
            return None
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"TEMPERATURE must be between 0.0 and 1.0, got {value}")
        return value

    @staticmethod
    def require_model(model: Optional[str]) -> str:
        """Validate the explicit model argument."""
        if not model:
            raise ValueError("model must be provided to process_request")
        return model

    @staticmethod
    def _load_response_format(settings: LLM_API_Settings) -> Optional[dict]:
        """Load response_format from the JSON file specified in settings.

        Returns:
            The response_format dict, or None if response_format_file is not set.
        """
        if not settings.response_format_file:
            return None
        with open(settings.response_format_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    @lru_cache
    def get(provider_name: str = None, model: str = None, **kwargs):
        """
        Factory method to get an LLM_API instance for a specific provider.
        """
        del model
        if provider_name is None:
            from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel
            provider_key = get_statek_settings().default_llm_api_provider.upper()
        else:
            provider_key = provider_name.upper()
        custom_provider = _CUSTOM_LLM_API_PROVIDERS.get(provider_key)
        if custom_provider is not None and custom_provider["settings"] is not None:
            settings = custom_provider["settings"]
        else:
            settings = get_provider_settings(provider_key)
        if not settings:
            raise ValueError(f"No settings found for {provider_key} provider.")

        if custom_provider is not None:
            provider_kwargs = {**custom_provider["kwargs"], **kwargs}
            return custom_provider["impl"](settings=settings, **provider_kwargs)

        compatible_provider_cls = OPENAI_COMPATIBLE_API_PROVIDERS.get(provider_key)
        if compatible_provider_cls is not None:
            return compatible_provider_cls(settings=settings, **kwargs)
        if provider_key in ('VERTEXAI', 'VERTEX_AI', 'GOOGLE_VERTEXAI', 'GOOGLE'):
            return VertexAI_API(settings=settings, **kwargs)
        if provider_key in ('CLAUDEAI', 'CLAUDE_AI', 'CLAUDE', 'ANTHROPIC'):
            return ClaudeAI_API(settings=settings, **kwargs)
        raise ValueError(f"Unsupported LLM API provider: {provider_name}")

class DefaultLLM_API_Impl(LLM_API):
    """Default OpenAI-compatible chat-completions implementation of LLM_API.

    This implementation is suitable for providers compatible with OpenAI's
    standard chat completions API v1.
    """

    def __init__(self, settings: LLM_API_Settings, **kwargs):
        """Initialize an OpenAI-compatible API client.

        Args:
            settings: LLM_API_Settings containing API URL and key
        """
        self.settings = settings
        self.response_format = LLM_API._load_response_format(settings)
        # additional kwargs that will be passed to request if needed
        self.kwargs = kwargs

        self.api_url = settings.api_url
        self.api_key = settings.api_key
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def build_messages(
        self,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the OpenAI-compatible messages list from a ``ChatHistoryItem`` stream.

        Delegates per-item formatting to :func:`format_chat_history_item`,
        which produces dicts in the OpenAI chat-completions schema.
        ``chat_style`` defaults to the global StatekSettings value
        when not supplied.
        """
        if chat_style is None:
            from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel
            chat_style = get_statek_settings().chat_style
        from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel
        settings = get_statek_settings()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if not chat_history:
            return messages
        for item in chat_history:
            reasoning_payload = _matching_reasoning_payload(
                getattr(item, "provider_reasoning_payload", None), provider, "openai")
            if item.role == ChatRole.ASSISTANT and item.content is None and item.tool_calls is None:
                fields = _reasoning_mapping(
                    reasoning_payload.get("fields") if reasoning_payload is not None else None)
                if fields is not None:
                    messages.append({"role": "assistant", "content": None, **fields})
                continue
            message = format_chat_history_item(item, chat_style, settings)
            fields = _reasoning_mapping(
                reasoning_payload.get("fields") if reasoning_payload is not None else None)
            if fields is not None:
                message.update(fields)
            messages.append(message)
        return messages

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict:
        """Build the OpenAI-compatible JSON payload."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        provider = select_model_provider(model, default_provider=(metadata or {}).get("PROVIDER"))
        model, reasoning_payload = resolve_reasoning_payload(
            self.require_model(model),
            provider_config,
            provider,
        )
        payload = {
            "model": self.require_model(model),
            "messages": self.build_messages(
                system_prompt=system_prompt,
                chat_history=chat_history,
                chat_style=chat_style,
                provider=provider,
            ),
        }
        if tools:
            payload["tools"] = [format_tool_spec(t) for t in tools]
        if self.response_format:
            payload["response_format"] = self.response_format
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(self.kwargs)
        if reasoning_payload is not None:
            payload = _deep_merge(payload, reasoning_payload)
        return payload

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> LLM_Response:
        """Process a request to an OpenAI-compatible API.

        Args:
            system_prompt: Optional system prompt
            model: Required provider model.
            metadata: Optional metadata unrelated to provider model selection.
            tools: Optional list of tool callables to include as formal tool definitions
                   in the OpenAI function-calling format.
            chat_history: Conversation history including the latest user message as
                         the final element

        Returns:
            LLM_Response with generated step data and usage stats.

        Raises:
            httpx.HTTPError: If the API request fails
            KeyError: If the response format is unexpected
        """
        payload = self._build_request_payload(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            tools=tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            provider_config=provider_config,
        )
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Measure bytes sent
        payload_bytes = json.dumps(payload).encode('utf-8')
        total_bytes_sent = len(payload_bytes)

        # Make the async HTTP request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Measure bytes received
            total_bytes_received = len(response.content)

            # Parse the response
            data = response.json()

            if "choices" not in data or not data["choices"]:
                error_detail = data.get("error", {}).get("message", str(data))
                raise RuntimeError(f"LLM API error: {error_detail}")

            message = data["choices"][0]["message"]

            # Parse tool calls when the LLM chose to invoke tools
            raw_tool_calls = message.get("tool_calls")
            call_requests = (
                [extract_call_params(tc) for tc in raw_tool_calls]
                if raw_tool_calls else None
            )

            response_text = message.get("content") or ""
            provider = select_model_provider(
                model,
                default_provider=(metadata or {}).get("PROVIDER"),
            )

            # When response_format is used, extract python_code from JSON
            if self.response_format and response_text:
                response_text = json.loads(response_text)["python_code"]

            _usage = data.get("usage", {})
            cost = _usage.get("cost")
            _details = _usage.get("prompt_tokens_details") or {}

            stats = LLM_Stats(
                total_bytes_sent=total_bytes_sent,
                total_bytes_received=total_bytes_received,
                cost=cost,
                input_tokens=_usage.get("prompt_tokens", 0),
                output_tokens=_usage.get("completion_tokens", 0),
                cached_tokens=_details.get("cached_tokens", 0),
            )

            return LLM_Response(
                step_data=LLM_StepData(
                    text=response_text,
                    call_requests=call_requests,
                    reasoning_payload={
                        "provider": provider.casefold() if isinstance(provider, str) else None,
                        "format": "openai",
                        "fields": {
                            key: message[key]
                            for key in ("reasoning_details", "reasoning", "reasoning_content")
                            if key in message
                        },
                    }
                    if any(
                        key in message
                        for key in ("reasoning_details", "reasoning", "reasoning_content")
                    )
                    else None,
                ),
                stats=stats,
            )


class OpenRouter_API(DefaultLLM_API_Impl):
    """OpenRouter API typed wrapper over the default chat-completions implementation."""


class OpenAI_API(DefaultLLM_API_Impl):
    """OpenAI API implementation using the chat completions wire format."""


class Groq_API(DefaultLLM_API_Impl):
    """Groq API typed wrapper over the default chat-completions implementation."""


class MistralAI_API(DefaultLLM_API_Impl):
    """Mistral AI API typed wrapper over the default chat-completions implementation."""


class DeepSeek_API(DefaultLLM_API_Impl):
    """DeepSeek API typed wrapper over the default chat-completions implementation."""


class XAI_API(DefaultLLM_API_Impl):
    """xAI API typed wrapper over the default chat-completions implementation."""


class TogetherAI_API(DefaultLLM_API_Impl):
    """Together AI API typed wrapper over the default chat-completions implementation."""


class FireworksAI_API(DefaultLLM_API_Impl):
    """Fireworks AI API typed wrapper over the default chat-completions implementation."""


class Cerebras_API(DefaultLLM_API_Impl):
    """Cerebras API typed wrapper over the default chat-completions implementation."""


class Perplexity_API(DefaultLLM_API_Impl):
    """Perplexity API typed wrapper over the default chat-completions implementation."""


class SambaNova_API(DefaultLLM_API_Impl):
    """SambaNova API typed wrapper over the default chat-completions implementation."""


class NvidiaNIM_API(DefaultLLM_API_Impl):
    """NVIDIA NIM API typed wrapper over the default chat-completions implementation."""


class Nebius_API(DefaultLLM_API_Impl):
    """Nebius API typed wrapper over the default chat-completions implementation."""


class Cohere_API(DefaultLLM_API_Impl):
    """Cohere compatibility API typed wrapper over the default implementation."""


class MoonshotAI_API(DefaultLLM_API_Impl):
    """Moonshot AI / Kimi API typed wrapper over the default implementation."""


class DashScope_API(DefaultLLM_API_Impl):
    """Alibaba Cloud DashScope API typed wrapper over the default implementation."""


class CloudflareWorkersAI_API(DefaultLLM_API_Impl):
    """Cloudflare Workers AI typed wrapper over the default implementation."""


class CloudflareAIGateway_API(DefaultLLM_API_Impl):
    """Cloudflare AI Gateway typed wrapper over the default implementation."""


class GitHubModels_API(DefaultLLM_API_Impl):
    """GitHub Models API typed wrapper over the default implementation."""


class Bedrock_API(DefaultLLM_API_Impl):
    """Amazon Bedrock OpenAI-compatible API typed wrapper over the default implementation."""


class MicrosoftFoundry_API(DefaultLLM_API_Impl):
    """Microsoft Foundry OpenAI-compatible API typed wrapper over the default implementation."""


class AzureOpenAI_API(DefaultLLM_API_Impl):
    """Azure OpenAI API typed wrapper over the default implementation."""


class GeminiEnterprise_API(DefaultLLM_API_Impl):
    """Gemini Enterprise OpenAI-compatible API typed wrapper over the default implementation."""


class Ollama_API(DefaultLLM_API_Impl):
    """Ollama OpenAI-compatible API typed wrapper over the default implementation."""


class LMStudio_API(DefaultLLM_API_Impl):
    """LM Studio OpenAI-compatible API typed wrapper over the default implementation."""


class VLLM_API(DefaultLLM_API_Impl):
    """vLLM OpenAI-compatible server typed wrapper over the default implementation."""


class SGLang_API(DefaultLLM_API_Impl):
    """SGLang OpenAI-compatible server typed wrapper over the default implementation."""


class LlamaCpp_API(DefaultLLM_API_Impl):
    """llama.cpp OpenAI-compatible server typed wrapper over the default implementation."""


OPENAI_COMPATIBLE_API_PROVIDERS = {
    "OPENAI": OpenAI_API,
    "OPENROUTER": OpenRouter_API,
    "GROQ": Groq_API,
    "MISTRAL": MistralAI_API,
    "MISTRALAI": MistralAI_API,
    "MISTRAL_AI": MistralAI_API,
    "DEEPSEEK": DeepSeek_API,
    "DEEP_SEEK": DeepSeek_API,
    "XAI": XAI_API,
    "X_AI": XAI_API,
    "GROK": XAI_API,
    "TOGETHER": TogetherAI_API,
    "TOGETHERAI": TogetherAI_API,
    "TOGETHER_AI": TogetherAI_API,
    "FIREWORKS": FireworksAI_API,
    "FIREWORKSAI": FireworksAI_API,
    "FIREWORKS_AI": FireworksAI_API,
    "CEREBRAS": Cerebras_API,
    "PERPLEXITY": Perplexity_API,
    "SAMBANOVA": SambaNova_API,
    "SAMBA_NOVA": SambaNova_API,
    "NVIDIA": NvidiaNIM_API,
    "NVIDIA_NIM": NvidiaNIM_API,
    "NIM": NvidiaNIM_API,
    "NEBIUS": Nebius_API,
    "COHERE": Cohere_API,
    "MOONSHOT": MoonshotAI_API,
    "MOONSHOTAI": MoonshotAI_API,
    "MOONSHOT_AI": MoonshotAI_API,
    "KIMI": MoonshotAI_API,
    "DASHSCOPE": DashScope_API,
    "DASH_SCOPE": DashScope_API,
    "ALIBABA": DashScope_API,
    "ALIBABA_CLOUD": DashScope_API,
    "QWEN": DashScope_API,
    "CLOUDFLARE": CloudflareWorkersAI_API,
    "CLOUDFLARE_WORKERS_AI": CloudflareWorkersAI_API,
    "WORKERS_AI": CloudflareWorkersAI_API,
    "CLOUDFLARE_AI_GATEWAY": CloudflareAIGateway_API,
    "AI_GATEWAY": CloudflareAIGateway_API,
    "GITHUB": GitHubModels_API,
    "GITHUB_MODELS": GitHubModels_API,
    "BEDROCK": Bedrock_API,
    "AMAZON_BEDROCK": Bedrock_API,
    "AWS_BEDROCK": Bedrock_API,
    "MICROSOFT_FOUNDRY": MicrosoftFoundry_API,
    "MS_FOUNDRY": MicrosoftFoundry_API,
    "AZURE_FOUNDRY": MicrosoftFoundry_API,
    "AZURE_OPENAI": AzureOpenAI_API,
    "AZURE_OPEN_AI": AzureOpenAI_API,
    "GEMINI_ENTERPRISE": GeminiEnterprise_API,
    "GOOGLE_GEMINI_ENTERPRISE": GeminiEnterprise_API,
    "GOOGLE_OPENAI": GeminiEnterprise_API,
    "OLLAMA": Ollama_API,
    "LMSTUDIO": LMStudio_API,
    "LM_STUDIO": LMStudio_API,
    "VLLM": VLLM_API,
    "SGLANG": SGLang_API,
    "LLAMA_CPP": LlamaCpp_API,
    "LLAMACPP": LlamaCpp_API,
    "LLAMA_CPP_PYTHON": LlamaCpp_API,
}


def add_provider(name: str, llm_api_impl: Type[LLM_API] = None, **kwargs):
    """Register a custom LLM API provider.

    Provider names are case-insensitive. The only registration-time
    validation is name uniqueness; implementation or settings problems surface
    later when the provider is used.
    """
    provider_key = name.upper()
    if (
        provider_key in OPENAI_COMPATIBLE_API_PROVIDERS
        or provider_key in _INTERNAL_LLM_API_PROVIDER_ALIASES
        or provider_key in _CUSTOM_LLM_API_PROVIDERS
    ):
        raise ValueError(f"LLM API provider already registered: {name}")

    settings_kwargs = {
        key: value for key, value in kwargs.items()
        if key in _SETTINGS_KWARGS
    }
    provider_kwargs = {
        key: value for key, value in kwargs.items()
        if key not in _SETTINGS_KWARGS
    }
    settings = None
    if "api_url" in settings_kwargs and "api_key" in settings_kwargs:
        settings = LLM_API_Settings(**settings_kwargs)

    _CUSTOM_LLM_API_PROVIDERS[provider_key] = {
        "impl": llm_api_impl or DefaultLLM_API_Impl,
        "settings": settings,
        "kwargs": provider_kwargs,
    }
    LLM_API.get.cache_clear()


class VertexAI_API(LLM_API):
    """Vertex AI Gemini implementation of LLM_API."""

    def __init__(self, settings: LLM_API_Settings, **kwargs):
        self.settings = settings
        self.response_format = LLM_API._load_response_format(settings)
        self.kwargs = kwargs
        self.api_url = settings.api_url
        self.api_key = settings.api_key

    @staticmethod
    def _content_text_for_item(item: "ChatHistoryItem", chat_style, settings) -> str:
        formatted = format_chat_history_item(item, chat_style, settings)
        content = formatted.get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _normalise_tool_calls(tool_calls) -> List:
        if tool_calls is None:
            return []
        if isinstance(tool_calls, list):
            return tool_calls
        return [tool_calls]

    def build_contents(
        self,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        provider: Optional[str] = None,
    ) -> List[Dict]:
        """Build Gemini ``contents`` from a ``ChatHistoryItem`` stream."""
        from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel
        if chat_style is None:
            chat_style = get_statek_settings().chat_style
        settings = get_statek_settings()

        if not chat_history:
            return []

        contents: List[Dict] = []
        for item in chat_history:
            if item.role == ChatRole.SYSTEM:
                continue
            if item.role == ChatRole.USER:
                text = self._content_text_for_item(item, chat_style, settings)
                if text:
                    contents.append({"role": "user", "parts": [{"text": text}]})
                continue
            if item.role == ChatRole.ASSISTANT:
                reasoning_payload = _matching_reasoning_payload(
                getattr(item, "provider_reasoning_payload", None),
                provider,
                "vertex",
            )
                parts = _reasoning_sequence(
                    reasoning_payload.get("parts") if reasoning_payload is not None else None)
                if parts is not None:
                    contents.append({"role": "model", "parts": parts})
                    continue
                parts = []
                if item.content:
                    parts.append({"text": self._content_text_for_item(item, chat_style, settings)})
                for call in self._normalise_tool_calls(item.tool_calls):
                    parts.append({
                        "functionCall": {
                            "name": call.func_name,
                            "args": dict(call.kwargs) if call.kwargs else {},
                        }
                    })
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue
            if item.role == ChatRole.TOOL:
                tool_calls = self._normalise_tool_calls(item.tool_calls)
                name = tool_calls[0].func_name if tool_calls else "tool_result"
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": {"content": item.content or ""},
                        }
                    }],
                })
        return contents

    @staticmethod
    def _build_system_instruction(system_prompt: Optional[str]) -> Optional[Dict]:
        if not system_prompt:
            return None
        return {"parts": [{"text": system_prompt}]}

    @staticmethod
    def _to_vertex_tool(spec: Dict) -> Dict:
        fn = spec["function"]
        return {
            "name": fn["name"],
            "description": fn["description"],
            "parameters": fn["parameters"],
        }

    def _model_resource(self, model: str) -> str:
        if model.startswith("projects/"):
            return model
        project = self.kwargs.get("project")
        location = self.kwargs.get("location")
        publisher = self.kwargs.get("publisher", "google")
        if project and location:
            return (
                f"projects/{project}/locations/{location}/publishers/"
                f"{publisher}/models/{model}"
            )
        return model

    def _request_url(self, model: str) -> str:
        if "{model}" in self.api_url:
            return self.api_url.format(model=self._model_resource(model))
        if self.api_url.endswith(":generateContent"):
            return self.api_url
        return f"{self.api_url.rstrip('/')}/{self._model_resource(model)}:generateContent"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.kwargs.get("auth") == "api_key":
            headers["x-goog-api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict:
        """Build the Vertex AI GenerateContent JSON payload."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        provider = select_model_provider(model, default_provider=(metadata or {}).get("PROVIDER"))
        model, reasoning_payload = resolve_reasoning_payload(
            self.require_model(model),
            provider_config,
            provider,
        )
        payload = {
            "contents": self.build_contents(
                chat_history,
                chat_style=chat_style,
                provider=provider,
            ),
        }
        system_instruction = self._build_system_instruction(system_prompt)
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    self._to_vertex_tool(format_tool_spec(t)) for t in tools
                ]
            }]
        generation_config = dict(self.kwargs.get("generation_config", {}))
        if temperature is not None:
            generation_config["temperature"] = temperature
        if generation_config:
            payload["generationConfig"] = generation_config

        for key, value in self.kwargs.items():
            if key not in ("project", "location", "publisher", "auth", "generation_config"):
                payload[key] = value
        self.require_model(model)
        if reasoning_payload is not None:
            payload = _deep_merge(payload, reasoning_payload)
        return payload

    @staticmethod
    def _parse_response(  # pylint: disable=too-many-locals
        data: Dict,
    ) -> Tuple[str, Optional[List[CallParams]], Optional[Dict]]:
        candidates = data.get("candidates") or []
        if not candidates:
            error_detail = data.get("error", {}).get("message", str(data))
            raise RuntimeError(f"VertexAI API error: {error_detail}")
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = ""
        call_requests = []
        for part in parts:
            if "text" in part and not part.get("thought"):
                response_text += part.get("text", "")
            if "functionCall" in part:
                function_call = part["functionCall"]
                name = function_call.get("name")
                kwargs = function_call.get("args", {})
                call_requests.append(
                    CallParams(
                        call_id=function_call.get("id") or name or "",
                        name=name,
                        args=[],
                        kwargs=kwargs if isinstance(kwargs, dict) else {},
                    )
                )
        reasoning_payload = {"format": "vertex", "parts": parts} if any(
            part.get("thought") or "thoughtSignature" in part for part in parts
        ) else None
        return response_text, call_requests or None, reasoning_payload

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> LLM_Response:
        """Process a request to Vertex AI Gemini GenerateContent."""
        payload = self._build_request_payload(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            tools=tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            provider_config=provider_config,
        )
        provider = select_model_provider(model, default_provider=(metadata or {}).get("PROVIDER"))
        model, _ = resolve_reasoning_payload(
            self.require_model(model),
            provider_config,
            provider,
        )

        payload_bytes = json.dumps(payload).encode('utf-8')
        total_bytes_sent = len(payload_bytes)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._request_url(model),
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            total_bytes_received = len(response.content)
            data = response.json()

            response_text, call_requests, response_reasoning_payload = self._parse_response(data)
            if response_reasoning_payload is not None:
                response_reasoning_payload["provider"] = provider
            if self.response_format and response_text:
                response_text = json.loads(response_text)["python_code"]
            _usage = data.get("usageMetadata", {})
            cost = _usage.get("cost")
            stats = LLM_Stats(
                total_bytes_sent=total_bytes_sent,
                total_bytes_received=total_bytes_received,
                cost=cost,
                input_tokens=_usage.get("promptTokenCount", 0),
                output_tokens=_usage.get("candidatesTokenCount", 0),
                cached_tokens=_usage.get("cachedContentTokenCount", 0),
            )
            return LLM_Response(
                step_data=LLM_StepData(
                    text=response_text,
                    call_requests=call_requests,
                    reasoning_payload=response_reasoning_payload,
                ),
                stats=stats,
            )


class ClaudeAI_API(LLM_API):
    """Claude API implementation of LLM_API.

    This class provides a concrete implementation for Anthropic's Claude service,
    using the Anthropic Python SDK with prompt caching for multi-turn conversations.
    """

    def __init__(self, settings: LLM_API_Settings,
                 use_prompt_caching: Optional[bool] = None, **kwargs):
        """Initialize Claude API client.

        Args:
            settings: LLM_API_Settings containing API URL and key
            use_prompt_caching: Whether to enable prompt caching for system prompts
                               and conversation history (reduces cost and latency).
                               If None, uses value from settings
                               (env var CLAUDE_USE_PROMPT_CACHING).
        """
        self.settings = settings
        self.response_format = LLM_API._load_response_format(settings)
        self.use_prompt_caching = (
            use_prompt_caching if use_prompt_caching is not None
            else settings.use_prompt_caching
        )
        self.kwargs = kwargs
        self.api_key = settings.api_key


    @staticmethod
    def _content_text_for_item(item: "ChatHistoryItem", chat_style, settings) -> str:
        """Return the text content for ``item`` after applying chat-style wrapping.
        Reuses ``format_chat_history_item`` for consistent formatting and
        extracts the resulting ``content`` string.
        """
        formatted = format_chat_history_item(item, chat_style, settings)
        content = formatted.get("content")
        return content if isinstance(content, str) else ""

    def build_messages(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        provider: Optional[str] = None,
    ) -> List[Dict]:
        """Build the Claude messages list from a ``ChatHistoryItem`` stream.

        SYSTEM items are skipped here — :meth:`extract_system_prompt` already
        consumed the first SYSTEM item to populate the top-level Claude
        ``system`` field.  Remaining items are emitted as Anthropic messages
        using ``tool_use`` / ``tool_result`` blocks where applicable.

        Consecutive TOOL items, optionally followed by a USER item, are
        merged into a single ``role: user`` message containing all of their
        ``tool_result`` blocks plus an optional trailing text block — the
        format Anthropic expects when reporting tool outputs.
        """
        from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel
        if chat_style is None:
            chat_style = get_statek_settings().chat_style
        settings = get_statek_settings()

        if not chat_history:
            return []

        history_list = [
            item for item in chat_history if item.role != ChatRole.SYSTEM
        ]

        # Locate the index of the last assistant text item — used to mark
        # cache_control when prompt caching is enabled.
        last_asst_idx = -1
        for i, item in enumerate(history_list):
            if item.role == ChatRole.ASSISTANT and item.content:
                last_asst_idx = i

        messages: List[Dict] = []
        i = 0
        while i < len(history_list):
            item = history_list[i]

            if item.role == ChatRole.USER:
                content = self._content_text_for_item(item, chat_style, settings)
                if content:
                    messages.append({"role": "user", "content": content})
                i += 1
                continue

            if item.role == ChatRole.TOOL:
                # Batch this and any following TOOL items + a trailing USER
                # item into one Anthropic user message.
                user_blocks: List[Dict] = []
                while i < len(history_list) and history_list[i].role == ChatRole.TOOL:
                    tool_item = history_list[i]
                    tcs = tool_item.tool_calls
                    tool_call_id = ""
                    if tcs is not None:
                        cs = tcs if not isinstance(tcs, list) else (tcs[0] if tcs else None)
                        if cs is not None:
                            tool_call_id = cs.id
                    user_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_item.content or "",
                    })
                    i += 1
                if i < len(history_list) and history_list[i].role == ChatRole.USER:
                    text = self._content_text_for_item(
                        history_list[i], chat_style, settings)
                    if text:
                        user_blocks.append({"type": "text", "text": text})
                    i += 1
                if user_blocks:
                    messages.append({"role": "user", "content": user_blocks})
                continue

            if item.role == ChatRole.ASSISTANT:
                reasoning_payload = _matching_reasoning_payload(
                    getattr(item, "provider_reasoning_payload", None),
                    provider,
                    "claude",
                )
                content = _reasoning_sequence(
                    reasoning_payload.get("content") if reasoning_payload is not None else None)
                if content is not None:
                    messages.append({"role": "assistant", "content": content})
                    i += 1
                    continue
                is_last_asst = i == last_asst_idx
                if item.tool_calls:
                    asst_content: List[Dict] = []
                    if item.content:
                        text_block = {"type": "text", "text": item.content}
                        if self.use_prompt_caching and is_last_asst:
                            text_block["cache_control"] = {"type": "ephemeral"}
                        asst_content.append(text_block)
                    tcs = item.tool_calls
                    if not isinstance(tcs, list):
                        tcs = [tcs]
                    for cs in tcs:
                        asst_content.append({
                            "type": "tool_use",
                            "id": cs.id,
                            "name": cs.func_name,
                            "input": dict(cs.kwargs) if cs.kwargs else {},
                        })
                    messages.append({"role": "assistant", "content": asst_content})
                elif item.content:
                    text = self._content_text_for_item(item, chat_style, settings)
                    if self.use_prompt_caching and is_last_asst:
                        content = [{
                            "type": "text", "text": text,
                            "cache_control": {"type": "ephemeral"},
                        }]
                    else:
                        content = text
                    messages.append({"role": "assistant", "content": content})
                i += 1
                continue

            # Unknown role — skip.
            i += 1

        return messages

    @staticmethod
    def extract_system_prompt(
        chat_history: Optional[Iterable["ChatHistoryItem"]],
    ) -> tuple:
        """Pull the leading SYSTEM item out of ``chat_history``.

        Returns ``(system_text, remaining_iterable)``: ``system_text`` is the
        content of the first SYSTEM item if it appears at the head of the
        stream (otherwise ``None``), and ``remaining_iterable`` is the rest
        of the items in original order.  Materialises the iterable into a
        list — Claude needs random access to compute prompt caching anyway.
        """
        if chat_history is None:
            return None, None
        items = list(chat_history)
        if items and items[0].role == ChatRole.SYSTEM:
            return items[0].content or "", items[1:]
        return None, items

    def _build_system_prompt(self, system_prompt: str) -> List[Dict]:
        """Build the system prompt with optional cache control.

        Args:
            system_prompt: The system prompt text

        Returns:
            System prompt as a list of content blocks (for caching support)
        """
        if self.use_prompt_caching:
            return [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        return system_prompt

    @staticmethod
    def _to_anthropic_tool(spec: Dict) -> Dict:
        """Convert an OpenAI-format tool spec to Anthropic's tool format.

        Args:
            spec: Tool spec dict in OpenAI function-calling format

        Returns:
            Tool dict in Anthropic format with ``input_schema`` instead of ``parameters``
        """
        fn = spec["function"]
        return {
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        }

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> Dict:
        """Build the Anthropic Messages API JSON payload."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        provider = select_model_provider(model, default_provider=(metadata or {}).get("PROVIDER"))
        model, reasoning_payload = resolve_reasoning_payload(
            self.require_model(model),
            provider_config,
            provider,
        )
        payload = {
            "model": self.require_model(model),
            "messages": self.build_messages(chat_history, chat_style=chat_style, provider=provider),
            "max_tokens": self.kwargs.get("max_tokens", 4096),
        }
        if tools:
            payload["tools"] = [
                self._to_anthropic_tool(format_tool_spec(t)) for t in tools
            ]
        if system_prompt:
            payload["system"] = self._build_system_prompt(system_prompt)
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_payload is not None:
            payload = _deep_merge(payload, reasoning_payload)
        return payload

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        provider_config: Optional[ProviderConfig] = None,
    ) -> LLM_Response:
        """Process a request to the Claude API using the Messages API."""
        payload = self._build_request_payload(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            tools=tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            provider_config=provider_config,
        )

        # Prepare headers for Claude API
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Add beta header for prompt caching
        if self.use_prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        # Measure bytes sent
        payload_bytes = json.dumps(payload).encode('utf-8')
        total_bytes_sent = len(payload_bytes)

        # Make the async HTTP request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Measure bytes received
            total_bytes_received = len(response.content)

            # Parse the response
            data = response.json()

            # Extract text and tool_use blocks from Claude's content array
            content_blocks = data.get("content", [])
            response_text = ""
            provider = select_model_provider(
                model,
                default_provider=(metadata or {}).get("PROVIDER"),
            )
            tool_use_blocks = []
            for block in content_blocks:
                if block.get("type") == "text":
                    response_text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_use_blocks.append(block)

            # Convert Claude tool_use blocks to CallParams via extract_call_params
            call_requests = (
                [extract_call_params(block) for block in tool_use_blocks]
                if tool_use_blocks else None
            )

            # When response_format is used, extract python_code from JSON
            if self.response_format and response_text:
                response_text = json.loads(response_text)["python_code"]

            _usage = data.get("usage", {})
            cost = _usage.get("cost")

            stats = LLM_Stats(
                total_bytes_sent=total_bytes_sent,
                total_bytes_received=total_bytes_received,
                cost=cost,
                input_tokens=_usage.get("input_tokens", 0),
                output_tokens=_usage.get("output_tokens", 0),
                cached_tokens=_usage.get("cache_read_input_tokens", 0),
            )

            return LLM_Response(
                step_data=LLM_StepData(
                    text=response_text,
                    call_requests=call_requests,
                    reasoning_payload={
                        "provider": provider.casefold() if isinstance(provider, str) else None,
                        "format": "claude", "content": content_blocks,
                    }
                    if any(
                        block.get("type") in ("thinking", "redacted_thinking")
                        for block in content_blocks
                    )
                    else None,
                ),
                stats=stats,
            )


Claude_API = ClaudeAI_API
