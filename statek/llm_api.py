"""LLM API abstraction layer for Statek."""

# pylint: disable=no-member

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Iterable, Sequence, List, Dict, Callable
import json
import httpx

from .settings import LLM_API_Settings, get_provider_settings, get_statek_logger
from .exceptions import InvalidFormat
from .chat_history import (
    ChatHistoryItem, ChatRole, format_chat_history_item,
)

STATEK_LOGGER = get_statek_logger()

LLM_Stats = namedtuple("LLM_Stats", ["total_bytes_sent", "total_bytes_received", "cost"])
# text: response text from the LLM (empty string when the LLM made tool calls instead)
# session_id: optional provider-managed session identifier
# stats: byte/cost accounting
# call_requests: list of CallParams when the LLM requested tool calls, else None
LLM_Response = namedtuple("LLM_Response", ["text", "session_id", "stats", "call_requests"])

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


class LLM_API(ABC):
    """Abstract base class for LLM API wrappers.

    This class provides a stateless interface to abstract away individual
    provider specifics and match them with Statek's requirements.
    A single LLM_API instance is intended for a single session with the LLM agent.
    """

    async def process_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        available_tools: Optional[Sequence[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        session_id: Optional[str] = None,
        chat_style=None
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
            metadata: Optional metadata key/value pairs. ``"MODEL"`` overrides
                the default model. ``"LLM_TOOLS_SCOPE"`` selects tools from
                ``available_tools`` to send to the provider.
            available_tools: All tools available in the agent's local context.
                Only used when ``"LLM_TOOLS_SCOPE"`` is set in metadata.
            chat_history: Conversation history as ``ChatHistoryItem`` objects.
            session_id: Provider-specific session ID (if continuation).
            chat_style: Optional ChatStyle. Threaded through for tool selection
                and message formatting; in DIRECT mode the history is rewritten
                so assistant code becomes ``python_cli`` tool calls.

        Returns:
            LLM_Response containing the response text, optional session_id, and stats.
        """
        from .system import select_tools, find_tools  # pylint: disable=import-outside-toplevel

        STATEK_LOGGER.debug("%s metadata: %s", self.__class__.__name__, metadata)
        STATEK_LOGGER.debug(
            "%s available_tools: %s",
            self.__class__.__name__,
            [t.__name__ for t in available_tools] if available_tools else None
        )

        tools_scope = metadata.get("LLM_TOOLS_SCOPE") if metadata else None
        if tools_scope and available_tools is not None:
            tools = select_tools(available_tools, tools_scope, chat_style=chat_style)
            if tools_scope in ("SYSTEM", "ALL", None):
                existing = {t.__name__ for t in tools}
                for rt in find_tools("SYSTEM", chat_style=chat_style):
                    if rt.__name__ not in existing:
                        existing.add(rt.__name__)
                        tools.append(rt)
        else:
            tools = None

        response = await self._process_request(
            system_prompt=system_prompt,
            metadata=metadata,
            tools=tools,
            chat_history=chat_history,
            session_id=session_id,
            chat_style=chat_style,
        )
        STATEK_LOGGER.debug("%s response: %s", self.__class__.__name__, response.text)
        if response.call_requests:
            STATEK_LOGGER.debug(
                "%s call_requests: %s",
                self.__class__.__name__,
                [cp.name for cp in response.call_requests]
            )
        return response

    @abstractmethod
    async def _process_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        session_id: Optional[str] = None,
        chat_style=None,
    ) -> LLM_Response:
        """Provider-specific request implementation. Called by process_request."""

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
    def get(provider_name: str = None, model: str = None, **kwargs):  # pylint: disable=unused-argument
        """
        Factory method to get an LLM_API instance for a specific provider.
        """
        if provider_name.upper() == 'OPENROUTER':
            settings = get_provider_settings('OPENROUTER')
            if not settings:
                raise ValueError("No settings found for OpenRouter provider.")
            return OpenRouter_API(settings=settings, model=model)
        if provider_name.upper() in ('CLAUDE', 'ANTHROPIC'):
            settings = get_provider_settings('CLAUDE')
            if not settings:
                raise ValueError("No settings found for Claude provider.")
            return Claude_API(settings=settings, model=model, **kwargs)
        raise ValueError(f"Unsupported LLM API provider: {provider_name}")


class OpenRouter_API(LLM_API):
    """OpenRouter API implementation of LLM_API.

    This class provides a concrete implementation for the OpenRouter service,
    which acts as a gateway to multiple LLM providers.
    """

    def __init__(self, settings: LLM_API_Settings, model: Optional[str] = None, **kwargs):
        """Initialize OpenRouter API client.

        Args:
            settings: LLM_API_Settings containing API URL and key
            model: Specific model to use.

        Raises:
            ValueError: If no model is specified
        """
        self.settings = settings
        self.model = model
        self.response_format = LLM_API._load_response_format(settings)
        # additional kwargs that will be passed to request if needed
        self.kwargs = kwargs
        if not self.model:
            raise ValueError(
                "No model specified. Please provide a model name."
            )

        self.api_url = settings.api_url
        self.api_key = settings.api_key
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def build_messages(
        self,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
    ) -> List[Dict[str, str]]:
        """Build the OpenRouter messages list from a ``ChatHistoryItem`` stream.

        Delegates per-item formatting to :func:`format_chat_history_item`,
        which produces dicts in the OpenAI / OpenRouter chat-completions
        schema.  ``chat_style`` defaults to the global StatekSettings value
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
        messages.extend(
            format_chat_history_item(item, chat_style, settings)
            for item in chat_history
        )
        return messages

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        session_id: Optional[str] = None,
        chat_style=None,
    ) -> LLM_Response:
        """Process a request to the OpenRouter API.

        Args:
            system_prompt: Optional system prompt
            metadata: Optional metadata. If it contains the "MODEL" key it overrides
                     the instance-level default model for this request.
            tools: Optional list of tool callables to include as formal tool definitions
                   in the OpenAI function-calling format.
            chat_history: Conversation history including the latest user message as
                         the final element
            session_id: Not used by OpenRouter (stateless)

        Returns:
            LLM_Response with the generated text and None for session_id

        Raises:
            httpx.HTTPError: If the API request fails
            KeyError: If the response format is unexpected
        """
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        messages = self.build_messages(
            system_prompt=system_prompt,
            chat_history=chat_history,
            chat_style=chat_style,
        )
        model = metadata.get('MODEL', self.model) if metadata else self.model

        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages
        }
        if tools:
            payload["tools"] = [format_tool_spec(t) for t in tools]
        if self.response_format:
            payload["response_format"] = self.response_format
        # set any additional parameters from kwargs
        payload.update(self.kwargs)
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Measure bytes sent
        payload_bytes = json.dumps(payload).encode('utf-8')
        total_bytes_sent = len(payload_bytes)
        if STATEK_LOGGER.isEnabledFor(10):  # logging.DEBUG
            STATEK_LOGGER.debug("OpenRouter payload: %s", json.dumps(payload, indent=2))

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
            STATEK_LOGGER.debug("OpenRouter response: %s", json.dumps(data))

            # Extract the response text
            # OpenRouter follows OpenAI's response format
            if "choices" not in data or not data["choices"]:
                error_detail = data.get("error", {}).get("message", str(data))
                raise RuntimeError(f"OpenRouter API error: {error_detail}")

            message = data["choices"][0]["message"]

            # Parse tool calls when the LLM chose to invoke tools
            raw_tool_calls = message.get("tool_calls")
            call_requests = (
                [extract_call_params(tc) for tc in raw_tool_calls]
                if raw_tool_calls else None
            )

            response_text = message.get("content") or ""

            # When response_format is used, extract python_code from JSON
            if self.response_format and response_text:
                response_text = json.loads(response_text)["python_code"]

            cost = data.get("usage", {}).get("cost")

            stats = LLM_Stats(
                total_bytes_sent=total_bytes_sent,
                total_bytes_received=total_bytes_received,
                cost=cost
            )

            # OpenRouter is stateless, so session_id is None
            return LLM_Response(
                text=response_text, session_id=None, stats=stats,
                call_requests=call_requests
            )


class Claude_API(LLM_API):
    """Claude API implementation of LLM_API.

    This class provides a concrete implementation for Anthropic's Claude service,
    using the Anthropic Python SDK with prompt caching for multi-turn conversations.
    """

    def __init__(self, settings: LLM_API_Settings, model: Optional[str] = None,
                 use_prompt_caching: Optional[bool] = None, **kwargs):
        """Initialize Claude API client.

        Args:
            settings: LLM_API_Settings containing API URL and key
            model: Specific model to use.
            use_prompt_caching: Whether to enable prompt caching for system prompts
                               and conversation history (reduces cost and latency).
                               If None, uses value from settings
                               (env var CLAUDE_USE_PROMPT_CACHING).

        Raises:
            ValueError: If no model is specified
        """
        self.settings = settings
        self.model = model
        self.response_format = LLM_API._load_response_format(settings)
        self.use_prompt_caching = (
            use_prompt_caching if use_prompt_caching is not None
            else settings.use_prompt_caching
        )
        self.kwargs = kwargs
        if not self.model:
            raise ValueError(
                "No model specified. Please provide a model name."
            )
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

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        session_id: Optional[str] = None,
        chat_style=None,
    ) -> LLM_Response:
        """Process a request to the Claude API using the Messages API."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        messages = self.build_messages(chat_history, chat_style=chat_style)
        model = metadata.get('MODEL', self.model) if metadata else self.model

        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.kwargs.get("max_tokens", 4096)
        }

        # Add tools in Anthropic format if provided
        if tools:
            payload["tools"] = [
                self._to_anthropic_tool(format_tool_spec(t)) for t in tools
            ]

        # Add system prompt if provided (Claude uses a separate 'system' field)
        if system_prompt:
            payload["system"] = self._build_system_prompt(system_prompt)

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
        if STATEK_LOGGER.isEnabledFor(10):  # logging.DEBUG
            STATEK_LOGGER.debug("Claude payload: %s", json.dumps(payload, indent=2))

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
            STATEK_LOGGER.debug("Claude response: %s", json.dumps(data))

            # Extract text and tool_use blocks from Claude's content array
            content_blocks = data.get("content", [])
            response_text = ""
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

            cost = data.get("usage", {}).get("cost")

            stats = LLM_Stats(
                total_bytes_sent=total_bytes_sent,
                total_bytes_received=total_bytes_received,
                cost=cost
            )

            # Claude Messages API is stateless, so session_id is None
            return LLM_Response(
                text=response_text, session_id=None, stats=stats,
                call_requests=call_requests
            )
