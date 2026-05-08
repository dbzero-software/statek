"""LLM API abstraction layer for Statek."""

# pylint: disable=no-member

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Iterable, Sequence, List, Dict, Callable, Tuple, Any
import json
import httpx

from .settings import LLM_API_Settings, get_provider_settings, get_statek_logger
from .exceptions import InvalidFormat
from .chat_history import (
    ChatHistoryItem, ChatRole, format_chat_history_item,
)

STATEK_LOGGER = get_statek_logger()


def _func_name_from_tool_calls(tool_calls) -> str:
    """Return the function name from a single CallSpec or the first item of a list."""
    if tool_calls is None:
        return ""
    cs = tool_calls if not isinstance(tool_calls, list) else (tool_calls[0] if tool_calls else None)
    return cs.func_name if cs is not None else ""

LLM_Stats = namedtuple(
    "LLM_Stats",
    ["total_bytes_sent", "total_bytes_received", "cost",
     "input_tokens", "output_tokens", "cached_tokens"],
    defaults=[0, 0, 0],
)
# text: response text from the LLM (empty string when the LLM made tool calls instead)
# stats: byte/cost accounting
# call_requests: list of CallParams when the LLM requested tool calls, else None
LLM_Response = namedtuple("LLM_Response", ["text", "stats", "call_requests"])

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


def select_request_tools(
    metadata: Optional[Dict[str, str]] = None,
    available_tools: Optional[Sequence[Callable]] = None,
    chat_style=None,
) -> Optional[List[Callable]]:
    """Return the tool callables that would be forwarded in an LLM request."""
    from .system import select_tools, find_tools  # pylint: disable=import-outside-toplevel

    tools_scope = metadata.get("LLM_TOOLS_SCOPE") if metadata else None
    if not tools_scope or available_tools is None:
        return None

    tools = list(select_tools(available_tools, tools_scope, chat_style=chat_style))
    if tools_scope in ("SYSTEM", "ALL", None):
        existing = {t.__name__ for t in tools}
        for rt in find_tools("SYSTEM", chat_style=chat_style):
            if rt.__name__ not in existing:
                existing.add(rt.__name__)
                tools.append(rt)
    return tools


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
        enable_reasoning: bool = False,
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
        return {
            "system_prompt": system_prompt,
            "model": model,
            "metadata": metadata,
            "tools": tools,
            "chat_history": chat_history,
            "chat_style": chat_style,
            "temperature": temperature,
            "enable_reasoning": enable_reasoning,
        }

    def preview_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        available_tools: Optional[Sequence[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        enable_reasoning: bool = False,
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
            enable_reasoning=enable_reasoning,
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
        enable_reasoning: bool = False,
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
            enable_reasoning: Whether provider reasoning / thinking should be
                enabled for this request.

        Returns:
            LLM_Response containing the response text, stats, and call requests.
        """
        STATEK_LOGGER.debug("%s metadata: %s", self.__class__.__name__, metadata)
        STATEK_LOGGER.debug(
            "%s available_tools: %s",
            self.__class__.__name__,
            [t.__name__ for t in available_tools] if available_tools else None
        )

        request_kwargs = self._prepare_request_kwargs(
            system_prompt=system_prompt,
            model=model,
            metadata=metadata,
            available_tools=available_tools,
            chat_history=chat_history,
            chat_style=chat_style,
            temperature=temperature,
            enable_reasoning=enable_reasoning,
        )

        response = await self._process_request(**request_kwargs)
        STATEK_LOGGER.debug("%s response: %s", self.__class__.__name__, response.text)
        if response.call_requests:
            STATEK_LOGGER.debug(
                "%s call_requests: %s",
                self.__class__.__name__,
                [cp.name for cp in response.call_requests]
            )
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
        enable_reasoning: bool = False,
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
        enable_reasoning: bool = False,
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
    def parse_enable_reasoning(value: Optional[str]) -> bool:
        """Convert metadata REASONING values into a boolean flag."""
        if value is None:
            return False
        normalized = value.strip().upper()
        if normalized in ("", "FALSE", "0", "NO", "OFF", "DISABLED", "NONE"):
            return False
        return True

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
        settings = get_provider_settings(provider_key)
        if not settings:
            raise ValueError(f"No settings found for {provider_key} provider.")

        if provider_key == 'OPENROUTER':
            return OpenRouter_API(settings=settings, **kwargs)
        if provider_key == 'OPENAI':
            return OpenAI_API(settings=settings, **kwargs)
        if provider_key in ('VERTEXAI', 'VERTEX_AI', 'GOOGLE_VERTEXAI', 'GOOGLE'):
            return VertexAI_API(settings=settings, **kwargs)
        if provider_key in ('CLAUDEAI', 'CLAUDE_AI', 'CLAUDE', 'ANTHROPIC'):
            return ClaudeAI_API(settings=settings, **kwargs)
        raise ValueError(f"Unsupported LLM API provider: {provider_name}")

class OpenRouter_API(LLM_API):
    """OpenRouter API implementation of LLM_API.

    This class provides a concrete implementation for the OpenRouter service,
    which acts as a gateway to multiple LLM providers.
    """

    def __init__(self, settings: LLM_API_Settings, **kwargs):
        """Initialize OpenRouter API client.

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

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        enable_reasoning: bool = False,
    ) -> Dict:
        """Build the OpenAI-compatible JSON payload for OpenRouter/OpenAI."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        del metadata
        payload = {
            "model": self.require_model(model),
            "messages": self.build_messages(
                system_prompt=system_prompt,
                chat_history=chat_history,
                chat_style=chat_style,
            ),
        }
        if tools:
            payload["tools"] = [format_tool_spec(t) for t in tools]
        if self.response_format:
            payload["response_format"] = self.response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if enable_reasoning:
            payload["reasoning"] = {}
        payload.update(self.kwargs)
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
        enable_reasoning: bool = False,
    ) -> LLM_Response:
        """Process a request to the OpenRouter API.

        Args:
            system_prompt: Optional system prompt
            model: Required provider model.
            metadata: Optional metadata unrelated to provider model selection.
            tools: Optional list of tool callables to include as formal tool definitions
                   in the OpenAI function-calling format.
            chat_history: Conversation history including the latest user message as
                         the final element

        Returns:
            LLM_Response with the generated text, stats, and tool calls

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
            enable_reasoning=enable_reasoning,
        )
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
                text=response_text, stats=stats,
                call_requests=call_requests
            )


class OpenAI_API(OpenRouter_API):
    """OpenAI API implementation using the chat completions wire format."""


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

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        enable_reasoning: bool = False,
    ) -> Dict:
        """Build the Vertex AI GenerateContent JSON payload."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        del metadata, enable_reasoning
        payload = {"contents": self.build_contents(chat_history, chat_style=chat_style)}
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
        return payload

    @staticmethod
    def _parse_response(data: Dict) -> Tuple[str, Optional[List[CallParams]]]:
        candidates = data.get("candidates") or []
        if not candidates:
            error_detail = data.get("error", {}).get("message", str(data))
            raise RuntimeError(f"VertexAI API error: {error_detail}")
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = ""
        call_requests = []
        for part in parts:
            if "text" in part:
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
        return response_text, call_requests or None

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        enable_reasoning: bool = False,
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
            enable_reasoning=enable_reasoning,
        )
        model = self.require_model(model)

        payload_bytes = json.dumps(payload).encode('utf-8')
        total_bytes_sent = len(payload_bytes)
        if STATEK_LOGGER.isEnabledFor(10):  # logging.DEBUG
            STATEK_LOGGER.debug("VertexAI payload: %s", json.dumps(payload, indent=2))

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._request_url(model),
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            total_bytes_received = len(response.content)
            data = response.json()
            STATEK_LOGGER.debug("VertexAI response: %s", json.dumps(data))

            response_text, call_requests = self._parse_response(data)
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
                text=response_text,
                stats=stats,
                call_requests=call_requests,
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

    def _build_request_payload(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable["ChatHistoryItem"]] = None,
        chat_style=None,
        temperature: Optional[float] = None,
        enable_reasoning: bool = False,
    ) -> Dict:
        """Build the Anthropic Messages API JSON payload."""
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        del metadata
        payload = {
            "model": self.require_model(model),
            "messages": self.build_messages(chat_history, chat_style=chat_style),
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
        if enable_reasoning:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 1024,
            }
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
        enable_reasoning: bool = False,
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
            enable_reasoning=enable_reasoning,
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
                text=response_text, stats=stats,
                call_requests=call_requests
            )


Claude_API = ClaudeAI_API
