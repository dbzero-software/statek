"""LLM API abstraction layer for Statek."""

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Iterable, Sequence, List, Dict, Callable, Union
import json
import httpx

from .settings import LLM_API_Settings, get_provider_settings, get_statek_logger
from .exceptions import InvalidFormat
from .utils import strip_markup

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


@dataclass
class ChatStepAssistantData:
    """Represents a single completed assistant step in the LLM conversation history.

    Each step captures the LLM's code response and the subsequent console
    output produced by executing that code. Tool calls made during the step
    are stored in tool_calls.
    """
    code: str
    """Code requested for execution in this step (LLM assistant response)."""
    console_output: str
    """The console output from code execution (user feedback message)."""
    tool_calls: Optional[Dict] = None
    """Tool call requests and their results: Dict[CallParams, str]."""


@dataclass
class ChatStepUserData:
    """Represents a single user message step in the LLM conversation history."""
    message: str
    """Message sent by the user."""


ChatStepData = Union[ChatStepAssistantData, ChatStepUserData]


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
        chat_history: Optional[Iterable[ChatStepData]] = None,
        session_id: Optional[str] = None,
        chat_style=None
    ) -> LLM_Response:
        """Process a request to the LLM API.

        Logs metadata before the call and the response text after. Delegates
        the actual provider interaction to _process_request.

        When metadata contains ``"LLM_TOOLS_SCOPE"``, the tools from
        ``available_tools`` are filtered by that scope (via ``select_tools``)
        and forwarded to the provider as a formal tools parameter.  When
        ``chat_style`` is also provided, tools are additionally filtered by
        their target chat style.

        Args:
            system_prompt: Optional system prompt to guide the LLM behavior
            metadata: Optional metadata key/value pairs. If the ``"MODEL"`` key
                     is present it overrides the class-level default model for
                     this request. If ``"LLM_TOOLS_SCOPE"`` is present, tools
                     from ``available_tools`` matching that scope are sent to
                     the provider.
            available_tools: All tools available in the agent's local context.
                     Only used when ``"LLM_TOOLS_SCOPE"`` is set in metadata.
            chat_history: Conversation history including the latest user message as the
                         final element. Depending on the provider, if session is not managed
                         on the provider side, this history needs to be included in the
                         message sent to the LLM API
            session_id: Provider-specific session ID (if request is a continuation)
            chat_style: Optional ChatStyle to filter tools by their target chat
                     style. Only applied when ``"LLM_TOOLS_SCOPE"`` is set.

        Returns:
            LLM_Response containing the response text, optional session_id, and stats

        Raises:
            Exception: If the API request fails or model cannot be determined
        """
        from .system import select_tools, find_tools  # pylint: disable=import-outside-toplevel

        STATEK_LOGGER.debug("%s metadata: %s", self.__class__.__name__, metadata)
        STATEK_LOGGER.debug(
            "%s available_tools: %s",
            self.__class__.__name__,
            [t.__name__ for t in available_tools] if available_tools else None
        )

        tools_scope = metadata.get("LLM_TOOLS_SCOPE") if metadata else None
        if tools_scope and available_tools:
            tools = select_tools(available_tools, tools_scope, chat_style=chat_style)
            # Merge system tools from the global registry that aren't already
            # in the agent's tool list (e.g. python_cli).
            if tools_scope in ("SYSTEM", "ALL", None):
                existing = {t.__name__ for t in tools}
                for rt in find_tools("SYSTEM", chat_style=chat_style):
                    if rt.__name__ not in existing:
                        existing.add(rt.__name__)
                        tools.append(rt)
        else:
            tools = None

        if chat_history is not None and chat_style is not None:
            from .chat_style import ChatStyle  # pylint: disable=import-outside-toplevel
            if chat_style == ChatStyle.DIRECT:  # pylint: disable=no-member
                chat_history = self._wrap_direct_chat_history(chat_history)

        response = await self._process_request(
            system_prompt=system_prompt,
            metadata=metadata,
            tools=tools,
            chat_history=chat_history,
            session_id=session_id
        )
        STATEK_LOGGER.debug("%s response: %s", self.__class__.__name__, response.text)
        if response.call_requests:
            STATEK_LOGGER.debug(
                "%s call_requests: %s",
                self.__class__.__name__,
                [cp.name for cp in response.call_requests]
            )
        return response

    @staticmethod
    def _wrap_direct_chat_history(
        chat_history: Iterable["ChatStepData"],
    ) -> Iterable["ChatStepData"]:
        """Rewrite chat history for DIRECT chat style.

        In DIRECT mode the LLM executes code exclusively via ``python_cli``
        tool calls.  Any ``ChatStepAssistantData`` step that carries ``code``
        is rewritten so the code appears as a ``python_cli`` tool call with
        the ``console_output`` as its result.  When the step already has
        ``tool_calls``, the new ``python_cli`` call is merged into them.
        This keeps the conversation history consistent with the actual chat
        style the LLM operates in.

        Steps without ``code`` or ``ChatStepUserData`` are yielded unchanged.
        """
        counter = 0
        for step in chat_history:
            if not isinstance(step, ChatStepAssistantData):
                yield step
                continue
            if not step.code:
                yield step
                continue
            # Strip markdown code fences (e.g. ```python\n...\n```)
            code = strip_markup(step.code, strict=True)
            # Wrap code as a python_cli tool call
            counter += 1
            cp = CallParams(
                call_id=f"STATEK-W-{counter:03d}",
                name="python_cli",
                args=[],
                kwargs={"code": code},
            )
            tool_calls = dict(step.tool_calls) if step.tool_calls else {}
            tool_calls[cp] = step.console_output
            yield ChatStepAssistantData(
                code="",
                console_output="",
                tool_calls=tool_calls,
            )

    @abstractmethod
    async def _process_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable[ChatStepData]] = None,
        session_id: Optional[str] = None
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
            model: Specific model to use. If None, uses default_model from settings

        Raises:
            ValueError: If no model is specified and no default is available
        """
        self.settings = settings
        self.model = model or settings.default_model
        self.response_format = LLM_API._load_response_format(settings)
        # additional kwargs that will be passed to request if needed
        self.kwargs = kwargs
        if not self.model:
            raise ValueError(
                "No model specified and no default model available in settings. "
                "Please provide a model name or configure default_model in settings."
            )

        self.api_url = settings.api_url
        self.api_key = settings.api_key
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def build_messages(
        self,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[ChatStepData]] = None
    ) -> List[Dict[str, str]]:
        """Build the messages list for the OpenRouter API request.

        Each ChatStepData expands to an optional assistant message (code) followed
        by an optional user message (console_output). When the step has tool_calls,
        the assistant message uses the OpenAI tool_calls format and each tool result
        is emitted as a separate ``role: tool`` message before the user message.

        Args:
            system_prompt: Optional system prompt
            chat_history: Conversation history as ChatStepData objects. The final
                         element's console_output is the current user message.

        Returns:
            List of message dictionaries with 'role' and 'content' fields
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if chat_history:
            for step in chat_history:
                if isinstance(step, ChatStepUserData):
                    if step.message:
                        messages.append({"role": "user", "content": step.message})
                    continue
                if step.tool_calls:
                    tool_call_list = [
                        {
                            "id": cp.id,
                            "type": "function",
                            "function": {
                                "name": cp.name,
                                "arguments": json.dumps(cp.kwargs or {})
                            }
                        }
                        for cp in step.tool_calls
                    ]
                    asst_msg = {"role": "assistant", "tool_calls": tool_call_list,
                                "content": step.code or None}
                    messages.append(asst_msg)
                    for cp, result in step.tool_calls.items():
                        messages.append({"role": "tool", "tool_call_id": cp.id, "content": result})
                elif step.code:
                    messages.append({"role": "assistant", "content": step.code})
                if step.console_output:
                    messages.append({"role": "user", "content": step.console_output})

        return messages

    async def _process_request(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tools: Optional[List[Callable]] = None,
        chat_history: Optional[Iterable[ChatStepData]] = None,
        session_id: Optional[str] = None
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

        messages = self.build_messages(system_prompt, chat_history)
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
            model: Specific model to use. If None, uses default_model from settings
            use_prompt_caching: Whether to enable prompt caching for system prompts
                               and conversation history (reduces cost and latency).
                               If None, uses value from settings
                               (env var CLAUDE_USE_PROMPT_CACHING).

        Raises:
            ValueError: If no model is specified and no default is available
        """
        self.settings = settings
        self.model = model or settings.default_model
        self.response_format = LLM_API._load_response_format(settings)
        self.use_prompt_caching = (
            use_prompt_caching if use_prompt_caching is not None
            else settings.use_prompt_caching
        )
        self.kwargs = kwargs
        if not self.model:
            raise ValueError(
                "No model specified and no default model available in settings. "
                "Please provide a model name or configure default_model in settings."
            )
        self.api_key = settings.api_key

    def _step_with_tool_calls(self, step: "ChatStepAssistantData", is_last: bool) -> List[Dict]:
        """Return the (assistant, user) message pair for a step that has tool calls."""
        asst_content = []
        if step.code:
            text_block = {"type": "text", "text": step.code}
            if self.use_prompt_caching and is_last:
                text_block["cache_control"] = {"type": "ephemeral"}
            asst_content.append(text_block)
        for cp in step.tool_calls:
            asst_content.append({
                "type": "tool_use", "id": cp.id,
                "name": cp.name, "input": dict(cp.kwargs) if cp.kwargs else {}
            })

        user_content = [
            {"type": "tool_result", "tool_use_id": cp.id, "content": result}
            for cp, result in step.tool_calls.items()
        ]
        if step.console_output:
            user_content.append({"type": "text", "text": step.console_output})

        msgs = [{"role": "assistant", "content": asst_content}]
        if user_content:
            msgs.append({"role": "user", "content": user_content})
        return msgs

    def _step_without_tool_calls(self, step: "ChatStepAssistantData", is_last: bool) -> List[Dict]:
        """Return the (assistant, user) message pair for a plain code step."""
        msgs = []
        if step.code:
            if self.use_prompt_caching and is_last:
                content = [{"type": "text", "text": step.code,
                            "cache_control": {"type": "ephemeral"}}]
            else:
                content = step.code
            msgs.append({"role": "assistant", "content": content})
        if step.console_output:
            msgs.append({"role": "user", "content": step.console_output})
        return msgs

    def build_messages(
        self,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[ChatStepData]] = None
    ) -> List[Dict]:
        """Build the messages list for the Claude API request.

        Each ChatStepData expands to an optional assistant message (code) followed
        by an optional user message (console_output). When prompt caching is enabled,
        cache_control is added to the last step's assistant message. When the step
        has tool_calls the messages use Anthropic's tool_use / tool_result format.

        Args:
            system_prompt: Optional system prompt (included for display/logging only;
                          the actual API call passes it as a top-level 'system' field)
            chat_history: Conversation history as ChatStepData objects. The final
                         element's console_output is the current user message.

        Returns:
            List of message dictionaries with 'role' and 'content' fields
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if not chat_history:
            return messages

        history_list = list(chat_history)
        for i, step in enumerate(history_list):
            is_last = i == len(history_list) - 1
            if isinstance(step, ChatStepUserData):
                if step.message:
                    messages.append({"role": "user", "content": step.message})
            elif step.tool_calls:
                messages.extend(self._step_with_tool_calls(step, is_last))
            else:
                messages.extend(self._step_without_tool_calls(step, is_last))

        return messages

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
        chat_history: Optional[Iterable[ChatStepData]] = None,
        session_id: Optional[str] = None
    ) -> LLM_Response:
        """Process a request to the Claude API using the Messages API.

        Args:
            system_prompt: Optional system prompt
            metadata: Optional metadata. If it contains the "MODEL" key it overrides
                     the instance-level default model for this request.
            tools: Optional list of tool callables to include as formal tool definitions
                   in Anthropic's tool format.
            chat_history: Conversation history including the latest user message as
                         the final element
            session_id: Not used (Claude Messages API is stateless)

        Returns:
            LLM_Response with the generated text and None for session_id

        Raises:
            httpx.HTTPError: If the API request fails
            KeyError: If the response format is unexpected
        """
        from .utils import format_tool_spec  # pylint: disable=import-outside-toplevel

        messages = self.build_messages(chat_history=chat_history)
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
