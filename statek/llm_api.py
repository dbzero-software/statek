"""LLM API abstraction layer for Statek."""

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Iterable, List, Dict
import json
import httpx

from .settings import LLM_API_Settings, get_provider_settings, get_statek_logger, statek_log

STATEK_LOGGER = get_statek_logger()

# Named tuple for LLM response
LLM_Response = namedtuple("LLM_Response", ["text", "session_id"])


class LLM_API(ABC):
    """Abstract base class for LLM API wrappers.

    This class provides a stateless interface to abstract away individual
    provider specifics and match them with Statek's requirements.
    A single LLM_API instance is intended for a single session with the LLM agent.
    """

    @abstractmethod
    async def process_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[str]] = None,
        session_id: Optional[str] = None
    ) -> LLM_Response:
        """Process a request to the LLM API.

        Args:
            prompt: The prompt to be sent to LLM (only the latest input)
            system_prompt: Optional system prompt to guide the LLM behavior
            chat_history: Optional conversation history so far. Depending on the provider,
                         if session is not managed on the provider side, this history
                         needs to be included in the message sent to the LLM API
            session_id: Provider-specific session ID (if request is a continuation)

        Returns:
            LLM_Response containing the response text and optional session_id

        Raises:
            Exception: If the API request fails or model cannot be determined
        """

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

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[str]] = None
    ) -> List[Dict[str, str]]:
        """Build the messages list for the OpenRouter API request.

        Args:
            prompt: The current user prompt
            system_prompt: Optional system prompt
            chat_history: Optional chat history

        Returns:
            List of message dictionaries with 'role' and 'content' fields
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add chat history if provided
        if chat_history:
            for i, message in enumerate(chat_history):
                # Alternate between user and assistant roles
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": message})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    async def process_request(  # pylint: disable=too-many-locals
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[str]] = None,
        session_id: Optional[str] = None
    ) -> LLM_Response:
        """Process a request to the OpenRouter API.

        Args:
            prompt: The prompt to be sent to LLM
            system_prompt: Optional system prompt
            chat_history: Optional conversation history
            session_id: Not used by OpenRouter (stateless)

        Returns:
            LLM_Response with the generated text and None for session_id

        Raises:
            httpx.HTTPError: If the API request fails
            KeyError: If the response format is unexpected
        """
        messages = self._build_messages(prompt, system_prompt, chat_history)

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages
        }
        if self.response_format:
            payload["response_format"] = self.response_format
        messages_str = "Sending request to OpenRouter with the following messages:\n"
        for message in messages:
            messages_str += f"Message role: {message['role']}, content: {message['content']}\n"
        statek_log(messages_str, level='debug')
        # set any additional parameters from kwargs
        payload.update(self.kwargs)
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Make the async HTTP request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Extract the response text
            # OpenRouter follows OpenAI's response format
            response_text = data["choices"][0]["message"]["content"]

            # When response_format is used, extract python_code from JSON
            if self.response_format:
                response_text = json.loads(response_text)["python_code"]

            # OpenRouter is stateless, so session_id is None
            return LLM_Response(text=response_text, session_id=None)


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

    def _build_messages(
        self,
        prompt: str,
        chat_history: Optional[Iterable[str]] = None
    ) -> List[Dict]:
        """Build the messages list for the Claude API request.

        Args:
            prompt: The current user prompt
            chat_history: Optional chat history

        Returns:
            List of message dictionaries with 'role' and 'content' fields
        """
        messages = []

        # Add chat history if provided
        if chat_history:
            history_list = list(chat_history)
            for i, message in enumerate(history_list):
                # Alternate between user and assistant roles
                role = "user" if i % 2 == 0 else "assistant"
                msg = {"role": role, "content": message}

                # Add cache_control to the last history message if caching is enabled
                if self.use_prompt_caching and i == len(history_list) - 1:
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": message,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]

                messages.append(msg)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

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

    async def process_request(  # pylint: disable=too-many-locals
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[Iterable[str]] = None,
        session_id: Optional[str] = None
    ) -> LLM_Response:
        """Process a request to the Claude API using the Messages API.

        Args:
            prompt: The prompt to be sent to LLM
            system_prompt: Optional system prompt
            chat_history: Optional conversation history
            session_id: Not used (Claude Messages API is stateless)

        Returns:
            LLM_Response with the generated text and None for session_id

        Raises:
            httpx.HTTPError: If the API request fails
            KeyError: If the response format is unexpected
        """
        messages = self._build_messages(prompt, chat_history)

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.kwargs.get("max_tokens", 4096)
        }

        # Add system prompt if provided (Claude uses a separate 'system' field)
        if system_prompt:
            payload["system"] = self._build_system_prompt(system_prompt)

        messages_str = "Sending request to Claude with the following messages:\n"
        for message in messages:
            messages_str += f"Message role: {message['role']}, content: {message['content']}\n"
        statek_log(messages_str, level='debug')

        # Prepare headers for Claude API
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        # Add beta header for prompt caching
        if self.use_prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        # Make the async HTTP request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Extract the response text from Claude's response format
            # Claude returns content as an array of content blocks
            content_blocks = data.get("content", [])
            response_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # When response_format is used, extract python_code from JSON
            if self.response_format:
                response_text = json.loads(response_text)["python_code"]

            # Claude Messages API is stateless, so session_id is None
            return LLM_Response(text=response_text, session_id=None)
