"""LLM API abstraction layer for Statek."""

from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Optional, Iterable, List, Dict
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

    async def process_request(
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

        # Log the user message at INFO level
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        if user_messages:
            last_user_message = user_messages[-1]['content']
            STATEK_LOGGER.info(f"User message to LLM:\\n{last_user_message}")

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages
        }
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
        import time
        start_time = time.time()
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
            
            # Log response time
            elapsed_time = time.time() - start_time
            STATEK_LOGGER.info(f"LLM response time: {elapsed_time:.2f} seconds")

            # OpenRouter is stateless, so session_id is None
            return LLM_Response(text=response_text, session_id=None)
