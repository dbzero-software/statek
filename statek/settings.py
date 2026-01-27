"""Settings module for Statek - LLM API configuration management."""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLM_API_Settings(BaseSettings):
    """Settings for a specific LLM provider (e.g. OpenAI, OpenRouter, Google).

    Attributes:
        api_url: The base URL for the LLM API
        api_key: The API key for authentication
        default_model: Optional default chat model name (e.g. gpt-4, gpt-3.5-turbo)
    """
    api_url: str
    api_key: str
    default_model: Optional[str] = None

    model_config = SettingsConfigDict(extra='ignore')


class StatekSettings(BaseSettings):
    """Main settings class for Statek, aggregating LLM API settings by provider.

    Environment variables should be prefixed with the provider name:
    - OPENROUTER_API_URL
    - OPENROUTER_API_KEY
    - OPENROUTER_DEFAULT_MODEL (optional)
    - OPENAI_API_URL
    - OPENAI_API_KEY
    - OPENAI_DEFAULT_MODEL (optional)

    Attributes:
        llm_api_settings: Dictionary of LLM_API_Settings by provider name
        default_llm_api_provider: The default provider to use
    """
    llm_api_settings: Dict[str, LLM_API_Settings] = Field(default_factory=dict)
    default_llm_api_provider: str = "OPENROUTER"

    model_config = SettingsConfigDict(extra='ignore')

    def __init__(self, **data):
        """Initialize StatekSettings by parsing environment variables.

        Automatically detects provider-prefixed environment variables and
        creates LLM_API_Settings instances for each provider.
        """
        super().__init__(**data)

        # Parse environment variables to build llm_api_settings dictionary
        if not self.llm_api_settings:
            self.llm_api_settings = self._parse_llm_providers_from_env()

    @staticmethod
    def _parse_llm_providers_from_env() -> Dict[str, LLM_API_Settings]:
        """Parse environment variables to extract LLM provider settings.

        Looks for environment variables with the pattern:
        {PROVIDER}_API_URL, {PROVIDER}_API_KEY, {PROVIDER}_DEFAULT_MODEL

        Returns:
            Dictionary mapping provider names to their LLM_API_Settings
        """
        providers: Dict[str, Dict[str, str]] = {}

        # Scan environment variables for provider-prefixed keys
        for key, value in os.environ.items():
            if '_API_URL' in key:
                provider = key.replace('_API_URL', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['api_url'] = value
            elif '_API_KEY' in key:
                provider = key.replace('_API_KEY', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['api_key'] = value
            elif '_DEFAULT_MODEL' in key:
                provider = key.replace('_DEFAULT_MODEL', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['default_model'] = value

        # Create LLM_API_Settings instances for each provider
        llm_settings = {}
        for provider_name, settings_dict in providers.items():
            # Only create settings if we have both required fields
            if 'api_url' in settings_dict and 'api_key' in settings_dict:
                llm_settings[provider_name] = LLM_API_Settings(**settings_dict)

        return llm_settings

    def get_provider_settings(self, provider: Optional[str] = None) -> Optional[LLM_API_Settings]:
        """Get settings for a specific provider or the default provider.

        Args:
            provider: Provider name (e.g. 'OPENAI', 'OPENROUTER').
                     If None, uses default_llm_api_provider.

        Returns:
            LLM_API_Settings for the requested provider, or None if not found
        """
        provider_name = provider or self.default_llm_api_provider
        return self.llm_api_settings.get(provider_name)



@lru_cache()
def get_provider_settings(provider: Optional[str] = None) -> Optional[LLM_API_Settings]:
    """Get LLM_API_Settings for a specific provider or the default provider."""
    settings = StatekSettings()
    return settings.get_provider_settings(provider)

@lru_cache()
def get_statek_settings() -> StatekSettings:
    """Get the cached StatekSettings instance."""
    return StatekSettings()


def configure_logging(level: str = "WARNING") -> None:
    """Configure logging for Statek only, without affecting other loggers.
    
    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to WARNING.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    # Get statek logger
    logger = logging.getLogger('statek')
    logger.setLevel(numeric_level)

    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        # Create console handler with formatting
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    # Prevent propagation to root logger to keep it separate
    logger.propagate = False


@lru_cache()
def get_statek_logger() -> logging.Logger:
    """Get the cached statek logger instance.

    Returns:
        logging.Logger: The statek logger instance.
    """
    return logging.getLogger('statek')


def statek_log(message: str, level: str = 'info') -> None:
    """Log a message with separator lines using STATEK_LOGGER.
    
    Args:
        message: The message to log
        level: The log level (info, debug, warning, error, critical). Defaults to 'info'.
    """
    logger = get_statek_logger()
    log_func = getattr(logger, level.lower(), logger.info)
    formatted_message = f"{'-'*40}\n{message}\n{'-'*40}"
    log_func(formatted_message)
