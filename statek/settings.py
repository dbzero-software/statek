"""Settings module for Statek - LLM API configuration management."""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from statek.chat_style import ChatStyle  # noqa: F401  # re-exported for backward compatibility
from statek.prompt_config import PromptDef, load_prompt_files
from statek.docstring import ACL_Item, Statek_ACL


class LLM_API_Settings(BaseSettings):
    """Settings for a specific LLM provider (e.g. OpenAI, OpenRouter, Google).

    Attributes:
        api_url: The base URL for the LLM API
        api_key: The API key for authentication
        default_model: Optional default chat model name (e.g. gpt-4, gpt-3.5-turbo)
        response_format_file: Optional path to a JSON file with custom response_format schema
        use_prompt_caching: Whether to enable prompt caching (Claude-specific)
    """
    api_url: str
    api_key: str
    default_model: Optional[str] = None
    response_format_file: Optional[str] = None
    use_prompt_caching: bool = False

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
    - STATEK_PROMPT_FILES_DIR (optional, for prompt definition files)

    Attributes:
        llm_api_settings: Dictionary of LLM_API_Settings by provider name
        default_llm_api_provider: The default provider to use
        prompt_files_dir: Location for prompt .md files (uses current dir if not set)
        prompt_defs: Dictionary of PromptDef parsed from prompt files
    """
    llm_api_settings: Dict[str, LLM_API_Settings] = Field(default_factory=dict)
    default_llm_api_provider: str = "OPENROUTER"
    prompt_files_dir: Optional[str] = None
    prompt_defs: Dict[str, PromptDef] = Field(default_factory=dict)
    logs_path: Optional[str] = None
    examples_dir: Optional[str] = None
    documents_dir: Optional[str] = None
    """The maximum allowed number of LLM turns per conversation"""
    max_turns: int = 5
    """The maximum allowed total number of exceptions per conversation"""
    max_exceptions: int = 3
    """Maximum number of consecutive exceptions per conversation"""
    max_consecutive_exceptions: int = 1
    """Maximum allowed total number of tokens per conversation"""
    max_token_usage: int = 10000
    chat_style: Optional[ChatStyle]= None  # pylint: disable=no-member
    """Optional style for formatting examples (overrides chat_style when set)"""
    examples_style: Optional[ChatStyle] = None  # pylint: disable=no-member
    """The boxing XML tag for console outputs"""
    xml_box_console: Optional[str] = None
    """The boxing XML tag for code examples"""
    xml_box_example: Optional[str] = None
    """Log level for statek logger (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
    log_level: str = "WARNING"
    """The default ACL mode string: GRANT or DENY (loaded from STATEK_DEFAULT_ACL)"""
    default_acl_str: str = "DENY"
    """Host for the STATEK RPC server"""
    statek_rpc_host: str = "0.0.0.0"
    """Port for the STATEK RPC server"""
    statek_rpc_port: int = 8300

    model_config = SettingsConfigDict(extra='ignore')

    def __init__(self, **data):  # pylint: disable=too-many-branches
        """Initialize StatekSettings by parsing environment variables.

        Automatically detects provider-prefixed environment variables and
        creates LLM_API_Settings instances for each provider.
        Also parses prompt definition files from the configured directory.
        """
        super().__init__(**data)

        # Parse environment variables to build llm_api_settings dictionary
        if not self.llm_api_settings:
            self.llm_api_settings = self._parse_llm_providers_from_env()

        if self.prompt_files_dir is None:
            self.prompt_files_dir = os.environ.get('STATEK_PROMPT_FILES_DIR')

        if self.logs_path is None:
            self.logs_path = os.environ.get('STATEK_LOGS_PATH')

        if self.examples_dir is None:
            self.examples_dir = os.environ.get('STATEK_EXAMPLES_DIR')

        if self.documents_dir is None:
            self.documents_dir = os.environ.get('STATEK_DOCUMENTS_DIR')

        # Parse STATEK_ prefixed env vars for harness settings
        for attr, env_var in [
            ('max_turns', 'STATEK_MAX_TURNS'),
            ('max_exceptions', 'STATEK_MAX_EXCEPTIONS'),
            ('max_consecutive_exceptions', 'STATEK_MAX_CONSECUTIVE_EXCEPTIONS'),
            ('max_token_usage', 'STATEK_MAX_TOKEN_USAGE'),
        ]:
            env_val = os.environ.get(env_var)
            if env_val is not None and attr not in data:
                setattr(self, attr, int(env_val))

        env_val = os.environ.get('STATEK_CHAT_STYLE')
        if env_val is not None:
            self.chat_style = ChatStyle[env_val.upper()]

        env_val = os.environ.get('STATEK_EXAMPLES_STYLE')
        if env_val is not None:
            self.examples_style = getattr(ChatStyle, env_val.upper())

        if self.xml_box_console is None:
            self.xml_box_console = os.environ.get('STATEK_XML_BOX_CONSOLE')

        if self.xml_box_example is None:
            self.xml_box_example = os.environ.get('STATEK_XML_BOX_EXAMPLE')

        env_val = os.environ.get('STATEK_LOG_LEVEL')
        if env_val is not None:
            self.log_level = env_val.upper()
        set_log_level(self.log_level)

        env_val = os.environ.get('STATEK_DEFAULT_ACL')
        if env_val is not None and 'default_acl_str' not in data:
            self.default_acl_str = env_val.upper()

        env_val = os.environ.get('STATEK_RPC_HOST')
        if env_val is not None and 'statek_rpc_host' not in data:
            self.statek_rpc_host = env_val

        env_val = os.environ.get('STATEK_RPC_PORT')
        if env_val is not None and 'statek_rpc_port' not in data:
            self.statek_rpc_port = int(env_val)

        if not self.prompt_defs:
            self.prompt_defs = (
                load_prompt_files(self.prompt_files_dir)
                if self.prompt_files_dir else {}
            )

    @staticmethod
    def _parse_llm_providers_from_env() -> Dict[str, LLM_API_Settings]:  # pylint: disable=too-many-branches
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
            elif '_RESPONSE_FORMAT_FILE' in key:
                provider = key.replace('_RESPONSE_FORMAT_FILE', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['response_format_file'] = value
            elif '_DEFAULT_MODEL' in key:
                provider = key.replace('_DEFAULT_MODEL', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['default_model'] = value
            elif '_USE_PROMPT_CACHING' in key:
                provider = key.replace('_USE_PROMPT_CACHING', '')
                if provider not in providers:
                    providers[provider] = {}
                providers[provider]['use_prompt_caching'] = value.lower() in ('true', '1', 'yes')

        # Create LLM_API_Settings instances for each provider
        llm_settings = {}
        for provider_name, settings_dict in providers.items():
            # Only create settings if we have both required fields
            if 'api_url' in settings_dict and 'api_key' in settings_dict:
                llm_settings[provider_name] = LLM_API_Settings(**settings_dict)

        return llm_settings



    def get_prompt_def(self, name: str) -> Optional[PromptDef]:
        """Get a prompt definition by name.

        Args:
            name: The prompt name (typically the agent role name)

        Returns:
            PromptDef if found, None otherwise
        """
        return self.prompt_defs.get(name)

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

    def get_xml_box_tags(self) -> Dict[str, str]:
        """For configured tags retrieves tag name -> tag box mapping.

        Returns a dict mapping logical names to configured XML tag names.
        Only includes entries for tags that have been configured (non-None).
        Logical names: 'console' for xml_box_console, 'example' for xml_box_example.
        """
        tags = {}
        if self.xml_box_console:
            tags["console"] = self.xml_box_console
        if self.xml_box_example:
            tags["example"] = self.xml_box_example
        return tags

    @property
    def default_acl(self) -> Statek_ACL:
        """The default ACL: GRANT grants all implicit access, DENY denies it.

        Returns:
            Statek_ACL with a single wildcard rule matching the configured mode.
        """
        access = self.default_acl_str.upper() == "GRANT"
        return Statek_ACL(acl=[ACL_Item(access=access, name="", is_prefix=True, scope=[])])



@lru_cache()
def get_provider_settings(provider: Optional[str] = None) -> Optional[LLM_API_Settings]:
    """Get LLM_API_Settings for a specific provider or the default provider."""
    settings = StatekSettings()
    return settings.get_provider_settings(provider)

@lru_cache()
def get_statek_settings() -> StatekSettings:
    """Get the cached StatekSettings instance."""
    return StatekSettings()


def get_prompt_def(name: str) -> Optional[PromptDef]:
    """Get a prompt definition by name from the cached settings.

    Args:
        name: The prompt name (typically the agent role name)

    Returns:
        PromptDef if found, None otherwise
    """
    settings = get_statek_settings()
    return settings.get_prompt_def(name)


def set_log_level(log_level: str = "WARNING") -> None:
    """Set the log level for STATEK routines.

    Enables or disables logs from run_agentic_loop and other STATEK routines.

    Args:
        log_level: One of NOTSET, DEBUG, INFO, WARNING, ERROR or CRITICAL.
                   Defaults to WARNING.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.WARNING)

    logger = logging.getLogger('statek')
    logger.setLevel(numeric_level)

    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Update existing handler level
        for handler in logger.handlers:
            handler.setLevel(numeric_level)

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
