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

"""Settings module for Statek - LLM API configuration management."""

import os
import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from statek.chat_style import ChatStyle  # noqa: F401  # re-exported for backward compatibility
from statek.multi_source_settings import (
    MultiSourceBaseSettings,
)
from statek.prompt_config import PromptDef, load_prompt_files
from statek.docstring import ACL_Item, Statek_ACL


class LLM_API_Settings(BaseSettings):
    """Settings for a specific LLM provider (e.g. OpenAI, OpenRouter, Google).

    Attributes:
        api_url: The base URL for the LLM API
        api_key: The API key for authentication
        response_format_file: Optional path to a JSON file with custom response_format schema
        use_prompt_caching: Whether to enable prompt caching (Claude-specific)
    """
    api_url: str
    api_key: str
    response_format_file: Optional[str] = None
    use_prompt_caching: bool = False

    model_config = SettingsConfigDict(extra='ignore')


class StatekSettings(MultiSourceBaseSettings):
    """Main settings class for Statek, aggregating LLM API settings by provider.

    Environment variables should be prefixed with the provider name:
    - OPENROUTER_API_URL
    - OPENROUTER_API_KEY
    - OPENAI_API_URL
    - OPENAI_API_KEY
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
    """Fraction by which limits are extended after each job completion"""
    limit_extension_per_completion: float = 0.0
    chat_style: Optional[ChatStyle]= None  # pylint: disable=no-member
    """Optional style for formatting examples (overrides chat_style when set)"""
    examples_style: Optional[ChatStyle] = None  # pylint: disable=no-member
    """The boxing XML tag for console outputs"""
    xml_box_console: Optional[str] = None
    """The boxing XML tag for code examples"""
    xml_box_example: Optional[str] = None
    """Log level for statek logger (INFO, WARNING, ERROR, CRITICAL)"""
    log_level: str = "ERROR"
    """The default ACL mode string: GRANT or DENY (loaded from STATEK_DEFAULT_ACL)"""
    default_acl_str: str = "DENY"
    """Host for the STATEK RPC server"""
    statek_rpc_host: Optional[str] = None
    """Port for the STATEK RPC server"""
    statek_rpc_port: Optional[int] = None
    """Default task difficulty used when neither example nor job metadata specifies one."""
    statek_default_difficulty: str = "M"
    """Directory containing model pricing files (.csv/.txt); scanned recursively on init."""
    statek_model_info_dir: Optional[str] = None
    """Path to a UTF-8 JSON file containing provider-specific request configuration."""
    statek_provider_config: Optional[str] = None
    """Directory containing agent warmup definition .py files."""
    warmup_defs_dir: Optional[str] = None
    """Python sandbox mode: restricted or off."""
    python_sandbox_mode: str = "restricted"
    """Maximum number of UTF-8 bytes accepted for an executed Python snippet."""
    python_sandbox_max_source_bytes: int = 200_000
    """Maximum AST nodes accepted for an executed Python snippet."""
    python_sandbox_max_ast_nodes: int = 20_000
    """Comma-separated import roots allowed in sandboxed Python."""
    python_sandbox_allowed_imports: str = (
        "datetime,calendar,time,re,math,decimal,fractions,statistics,collections,"
        "itertools,functools,operator,json"
    )
    """Comma-separated additional hidden/internal tool names allowed in sandboxed Python."""
    python_sandbox_allowed_tools: str = ""

    model_config = SettingsConfigDict(extra='ignore')

    def __init__(self, **data):  # pylint: disable=too-many-branches,too-many-statements
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

        if self.statek_model_info_dir is None:
            self.statek_model_info_dir = os.environ.get('STATEK_MODEL_INFO_DIR')

        if self.warmup_defs_dir is None:
            self.warmup_defs_dir = os.environ.get('STATEK_WARMUP_DEFS_DIR')

        env_val = os.environ.get('STATEK_PYTHON_SANDBOX_MODE')
        if env_val is not None and 'python_sandbox_mode' not in data:
            self.python_sandbox_mode = env_val.lower()

        # Parse STATEK_ prefixed env vars for harness settings
        for attr, env_var, conv in [
            ('max_turns', 'STATEK_MAX_TURNS', int),
            ('max_exceptions', 'STATEK_MAX_EXCEPTIONS', int),
            ('max_consecutive_exceptions', 'STATEK_MAX_CONSECUTIVE_EXCEPTIONS', int),
            ('max_token_usage', 'STATEK_MAX_TOKEN_USAGE', int),
            ('limit_extension_per_completion', 'STATEK_LIMIT_EXTENSION_PER_COMPLETION', float),
            ('python_sandbox_max_source_bytes', 'STATEK_PYTHON_SANDBOX_MAX_SOURCE_BYTES', int),
            ('python_sandbox_max_ast_nodes', 'STATEK_PYTHON_SANDBOX_MAX_AST_NODES', int),
        ]:
            env_val = os.environ.get(env_var)
            if env_val is not None and attr not in data:
                setattr(self, attr, conv(env_val))

        env_val = os.environ.get('STATEK_PYTHON_SANDBOX_ALLOWED_IMPORTS')
        if env_val is not None and 'python_sandbox_allowed_imports' not in data:
            self.python_sandbox_allowed_imports = env_val

        env_val = os.environ.get('STATEK_PYTHON_SANDBOX_ALLOWED_TOOLS')
        if env_val is not None and 'python_sandbox_allowed_tools' not in data:
            self.python_sandbox_allowed_tools = env_val

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

        env_val = os.environ.get('STATEK_DEFAULT_DIFFICULTY')
        if env_val is not None and 'statek_default_difficulty' not in data:
            self.statek_default_difficulty = env_val

        if not self.prompt_defs:
            self.prompt_defs = (
                load_prompt_files(self.prompt_files_dir)
                if self.prompt_files_dir else {}
            )

    def _get_value_from_sources_or_env(self, env_var: str):
        value = self.get_value_from_sources(env_var)
        if value is not None:
            return value

        return os.environ.get(env_var)

    def _parse_llm_providers_from_env(self) -> Dict[str, LLM_API_Settings]:
        """Parse environment variables to extract LLM provider settings.

        Looks for environment variables with the pattern:
        {PROVIDER}_API_URL, {PROVIDER}_API_KEY

        Returns:
            Dictionary mapping provider names to their LLM_API_Settings
        """
        providers: Dict[str, Dict[str, Any]] = {}

        api_url_suffix = '_API_URL'
        for key in os.environ:
            if not key.endswith(api_url_suffix):
                continue

            provider = key[:-len(api_url_suffix)]
            settings_dict: Dict[str, Any] = {}

            for field_name, env_suffix in [
                ('api_url', '_API_URL'),
                ('api_key', '_API_KEY'),
                ('response_format_file', '_RESPONSE_FORMAT_FILE'),
                ('use_prompt_caching', '_USE_PROMPT_CACHING'),
            ]:
                value = self._get_value_from_sources_or_env(f'{provider}{env_suffix}')
                if value is None:
                    continue

                if field_name == 'use_prompt_caching':
                    settings_dict[field_name] = str(value).lower() in ('true', '1', 'yes')
                else:
                    settings_dict[field_name] = value

            providers[provider] = settings_dict

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



_ACTIVE_STATEK_SETTINGS: Optional[StatekSettings] = None


def set_statek_settings(settings: Optional[StatekSettings]) -> None:
    """Set the process-active Statek settings instance."""
    global _ACTIVE_STATEK_SETTINGS  # pylint: disable=global-statement
    _ACTIVE_STATEK_SETTINGS = settings
    get_statek_settings.cache_clear()
    from statek.python_sandbox import reset_sandbox_policy  # pylint: disable=import-outside-toplevel

    reset_sandbox_policy()


@lru_cache()
def get_provider_settings(provider: Optional[str] = None) -> Optional[LLM_API_Settings]:
    """Get LLM_API_Settings for a specific provider or the default provider."""
    settings = StatekSettings()
    return settings.get_provider_settings(provider)

@lru_cache()
def get_statek_settings() -> StatekSettings:
    """Get the cached StatekSettings instance."""
    if _ACTIVE_STATEK_SETTINGS is not None:
        return _ACTIVE_STATEK_SETTINGS
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


def set_log_level(log_level: str = "ERROR") -> None:
    """Set the log level for STATEK routines.

    Enables or disables logs from run_agentic_loop and other STATEK routines.

    Args:
        log_level: One of INFO, WARNING, ERROR or CRITICAL.
                   Defaults to ERROR.
    """
    supported_levels = {
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level = supported_levels.get(log_level.upper(), logging.ERROR)

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
        level: The log level (info, warning, error, critical). Defaults to 'info'.
    """
    logger = get_statek_logger()
    log_func = getattr(logger, level.lower(), logger.info)
    formatted_message = f"{'-'*40}\n{message}\n{'-'*40}"
    log_func(formatted_message)


_TRACE_SECRET_KEYS = {"api_key", "authorization", "cookie", "password", "secret", "token"}
_FULL_AGENT_TRACE_LOGGER_NAME = "statek.full_agent_trace"
_DEFAULT_FULL_AGENT_TRACE_PATH = "full_agent_trace.jsonl"
_DEFAULT_FULL_AGENT_TRACE_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_FULL_AGENT_TRACE_BACKUP_COUNT = 3


def _is_trace_secret_key(key: object) -> bool:
    """Return whether a trace field name is likely to hold credentials."""
    normalized_key = str(key).casefold()
    return any(secret_key in normalized_key for secret_key in _TRACE_SECRET_KEYS)


def _redact_trace_data(value: Any) -> Any:
    """Return trace data with credential-bearing fields replaced by a redaction marker."""
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if _is_trace_secret_key(key)
            else _redact_trace_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_trace_data(item) for item in value]
    return value


def _get_full_agent_trace_logger() -> logging.Logger:
    """Return the dedicated rotating JSONL logger for full agent traces."""
    trace_path = Path(os.getenv("FULL_AGENT_TRACE_PATH", _DEFAULT_FULL_AGENT_TRACE_PATH)).resolve()
    max_bytes = int(
        os.getenv("FULL_AGENT_TRACE_MAX_BYTES", str(_DEFAULT_FULL_AGENT_TRACE_MAX_BYTES))
    )
    backup_count = int(
        os.getenv("FULL_AGENT_TRACE_BACKUP_COUNT", str(_DEFAULT_FULL_AGENT_TRACE_BACKUP_COUNT))
    )
    logger = logging.getLogger(_FULL_AGENT_TRACE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    matching_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == trace_path
        ),
        None,
    )
    if matching_handler is not None:
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        trace_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def full_agent_trace(event: str, details: Dict[str, Any]) -> None:
    """Emit opt-in structured agent diagnostics without credentials.

    Full payload tracing is intentionally disabled unless ``FULL_AGENT_TRACE`` is
    set to ``true``. Trace output can contain user content and must only be enabled
    while diagnosing a local issue.
    """
    if os.getenv("FULL_AGENT_TRACE", "").casefold() != "true":
        return
    message = json.dumps(
        {"event": event, "details": _redact_trace_data(details)},
        default=str,
        ensure_ascii=False,
    )
    _get_full_agent_trace_logger().info(message)
