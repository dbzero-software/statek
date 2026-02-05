"""Settings module for Statek - LLM API configuration management."""

import os
import logging
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Iterable
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Named tuple for prompt definitions parsed from .md files
PromptDef = namedtuple("PromptDef", ["system", "template"])


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

    model_config = SettingsConfigDict(extra='ignore')

    def __init__(self, **data):
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

        if not self.prompt_defs:
            self.prompt_defs = self._parse_prompt_files()

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

    def _parse_prompt_files(self) -> Dict[str, PromptDef]:
        """Parse prompt definition files from the configured directory.

        Prompt files are .md files with two top-level sections:
        - # System - the system prompt
        - # Template - the prompt template

        Sections can be separated by `---` which will not be included in the prompt.

        Returns:
            Dictionary mapping prompt names (filename without extension) to PromptDef
        """
        prompt_defs = {}

        if self.prompt_files_dir is None:
            return prompt_defs

        prompt_dir = Path(self.prompt_files_dir)
        if not prompt_dir.exists() or not prompt_dir.is_dir():
            return prompt_defs

        for md_file in prompt_dir.glob("*.md"):
            try:
                prompt_def = self._parse_prompt_file(md_file)
                if prompt_def:
                    # Use filename without extension as the prompt name
                    prompt_name = md_file.stem
                    prompt_defs[prompt_name] = prompt_def
            except Exception:  # pylint: disable=broad-exception-caught
                # Skip files that fail to parse
                continue

        return prompt_defs

    @staticmethod
    def _parse_prompt_file(file_path: Path) -> Optional[PromptDef]:
        """Parse a single prompt definition file.

        Args:
            file_path: Path to the .md file

        Returns:
            PromptDef with system and template, or None if parsing fails
        """
        content = file_path.read_text(encoding='utf-8')
        
        current_section = None
        sections = {'system': [], 'template': []}
        
        for line in content.split('\n'):
            stripped = line.strip().lower()
            
            if stripped == '# system':
                current_section = 'system'
            elif stripped == '# template':
                current_section = 'template'
            elif stripped == '---':
                continue
            elif current_section:
                sections[current_section].append(line)
        
        system_prompt = '\n'.join(sections['system']).strip() if sections['system'] else None
        template = '\n'.join(sections['template']).strip() if sections['template'] else None
        
        if system_prompt:
            return PromptDef(system=system_prompt, template=template or "")
        
        return None

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


def update_prompt_config(prompt_defs: Dict[str, PromptDef], 
                         agents: 'Iterable[Agent]' = None) -> None:
    """Apply updated prompt definitions to Agent instances.
    
    This function updates the system prompt and prompt template of agents
    based on the provided prompt definitions. The agent's role name is used
    to look up the corresponding prompt definition.
    
    Args:
        prompt_defs: Dictionary mapping prompt names to PromptDef instances
                    (e.g. from StatekSettings.prompt_defs)
        agents: Optional iterable of Agent instances to update.
               If None, looks up all agents using db0.find(Agent).
    
    Note:
        If a prompt definition is not found for a specific agent's role,
        that agent is quietly skipped (no error raised).
    """
    import dbzero as db0  # pylint: disable=import-outside-toplevel
    from statek.agents.agent import Agent  # pylint: disable=import-outside-toplevel
    
    # If agents not provided, find all agents in db0
    if agents is None:
        agents = db0.find(Agent)
    
    for agent in agents:
        # Look up prompt definition by agent's role name
        prompt_def = prompt_defs.get(agent.role)
        
        if prompt_def is None:
            # Quietly skip agents without matching prompt definitions
            continue
        
        # Update agent's system prompt and template
        if prompt_def.system:
            agent._system_prompt = prompt_def.system
        if prompt_def.template:
            agent._prompt_template = prompt_def.template
