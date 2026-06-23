"""statek package."""

from typing import Optional

from .settings import LLM_API_Settings, StatekSettings, get_statek_settings
from .multi_source_settings import (
    AwsSecretsManagerSource,
    MultiSourceBaseSettings,
    SettingValuesSource,
)
from .prompt_config import PromptDef, update_prompt_config
from .llm_api import (
    LLM_API, OpenRouter_API, OpenAI_API, VertexAI_API, ClaudeAI_API,
    Claude_API, LLM_Response, LLM_Stats,
)
from .exceptions import LLM_HarnessError
from .llm_harness import LLM_Harness, get_llm_harness
from .system import (tool, subtask, docstr, get_any, get_all, error_handler,
                     is_valid_error_handler, docs_style, find_sub_task_handler)
from .shared_context import init_shared_context, print_locals, shared_context_set_var
from .utils import (statek_print, format_default_llm_repr,
                    get_current_agent, get_current_agent_name, get_current_job)
from .task import (
    SubTaskHandler,
    SubTaskState,
    TaskError,
    complete_sub_task,
    create_new_job,
    create_sub_task,
)
from . import task

__version__ = "0.1.0"


def init(settings: Optional[StatekSettings] = None) -> None:
    """Initialize statek before first use.

    Loads model pricing from statek_model_info_dir when configured.
    """
    if settings is None:
        settings = get_statek_settings()
    if settings.statek_model_info_dir:
        from .model_pricing import init_model_pricing  # pylint: disable=import-outside-toplevel
        init_model_pricing(settings.statek_model_info_dir)


__all__ = [
    "init",
    "LLM_API_Settings",
    "StatekSettings",
    "AwsSecretsManagerSource",
    "MultiSourceBaseSettings",
    "SettingValuesSource",
    "PromptDef",
    "update_prompt_config",
    "LLM_API",
    "OpenRouter_API",
    "OpenAI_API",
    "VertexAI_API",
    "ClaudeAI_API",
    "Claude_API",
    "LLM_Response",
    "LLM_Stats",
    "LLM_Harness",
    "LLM_HarnessError",
    "get_llm_harness",
    "subtask",
    "SubTaskHandler",
    "SubTaskState",
    "TaskError",
    "complete_sub_task",
    "create_new_job",
    "create_sub_task",
    "find_sub_task_handler",
    "init_shared_context",
    "print_locals",
    "shared_context_set_var",
    "statek_print",
    "format_default_llm_repr",
    "get_current_agent",
    "get_current_agent_name",
    "get_current_job",
]
