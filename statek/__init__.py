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

"""statek package."""

from typing import Optional

try:
    import dbzero  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "statek requires dbzero. Install it with `pip install statek[dbzero]` "
        "or install dbzero-pro with `pip install statek[dbzero-pro]`."
    ) from exc

from .settings import (
    LLM_API_Settings,
    StatekSettings,
    get_statek_settings,
    set_statek_settings,
)
from .multi_source_settings import (
    MultiSourceBaseSettings,
    SettingValuesSource,
)
from .prompt_config import PromptDef, update_prompt_config
from .llm_api import (
    LLM_API, OpenRouter_API, OpenAI_API, Groq_API, MistralAI_API,
    DeepSeek_API, XAI_API, TogetherAI_API, FireworksAI_API, Cerebras_API,
    Perplexity_API, SambaNova_API, NvidiaNIM_API, Nebius_API, Cohere_API,
    MoonshotAI_API, DashScope_API, CloudflareWorkersAI_API,
    CloudflareAIGateway_API, GitHubModels_API, Bedrock_API,
    MicrosoftFoundry_API, AzureOpenAI_API, GeminiEnterprise_API, Ollama_API,
    LMStudio_API, VLLM_API, SGLang_API, LlamaCpp_API, VertexAI_API,
    ClaudeAI_API, Claude_API, LLM_Response, LLM_Stats, add_provider,
)
from .exceptions import LLM_HarnessError
from .llm_harness import LLM_Harness, get_llm_harness
from .system import (tool, subtask, docstr, get_any, get_all, error_handler,
                     is_valid_error_handler, docs_style, find_sub_task_handler,
                     find_tools)
from .shared_context import (
    ContextCategory,
    ContextCategoryDict,
    ContextVar,
    init_shared_context,
    print_locals,
    shared_context_set_var,
)
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
from .runner import start_statek, start_statek_async
from . import task
from .dbzero_restricted import (
    DbzeroRestrictedModeError,
    dbzero_restricted_enabled,
    open_prefix,
    validate_current_prefix_restricted,
    validate_dbzero_restricted,
)

__version__ = "0.1.0"


def init(settings: Optional[StatekSettings] = None, restricted: bool = True) -> None:
    """Initialize statek before first use.

    Loads model pricing from statek_model_info_dir when configured.
    """
    if settings is None:
        settings = get_statek_settings()
    if not restricted:
        settings.python_sandbox_mode = "off"
    elif settings.python_sandbox_mode.lower() != "off":
        settings.python_sandbox_mode = "restricted"
    set_statek_settings(settings)
    validate_dbzero_restricted(settings)
    from .python_sandbox import configure_sandbox  # pylint: disable=import-outside-toplevel

    configure_sandbox(
        settings,
        blocked_tools={
            fn.__name__
            for fn in find_tools(None, include_hidden=True)
            if getattr(fn, "tool_hidden", False)
        },
    )
    if settings.statek_model_info_dir:
        from .model_pricing import init_model_pricing  # pylint: disable=import-outside-toplevel
        init_model_pricing(settings.statek_model_info_dir)


__all__ = [
    "init",
    "start_statek",
    "start_statek_async",
    "DbzeroRestrictedModeError",
    "dbzero_restricted_enabled",
    "open_prefix",
    "validate_current_prefix_restricted",
    "validate_dbzero_restricted",
    "LLM_API_Settings",
    "StatekSettings",
    "MultiSourceBaseSettings",
    "SettingValuesSource",
    "PromptDef",
    "update_prompt_config",
    "LLM_API",
    "OpenRouter_API",
    "OpenAI_API",
    "Groq_API",
    "MistralAI_API",
    "DeepSeek_API",
    "XAI_API",
    "TogetherAI_API",
    "FireworksAI_API",
    "Cerebras_API",
    "Perplexity_API",
    "SambaNova_API",
    "NvidiaNIM_API",
    "Nebius_API",
    "Cohere_API",
    "MoonshotAI_API",
    "DashScope_API",
    "CloudflareWorkersAI_API",
    "CloudflareAIGateway_API",
    "GitHubModels_API",
    "Bedrock_API",
    "MicrosoftFoundry_API",
    "AzureOpenAI_API",
    "GeminiEnterprise_API",
    "Ollama_API",
    "LMStudio_API",
    "VLLM_API",
    "SGLang_API",
    "LlamaCpp_API",
    "VertexAI_API",
    "ClaudeAI_API",
    "Claude_API",
    "LLM_Response",
    "LLM_Stats",
    "add_provider",
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
    "ContextCategory",
    "ContextCategoryDict",
    "ContextVar",
    "init_shared_context",
    "print_locals",
    "shared_context_set_var",
    "statek_print",
    "format_default_llm_repr",
    "get_current_agent",
    "get_current_agent_name",
    "get_current_job",
]
