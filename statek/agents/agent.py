from dataclasses import dataclass
from typing import List, Callable, Dict, Optional
import dbzero as db0
from statek.utils import format_callable_decl
from statek.executors.job import JobDef

@db0.memo
@dataclass
class Agent:
    """
        This is the fundamental class to hold the workflow specification and available tools.
    """
    role: str  # An arbitrary role name
    _system_prompt: str  # f-string with the {tools} placeholder
    _prompt_template: str  # Agent's prompt template / to be formatted with job-specific params
    _tools: List[Callable]
    _X__context: Optional[Dict] = None  # Agent's specific context (e.g. with private tools)

    def __post_init__(self):
        """
        Apply prompt configuration from StatekSettings after initialization.

        If a prompt definition exists for this agent's role, it will override
        the _system_prompt and _prompt_template values.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from statek.prompt_config import load_prompt_files
        from statek.settings import get_statek_settings

        settings = get_statek_settings()

        # Load prompt defs if they haven't been loaded yet
        if not settings.prompt_defs and settings.prompt_files_dir:
            settings.prompt_defs = load_prompt_files(settings.prompt_files_dir)

        prompt_def = settings.get_prompt_def(self.role)

        if prompt_def is not None:
            if prompt_def.system:
                self._system_prompt = prompt_def.system
            if prompt_def.template:
                self._prompt_template = prompt_def.template

    @property
    def system_prompt(self) -> str:
        """
        Format system_prompt with tool descriptions.

        Places all available tool descriptions in the placeholder using
        newlines and > character to start each line.
        """
        tools_str = "\n".join(">" + format_callable_decl(tool) for tool in self._tools)
        return self._system_prompt.format(tools=tools_str)

    @property
    def context(self) -> Optional[Dict]:
        """
        Get agent's private context.
        """
        return self._X__context

    def prompt(self, job_params: Dict = None, **kwargs) -> str:
        """
        Format prompt message from prompt template by filling in job specific parameters.

        Args:
            job_params: optional context for format (e.g. job local variables)
            kwargs: optional additional params

        Returns:
            Formatted prompt string
        """
        format_ctx = {}
        if job_params:
            format_ctx.update(job_params)
        if kwargs:
            format_ctx.update(kwargs)

        if format_ctx:
            return self._prompt_template.format_map(format_ctx)
        return self._prompt_template

@db0.memo
class SupervisedAgent(Agent):
    """
    Base class for implementing agents initiated and supervised by other agents or system functions
    """

    def create_job_def(
        self,
        tools: Optional[List[Callable]] = None,
        warmup_code: str = None,
        **kwargs
    ) -> JobDef:
        # pylint: disable=unused-argument
        """
        Create a job definition with job-specific parameters.

        Args:
            tools: agent's tools additional tools (currently not used)
            warmup_code: optional initialization code
            kwargs: job specific parameters for prompt formatting (i.e. job_params)

        Returns:
            A new job definition object with specific job_params
        """
        # kwargs become job_params
        job_params = kwargs if kwargs else None

        return JobDef(
            agent=self,
            job_params=job_params,
            warmup_code=warmup_code
        )
