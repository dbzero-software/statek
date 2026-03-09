from dataclasses import dataclass
import re
from typing import Iterable, List, Callable, Dict, Optional, Sequence, Union
import dbzero as db0
from statek.utils import block_comment, find_locals, _get_class_name
from statek.system import tool
from statek.docstring import parse_tool_docstring, format_docstring
from statek.executors.job import JobDef, parse_warmup_code
from statek.settings import get_statek_logger

STATEK_LOGGER = get_statek_logger()


@tool(system=True)
def list_of_examples(start_index: int = 0, limit: int = 10, **kwargs):  # pylint: disable=unused-argument
    """Lists available examples for this agent.

    Results are printed as a numbered list (index: name).

    Args:
        start_index: Index of the first example to show (default: 0).
        limit: Maximum number of examples to show (default: 10).
    """
    from statek.agents.list_of_examples import list_of_examples as _impl  # pylint: disable=import-outside-toplevel
    agent_name = next(iter(find_locals(var_name="agent_name")), None)
    _impl(agent_name, start_index, limit)


@tool(system=True)
def show_example(example_id: int, **kwargs):  # pylint: disable=unused-argument
    """Shows the content of a specific example by its index.

    Args:
        example_id: Index of the example to show (as listed by list_of_examples).
    """
    from statek.agents.list_of_examples import show_example as _impl  # pylint: disable=import-outside-toplevel
    agent_name = next(iter(find_locals(var_name="agent_name")), None)
    _impl(agent_name, example_id)


@db0.memo
@dataclass
class Agent:
    """
        This is the fundamental class to hold the workflow specification and available tools.
    """
    role: str  # An arbitrary role name
    _system_prompt: str  # f-string with the {tools} placeholder
    _tools: List[Callable]
    # NOTE: dynamically created tools are stored by their name
    _tools_by_name: Optional[List[str]] = None
    _metadata: Optional[Dict[str, str]] = None  # prompt meta-data as key/value pairs
    _X__context: Optional[Dict] = None  # Agent's specific context (e.g. with private tools)

    def __post_init__(self):
        """
        Apply prompt configuration from StatekSettings after initialization.

        If a prompt definition exists for this agent's role, it will override
        the _system_prompt and _metadata values.
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
            if prompt_def.metadata:
                self.update_metadata(prompt_def.metadata)
        self.append_tool(list_of_examples)
        self.append_tool(show_example)


    def update_system_prompt(self, new_prompt: str) -> bool:
        """Update _system_prompt only if it differs from the current value.

        Args:
            new_prompt: New system prompt string to apply.

        Returns:
            True if the prompt was updated, False if it was already up to date.
        """
        if self._system_prompt == new_prompt:
            return False
        self._system_prompt = new_prompt
        STATEK_LOGGER.debug("Agent '%s' system prompt updated", self.role)
        return True

    def update_metadata(self, new_metadata: Dict[str, str]) -> bool:
        """Update _metadata only if keys or values differ from the current state.

        Args:
            new_metadata: Metadata dict to apply.

        Returns:
            True if metadata was updated, False if it was already up to date.
        """
        if self._metadata == new_metadata:
            return False
        self._metadata = new_metadata
        STATEK_LOGGER.debug("Agent '%s' metadata updated: %s", self.role, self._metadata)
        return True

    def _expand_tool_placeholders(self, text: str) -> Optional[str]:
        """Expand {tools}, {brief_tools}, {detailed_tools} placeholders in text.

        Lines starting with '#' before a placeholder are embedded as a block comment.
        """
        if text is None:
            return None
        placeholders = [
            ('tools', True, False),
            ('brief_tools', True, False),
            ('detailed_tools', False, True),
        ]
        for name, brief, py_syntax in placeholders:
            pattern = re.compile(rf'^(\s*#\s*)\{{{name}\}}', re.MULTILINE)
            if pattern.search(text):
                tools_str = self._format_tools(brief, py_syntax)
                text = pattern.sub(block_comment(tools_str), text)
            elif f'{{{name}}}' in text:
                tools_str = self._format_tools(brief, py_syntax)
                text = text.replace(f'{{{name}}}', tools_str)
        return text

    def system_prompt(self, job_params: Dict = None, **kwargs) -> str:
        """
        Format system_prompt with tool descriptions and job-specific parameters.

        Supports tool placeholders: {tools}, {brief_tools}, {detailed_tools}
        - tools/brief_tools: formatted with brief=True, py_syntax=False
        - detailed_tools: formatted with brief=False, py_syntax=True

        If the placeholder line starts with '#', the result is embedded in a block comment.

        Additional {key} placeholders are resolved via job_params and kwargs using
        format_map, allowing job-specific values to be injected into the system prompt.

        Args:
            job_params: optional context for format (e.g. job local variables)
            kwargs: optional additional params

        Returns:
            Formatted system prompt string
        """
        if self._system_prompt is None:
            return ""
        result = self._expand_tool_placeholders(self._system_prompt)
        format_ctx = {}
        if job_params:
            format_ctx.update(job_params)
        if kwargs:
            format_ctx.update(kwargs)
        if format_ctx:
            return result.format_map(format_ctx)
        return result

    def _format_tools(self, brief: bool, py_syntax: bool) -> str:
        """Format all tools with the specified settings."""
        agent_name = _get_class_name(self)
        formatted = []
        def inner_format_tool(fn: Callable) -> str:
            parsed = parse_tool_docstring(fn)
            return format_docstring(parsed, brief=brief, py_syntax=py_syntax,
                                    agent=agent_name)

        formatted = [inner_format_tool(fn) for fn in self._tools]
        # also process tools specified by name
        if self._tools_by_name:
            for tool_name in self._tools_by_name:
                fn = self.context.get(tool_name)
                if fn is None:
                    raise ValueError(f'Missing tool defined by name "{tool_name}"')
                formatted.append(inner_format_tool(fn))

        return '\n\n'.join(formatted)

    def init_context(self):
        """Initialize context. Override in subclasses to add agent-specific context."""
        if self._X__context is None:
            self._X__context = {}
        self._X__context["agent_name"] = self.role

    @property
    def context(self) -> Optional[Dict]:
        """
        Get agent's private context.
        """
        if self._X__context is None:
            self.init_context()
        return self._X__context

    @property
    def all_tools(self) -> List[Callable]:
        """Return all tools assigned to this agent (both _tools list and named context tools)."""
        result = list(self._tools)
        if self._tools_by_name:
            for tool_name in self._tools_by_name:
                fn = self.context.get(tool_name)
                if fn is not None:
                    result.append(fn)
        return result

    @property
    def system_tools(self) -> Iterable[Callable]:
        """Return agent-assigned tools marked with system=True."""
        result = [fn for fn in self._tools if getattr(fn, 'tool_system', False)]
        if self._tools_by_name:
            for tool_name in self._tools_by_name:
                fn = self.context.get(tool_name)
                if fn is not None and getattr(fn, 'tool_system', False):
                    result.append(fn)
        return result

    def append_tool(self, tool_or_name: Callable | str):
        """
        Add a tool to the agent's toolset.

        Args:
            tool_or_name: a callable or the name of a callable to be added as a tool
        """
        if isinstance(tool_or_name, str):
            if not self._tools_by_name:
                self._tools_by_name = []
            self._tools_by_name.append(tool_or_name)
        else:
            self._tools.append(tool_or_name)

@db0.memo
class SupervisedAgent(Agent):
    """
    Base class for implementing agents initiated and supervised by other agents or system functions
    """

    def create_job_def(
        self,
        tools: Optional[List[Callable]] = None,
        warmup_code: Optional[Union[str, Sequence[str]]] = None,
        **kwargs
    ) -> JobDef:
        # pylint: disable=unused-argument
        """
        Create a job definition with job-specific parameters.

        Args:
            tools: agent's tools additional tools (currently not used)
            warmup_code: optional initialization code (single block or sequence of blocks)
            kwargs: job specific parameters for prompt formatting (i.e. job_params)

        Returns:
            A new job definition object with specific job_params
        """
        # kwargs become job_params
        job_params = kwargs if kwargs else None

        return JobDef(
            agent=self,
            job_params=job_params,
            warmup_code=parse_warmup_code(warmup_code)
        )
