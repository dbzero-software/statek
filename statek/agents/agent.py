from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, List, Callable, Dict, Optional, Sequence, Union
import dbzero as db0
from statek.utils import block_comment, get_current_agent, _get_class_name
from statek.system import tool
from statek.docstring import parse_tool_docstring, format_docstring
from statek.utils import CodeBlock
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
    agent = get_current_agent()
    agent_name = agent.role if agent else None
    STATEK_LOGGER.debug("list_of_examples invoked for agent: %s", agent_name)
    _impl(agent_name, start_index, limit)


@tool(system=True)
def list_of_documents(topic=None, start_index: int = 0, limit: int = 25, **kwargs):  # pylint: disable=unused-argument
    """Lists available topics or documents within a topic.

    When called without a topic, lists all topics. When called with a topic
    (ID, name, or fragment), lists documents in that topic.

    Args:
        topic: Optional topic ID (int), name, or name fragment (str).
        start_index: Index of the first item to show (default: 0).
        limit: Maximum number of items to show (default: 25).
    """
    from statek.agents.list_of_documents import list_of_documents as _impl  # pylint: disable=import-outside-toplevel
    from statek.settings import get_statek_settings  # pylint: disable=import-outside-toplevel
    agent = get_current_agent()
    agent_name = agent.role if agent else None
    documents_dir = get_statek_settings().documents_dir
    STATEK_LOGGER.debug("list_of_documents invoked for agent: %s", agent_name)
    _impl(agent_name, documents_dir, topic=topic, start_index=start_index, limit=limit)


@tool(system=True)
def show_document(key, topic=None, start_from: int = 0, limit: int = 50, **kwargs):  # pylint: disable=unused-argument
    """Shows the contents of a specific document.

    Locates and prints a document from a specific topic or the last accessed topic.

    Args:
        key: Document ID (int), title, or title fragment (str).
        topic: Optional topic ID (int), name, or name fragment (str).
        start_from: First line number to print (default: 0).
        limit: Maximum number of lines to print (default: 50).
    """
    from statek.agents.list_of_documents import show_document as _impl  # pylint: disable=import-outside-toplevel
    from statek.settings import get_statek_settings  # pylint: disable=import-outside-toplevel
    agent = get_current_agent()
    agent_name = agent.role if agent else None
    documents_dir = get_statek_settings().documents_dir
    STATEK_LOGGER.debug("show_document invoked for agent: %s", agent_name)
    _impl(agent_name, documents_dir, key=key, topic=topic, start_from=start_from, limit=limit)


@tool(system=True)
def show_example(example_id: int, **kwargs):  # pylint: disable=unused-argument
    """Shows the content of a specific example by its index.

    Args:
        example_id: Index of the example to show (as listed by list_of_examples).
    """
    from statek.agents.list_of_examples import show_example as _impl  # pylint: disable=import-outside-toplevel
    agent = get_current_agent()
    _impl(agent.role if agent else None, example_id)


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
    _tools_by_name: Optional[List[str]] = field(default_factory=list)
    # Internal tools (name starts with '_'): available in execution context but not reported to LLM
    _internal_tools: Optional[List[Callable]] = field(default_factory=list)
    _internal_tools_by_name: Optional[List[str]] = field(default_factory=list)
    _metadata: Optional[Dict[str, str]] = None  # prompt meta-data as key/value pairs
    _X__context: Optional[Dict] = None  # Agent's specific context (e.g. with private tools)
    _description: Optional[str] = None  # optional LLM-friendly description of agent capabilities

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
        # Migrate any '_'-prefixed tools passed directly to _tools into _internal_tools
        internal = [fn for fn in self._tools if fn.__name__.startswith('_')]
        if internal:
            if not self._internal_tools:
                self._internal_tools = []
            for fn in internal:
                self._tools.remove(fn)
                self._internal_tools.append(fn)


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

    @property
    def description(self) -> Optional[str]:
        """Return the short description of the agent's capabilities."""
        return self._description

    def update_description(self, new_description: str) -> bool:
        """Update _description only if it differs from the current value.

        Args:
            new_description: New description string to apply.

        Returns:
            True if the description was updated, False if it was already up to date.
        """
        if self._description == new_description:
            return False
        self._description = new_description
        STATEK_LOGGER.debug("Agent '%s' description updated", self.role)
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
        # Resolve {shared_var_names} to comma-delimited list of shared_vars keys
        if '{shared_var_names}' in result:
            shared_vars = format_ctx.get('shared_vars', [])
            format_ctx['shared_var_names'] = ', '.join(shared_vars) if shared_vars else ''
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
        """Return all tools assigned to this agent (both regular and internal)."""
        result = list(self._tools)
        if self._tools_by_name:
            for tool_name in self._tools_by_name:
                fn = self.context.get(tool_name)
                if fn is not None:
                    result.append(fn)
        if self._internal_tools:
            result.extend(self._internal_tools)
        if self._internal_tools_by_name:
            for tool_name in self._internal_tools_by_name:
                fn = self.context.get(tool_name)
                if fn is not None:
                    result.append(fn)
        return result

    def _find_tool(self, name: str):
        """Find a tool by name in _tools.

        Args:
            name: The tool name to search for.

        Returns:
            The tool callable if found, None otherwise.
        """
        for t in self._tools:
            if getattr(t, '__name__', '') == name:
                return t
        return None

    def append_tool(self, tool_or_name: Callable | str):
        """
        Add a tool to the agent's toolset.

        Tools whose name starts with '_' are registered as internal tools:
        available in the job's local execution context but not reported to the LLM
        (i.e. excluded from {tools}, {brief_tools} and {detailed_tools} placeholders).

        Args:
            tool_or_name: a callable or the name of a callable to be added as a tool
        """
        if isinstance(tool_or_name, str):
            if tool_or_name.startswith('_'):
                self._internal_tools_by_name.append(tool_or_name)
            else:
                self._tools_by_name.append(tool_or_name)
        else:
            tool_name = getattr(tool_or_name, '__name__', '')
            if tool_name.startswith('_'):
                try:
                    self._internal_tools.append(tool_or_name)
                except RuntimeError:
                    # db0 rejects nested/decorated functions in persistent lists;
                    # fall back to name-based registration via context.
                    self._internal_tools_by_name.append(tool_name)
                    self.context[tool_name] = tool_or_name
            else:
                try:
                    self._tools.append(tool_or_name)
                except RuntimeError:
                    # db0 rejects nested/decorated functions in persistent lists;
                    # fall back to name-based registration via context.
                    self._tools_by_name.append(tool_name)
                    self.context[tool_name] = tool_or_name

@db0.memo
@dataclass
class WarmupDef:
    """Holds warmup code blocks retrieved from configuration files."""
    warmup_code: Optional[Union[str, CodeBlock, Sequence[Union[str, CodeBlock]]]] = None


@db0.memo
@dataclass
class SupervisedAgent(Agent):
    """
    Base class for implementing agents initiated and supervised by other agents or system functions
    """
    warmup_def: Optional[WarmupDef] = None

    def update_warmup_def(self, unparsed_warmup_def: str) -> bool:
        """Set or update warmup_def only if it has changed.

        Parses the input through parse_warmup_code before comparing with the
        current warmup_def's warmup_code.

        Args:
            unparsed_warmup_def: Raw warmup code string, or None.

        Returns:
            True if warmup_def was updated, False if it was already up to date.
        """
        new_value = parse_warmup_code(unparsed_warmup_def)
        if new_value is None and self.warmup_def is None:
            return False
        if new_value is None:
            self.warmup_def = None
            STATEK_LOGGER.debug("Agent '%s' warmup_def cleared", self.role)
            return True
        if self.warmup_def is not None and self.warmup_def.warmup_code == new_value:
            return False
        self.warmup_def = WarmupDef(warmup_code=new_value)
        STATEK_LOGGER.debug("Agent '%s' warmup_def updated", self.role)
        return True

    def create_job_def(
        self,
        tools: Optional[List[Callable]] = None,
        warmup_code: Optional[Union[str, Sequence[str]]] = None,
        shared_vars: Optional[Dict[str, Any]] = None,
        locale=None,
        **kwargs
    ) -> JobDef:
        # pylint: disable=unused-argument
        """
        Create a job definition with job-specific parameters.

        Args:
            tools: agent's tools additional tools (currently not used)
            warmup_code: optional initialization code (single block or sequence of blocks)
            shared_vars: optional ``{var_name: value}`` mapping of variables
                additionally shared by the parent job. Its keys are reported
                as a ``shared_vars`` entry in ``job_params`` so they can be
                referenced from the agent's system prompt template.
            locale: optional locale for job execution
            kwargs: job specific parameters for prompt formatting (i.e. job_params)

        Returns:
            A new job definition object with specific job_params
        """
        # kwargs become job_params
        job_params = dict(kwargs) if kwargs else {}

        if shared_vars:
            job_params["shared_vars"] = list(shared_vars.keys())

        if not job_params:
            job_params = None

        combined = self._combine_warmup_code(warmup_code)
        metadata = self._metadata or {}
        model = metadata.get('MODEL')
        if model is None:
            raise ValueError(
                f"Agent '{self.role}' is missing required metadata field 'MODEL'"
            )
        model_family = metadata.get('MODEL_FAMILY')
        if model_family is None and '/' in model:
            model_family = model.split('/', 1)[0]

        return JobDef(
            agent=self,
            job_params=job_params,
            warmup_code=combined,
            locale=locale,
            model_family=model_family,
            model=model,
        )

    def _combine_warmup_code(self, warmup_code):
        """Combine dynamic warmup_code with agent's warmup_def blocks.

        Dynamic warmup_code comes first, followed by warmup_def blocks.

        Returns:
            Combined warmup code, or None if both are absent.
        """
        dynamic = parse_warmup_code(warmup_code)
        agent_warmup = self.warmup_def.warmup_code if self.warmup_def else None

        if dynamic is None and agent_warmup is None:
            return None
        if dynamic is None:
            return agent_warmup
        if agent_warmup is None:
            return dynamic

        # Both present — combine into a list (dynamic first, then agent's)
        def _as_list(val):
            if isinstance(val, (str, CodeBlock)):
                return [val]
            return list(val)

        return _as_list(dynamic) + _as_list(agent_warmup)


def update_warmup_defs(path: Path):
    """Load warmup definitions from .py files and apply to matching SupervisedAgents.

    Files are matched to agents by convention: the filename (without .py) must
    equal the agent's role (e.g. information_retriever.py -> role "information_retriever").

    Args:
        path: Directory to scan for warmup definition files.
    """
    agents_by_role = {agent.role: agent for agent in db0.find(SupervisedAgent)}  # pylint: disable=no-member
    for py_file in path.glob("*.py"):
        role = py_file.stem
        agent = agents_by_role.get(role)
        if agent is None:
            continue
        content = py_file.read_text(encoding="utf-8")
        agent.update_warmup_def(content)
