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

"""Prompt configuration utilities to avoid cyclic imports."""

import re
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import dbzero as db0
from statek.task_difficulty import TaskDifficulty


# system = the system prompt
# metadata = prompt's metadata (as key-value dictionary)
PromptDef = namedtuple("PromptDef", ["system", "metadata"])


@dataclass
class PromptSectionData:
    """Volatile representation of a parsed system prompt section."""
    title: str
    contents: str
    target_difficulties: Optional[Set[TaskDifficulty]] = None


@db0.memo
@dataclass
class PromptSection:
    """Persistent representation of a system prompt section."""
    title: str
    contents: str
    target_difficulties: Optional[Set[TaskDifficulty]] = None


@dataclass
class SystemPromptData:
    """Volatile representation of a parsed system prompt."""
    intro: str
    sections: List[PromptSectionData]


@db0.memo
@dataclass
class SystemPrompt:
    """Persistent representation of a parsed system prompt."""
    intro: str
    sections: List[PromptSection]


PromptSectionLike = Union[PromptSection, PromptSectionData]
SystemPromptLike = Union[SystemPrompt, SystemPromptData]
SystemPromptInput = Union[str, SystemPromptLike]


@db0.enum(values=["XML", "DASHED", "ASTERISK", "MARKDOWN"])
class PromptStyle:
    """Supported system prompt section formatting styles."""


_LINE_SECTION_RE = re.compile(
    r"^[ \t]*(?:"
    r"-{3,}[ \t]*(?P<dash>.+?)[ \t]*-{3,}|"
    r"\*{3,}[ \t]*(?P<star>.+?)[ \t]*\*{3,}|"
    r"#{2,6}[ \t]+(?P<markdown>.+?)[ \t]*#*"
    r")[ \t]*$",
    re.MULTILINE,
)
_XML_SECTION_RE = re.compile(
    r"^[ \t]*<(?P<tag>[A-Za-z_][\w.-]*)(?P<attrs>(?:\s+[^<>]*?)?)\s*>"
    r"(?P<contents>.*?)"
    r"</(?P=tag)>[ \t]*(?:\r?\n)?",
    re.DOTALL | re.MULTILINE,
)
_XML_DIFFICULTY_ATTR_RE = re.compile(
    r"\b(?:target_difficulties|difficulty)\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)')",
    re.IGNORECASE,
)
_TASK_DIFFICULTY_BY_LABEL = {
    "L": TaskDifficulty.low,  # pylint: disable=no-member
    "LOW": TaskDifficulty.low,  # pylint: disable=no-member
    "M": TaskDifficulty.medium,  # pylint: disable=no-member
    "MEDIUM": TaskDifficulty.medium,  # pylint: disable=no-member
    "H": TaskDifficulty.high,  # pylint: disable=no-member
    "HIGH": TaskDifficulty.high,  # pylint: disable=no-member
}


def _strip_prompt_part(value: str) -> str:
    """Trim outer whitespace while preserving internal formatting."""
    return value.strip()


def _parse_target_difficulties(value: str) -> Set[TaskDifficulty]:
    """Parse compact or delimited task difficulty metadata."""
    compact = value.strip()
    if not compact:
        raise ValueError("Invalid task difficulty: ''")

    if re.fullmatch(r"[LMHlmh]+", compact):
        return {_parse_task_difficulty_label(char) for char in compact}

    parts = [part for part in re.split(r"[\s,|/]+", compact) if part]
    if len(parts) > 1:
        return {_parse_task_difficulty_label(part) for part in parts}

    return {_parse_task_difficulty_label(compact)}


def _parse_task_difficulty_label(value: str) -> TaskDifficulty:
    """Parse a task difficulty label without requiring enum scope resolution."""
    label = value.strip().upper()
    try:
        return _TASK_DIFFICULTY_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Invalid task difficulty: {value!r}") from exc


def _parse_title_and_difficulties(title: str) -> Tuple[str, Optional[Set[TaskDifficulty]]]:
    """Split an optional difficulty prefix from a section title."""
    clean_title = title.strip()
    if ":" not in clean_title:
        return clean_title, None

    prefix, section_title = clean_title.split(":", 1)
    try:
        target_difficulties = _parse_target_difficulties(prefix)
    except ValueError:
        return clean_title, None

    return section_title.strip(), target_difficulties


def _parse_xml_target_difficulties(attrs: str) -> Optional[Set[TaskDifficulty]]:
    """Extract target difficulty metadata from XML attributes."""
    match = _XML_DIFFICULTY_ATTR_RE.search(attrs)
    if not match:
        return None
    value = match.group("double") if match.group("double") is not None else match.group("single")
    return _parse_target_difficulties(value)


def _next_section_match(input: str, pos: int):  # pylint: disable=redefined-builtin
    """Return the next XML or line section delimiter match."""
    line_match = _LINE_SECTION_RE.search(input, pos)
    xml_match = _XML_SECTION_RE.search(input, pos)

    if line_match is None:
        return xml_match, "xml" if xml_match is not None else None
    if xml_match is None:
        return line_match, "line"
    if xml_match.start() < line_match.start():
        return xml_match, "xml"
    return line_match, "line"


def parse_system_prompt(input: str) -> SystemPromptData:  # pylint: disable=redefined-builtin
    """Parse a system prompt into intro text and annotated sections.

    Supported section delimiters are XML boxes, dashed headers, starred headers,
    and markdown headings (``##`` through ``######``). Non-XML section titles may
    be prefixed with target difficulties, for example ``MH:CONFIDENTIALITY``.
    XML sections accept ``target_difficulties`` and ``difficulty`` attributes.
    """
    intro_parts: List[str] = []
    sections: List[PromptSectionData] = []
    current_title: Optional[str] = None
    current_target_difficulties: Optional[Set[TaskDifficulty]] = None
    current_contents: List[str] = []

    def flush_current_section():
        nonlocal current_title, current_target_difficulties, current_contents
        if current_title is None:
            return
        sections.append(PromptSectionData(
            title=current_title,
            contents=_strip_prompt_part("".join(current_contents)),
            target_difficulties=current_target_difficulties,
        ))
        current_title = None
        current_target_difficulties = None
        current_contents = []

    def append_plain_text(text: str):
        if current_title is not None:
            current_contents.append(text)
        elif sections and _strip_prompt_part(text):
            last_section = sections[-1]
            contents = "\n".join(
                part for part in [last_section.contents, _strip_prompt_part(text)]
                if part
            )
            last_section.contents = contents
        else:
            intro_parts.append(text)

    pos = 0
    while pos < len(input):
        match, match_type = _next_section_match(input, pos)
        if match is None:
            break

        append_plain_text(input[pos:match.start()])

        if match_type == "xml":
            flush_current_section()
            sections.append(PromptSectionData(
                title=match.group("tag").strip(),
                contents=_strip_prompt_part(match.group("contents")),
                target_difficulties=_parse_xml_target_difficulties(match.group("attrs") or ""),
            ))
        else:
            flush_current_section()
            raw_title = match.group("dash") or match.group("star") or match.group("markdown")
            current_title, current_target_difficulties = _parse_title_and_difficulties(raw_title)
            current_contents = []

        pos = match.end()

    append_plain_text(input[pos:])
    flush_current_section()

    return SystemPromptData(
        intro=_strip_prompt_part("".join(intro_parts)),
        sections=sections,
    )


def make_system_prompt(prompt: SystemPromptInput) -> SystemPrompt:
    """Build a persistent SystemPrompt from raw text or parsed prompt data."""
    if isinstance(prompt, SystemPrompt):
        return prompt
    if isinstance(prompt, str):
        prompt = parse_system_prompt(prompt)

    return SystemPrompt(
        intro=prompt.intro,
        sections=[
            section if isinstance(section, PromptSection) else PromptSection(
                title=section.title,
                contents=section.contents,
                target_difficulties=section.target_difficulties,
            )
            for section in prompt.sections
        ],
    )


def _section_matches_difficulty(
    section: PromptSectionLike,
    task_difficulty: TaskDifficulty,
) -> bool:
    """Return whether a section should be included for the target difficulty."""
    return (
        section.target_difficulties is None
        or task_difficulty in section.target_difficulties
    )


def _format_section_contents(
    section: PromptSectionLike,
    section_formatter: Optional[Callable[[str], str]],
    prompt_part_formatter: Optional[Callable[[str], str]],
) -> str:
    """Apply optional section content formatting and trim outer whitespace."""
    contents = section.contents
    if section_formatter is not None:
        contents = section_formatter(contents)
    if prompt_part_formatter is not None:
        contents = prompt_part_formatter(contents)
    return _strip_prompt_part(contents)


def _format_xml_section(title: str, contents: str) -> str:
    """Format a section as an XML box."""
    if not re.fullmatch(r"[A-Za-z_][\w.-]*", title):
        raise ValueError(f"Invalid XML section title: {title!r}")
    if "\n" in contents:
        return f"<{title}>\n{contents}\n</{title}>"
    return f"<{title}>{contents}</{title}>"


def _format_prompt_section(
    section: PromptSectionLike,
    style: PromptStyle,
    section_formatter: Optional[Callable[[str], str]],
    prompt_part_formatter: Optional[Callable[[str], str]],
) -> str:
    """Format one prompt section using the requested delimiter style."""
    title = section.title.strip()
    contents = _format_section_contents(section, section_formatter, prompt_part_formatter)
    if style == PromptStyle.XML:  # pylint: disable=no-member
        return _format_xml_section(title, contents)
    if style == PromptStyle.DASHED:  # pylint: disable=no-member
        return f"--- {title} ---\n{contents}"
    if style == PromptStyle.ASTERISK:  # pylint: disable=no-member
        return f"*** {title} ***\n{contents}"
    if style == PromptStyle.MARKDOWN:  # pylint: disable=no-member
        return f"## {title}\n{contents}"
    raise ValueError(f"Unsupported prompt style: {style!r}")


def format_system_prompt(
    prompt: SystemPromptLike,
    task_difficulty: TaskDifficulty,
    style: PromptStyle = PromptStyle.DASHED,  # pylint: disable=no-member
    section_formatter: Optional[Callable[[str], str]] = None,
    prompt_part_formatter: Optional[Callable[[str], str]] = None,
) -> str:
    """Format a system prompt for a target task difficulty.

    The intro is always included. Sections with no target difficulty metadata
    are included for every task. Sections with metadata are included only when
    ``task_difficulty`` is one of the section's target difficulties.
    ``prompt_part_formatter`` is applied to the intro and each selected section's
    contents before prompt parts are joined.
    """
    parts = []
    intro = prompt.intro
    if prompt_part_formatter is not None:
        intro = prompt_part_formatter(intro)
    intro = _strip_prompt_part(intro)
    if intro:
        parts.append(intro)

    parts.extend(
        _format_prompt_section(section, style, section_formatter, prompt_part_formatter)
        for section in prompt.sections
        if _section_matches_difficulty(section, task_difficulty)
    )
    return "\n\n".join(parts)


def _normalize_prompt_section(section: PromptSectionLike):
    """Return a stable structural representation of a prompt section."""
    target_difficulties = section.target_difficulties
    if target_difficulties is not None:
        target_difficulties = frozenset(target_difficulties)
    return (
        section.title,
        section.contents,
        target_difficulties,
    )


def _normalize_system_prompt(prompt: SystemPromptLike):
    """Return a stable structural representation of a prompt."""
    return (
        prompt.intro,
        tuple(_normalize_prompt_section(section) for section in prompt.sections),
    )


def compare_prompts(prompt_1: SystemPromptLike, prompt_2: SystemPromptLike) -> bool:
    """Return True when two volatile or persistent system prompts are identical."""
    return _normalize_system_prompt(prompt_1) == _normalize_system_prompt(prompt_2)


def parse_prompt_file(file_path: Path) -> PromptDef:
    """Parse a single prompt definition file.

    The file format uses # KEY: value comment lines for metadata and a
    # System Prompt section for the system prompt content.

    Args:
        file_path: Path to the .md file

    Returns:
        PromptDef with system and metadata

    Raises:
        ValueError: If the file does not contain a '# System Prompt' section
    """
    content = file_path.read_text(encoding='utf-8')

    metadata = {}
    system_lines = []
    in_system = False

    for line in content.split('\n'):
        if in_system:
            system_lines.append(line)
        else:
            # Check for System Prompt section header
            if re.match(r'^#\s+System Prompt\s*$', line, re.IGNORECASE):
                in_system = True
            else:
                # Check for metadata comment: # KEY: value
                meta_match = re.match(r'^#\s+(\w+):\s+(.+)$', line)
                if meta_match:
                    metadata[meta_match.group(1).upper()] = meta_match.group(2).strip()

    if not in_system:
        raise ValueError(
            f"Prompt file '{file_path}' is missing the '# System Prompt' section."
        )

    system_prompt = parse_system_prompt('\n'.join(system_lines).strip())
    return PromptDef(system=system_prompt, metadata=metadata)


def load_prompt_files(prompt_files_dir: str) -> Dict[str, PromptDef]:
    """Load all prompt definition files from a directory.

    Args:
        prompt_files_dir: Path to directory containing .md prompt files

    Returns:
        Dictionary mapping role names to PromptDef objects
    """
    if not prompt_files_dir:
        return {}

    prompt_dir = Path(prompt_files_dir)
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        return {}

    prompt_defs = {}
    for file_path in prompt_dir.glob('*.md'):
        role = file_path.stem
        prompt_defs[role] = parse_prompt_file(file_path)

    return prompt_defs


def update_prompt_config(prompt_defs: Dict[str, PromptDef], agents=None):
    """Update agent prompts and associated JobDef fields from prompt definitions.

    Updates agent system prompts and metadata, and propagates the CHAT_STYLE
    metadata key to all JobDef instances associated with each agent.

    Args:
        prompt_defs: Dictionary of role -> PromptDef mappings
        agents: List of agents to update. If None, updates all agents in db0.
    """
    from statek.agents.agent import Agent  # pylint: disable=import-outside-toplevel,cyclic-import
    from statek.executors.job import JobDef  # pylint: disable=import-outside-toplevel,cyclic-import
    from statek.executors.post_processor import (  # pylint: disable=import-outside-toplevel
        resolve_post_processors_from_metadata,
    )
    from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

    # If agents not provided, find all agents in db0
    if agents is None:
        agents = db0.find(Agent)  # pylint: disable=no-member

    for agent in agents:
        # Look up prompt definition by agent's role name
        prompt_def = prompt_defs.get(agent.role)

        if prompt_def is None:
            # Quietly skip agents without matching prompt definitions
            continue

        job_defs = tuple(db0.find(JobDef, db0.as_tag(agent)))  # pylint: disable=no-member

        # Update agent's system prompt if changed
        if prompt_def.system:
            agent.update_system_prompt(prompt_def.system)

        # Update agent's metadata if changed
        if prompt_def.metadata:
            agent.update_metadata(prompt_def.metadata)

        # Update agent description from DESCRIPTION metadata key
        description = prompt_def.metadata.get('DESCRIPTION') if prompt_def.metadata else None
        if description:
            agent.update_description(description)

        # Propagate CHAT_STYLE to associated JobDefs
        chat_style_str = prompt_def.metadata.get('CHAT_STYLE') if prompt_def.metadata else None
        new_chat_style = ChatStyle[chat_style_str.upper()] if chat_style_str else None  # pylint: disable=no-member
        new_post_processing = resolve_post_processors_from_metadata(prompt_def.metadata)
        for job_def in job_defs:
            job_def.set_chat_style(new_chat_style)
            job_def.set_post_processing(new_post_processing)
