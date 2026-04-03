"""Prompt configuration utilities to avoid cyclic imports."""

import re
from collections import namedtuple
from pathlib import Path
from typing import Dict


# system = the system prompt
# metadata = prompt's metadata (as key-value dictionary)
PromptDef = namedtuple("PromptDef", ["system", "metadata"])

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

    system_prompt = '\n'.join(system_lines).strip()
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
    import dbzero as db0  # pylint: disable=import-outside-toplevel
    from statek.agents.agent import Agent  # pylint: disable=import-outside-toplevel,cyclic-import
    from statek.executors.job import JobDef  # pylint: disable=import-outside-toplevel,cyclic-import
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
        for job_def in db0.find(JobDef, agent):  # pylint: disable=no-member
            job_def.set_chat_style(new_chat_style)
