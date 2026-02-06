"""Prompt configuration utilities to avoid cyclic imports."""

from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PromptDef:
    """Definition for an agent prompt."""
    system: str
    template: str
    role: Optional[str] = None


def parse_prompt_file(file_path: Path) -> Optional[PromptDef]:
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
        prompt_def = parse_prompt_file(file_path)
        if prompt_def:
            prompt_def.role = role
            prompt_defs[role] = prompt_def

    return prompt_defs


def update_prompt_config(prompt_defs: Dict[str, PromptDef], agents=None):
    """Update agent prompts from prompt definitions.

    Args:
        prompt_defs: Dictionary of role -> PromptDef mappings
        agents: List of agents to update. If None, updates all agents in db0.
    """
    import dbzero as db0  # pylint: disable=import-outside-toplevel
    from statek.agents.agent import Agent  # pylint: disable=import-outside-toplevel,cyclic-import

    # If agents not provided, find all agents in db0
    if agents is None:
        agents = db0.find(Agent)  # pylint: disable=no-member

    for agent in agents:
        # Look up prompt definition by agent's role name
        prompt_def = prompt_defs.get(agent.role)

        if prompt_def is None:
            # Quietly skip agents without matching prompt definitions
            continue

        # Update agent's system prompt and template only if changed
        if prompt_def.system and agent._system_prompt != prompt_def.system:  # pylint: disable=protected-access
            agent._system_prompt = prompt_def.system  # pylint: disable=protected-access
        if prompt_def.template and agent._prompt_template != prompt_def.template:  # pylint: disable=protected-access
            agent._prompt_template = prompt_def.template  # pylint: disable=protected-access
