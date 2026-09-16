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

"""
list_of_examples meta-tool for STATEK agents.

Provides base tools for listing and showing examples for a given agent from the configured
examples directory. The base directory is read from StatekSettings.examples_dir
(env: STATEK_EXAMPLES_DIR). Examples for each agent live under <examples_dir>/<agent_name>/.

These tools are registered in the global tool registry via ``@tool(system=True)``.
The wrappers in agent.py resolve agent_name at call time via get_current_agent() and delegate here.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from statek.executors.example import Example, load_examples, format_example
from statek.settings import get_statek_settings
from statek.system import tool
from statek.task_difficulty import TaskDifficulty
from statek.utils import find_locals, perm_ctx_set


def _get_examples_dir() -> Optional[str]:
    """Return the examples base directory from StatekSettings, or None if not set."""
    return get_statek_settings().examples_dir


@dataclass(frozen=True)
class SourcedExample:
    """An example paired with its configured agent role and local index."""

    agent_name: str
    example_id: int
    example: Example


def _agent_names(agent_name: str | Sequence[str]) -> list[str]:
    """Normalize one or more role names while preserving their first occurrence."""
    names = [agent_name] if isinstance(agent_name, str) else list(agent_name)
    return list(dict.fromkeys(name for name in names if name))


def get_sourced_examples(agent_name: str | Sequence[str]) -> list[SourcedExample]:
    """Return a fresh receiver-first combined view of examples and their sources."""
    examples_dir = _get_examples_dir()
    if not examples_dir:
        return []

    result = []
    for role in _agent_names(agent_name):
        path = os.path.join(examples_dir, role)
        if not os.path.isdir(path):
            continue
        result.extend(
            SourcedExample(role, example_id, example)
            for example_id, example in enumerate(load_examples(path))
        )
    return result


def _get_example(agent_name: str, example_id: int, logs: Optional[list[str]] = None):
    """Return an example by agent/id, optionally collecting user-facing lookup messages."""
    examples_dir = _get_examples_dir()
    if not examples_dir:
        if logs is not None:
            logs.append("# No examples found")
        return None

    path = os.path.join(examples_dir, agent_name)
    if not os.path.isdir(path):
        if logs is not None:
            logs.append("# No examples found")
        return None

    examples = load_examples(path)
    if example_id < 0 or example_id >= len(examples):
        if logs is not None:
            logs.append(f"# Example {example_id} not found (total: {len(examples)})")
        return None

    return examples[example_id]


def get_example_names(agent_name: str) -> list:
    """Return the list of example names for an agent.

    The list index corresponds to the example ID (0-based).

    Args:
        agent_name: The agent role (subdirectory name under examples_dir).

    Returns:
        List of example name strings. Empty list if no examples are
        configured or the agent's directory does not exist.
    """
    examples_dir = _get_examples_dir()
    if not examples_dir:
        return []
    path = os.path.join(examples_dir, agent_name)
    if not os.path.isdir(path):
        return []
    examples = load_examples(path)
    return [ex.example_metadata.get("name", "") for ex in examples]


def get_example_difficulty(agent_name: str, example_id: int) -> Optional[TaskDifficulty]:
    """Return a parsed example difficulty for an agent example, if configured."""
    example = _get_example(agent_name, example_id)
    return example.difficulty if example is not None else None


@tool(system=True)
def list_of_examples(  # pylint: disable=unused-argument
    agent_name: str | Sequence[str], start_index: int = 0, limit: int = 10, **kwargs,
):
    """Lists available examples for a given agent.

    Results are printed as a numbered list (index: name).

    Args:
        agent_name: The agent role (used as the subdirectory in the examples path).
        start_index: Index of the first example to show (default: 0).
        limit: Maximum number of examples to show (default: 10).

    Returns:
        None. Prints the list of examples to console.

    Examples:
        list_of_examples(agent_name="coordinator")
        list_of_examples(agent_name="information_retriever", start_index=10, limit=5)
    """
    sourced_examples = get_sourced_examples(agent_name)
    if not sourced_examples:
        print("# No examples found")
        return
    total = len(sourced_examples)
    print(f"# Example ID: Example name ({total} total)")
    for i, sourced in enumerate(sourced_examples[start_index:start_index + limit]):
        idx = start_index + i
        print(f"{idx}: {sourced.example.example_metadata.get('name', '')}")


@tool(system=True)
def show_example(  # pylint: disable=unused-argument
    agent_name: str | Sequence[str], example_id: Optional[int] = None, **kwargs,
):
    """Shows a specific example for a given agent.

    Prints the example content formatted using the current chat style setting.

    Args:
        agent_name: The agent role (used as the subdirectory in the examples path).
        example_id: Optional example index as reported by list_of_examples.
            If not provided, uses default_example_id from the local context.

    Returns:
        None. Prints the example to console.

    Examples:
        show_example(agent_name="coordinator", example_id=0)
        show_example(agent_name="information_retriever", example_id=3)
    """
    if example_id is None:
        defaults = list(find_locals(var_name="default_example_id"))
        if not defaults:
            print("# Example not found")
            return
        try:
            example_id = int(defaults[0])
        except (TypeError, ValueError):
            print("# Example not found")
            return
    sourced_examples = get_sourced_examples(agent_name)
    if example_id < 0 or example_id >= len(sourced_examples):
        if sourced_examples:
            print(f"# Example {example_id} not found (total: {len(sourced_examples)})")
        else:
            print("# No examples found")
        return
    sourced = sourced_examples[example_id]
    example = sourced.example
    settings = get_statek_settings()
    style = settings.examples_style or settings.chat_style
    name = example.example_metadata.get("name", "")
    try:
        perm_ctx_set(
            last_example_id=example_id,
            last_example_source={
                "agent_name": sourced.agent_name,
                "example_id": sourced.example_id,
            },
        )
    except RuntimeError:
        pass
    if settings.xml_box_example:
        print(format_example(example, style, xml_tags={"example": settings.xml_box_example}))
    else:
        print(f"# --- EXAMPLE: {name} ---")
        print(format_example(example, style))
        print("# --- END OF EXAMPLE ---")
