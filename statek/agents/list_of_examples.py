"""
list_of_examples meta-tool for STATEK agents.

Provides base tools for listing and showing examples for a given agent from the configured
examples directory. The base directory is read from StatekSettings.examples_dir
(env: STATEK_EXAMPLES_DIR). Examples for each agent live under <examples_dir>/<agent_name>/.

These tools are registered in the global tool registry via ``@tool(system=True)``.
The wrappers in agent.py resolve agent_name at call time via get_current_agent() and delegate here.
"""

import os
from typing import Optional

from statek.executors.example import load_examples, format_example
from statek.settings import get_statek_settings
from statek.system import tool
from statek.task_difficulty import TaskDifficulty
from statek.utils import find_locals, perm_ctx_set


def _get_examples_dir() -> Optional[str]:
    """Return the examples base directory from StatekSettings, or None if not set."""
    return get_statek_settings().examples_dir


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
def list_of_examples(agent_name: str, start_index: int = 0, limit: int = 10, **kwargs):  # pylint: disable=unused-argument
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
    names = get_example_names(agent_name)
    if not names:
        print("# No examples found")
        return
    total = len(names)
    print(f"# Example ID: Example name ({total} total)")
    for i, name in enumerate(names[start_index:start_index + limit]):
        idx = start_index + i
        print(f"{idx}: {name}")


@tool(system=True)
def show_example(agent_name: str, example_id: Optional[int] = None, **kwargs):  # pylint: disable=unused-argument
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
    lookup_logs = []
    example = _get_example(agent_name, example_id, logs=lookup_logs)
    if example is None:
        for message in lookup_logs:
            print(message)
        return
    settings = get_statek_settings()
    style = settings.examples_style or settings.chat_style
    name = example.example_metadata.get("name", "")
    try:
        perm_ctx_set(sync=True, last_example_id=example_id)
    except RuntimeError:
        pass
    if settings.xml_box_example:
        print(format_example(example, style, xml_tags={"example": settings.xml_box_example}))
    else:
        print(f"# --- EXAMPLE: {name} ---")
        print(format_example(example, style))
        print("# --- END OF EXAMPLE ---")
