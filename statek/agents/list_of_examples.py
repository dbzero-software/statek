"""
list_of_examples meta-tool for STATEK agents.

Provides base tools for listing and showing examples for a given agent from the configured
examples directory. The base directory is read from StatekSettings.examples_dir
(env: STATEK_EXAMPLES_DIR). Examples for each agent live under <examples_dir>/<agent_name>/.

These tools are registered on every Agent instance from Agent.__post_init__. The wrappers
in agent.py resolve agent_name at call time via find_locals() and delegate here.
"""

import os
from typing import Optional

from statek.executors.example import load_examples, format_example
from statek.settings import get_statek_settings
from statek.system import tool


def _get_examples_dir() -> Optional[str]:
    """Return the examples base directory from StatekSettings, or None if not set."""
    return get_statek_settings().examples_dir


@tool
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
    examples_dir = _get_examples_dir()
    if not examples_dir:
        print("# No examples found")
        return
    path = os.path.join(examples_dir, agent_name)
    if not os.path.isdir(path):
        print("# No examples found")
        return
    examples = load_examples(path)
    total = len(examples)
    print(f"# Example ID: Example name ({total} total)")
    for i, ex in enumerate(examples[start_index:start_index + limit]):
        idx = start_index + i
        name = ex.example_metadata.get("name", "")
        print(f"{idx}: {name}")

@tool
def show_example(agent_name: str, example_id: int, **kwargs):  # pylint: disable=unused-argument
    """Shows a specific example for a given agent.

    Prints the example content formatted using the current chat style setting.

    Args:
        agent_name: The agent role (used as the subdirectory in the examples path).
        example_id: The example index as reported by list_of_examples.

    Returns:
        None. Prints the example to console.

    Examples:
        show_example(agent_name="coordinator", example_id=0)
        show_example(agent_name="information_retriever", example_id=3)
    """
    examples_dir = _get_examples_dir()
    if not examples_dir:
        print("# No examples found")
        return
    path = os.path.join(examples_dir, agent_name)
    if not os.path.isdir(path):
        print("# No examples found")
        return
    examples = load_examples(path)
    if example_id < 0 or example_id >= len(examples):
        print(f"# Example {example_id} not found (total: {len(examples)})")
        return
    settings = get_statek_settings()
    style = settings.examples_style or settings.chat_style
    example = examples[example_id]
    if settings.xml_box_example:
        print(format_example(example, style, xml_tags={"example": settings.xml_box_example}))
    else:
        name = example.example_metadata.get("name", "")
        print(f"# --- EXAMPLE: {name} ---")
        print(format_example(example, style))
        print("# --- END OF EXAMPLE ---")
