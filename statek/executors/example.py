import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

from statek.executors.job import Job
from statek.settings import ChatStyle

# Separator inserted between the warmup section and the example section in the
# formatted output so that parse_example can correctly reconstruct both lists.
_WARMUP_SEPARATOR = "# ----------"


@dataclass
class Example:
    """Metadata as key-values"""
    example_metadata: Dict
    """Items starting from python code blocks interleaved with console blocks"""
    warmup_items: List[str]
    example_items: List[str]


def extract_example(job: Job, name: str) -> Example:
    """Retrieves an Example object from a specific job instance.

    Args:
        job: a job to extract the example from
        name: the example name to be assigned

    Returns:
        Example: the extracted example object
    """
    example_metadata = {
        "name": name,
        "model": job.model,
        "model_family": job.model_family,
        "num_turns": job.num_turns,
        "exception_count": job.exception_count,
        "exit_status": job.py_env.exit_status,
    }
    if job.job_def.agent is not None:
        example_metadata["agent_role"] = job.job_def.agent.role
    if job.job_def.job_params is not None:
        example_metadata["job_params"] = job.job_def.job_params

    return Example(
        example_metadata=example_metadata,
        warmup_items=_extract_warmup_items(job),
        example_items=_extract_example_items(job),
    )


def _extract_warmup_items(job: Job) -> List[str]:
    warmup_code = job.job_def.warmup_code
    if warmup_code is None:
        return []

    blocks = [warmup_code] if isinstance(warmup_code, str) else list(warmup_code)

    console = job.py_env.console or []
    warmup_console_end = job.chat_log[0].console_pos if job.chat_log else len(console)
    positions = job.warmup_console_positions

    items = []
    prev_pos = 0
    for i, block in enumerate(blocks):
        items.append(block)
        if i < len(positions):
            end_pos = min(positions[i], warmup_console_end)
            items.append("\n".join(console[prev_pos:end_pos]))
            prev_pos = end_pos
        elif i == len(blocks) - 1:
            # Fallback: last block gets remaining console
            items.append("\n".join(console[prev_pos:warmup_console_end]))
        else:
            items.append("")

    return items


def format_example(example: Example, chat_style: ChatStyle,
                   include_warmup: bool = True,
                   include_metadata: bool = False) -> str:
    """Format an example as a string using the specified chat style.

    Args:
        example: the example object instance
        chat_style: formatting style (CONSOLE or MARKDOWN)
        include_warmup: flag indicating if warmup block should be included
        include_metadata: flag indicating if metadata should be included

    Returns:
        str: the formatted string
    """
    sections = []

    if include_metadata and example.example_metadata:
        sections.append(_format_metadata(example.example_metadata))

    if include_warmup and example.warmup_items:
        warmup_str = _format_items(example.warmup_items, chat_style)
        if warmup_str:
            sections.append(warmup_str)
            sections.append(_WARMUP_SEPARATOR)

    example_str = _format_items(example.example_items, chat_style)
    if example_str:
        sections.append(example_str)

    return "\n".join(sections)


def _format_metadata(metadata: Dict) -> str:
    lines = []
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    return "\n".join(lines)


def _format_items(items: List[str], chat_style: ChatStyle) -> str:
    """Format alternating [code, console, code, console, ...] items."""
    formatted = []
    for i, item in enumerate(items):
        if i % 2 == 0:
            part = _format_code(item, chat_style)
        else:
            part = _format_console(item, chat_style)
        if part:
            formatted.append(part)
    return "\n".join(formatted)


def _format_code(code: str, chat_style: ChatStyle) -> str:
    if not code:
        return ""
    if chat_style == ChatStyle.MARKDOWN:  # pylint: disable=no-member
        return f"```python\n{code}\n```"
    return code


def _format_console(console: str, chat_style: ChatStyle) -> str:
    if not console:
        return ""
    if chat_style == ChatStyle.CONSOLE:  # pylint: disable=no-member
        return "\n".join(f"> {line}" for line in console.split("\n"))
    return console


def parse_example(example_md: str) -> Example:
    """Parse a valid example markdown file into an Example instance.

    Handles both MARKDOWN style (```python fences) and CONSOLE style (> prefixed
    console lines).  Lines of the form ``# key: value`` outside of code fences are
    treated as metadata and stripped from the content.  A ``# ----------`` separator
    line between the warmup and example sections reconstructs warmup_items and
    example_items separately.  If no separator is present all items are placed in
    example_items.

    Args:
        example_md: contents of the input markdown file

    Returns:
        Example: properly parsed Example instance

    Raises:
        ValueError: if the content before the first code block is non-empty
    """
    metadata: Dict = {}
    kept_lines = []
    in_code_block = False

    for line in example_md.split('\n'):
        if line.startswith('```'):
            in_code_block = not in_code_block
            kept_lines.append(line)
            continue
        if not in_code_block and line.startswith('# '):
            meta_part = line[2:]  # strip leading '# '
            key, sep, value = meta_part.partition(': ')
            if sep:
                metadata[key] = _parse_metadata_value(value)
                continue  # strip metadata line from content
        kept_lines.append(line)

    content = '\n'.join(kept_lines).lstrip('\n')

    # Split on the warmup/example separator when present
    separator_line = f'\n{_WARMUP_SEPARATOR}\n'
    if separator_line in content:
        warmup_content, example_content = content.split(separator_line, 1)
        warmup_items = _parse_items(warmup_content.lstrip('\n'))
        example_items = _parse_items(example_content.lstrip('\n'))
    else:
        warmup_items = []
        example_items = _parse_items(content)

    return Example(
        example_metadata=metadata,
        warmup_items=warmup_items,
        example_items=example_items,
    )


@lru_cache
def load_examples(path: str) -> List[Example]:
    """Load all example files from a directory (including subdirectories).

    Results are cached — repeated calls with the same path return the same list
    without re-reading the filesystem.

    Args:
        path: root directory to search for example files

    Returns:
        List of parsed Example instances sorted by metadata ``seq_id`` ascending.
        Examples without a ``seq_id`` are appended at the end in the order they
        were found.

    Raises:
        ValueError: if any .md file cannot be parsed
    """
    examples: List[Example] = []

    for root, dirs, files in os.walk(path):
        dirs.sort()
        for filename in sorted(files):
            if not filename.endswith('.md'):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            examples.append(parse_example(content))

    def _sort_key(ex: Example):
        seq_id = ex.example_metadata.get('seq_id')
        if seq_id is None:
            return (1, 0)
        return (0, int(seq_id))

    examples.sort(key=_sort_key)
    return examples


def _parse_metadata_value(value: str):
    """Convert a metadata string value to int if possible, else return as str."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_items(content: str) -> List[str]:
    """Parse alternating [code, console, ...] items from content.

    Dispatches to MARKDOWN or CONSOLE parser based on whether ```python fences
    are present.
    """
    if '```python' in content:
        return _parse_markdown_items(content)
    return _parse_console_items(content)


def _parse_markdown_items(content: str) -> List[str]:
    """Parse alternating [code, console, ...] items from MARKDOWN-style content.

    Uses re.split on ```python...``` fences.  The text between two consecutive
    code fences strips to an empty string, correctly representing the empty
    console item that _format_items skipped when writing.
    """
    parts = re.split(r'```python\n(.*?)\n```', content, flags=re.DOTALL)

    items: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Even positions: console text (or the gap before the very first fence)
            console = part.strip('\n')
            if not items:
                if console:
                    raise ValueError(f"Unexpected content before first code block: {console!r}")
                continue
            items.append(console)
        else:
            # Odd positions: captured code block content
            items.append(part)

    return items


def _parse_console_items(content: str) -> List[str]:
    """Parse alternating [code, console, ...] items from CONSOLE-style content.

    Lines starting with '> ' are console (prefix stripped); all other lines
    are code.  Consecutive same-type lines are merged into one segment.
    """
    lines = content.split('\n')
    items: List[str] = []
    current: List[str] = []
    in_console = False

    for line in lines:
        is_console_line = line.startswith('> ')

        if is_console_line != in_console:
            segment = '\n'.join(current)
            if in_console:
                items.append(segment)
            else:
                stripped = segment.strip('\n')
                if stripped:
                    items.append(stripped)
                elif items:
                    items.append('')  # empty code placeholder
            current = []
            in_console = is_console_line

        current.append(line[2:] if is_console_line else line)

    if current:
        segment = '\n'.join(current)
        if in_console:
            items.append(segment)
        else:
            stripped = segment.strip('\n')
            if stripped:
                items.append(stripped)

    return items


def _extract_example_items(job: Job) -> List[str]:
    console = job.py_env.console or []

    if not job.chat_log:
        return []

    items = []
    for i, chat_item in enumerate(job.chat_log):
        items.append(chat_item.llm_resp)
        end = job.chat_log[i + 1].console_pos if i + 1 < len(job.chat_log) else len(console)
        items.append("\n".join(console[chat_item.console_pos:end]))

    return items
