from dataclasses import dataclass
from typing import Dict, List

from statek.executors.job import Job
from statek.settings import ChatStyle


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
    warmup_console = "\n".join(console[:warmup_console_end])

    items = []
    for i, block in enumerate(blocks):
        items.append(block)
        items.append(warmup_console if i == len(blocks) - 1 else "")

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

    example_str = _format_items(example.example_items, chat_style)
    if example_str:
        sections.append(example_str)

    return "\n".join(sections)


def _format_metadata(metadata: Dict) -> str:
    lines = ["# Example metadata:"]
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


def _extract_example_items(job: Job) -> List[str]:
    console = job.py_env.console or []
    prompt = job.job_def.prompt()

    if not job.chat_log:
        items = [prompt, "\n".join(console)]
        return items

    items = [prompt, "\n".join(console[:job.chat_log[0].console_pos])]

    for i, chat_item in enumerate(job.chat_log):
        items.append(chat_item.llm_resp)
        end = job.chat_log[i + 1].console_pos if i + 1 < len(job.chat_log) else len(console)
        items.append("\n".join(console[chat_item.console_pos:end]))

    return items
