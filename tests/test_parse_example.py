"""Tests for parse_example and the warmup separator in format_example."""

# pylint: disable=no-member

from statek.executors.example import (
    Example, extract_example, format_example, parse_example, _WARMUP_SEPARATOR,
)
from statek.executors.job import Job, JobDef, JobStatus
from statek.agents.agent import Agent
from statek.settings import ChatStyle
from tests.conftest import create_chat_log_item


def _example(warmup_items, example_items, metadata=None) -> Example:
    return Example(
        example_metadata=metadata or {},
        warmup_items=warmup_items,
        example_items=example_items,
    )


def _fmt(example, style=ChatStyle.MARKDOWN, warmup=True, metadata=False) -> str:
    return format_example(example, style, include_warmup=warmup, include_metadata=metadata)


def _job_with_warmup():
    agent = Agent(role="test", _system_prompt="sys", _prompt_template="task", _tools=[])
    job_def = JobDef(agent=agent, warmup_code="setup = True")
    job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
    job.py_env.console = ["True", "after_llm"]
    job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="result = 42"))
    return job


def test_metadata_int_values_parsed_as_int():
    """num_turns and exception_count are reconstructed as int, not str."""
    meta = {"num_turns": 3, "exception_count": 1}
    result = parse_example(_fmt(_example([], ["x = 1", ""], metadata=meta), metadata=True))
    assert result.example_metadata["num_turns"] == 3
    assert isinstance(result.example_metadata["num_turns"], int)


def test_metadata_value_with_colon_preserved():
    """Values that contain ':' are not truncated at the first colon."""
    meta = {"exit_status": "Error: Maximum turns exceeded: 5/5"}
    result = parse_example(_fmt(_example([], ["x = 1", ""], metadata=meta), metadata=True))
    assert result.example_metadata["exit_status"] == "Error: Maximum turns exceeded: 5/5"


def test_consecutive_code_blocks_yield_empty_console_between():
    """Two adjacent code fences imply an empty string at the odd (console) index."""
    ex = _example([], ["code1", "", "code2", "out"])
    result = parse_example(_fmt(ex))
    assert result.example_items == ["code1", "", "code2", "out"]


def test_trailing_console_after_last_code_block():
    ex = _example([], ["prompt", "", "resp", "final output"])
    result = parse_example(_fmt(ex))
    assert result.example_items[3] == "final output"


def test_console_style_basic_parsing():
    """CONSOLE-style content (> prefixed lines) is correctly reconstructed."""
    ex = _example([], ["print(1)", "1"])
    result = parse_example(_fmt(ex, style=ChatStyle.CONSOLE))
    assert result.example_items[0] == "print(1)"
    assert result.example_items[1] == "1"


def test_format_emits_separator_only_when_warmup_present():
    assert _WARMUP_SEPARATOR in _fmt(_example(["setup()", ""], ["prompt", "out"]))
    assert _WARMUP_SEPARATOR not in _fmt(_example([], ["prompt", "out"]))


def test_parse_reconstructs_warmup_and_example_items_from_separator():
    ex = _example(["setup()", "warmup_out"], ["prompt", "out"])
    result = parse_example(_fmt(ex))
    assert result.warmup_items == ["setup()", "warmup_out"]
    assert result.example_items == ["prompt", "out"]


def test_parse_without_separator_puts_all_in_example_items():
    md = "```python\ncode\n```\nconsole text"
    result = parse_example(md)
    assert not result.warmup_items
    assert result.example_items[0] == "code"
    assert result.example_items[1] == "console text"


def test_roundtrip_markdown_with_warmup_and_metadata():
    meta = {"name": "", "model": "gpt-4o", "num_turns": 2}
    ex = _example(["warmup = 1", ""], ["prompt", "c0", "resp", "c1"], metadata=meta)
    original = _fmt(ex, metadata=True)
    assert _fmt(parse_example(original), metadata=True) == original


def test_roundtrip_from_real_job(job_factory):
    """Round-trip starting from an actual extracted job example."""
    job = job_factory()
    job.py_env.console = ["warmup_line", "exec_line", "result_line"]
    job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="resp1 = 1"))
    job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp2 = 2"))

    ex = extract_example(job, name="")
    fmt_kwargs = {"include_warmup": True, "include_metadata": True}
    original = format_example(ex, ChatStyle.MARKDOWN, **fmt_kwargs)
    parsed = parse_example(original)
    assert format_example(parsed, ChatStyle.MARKDOWN, **fmt_kwargs) == original
