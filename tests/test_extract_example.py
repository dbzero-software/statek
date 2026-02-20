"""Tests for extract_example and format_example functions."""

# pylint: disable=no-member

import pytest

from tests.conftest import create_chat_log_item
from statek.executors.example import Example, extract_example, format_example
from statek.settings import ChatStyle


class TestExtractExampleMetadata:
    """Tests for example_metadata field."""

    def test_name_is_set(self, job_factory):
        result = extract_example(job_factory(), "my_example")
        assert result.example_metadata["name"] == "my_example"

    def test_model_and_model_family(self, job_factory):
        job = job_factory(model_family="OpenAI", model="gpt-4")
        result = extract_example(job, "x")
        assert result.example_metadata["model"] == "gpt-4"
        assert result.example_metadata["model_family"] == "OpenAI"

    def test_num_turns_zero(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.example_metadata["num_turns"] == 0

    def test_num_turns_reflects_chat_log(self, job_factory):
        job = job_factory()
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 1"))
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 2"))
        result = extract_example(job, "x")
        assert result.example_metadata["num_turns"] == 2

    def test_exception_count_zero(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.example_metadata["exception_count"] == 0

    def test_exit_status_none(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.example_metadata["exit_status"] is None

    def test_exit_status_set(self, job_factory):
        job = job_factory()
        job.py_env.exit_status = "done"
        result = extract_example(job, "x")
        assert result.example_metadata["exit_status"] == "done"

    def test_agent_role_included(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.example_metadata["agent_role"] == "test"

    def test_job_params_included_when_set(self, job_def_factory, job_factory):
        job = job_factory(job_params={"goal": "analyze"})
        result = extract_example(job, "x")
        assert result.example_metadata["job_params"] == {"goal": "analyze"}

    def test_job_params_absent_when_none(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert "job_params" not in result.example_metadata

    def test_returns_example_instance(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert isinstance(result, Example)


class TestExtractWarmupItems:
    """Tests for warmup_items field."""

    def test_no_warmup_code_returns_empty(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.warmup_items == []

    def test_single_block_no_console(self, job_def_factory):
        job_def = job_def_factory(warmup_code="x = 1")
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        result = extract_example(job, "x")
        assert result.warmup_items == ["x = 1", ""]

    def test_single_block_with_console_no_chat_log(self, job_def_factory):
        job_def = job_def_factory(warmup_code="print('hi')")
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        job.py_env.console_append("hi")
        result = extract_example(job, "x")
        assert result.warmup_items == ["print('hi')", "hi"]

    def test_single_block_console_boundary_from_chat_log(self, job_def_factory):
        """console_pos of first chat item marks end of warmup console."""
        job_def = job_def_factory(warmup_code="x = 1")
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        job.py_env.console = ["warmup_out", "llm_out"]
        # First LLM turn recorded after warmup_out (pos=1)
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="y = 2"))
        result = extract_example(job, "x")
        # Only "warmup_out" belongs to warmup
        assert result.warmup_items == ["x = 1", "warmup_out"]

    def test_single_block_multiple_warmup_console_lines(self, job_def_factory):
        """Multiple console lines from a single warmup block are joined."""
        job_def = job_def_factory(warmup_code="print(1)\nprint(2)\nprint(3)")
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        job.py_env.console = ["1", "2", "3"]
        result = extract_example(job, "x")
        assert result.warmup_items == ["print(1)\nprint(2)\nprint(3)", "1\n2\n3"]

    def test_multiple_warmup_blocks_last_gets_console(self, job_def_factory):
        """Only the last warmup block receives the accumulated console output."""
        job_def = job_def_factory(warmup_code=["a = 1", "b = 2", "print(b)"])
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        job.py_env.console_append("2")
        result = extract_example(job, "x")
        assert result.warmup_items[0] == "a = 1"
        assert result.warmup_items[1] == ""      # no per-block console
        assert result.warmup_items[2] == "b = 2"
        assert result.warmup_items[3] == ""      # no per-block console
        assert result.warmup_items[4] == "print(b)"
        assert result.warmup_items[5] == "2"

    def test_multiple_warmup_blocks_length(self, job_def_factory):
        """warmup_items has exactly 2*N entries for N warmup blocks."""
        job_def = job_def_factory(warmup_code=["a = 1", "b = 2"])
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        result = extract_example(job, "x")
        assert len(result.warmup_items) == 4

    def test_warmup_items_no_formatting_characters(self, job_def_factory):
        """warmup_items must not contain markdown fences or '> ' prefixes."""
        job_def = job_def_factory(warmup_code="x = 1\nprint(x)")
        from statek.executors.job import Job, JobStatus
        job = Job(job_def=job_def, model_family="t", model="t", job_status=JobStatus.READY)
        job.py_env.console_append("1")
        result = extract_example(job, "x")
        for item in result.warmup_items:
            assert "```" not in item
            assert not item.startswith("> ")


class TestExtractExampleItems:
    """Tests for example_items field."""

    def test_no_chat_log_no_console(self, job_factory):
        result = extract_example(job_factory(), "x")
        assert result.example_items == ["Test task", ""]

    def test_no_chat_log_with_console(self, job_factory):
        job = job_factory()
        job.py_env.console = ["out1", "out2"]
        result = extract_example(job, "x")
        assert result.example_items == ["Test task", "out1\nout2"]

    def test_single_turn_structure(self, job_factory):
        """Single LLM turn → [prompt, pre_console, llm_resp, post_console]."""
        job = job_factory()
        job.py_env.console = ["warmup_line", "exec_line"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="y = 42"))
        result = extract_example(job, "x")
        assert len(result.example_items) == 4
        assert result.example_items[0] == "Test task"
        assert result.example_items[1] == "warmup_line"
        assert result.example_items[2] == "y = 42"
        assert result.example_items[3] == "exec_line"

    def test_single_turn_multiple_pre_console_lines(self, job_factory):
        """Multiple console lines before first LLM turn are joined."""
        job = job_factory()
        job.py_env.console = ["line1", "line2", "line3", "after_llm"]
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="code"))
        result = extract_example(job, "x")
        assert result.example_items[1] == "line1\nline2\nline3"

    def test_single_turn_multiple_post_console_lines(self, job_factory):
        """Multiple console lines produced after one LLM response are joined."""
        job = job_factory()
        job.py_env.console = ["before", "after1", "after2", "after3"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code"))
        result = extract_example(job, "x")
        assert result.example_items[3] == "after1\nafter2\nafter3"

    def test_multiple_turns_console_correctly_segmented(self, job_factory):
        """Each turn's console slice is bounded by adjacent console_pos values."""
        job = job_factory()
        job.py_env.console = ["w", "t1a", "t1b", "t2a", "t2b", "t2c"]
        #                       0    1      2      3      4      5
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp2"))
        result = extract_example(job, "x")
        # [prompt, console[0:1], resp1, console[1:3], resp2, console[3:6]]
        assert result.example_items[0] == "Test task"
        assert result.example_items[1] == "w"
        assert result.example_items[2] == "resp1"
        assert result.example_items[3] == "t1a\nt1b"
        assert result.example_items[4] == "resp2"
        assert result.example_items[5] == "t2a\nt2b\nt2c"

    def test_three_turns_length(self, job_factory):
        """N LLM turns → 2 + 2*N items (prompt + initial_console + N*(code+console))."""
        job = job_factory()
        job.py_env.console = ["c0", "c1", "c2", "c3"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="r1"))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="r2"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="r3"))
        result = extract_example(job, "x")
        assert len(result.example_items) == 8  # 2 + 2*3

    def test_empty_console_between_turns_is_empty_string(self, job_factory):
        """If no console output between turns, item is empty string not omitted."""
        job = job_factory()
        job.py_env.console = ["c0"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="r1"))
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="r2"))
        result = extract_example(job, "x")
        assert result.example_items[3] == ""   # no new console between r1 and r2

    def test_last_turn_trailing_console_to_end(self, job_factory):
        """Last LLM response collects all console lines after its console_pos."""
        job = job_factory()
        job.py_env.console = ["pre", "post1", "post2", "post3"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code"))
        result = extract_example(job, "x")
        assert result.example_items[3] == "post1\npost2\npost3"

    def test_example_items_no_formatting_characters(self, job_factory):
        """example_items must contain plain text without markdown or '> ' prefixes."""
        job = job_factory()
        job.py_env.console = ["output line"]
        job.chat_log.append(create_chat_log_item(
            console_pos=1, llm_resp="print('hello')"
        ))
        result = extract_example(job, "x")
        for item in result.example_items:
            assert "```" not in item
            assert not item.startswith("> ")

    def test_example_items_first_element_is_code(self, job_factory):
        """First element of example_items is always the prompt (code block)."""
        job = job_factory()
        result = extract_example(job, "x")
        assert result.example_items[0] == "Test task"

    def test_example_items_even_length(self, job_factory):
        """example_items always has an even number of elements (pairs of code+console)."""
        job = job_factory()
        job.py_env.console = ["a", "b", "c"]
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="r1"))
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="r2"))
        result = extract_example(job, "x")
        assert len(result.example_items) % 2 == 0


class TestFormatExample:
    """Tests for format_example function."""

    def _make_example(self, job_factory, warmup_code=None, console=None, turns=()):
        """Helper: build an Example from a job with given console and turns."""
        from statek.executors.job import Job, JobStatus, JobDef
        if warmup_code is not None:
            from tests.conftest import create_chat_log_item as _ccli
            from statek.agents.agent import Agent
            agent = Agent(role="test", _system_prompt="Test agent",
                          _prompt_template="Test task", _tools=[])
            from statek.executors.job import JobDef, JobStatus
            job_def = JobDef(agent=agent, warmup_code=warmup_code)
            job = Job(job_def=job_def, model_family="t", model="t",
                      job_status=JobStatus.READY)
        else:
            job = job_factory()
        if console is not None:
            job.py_env.console = console
        for console_pos, llm_resp in turns:
            job.chat_log.append(create_chat_log_item(console_pos, llm_resp))
        return extract_example(job, "test")

    def test_console_style_code_is_plain(self, job_factory):
        """CONSOLE style: code blocks appear as plain text (no fences)."""
        example = self._make_example(job_factory)
        result = format_example(example, ChatStyle.CONSOLE)
        assert "```" not in result
        assert "Test task" in result

    def test_console_style_console_prefixed(self, job_factory):
        """CONSOLE style: each console line is prefixed with '> '."""
        example = self._make_example(
            job_factory,
            console=["out1", "out2"],
            turns=[(2, "x = 1")]
        )
        result = format_example(example, ChatStyle.CONSOLE)
        assert "> out1" in result
        assert "> out2" in result

    def test_markdown_style_code_fenced(self, job_factory):
        """MARKDOWN style: code blocks are wrapped in ```python fences."""
        example = self._make_example(job_factory)
        result = format_example(example, ChatStyle.MARKDOWN)
        assert "```python\nTest task\n```" in result

    def test_markdown_style_console_plain(self, job_factory):
        """MARKDOWN style: console lines are not prefixed with '> '."""
        example = self._make_example(
            job_factory,
            console=["output_line"],
            turns=[(1, "code")]
        )
        result = format_example(example, ChatStyle.MARKDOWN)
        assert "output_line" in result
        assert "> output_line" not in result

    def test_include_warmup_false_omits_warmup(self, job_factory):
        """include_warmup=False: warmup code does not appear in output."""
        example = self._make_example(job_factory, warmup_code="setup = True")
        result_with = format_example(example, ChatStyle.CONSOLE, include_warmup=True)
        result_without = format_example(example, ChatStyle.CONSOLE, include_warmup=False)
        assert "setup = True" in result_with
        assert "setup = True" not in result_without

    def test_include_metadata_true_adds_metadata(self, job_factory):
        """include_metadata=True: metadata key-value pairs appear in output."""
        example = self._make_example(job_factory)
        result = format_example(example, ChatStyle.CONSOLE, include_metadata=True)
        assert "# Example metadata:" in result
        assert "name: test" in result

    def test_include_metadata_false_omits_metadata(self, job_factory):
        """include_metadata=False (default): no metadata header in output."""
        example = self._make_example(job_factory)
        result = format_example(example, ChatStyle.CONSOLE)
        assert "# Example metadata:" not in result
