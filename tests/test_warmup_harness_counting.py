"""Tests for harness counting: warmup exclusions and tool-exception inclusion."""
# pylint: disable=no-member,redefined-outer-name

from typing import Callable

import dbzero as db0
import pytest

from statek.executors.job import Job, JobStatus
from statek.executors.chat_log_item import LLM_LogItem, ToolError, WarmupLogItem
from statek.llm_harness import LLM_Harness
from statek.pyenv import Error, ErrorKind
from tests.conftest import DB0_DIR


@pytest.fixture
def make_job(job_def_factory):
    """Create a Job with a given chat_log and exceptions dict."""
    def _make(chat_log_items, exceptions=None):
        job_def = job_def_factory()
        job = Job(
            job_def=job_def,
            model_family="test",
            model="test-model",
            job_status=JobStatus.STARTED,
        )
        for item in chat_log_items:
            job.chat_log.append(item)
        if exceptions:
            job.py_env.exceptions = {
                key: Error(ErrorKind.EXECUTION, message) for key, message in exceptions.items()
            }
        return job
    return _make


class TestNumTurnsExcludesWarmup:
    """num_turns should only count LLM_LogItem entries, not WarmupLogItem."""

    def test_no_items(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([])
        assert job.num_turns == 0

    def test_only_llm_items(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [LLM_LogItem(console_pos=0), LLM_LogItem(console_pos=10)]
        job = make_job(items)
        assert job.num_turns == 2

    def test_only_warmup_items(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            WarmupLogItem(console_pos=5, warmup_block_num=1),
        ]
        job = make_job(items)
        assert job.num_turns == 0

    def test_mixed_warmup_and_llm(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """5 warmup + 3 LLM items should report num_turns == 3."""
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            WarmupLogItem(console_pos=1, warmup_block_num=1),
            WarmupLogItem(console_pos=2, warmup_block_num=2),
            WarmupLogItem(console_pos=3, warmup_block_num=3),
            WarmupLogItem(console_pos=4, warmup_block_num=4),
            LLM_LogItem(console_pos=5),
            LLM_LogItem(console_pos=6),
            LLM_LogItem(console_pos=7),
        ]
        job = make_job(items)
        assert job.num_turns == 3


class TestExceptionCountExcludesWarmup:
    """exception_count should not include exceptions from warmup blocks."""

    @pytest.mark.parametrize("message", [
        "LLM_HarnessError: Maximum token usage exceeded: 55320/50002.0",
        "LLM_HarnessError: Maximum number of exceptions exceeded: 8/6.0",
        "LLM_HarnessError: Maximum consecutive exceptions exceeded: 8/6.0",
        "LLM_HarnessError: Maximum number of turns exceeded: 101/100.0",
        "Dowolny komunikat zatrzymania",
    ])
    def test_terminal_diagnostic_preserves_shared_position_accounting(
        self, make_job: Callable[..., Job], message: str,
    ) -> None:
        """A terminal diagnostic must not duplicate events or join real-error streaks."""
        items = [LLM_LogItem(console_pos=0), LLM_LogItem(console_pos=1),
                 LLM_LogItem(console_pos=1), LLM_LogItem(console_pos=2)]
        items[0].push_tool_result(ToolError(err_message="tool error"))
        job = make_job(items)
        job.console_append("output")
        job.console_append("code error", error=Error(ErrorKind.EXECUTION, "code error"))
        job.console_append("another error", error=Error(ErrorKind.EXECUTION, "another error"))
        assert job.exception_count == 3
        assert job.max_consecutive_exceptions == 2

        job.console_append(message, error=Error(ErrorKind.HARNESS, message))

        assert job.py_env.exceptions[3].message == message
        assert job.py_env.exceptions[3].kind == ErrorKind.HARNESS
        assert job.exception_count == 3
        assert job.max_consecutive_exceptions == 2

    @pytest.mark.parametrize("with_execution_error", [False, True])
    def test_distinct_diagnostics_retain_unique_console_keys(
        self, make_job: Callable[..., Job], with_execution_error: bool,
    ) -> None:
        """Successive diagnostics cannot replace an earlier real error or each other."""
        job = make_job([LLM_LogItem(console_pos=0)])
        expected = {}
        if with_execution_error:
            job.console_append(
                "ValueError: bad code", error=Error(ErrorKind.EXECUTION, "ValueError: bad code"),
            )
            expected[0] = "ValueError: bad code"
        for message in (
            "LLM_HarnessError: Maximum token usage exceeded: 55320/50002.0",
            "LLM_HarnessError: Maximum number of turns exceeded: 101/100.0",
        ):
            expected[len(expected)] = message
            job.console_append(message, error=Error(ErrorKind.HARNESS, message))

        assert {key: error.message for key, error in job.py_env.exceptions.items()} == expected
        assert list(job.py_env.console) == list(expected.values())
        assert job.exception_count == int(with_execution_error)
        assert job.max_consecutive_exceptions == int(with_execution_error)

    @pytest.mark.parametrize("message", [
        "ValueError: Maximum token usage exceeded: application quota",
        "LLM_HarnessError: application failure",
        "LLM_HarnessError: Maximum token usage exceeded: 55320/50002.0",
    ])
    def test_application_errors_are_not_filtered(
        self, make_job: Callable[..., Job], message: str,
    ) -> None:
        """Message wording cannot exclude an application error from accounting."""
        job = make_job([LLM_LogItem(console_pos=0)])
        job.console_append(message, error=Error(ErrorKind.EXECUTION, message))
        assert job.exception_count == 1
        assert job.max_consecutive_exceptions == 1

    def test_shared_position_counts_one_event(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([LLM_LogItem(console_pos=1) for _ in range(8)],
                       exceptions={1: "ValueError: bad code"})
        assert job.exception_count == 1
        assert job.max_consecutive_exceptions == 1

    def test_new_errors_have_unique_console_keys(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([LLM_LogItem(console_pos=0)])
        job.console_append("output")
        job.console_append("first", error=Error(ErrorKind.EXECUTION, "same error"))
        job.console_append("second", error=Error(ErrorKind.EXECUTION, "same error"))
        assert {key: error.message for key, error in job.py_env.exceptions.items()} == {
            1: "same error", 2: "same error",
        }
        assert job.exception_count == 2
        assert job.max_consecutive_exceptions == 1

    def test_terminal_diagnostics_do_not_poison_resume(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([LLM_LogItem(console_pos=0) for _ in range(8)])
        for message in ("Token budget exhausted", "Execution stopped"):
            job.console_append(message, error=Error(ErrorKind.HARNESS, message))
        assert job.exception_count == 0
        assert job.max_consecutive_exceptions == 0
        job.set_status(JobStatus.DONE)
        job.push_user_message("continue")
        LLM_Harness(None, 6, 6, None).check_before_step(job)

    def test_shared_warmup_boundary_has_single_owner(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([LLM_LogItem(console_pos=0),
                        WarmupLogItem(console_pos=0, warmup_block_num=0)],
                       exceptions={0: "warmup error"})
        assert job.exception_count == 0
        assert job.max_consecutive_exceptions == 0

    def test_diagnostic_classification_survives_reopen(self, make_job: Callable[..., Job]) -> None:
        """Persist provenance independently of identical diagnostic and error text."""
        job = make_job([LLM_LogItem(console_pos=0)])
        message = "LLM_HarnessError: Maximum token usage exceeded"
        job.console_append(message, error=Error(ErrorKind.HARNESS, message))
        job.console_append(message, error=Error(ErrorKind.EXECUTION, message))
        job_id = db0.uuid(job)
        db0.close()
        db0.init(DB0_DIR, read_write=True)
        db0.open("test_prefix", "rw")
        restored = db0.fetch(job_id)
        assert list(restored.py_env.console) == [message, message]
        assert restored.py_env.exceptions[0].kind == ErrorKind.HARNESS
        assert restored.py_env.exceptions[1].kind == ErrorKind.EXECUTION
        assert [error.message for error in restored.py_env.exceptions.values()] == [
            message, message,
        ]
        assert restored.exception_count == 1
        assert restored.max_consecutive_exceptions == 1

    def test_success_breaks_shared_position_streak(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [LLM_LogItem(console_pos=0), LLM_LogItem(console_pos=1),
                 LLM_LogItem(console_pos=1), LLM_LogItem(console_pos=2)]
        items[0].push_tool_result(ToolError(err_message="tool error"))
        job = make_job(items, exceptions={1: "code error"})
        assert job.exception_count == 2
        assert job.max_consecutive_exceptions == 1

    def test_no_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        job = make_job([LLM_LogItem(console_pos=0)])
        assert job.exception_count == 0

    def test_only_llm_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            LLM_LogItem(console_pos=5),
            LLM_LogItem(console_pos=10),
        ]
        # Exceptions keyed by console_pos of LLM items
        job = make_job(items, exceptions={5: "err1", 10: "err2"})
        assert job.exception_count == 2

    def test_only_warmup_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            WarmupLogItem(console_pos=5, warmup_block_num=1),
            LLM_LogItem(console_pos=10),
        ]
        # Exceptions keyed by console_pos of warmup items
        job = make_job(items, exceptions={0: "err1", 5: "err2"})
        assert job.exception_count == 0

    def test_mixed_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """Warmup exceptions should not count; only LLM exceptions should."""
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            WarmupLogItem(console_pos=1, warmup_block_num=1),
            WarmupLogItem(console_pos=2, warmup_block_num=2),
            LLM_LogItem(console_pos=3),
            LLM_LogItem(console_pos=4),
            LLM_LogItem(console_pos=5),
        ]
        # Exceptions keyed by console_pos: 0,1 (warmup) and 4 (LLM)
        job = make_job(items, exceptions={0: "warmup_err", 1: "warmup_err", 4: "llm_err"})
        assert job.exception_count == 1


class TestMaxConsecutiveExceptionsExcludesWarmup:
    """max_consecutive_exceptions should only consider LLM items."""

    def test_consecutive_llm_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            LLM_LogItem(console_pos=1),
            LLM_LogItem(console_pos=2),
            LLM_LogItem(console_pos=3),
        ]
        # Exceptions keyed by console_pos of LLM items at pos 1,2
        job = make_job(items, exceptions={1: "e1", 2: "e2"})
        assert job.max_consecutive_exceptions == 2

    def test_warmup_exceptions_not_consecutive_with_llm(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """Warmup exception followed by LLM exception is NOT consecutive."""
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            LLM_LogItem(console_pos=1),
            LLM_LogItem(console_pos=2),
        ]
        # Exception keyed by console_pos: warmup at 0, LLM at 1
        job = make_job(items, exceptions={0: "warmup", 1: "llm"})
        assert job.max_consecutive_exceptions == 1

    def test_warmup_exceptions_only(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            WarmupLogItem(console_pos=0, warmup_block_num=0),
            WarmupLogItem(console_pos=1, warmup_block_num=1),
            LLM_LogItem(console_pos=2),
        ]
        # Exceptions keyed by console_pos of warmup items
        job = make_job(items, exceptions={0: "e1", 1: "e2"})
        assert job.max_consecutive_exceptions == 0


class TestExceptionCountIncludesToolExceptions:
    """exception_count should include tool-call exceptions from ChatLogItem.tool_exceptions."""

    def test_tool_exception_on_llm_item_counted(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A tool exception on an LLM turn counts toward exception_count."""
        llm_item = LLM_LogItem(console_pos=0)
        llm_item.push_tool_result(ToolError(err_message="ValueError: boom"))
        job = make_job([llm_item])
        assert job.exception_count == 1

    def test_tool_exception_on_warmup_not_counted(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A tool exception on a warmup item does NOT count toward exception_count."""
        warmup = WarmupLogItem(console_pos=0, warmup_block_num=0)
        warmup.push_tool_result(ToolError(err_message="RuntimeError: warmup err"))
        llm_item = LLM_LogItem(console_pos=1)
        job = make_job([warmup, llm_item])
        assert job.exception_count == 0

    def test_code_and_tool_exceptions_both_counted(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A turn with both a code exception and a tool exception counts as 2."""
        llm_item = LLM_LogItem(console_pos=5)
        llm_item.push_tool_result(ToolError(err_message="ValueError: tool err"))
        job = make_job([llm_item], exceptions={5: "code err"})
        assert job.exception_count == 2

    def test_tool_exception_only_no_code_exception(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A turn with only a tool exception (no code exception) is still counted."""
        llm_item = LLM_LogItem(console_pos=5)
        llm_item.push_tool_result(ToolError(err_message="NameError: missing"))
        job = make_job([llm_item])
        # No py_env.exceptions at all, but tool_exceptions present
        assert job.exception_count == 1

    def test_multiple_tool_exceptions_on_single_turn(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A turn with 3 tool exceptions counts as 3."""
        llm_item = LLM_LogItem(console_pos=0)
        llm_item.push_tool_result(ToolError(err_message="err1"))
        llm_item.push_tool_result(ToolError(err_message="err2"))
        llm_item.push_tool_result(ToolError(err_message="err3"))
        job = make_job([llm_item])
        assert job.exception_count == 3

    def test_mixed_turns_some_with_tool_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """Only turns that have exceptions (code or tool) are counted."""
        items = [
            LLM_LogItem(console_pos=0),  # no exception
            LLM_LogItem(console_pos=1),  # tool exception only
            LLM_LogItem(console_pos=2),  # no exception
        ]
        items[1].push_tool_result(ToolError(err_message="err"))
        job = make_job(items)
        assert job.exception_count == 1


class TestMaxConsecutiveIncludesToolExceptions:
    """max_consecutive_exceptions should include tool-call exceptions."""

    def test_consecutive_tool_exceptions(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        items = [
            LLM_LogItem(console_pos=0),
            LLM_LogItem(console_pos=1),
            LLM_LogItem(console_pos=2),
        ]
        items[0].push_tool_result(ToolError(err_message="e1"))
        items[1].push_tool_result(ToolError(err_message="e2"))
        job = make_job(items)
        assert job.max_consecutive_exceptions == 2

    def test_tool_exception_breaks_clean_streak(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """A tool exception between clean turns resets the streak."""
        items = [
            LLM_LogItem(console_pos=0),  # clean
            LLM_LogItem(console_pos=1),  # tool exception
            LLM_LogItem(console_pos=2),  # clean
        ]
        items[1].push_tool_result(ToolError(err_message="err"))
        job = make_job(items)
        assert job.max_consecutive_exceptions == 1

    def test_code_and_tool_exception_mix(self, db0_fixture, make_job):  # pylint: disable=unused-argument
        """Code exception followed by tool exception counts as 2 consecutive."""
        items = [
            LLM_LogItem(console_pos=0),  # code exception
            LLM_LogItem(console_pos=1),  # tool exception
            LLM_LogItem(console_pos=2),  # clean
        ]
        items[1].push_tool_result(ToolError(err_message="tool err"))
        job = make_job(items, exceptions={0: "code err"})
        assert job.max_consecutive_exceptions == 2
