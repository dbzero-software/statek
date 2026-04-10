"""Tests for LLM_Harness limit checks."""

from unittest.mock import MagicMock

import pytest

from statek.exceptions import LLM_HarnessError
from statek.llm_harness import LLM_Harness


def _make_job(num_turns=0, exception_count=0, max_consecutive_exceptions=0,
              approx_token_usage=0):
    job = MagicMock()
    job.num_turns = num_turns
    job.exception_count = exception_count
    job.max_consecutive_exceptions = max_consecutive_exceptions
    job.approx_token_usage = approx_token_usage
    return job


class TestCheckBeforeStep:
    """Limits use 'allow N, fail on N+1' semantics."""

    def test_max_turns_allows_up_to_limit(self):
        """max_turns=3: turns 1-3 allowed, 4th rejected."""
        harness = LLM_Harness(max_turns=3, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=None)
        harness.check_before_step(_make_job(num_turns=2))
        harness.check_before_step(_make_job(num_turns=3))
        with pytest.raises(LLM_HarnessError, match="turns"):
            harness.check_before_step(_make_job(num_turns=4))

    def test_max_exceptions_allows_up_to_limit(self):
        """max_exceptions=3: 3 exceptions allowed, 4th rejected."""
        harness = LLM_Harness(max_turns=None, max_exceptions=3,
                              max_consecutive_exceptions=100, max_token_usage=None)
        harness.check_before_step(_make_job(exception_count=2))
        harness.check_before_step(_make_job(exception_count=3))
        with pytest.raises(LLM_HarnessError, match="exceptions"):
            harness.check_before_step(_make_job(exception_count=4))

    def test_max_consecutive_exceptions_allows_up_to_limit(self):
        """max_consecutive_exceptions=2: 2 consecutive allowed, 3rd rejected."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=2, max_token_usage=None)
        harness.check_before_step(_make_job(max_consecutive_exceptions=1))
        harness.check_before_step(_make_job(max_consecutive_exceptions=2))
        with pytest.raises(LLM_HarnessError, match="consecutive"):
            harness.check_before_step(_make_job(max_consecutive_exceptions=3))

    def test_no_limits_does_not_raise(self):
        """With None limits, nothing should raise."""
        harness = LLM_Harness(max_turns=None, max_exceptions=None,
                              max_consecutive_exceptions=None, max_token_usage=None)
        harness.check_before_step(_make_job(num_turns=999, exception_count=999,
                                            max_consecutive_exceptions=999))


class TestCheckAfterStep:
    """check_after_step should raise only when token usage exceeds the limit."""

    def test_token_usage_allows_equal(self):
        """Exactly at the limit should be fine; over should raise."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=1000)
        harness.check_after_step(_make_job(approx_token_usage=1000))
        with pytest.raises(LLM_HarnessError, match="token"):
            harness.check_after_step(_make_job(approx_token_usage=1001))

    def test_no_token_limit_does_not_raise(self):
        """With None token limit, high usage should not raise."""
        harness = LLM_Harness(max_turns=None, max_exceptions=None,
                              max_consecutive_exceptions=None, max_token_usage=None)
        harness.check_after_step(_make_job(approx_token_usage=999_999))

    def test_max_exceptions_checked_after_step(self):
        """check_after_step should also enforce max_exceptions."""
        harness = LLM_Harness(max_turns=None, max_exceptions=3,
                              max_consecutive_exceptions=100, max_token_usage=None)
        harness.check_after_step(_make_job(exception_count=3))
        with pytest.raises(LLM_HarnessError, match="exceptions"):
            harness.check_after_step(_make_job(exception_count=4))

    def test_max_consecutive_exceptions_checked_after_step(self):
        """check_after_step should also enforce max_consecutive_exceptions."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=2, max_token_usage=None)
        harness.check_after_step(_make_job(max_consecutive_exceptions=2))
        with pytest.raises(LLM_HarnessError, match="consecutive"):
            harness.check_after_step(_make_job(max_consecutive_exceptions=3))
