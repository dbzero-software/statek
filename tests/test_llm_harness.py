"""Tests for LLM_Harness limit checks."""

from unittest.mock import MagicMock

import pytest

from statek.exceptions import LLM_HarnessError
from statek.llm_harness import LLM_Harness


def _make_job(num_turns=0, exception_count=0, max_consecutive_exceptions=0,
              approx_token_usage=0, num_completions=None):
    job = MagicMock()
    job.num_turns = num_turns
    job.exception_count = exception_count
    job.max_consecutive_exceptions = max_consecutive_exceptions
    job.approx_token_usage = approx_token_usage
    job.num_completions = num_completions
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


class TestEffectiveLimit:  # pylint: disable=protected-access
    """Tests for _effective_limit calculation.

    Formula: limit = base_limit + (1.0 + extension) * num_completions
    """

    def test_none_base_returns_none(self):
        """Unlimited base stays unlimited regardless of completions."""
        harness = LLM_Harness(max_turns=None, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        assert harness._effective_limit(None, 5) is None

    def test_none_num_completions_returns_base(self):
        """When num_completions is None, return base_limit unchanged."""
        harness = LLM_Harness(max_turns=5, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        assert harness._effective_limit(5, None) == 5

    def test_zero_completions_returns_base(self):
        """Zero completions should not extend the limit."""
        harness = LLM_Harness(max_turns=5, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        assert harness._effective_limit(5, 0) == 5

    def test_one_completion_extends_by_one_plus_fraction(self):
        """1 completion with extension=0.5 adds 1.5 to base."""
        harness = LLM_Harness(max_turns=5, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        assert harness._effective_limit(5, 1) == 6.5

    def test_two_completions(self):
        """2 completions with extension=0.5 adds 3.0 to base."""
        harness = LLM_Harness(max_turns=5, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        assert harness._effective_limit(5, 2) == 8.0

    def test_zero_extension(self):
        """Extension=0.0 means each completion adds exactly 1.0."""
        harness = LLM_Harness(max_turns=10, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.0)
        assert harness._effective_limit(10, 3) == 13.0


class TestLimitExtensionBeforeStep:
    """check_before_step uses extended limits when num_completions is set."""

    def test_turns_extended_after_completions(self):
        """max_turns=3, extension=0.5, 2 completions → effective=6.0."""
        harness = LLM_Harness(max_turns=3, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        # 4 turns would fail without completions, but with 2 completions effective=6.0
        harness.check_before_step(_make_job(num_turns=4, num_completions=2))
        harness.check_before_step(_make_job(num_turns=6, num_completions=2))
        with pytest.raises(LLM_HarnessError, match="turns"):
            harness.check_before_step(_make_job(num_turns=7, num_completions=2))

    def test_exceptions_extended_after_completions(self):
        """max_exceptions=3, extension=0.5, 1 completion → effective=4.5."""
        harness = LLM_Harness(max_turns=None, max_exceptions=3,
                              max_consecutive_exceptions=100, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        # 4 exceptions OK with 1 completion (effective=4.5)
        harness.check_before_step(_make_job(exception_count=4, num_completions=1))
        with pytest.raises(LLM_HarnessError, match="exceptions"):
            harness.check_before_step(_make_job(exception_count=5, num_completions=1))

    def test_consecutive_exceptions_extended(self):
        """max_consecutive_exceptions=1, extension=0.0, 2 completions → effective=3.0."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=1, max_token_usage=None,
                              limit_extension_per_completion=0.0)
        harness.check_before_step(_make_job(max_consecutive_exceptions=3, num_completions=2))
        with pytest.raises(LLM_HarnessError, match="consecutive"):
            harness.check_before_step(_make_job(max_consecutive_exceptions=4, num_completions=2))

    def test_none_completions_uses_base_limits(self):
        """When num_completions is None, base limits apply (backward compat)."""
        harness = LLM_Harness(max_turns=3, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        harness.check_before_step(_make_job(num_turns=3, num_completions=None))
        with pytest.raises(LLM_HarnessError, match="turns"):
            harness.check_before_step(_make_job(num_turns=4, num_completions=None))


class TestLimitExtensionAfterStep:
    """check_after_step uses extended limits when num_completions is set."""

    def test_token_usage_extended_after_completions(self):
        """max_token_usage=1000, extension=0.5, 2 completions → effective=1003.0."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=1000,
                              limit_extension_per_completion=0.5)
        harness.check_after_step(_make_job(approx_token_usage=1003, num_completions=2))
        with pytest.raises(LLM_HarnessError, match="token"):
            harness.check_after_step(_make_job(approx_token_usage=1004, num_completions=2))

    def test_exceptions_extended_after_step(self):
        """max_exceptions=3, extension=0.5, 1 completion → effective=4.5."""
        harness = LLM_Harness(max_turns=None, max_exceptions=3,
                              max_consecutive_exceptions=100, max_token_usage=None,
                              limit_extension_per_completion=0.5)
        harness.check_after_step(_make_job(exception_count=4, num_completions=1))
        with pytest.raises(LLM_HarnessError, match="exceptions"):
            harness.check_after_step(_make_job(exception_count=5, num_completions=1))

    def test_none_completions_uses_base_token_limit(self):
        """When num_completions is None, base token limit applies."""
        harness = LLM_Harness(max_turns=None, max_exceptions=100,
                              max_consecutive_exceptions=100, max_token_usage=1000,
                              limit_extension_per_completion=0.5)
        harness.check_after_step(_make_job(approx_token_usage=1000, num_completions=None))
        with pytest.raises(LLM_HarnessError, match="token"):
            harness.check_after_step(_make_job(approx_token_usage=1001, num_completions=None))


class TestDefaultExtension:
    """LLM_Harness without limit_extension_per_completion defaults to 0.0."""

    def test_default_extension_is_zero(self):
        harness = LLM_Harness(max_turns=5, max_exceptions=3,
                              max_consecutive_exceptions=1, max_token_usage=1000)
        assert harness.limit_extension_per_completion == 0.0
