"""LLM Harness - guards against catastrophic situations such as excessive token usage."""

import logging
from functools import lru_cache
from typing import Optional

from .exceptions import LLM_HarnessError
from .settings import get_statek_settings

logger = logging.getLogger(__name__)


class LLM_Harness:
    """Guards against catastrophic situations such as excessive token usage.

    Currently defined globally (singleton) and applied to all LLM_API instances.
    In the future it may be scoped per-provider, model, or agent type.

    Args:
        max_turns: Maximum number of allowed LLM turns (None for unlimited)
        max_exceptions: Maximum total number of exceptions per conversation
        max_consecutive_exceptions: Maximum allowed consecutive exceptions
        max_token_usage: Total allowed maximum token usage (None for unlimited)
    """

    def __init__(self, max_turns: Optional[int], max_exceptions: int,
                 max_consecutive_exceptions: int, max_token_usage: Optional[int]):
        self.max_turns = max_turns
        self.max_exceptions = max_exceptions
        self.max_consecutive_exceptions = max_consecutive_exceptions
        self.max_token_usage = max_token_usage

    def check_before_step(self, job):
        """Check constraints before executing a job step.

        Raises:
            LLM_HarnessError: If any pre-step limit has been exceeded.
        """
        if self.max_turns is not None and job.num_turns > self.max_turns:
            raise LLM_HarnessError(
                f"Maximum number of turns exceeded: {job.num_turns}/{self.max_turns}")
        if self.max_exceptions is not None and job.exception_count > self.max_exceptions:
            raise LLM_HarnessError(
                f"Maximum number of exceptions exceeded: {job.exception_count}/{self.max_exceptions}") # pylint: disable=line-too-long
        if (self.max_consecutive_exceptions is not None
                and job.max_consecutive_exceptions > self.max_consecutive_exceptions):
            raise LLM_HarnessError(
                f"Maximum consecutive exceptions exceeded: "
                f"{job.max_consecutive_exceptions}/{self.max_consecutive_exceptions}")

    def check_after_step(self, job):
        """Check constraints after executing a job step.

        Raises:
            LLM_HarnessError: If any post-step limit has been exceeded.
        """
        if self.max_token_usage is not None and job.approx_token_usage > self.max_token_usage:
            raise LLM_HarnessError(
                f"Maximum token usage exceeded: {job.approx_token_usage}/{self.max_token_usage}")


@lru_cache()
def get_llm_harness() -> LLM_Harness:
    """Get the LLM_Harness instance initialized from StatekSettings."""
    settings = get_statek_settings()
    harness = LLM_Harness(
        max_turns=settings.max_turns,
        max_exceptions=settings.max_exceptions,
        max_consecutive_exceptions=settings.max_consecutive_exceptions,
        max_token_usage=settings.max_token_usage,
    )
    logger.info(
        "LLM_Harness initialized: max_turns=%s, max_exceptions=%s, "
        "max_consecutive_exceptions=%s, max_token_usage=%s",
        harness.max_turns,
        harness.max_exceptions,
        harness.max_consecutive_exceptions,
        harness.max_token_usage,
    )
    return harness
