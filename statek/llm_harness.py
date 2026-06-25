# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        limit_extension_per_completion: Fraction by which limits are extended
            after each job completion. Formula:
            effective = base + (1.0 + extension) * num_completions
    """

    def __init__(self, max_turns: Optional[int], max_exceptions: int,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 max_consecutive_exceptions: int, max_token_usage: Optional[int],
                 limit_extension_per_completion: float = 0.0):
        self.max_turns = max_turns
        self.max_exceptions = max_exceptions
        self.max_consecutive_exceptions = max_consecutive_exceptions
        self.max_token_usage = max_token_usage
        self.limit_extension_per_completion = limit_extension_per_completion

    def _effective_limit(self, base_limit: Optional[int],
                         num_completions: Optional[int]) -> Optional[float]:
        """Compute the effective limit after job completions.

        Formula: base_limit + (1.0 + limit_extension_per_completion) * num_completions

        Returns None when base_limit is None (unlimited), or base_limit
        unchanged when num_completions is None (not tracked).
        """
        if base_limit is None:
            return None
        if num_completions is None:
            return base_limit
        return base_limit + (1.0 + self.limit_extension_per_completion) * num_completions

    def _check_exception_limits(self, job, num_completions):
        """Check exception-count limits shared by before- and after-step checks."""
        eff_exceptions = self._effective_limit(self.max_exceptions, num_completions)
        eff_consecutive = self._effective_limit(
            self.max_consecutive_exceptions, num_completions)

        if eff_exceptions is not None and job.exception_count > eff_exceptions:
            raise LLM_HarnessError(
                f"Maximum number of exceptions exceeded: "
                f"{job.exception_count}/{eff_exceptions}")
        if eff_consecutive is not None and job.max_consecutive_exceptions > eff_consecutive:
            raise LLM_HarnessError(
                f"Maximum consecutive exceptions exceeded: "
                f"{job.max_consecutive_exceptions}/{eff_consecutive}")

    def check_before_step(self, job):
        """Check constraints before executing a job step.

        Raises:
            LLM_HarnessError: If any pre-step limit has been exceeded.
        """
        nc = getattr(job, 'num_completions', None)
        eff_turns = self._effective_limit(self.max_turns, nc)

        if eff_turns is not None and job.num_turns > eff_turns:
            raise LLM_HarnessError(
                f"Maximum number of turns exceeded: {job.num_turns}/{eff_turns}")
        self._check_exception_limits(job, nc)

    def check_after_step(self, job):
        """Check constraints after executing a job step.

        Raises:
            LLM_HarnessError: If any post-step limit has been exceeded.
        """
        nc = getattr(job, 'num_completions', None)
        eff_tokens = self._effective_limit(self.max_token_usage, nc)

        if eff_tokens is not None and job.approx_token_usage > eff_tokens:
            raise LLM_HarnessError(
                f"Maximum token usage exceeded: "
                f"{job.approx_token_usage}/{eff_tokens}")
        self._check_exception_limits(job, nc)


@lru_cache()
def get_llm_harness() -> LLM_Harness:
    """Get the LLM_Harness instance initialized from StatekSettings."""
    settings = get_statek_settings()
    kwargs = {
        'max_turns': settings.max_turns,
        'max_exceptions': settings.max_exceptions,
        'max_consecutive_exceptions': settings.max_consecutive_exceptions,
        'max_token_usage': settings.max_token_usage,
        'limit_extension_per_completion': settings.limit_extension_per_completion,
    }
    harness = LLM_Harness(**kwargs)
    logger.info("LLM_Harness initialized: %s",
                ", ".join(f"{k}={v}" for k, v in kwargs.items()))
    return harness
