"""Public testing helpers for Statek execution context setup."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from statek.utils import _statek_ctx_for_job, _statek_ctx_scope


@contextmanager
def statek_context(job: Optional[Any] = None, agent: Optional[Any] = None) -> Iterator[dict]:
    """Establish a Statek execution context for focused tests.

    Args:
        job: Optional job to expose through Statek current-job helpers.
        agent: Optional agent override to expose through Statek current-agent helpers.

    Yields:
        The context dictionary active for the test scope.
    """
    ctx = _statek_ctx_for_job(job) if job is not None else {}
    if agent is not None:
        ctx["agent"] = agent

    with _statek_ctx_scope(ctx):
        yield ctx
