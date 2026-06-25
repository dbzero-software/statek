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
