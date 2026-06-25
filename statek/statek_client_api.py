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

from typing import Any, Dict, List, Optional, Union
import dbzero as db0
from .rpc_integration import rpc
from .agents.agent import SupervisedAgent
from .executors.job import Job
from .locale import StatekLocale, resolve_locale
from .task import submit_new_job as _submit_new_job
from .task import submit_new_jobs_batch as _submit_new_jobs_batch


@db0.memo(singleton=True)
class StatekClientAPI:
    """Client API for creating STATEK jobs from external processes."""

    @rpc.remote
    def submit_new_job(
        self,
        agent: SupervisedAgent,
        shared_vars: Optional[Dict[str, Any]] = None,
        locale: Optional[Union[StatekLocale, str]] = None,
        **kwargs
    ) -> Job:
        """Create a new STATEK job for the given agent.

        Args:
            agent: The agent (must inherit from SupervisedAgent).
            shared_vars: Optional ``{var_name: object}`` mapping of
                variables to share with the job.
            locale: Optional locale for job execution.  May be a
                :class:`~statek.locale.StatekLocale` instance or a
                ``"LANG-COUNTRY"`` string (e.g. ``"PL-PL"``).  Strings
                are resolved on the writer side via
                :func:`~statek.locale.resolve_locale`.
            kwargs: Agent-specific job parameters.

        Returns:
            The newly created Job instance.
        """
        if isinstance(locale, str):
            locale = resolve_locale(locale)
        return _submit_new_job(
            agent, shared_vars=shared_vars, locale=locale, **kwargs
        )

    @rpc.remote
    def submit_new_jobs_batch(
        self,
        agent: SupervisedAgent,
        shared_vars_batch: List[Optional[Dict[str, Any]]],
        locale: Optional[Union[StatekLocale, str]] = None,
        **kwargs
    ) -> List[Job]:
        """Create multiple STATEK jobs with different shared variables.

        Args:
            agent: The agent (must inherit from SupervisedAgent).
            shared_vars_batch: A list of shared_vars entries — one per job.
            locale: Optional locale for job execution.  May be a
                :class:`~statek.locale.StatekLocale` instance or a
                ``"LANG-COUNTRY"`` string (e.g. ``"PL-PL"``).  Strings
                are resolved on the writer side via
                :func:`~statek.locale.resolve_locale`.
            kwargs: Agent-specific job parameters (same for all jobs).

        Returns:
            A list of newly created Job instances.
        """
        if isinstance(locale, str):
            locale = resolve_locale(locale)
        return _submit_new_jobs_batch(
            agent,
            shared_vars_batch=shared_vars_batch,
            locale=locale,
            **kwargs
        )
