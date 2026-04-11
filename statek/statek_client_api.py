from typing import Any, Dict, List, Optional
import dbzero as db0
from .rpc_integration import rpc
from .agents.agent import SupervisedAgent
from .executors.job import Job
from .locale import StatekLocale
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
        locale: Optional[StatekLocale] = None,
        **kwargs
    ) -> Job:
        """Create a new STATEK job for the given agent.

        Args:
            agent: The agent (must inherit from SupervisedAgent).
            shared_vars: Optional ``{var_name: object}`` mapping of
                variables to share with the job.
            locale: Optional locale for job execution.
            kwargs: Agent-specific job parameters.

        Returns:
            The newly created Job instance.
        """
        return _submit_new_job(
            agent, shared_vars=shared_vars, locale=locale, **kwargs
        )

    @rpc.remote
    def submit_new_jobs_batch(
        self,
        agent: SupervisedAgent,
        shared_vars_batch: List[Optional[Dict[str, Any]]],
        locale: Optional[StatekLocale] = None,
        **kwargs
    ) -> List[Job]:
        """Create multiple STATEK jobs with different shared variables.

        Args:
            agent: The agent (must inherit from SupervisedAgent).
            shared_vars_batch: A list of shared_vars entries — one per job.
            locale: Optional locale for job execution.
            kwargs: Agent-specific job parameters (same for all jobs).

        Returns:
            A list of newly created Job instances.
        """
        return _submit_new_jobs_batch(
            agent,
            shared_vars_batch=shared_vars_batch,
            locale=locale,
            **kwargs
        )
