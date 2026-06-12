"""Read-only db0 access helpers for the Statek web UI."""

from typing import Iterable, Optional
import dbzero as db0

from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus


def get_all_agents() -> Iterable[Agent]:
    """Return all Agent objects from the database."""
    return db0.find(Agent)


def get_all_job_defs() -> Iterable[JobDef]:
    """Return all JobDef objects from the database."""
    return db0.find(JobDef)


def get_all_jobs() -> Iterable[Job]:
    """Return all Job objects from the database (irrespective of state)."""
    return db0.find(Job)


def get_jobs_by_status(status: JobStatus) -> Iterable[Job]:
    """Return all Job objects with a given status tag."""
    return db0.find(Job, status)


def get_job_stats() -> dict:
    """Return a dict mapping each JobStatus value to its job count."""
    return {status: len(list(get_jobs_by_status(status))) for status in JobStatus.values()}
