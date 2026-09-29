"""Behavioral tests for job notification system tools."""

# pylint: disable=no-member,protected-access

import pytest

from statek import get_pending_notifications
from statek.executors.chat_log_item import SubTaskLogItem
from statek.executors.utils import _setup_execution_context
from statek.llm_api import select_request_tools
from statek.system import find_tools, get_pending_notifications as system_get_pending_notifications
from statek.task import SubTaskHandler
from statek.utils import _statek_ctx_scope


def test_pending_notifications_empty_and_repeatable(job_factory):
    """Reading an empty pending queue never changes the job."""
    job = job_factory()
    with _statek_ctx_scope({"job": job}):
        assert not get_pending_notifications()
        assert not get_pending_notifications()
    assert job.chat_log == []


def test_pending_notifications_are_ordered_detached_and_non_consuming(job_factory):
    """Returned collection is independent while its entries retain identity."""
    job = job_factory()
    first = SubTaskLogItem(console_pos=0, handler=SubTaskHandler(job=job, id="first"))
    second = SubTaskLogItem(console_pos=0, handler=SubTaskHandler(job=job, id="second"))
    job._pending_notifications().extend([first, second])
    initial_status = job.status

    with _statek_ctx_scope({"job": job}):
        snapshot = get_pending_notifications()
        assert snapshot == [first, second]
        snapshot.clear()
        assert get_pending_notifications() == [first, second]

    assert job.status == initial_status
    assert job.chat_log == []
    assert job._pending_notifications() == [first, second]
    job._process_pending_notifications()
    with _statek_ctx_scope({"job": job}):
        assert not get_pending_notifications()


def test_pending_notifications_initializes_old_job_queue(job_factory):
    """Previously stored jobs without an initialized queue can be inspected."""
    job = job_factory()
    job._Job__pending_chat_log = None
    with _statek_ctx_scope({"job": job}):
        assert not get_pending_notifications()


def test_pending_notifications_requires_current_job():
    """Missing execution context is reported rather than treated as empty."""
    with _statek_ctx_scope({}):
        with pytest.raises(RuntimeError, match="current job"):
            get_pending_notifications()


def test_pending_notifications_system_discovery_and_scope():
    """Registered tool is selectable in system scopes but not application-only."""
    assert get_pending_notifications is system_get_pending_notifications
    assert get_pending_notifications in find_tools("SYSTEM")
    assert get_pending_notifications in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "SYSTEM"}, available_tools=[]
    )
    assert get_pending_notifications in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "+get_pending_notifications"}, available_tools=[]
    )
    assert get_pending_notifications not in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION"}, available_tools=[]
    )


def test_pending_notifications_is_injected_into_python_execution(job_factory):
    """A job's Python environment exposes the registered system tool."""
    job = job_factory()
    global_context = {}
    with _setup_execution_context(job, global_context, {}):
        assert global_context["get_pending_notifications"]() == []
