"""Behavioral tests for job notification system tools."""

# pylint: disable=no-member,protected-access

from unittest.mock import patch

import pytest

from statek import get_pending_notifications, send_notification_to
from statek.executors.chat_log_item import NotificationLogItem, SubTaskLogItem
from statek.executors.job import Job
from statek.executors.utils import _setup_execution_context, exec_tool
from statek.llm_api import select_request_tools
from statek.system import (
    brief,
    docstr,
    find_tools,
    get_pending_notifications as system_get_pending_notifications,
)
from statek.task import SubTaskHandler
from statek.utils import CallSpec, _statek_ctx_scope, format_tool_spec


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


def test_pending_notifications_includes_mixed_items(job_factory):
    """The getter retains text and subtask notifications in arrival order."""
    job = job_factory()
    subtask = SubTaskLogItem(console_pos=0, handler=SubTaskHandler(job=job))
    message = NotificationLogItem(console_pos=0, message="hello")
    job._pending_notifications().extend([message, subtask])

    with _statek_ctx_scope({"job": job}):
        assert get_pending_notifications() == [message, subtask]
        assert get_pending_notifications() == [message, subtask]

    assert not job.chat_log


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


def test_send_notification_to_job_calls_delivery_once(job_factory):
    """Forwarding a direct Job target preserves the exact message."""
    target = job_factory()
    with patch.object(Job, "send_notification", autospec=True) as send:
        assert send_notification_to(target, "Żółw\nexit('not code')") is None
    send.assert_called_once_with(target, "Żółw\nexit('not code')")


def test_send_notification_to_unfinished_handler_uses_its_job(job_factory):
    """A handler can be used while its child job is still running."""
    target = job_factory()
    handler = SubTaskHandler(job=target, id="child")
    send_notification_to(handler, "hello")

    assert handler.is_completed is False
    assert target._pending_notifications()[0].message == "hello"
    assert not target.chat_log


@pytest.mark.parametrize("target", ["unknown_job", None, 17])
def test_send_notification_to_rejects_invalid_external_targets(job_factory, target):
    """Unsupported tool input does not partly deliver a notification."""
    sender = job_factory()
    with pytest.raises(TypeError, match="target"):
        send_notification_to(target, "hello")
    assert not sender.chat_log
    assert not sender._pending_notifications()


def test_send_notification_to_is_registered_and_documented(capsys):
    """Agents can discover the tool, its signature, and brief/full help."""
    assert send_notification_to in find_tools("SYSTEM")
    assert send_notification_to not in find_tools("APPLICATION")
    assert send_notification_to.__annotations__["target"] == SubTaskHandler | Job
    assert send_notification_to in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "SYSTEM"}, available_tools=[]
    )
    assert send_notification_to in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "+send_notification_to"}, available_tools=[]
    )
    assert send_notification_to not in select_request_tools(
        metadata={"LLM_TOOLS_SCOPE": "APPLICATION"}, available_tools=[]
    )
    brief(send_notification_to)
    brief_output = capsys.readouterr().out
    docstr(send_notification_to)
    doc_output = capsys.readouterr().out
    assert "send_notification_to" in brief_output
    assert "send_notification_to" in doc_output
    assert len(doc_output) > len(brief_output)


def test_send_notification_to_formal_schema_accepts_local_target_name():
    """The provider must request a name resolvable to a Job or handler, not an array."""
    spec = format_tool_spec(send_notification_to)["function"]["parameters"]

    assert spec["properties"]["target"]["type"] == "string"
    assert spec["properties"]["message"]["type"] == "string"
    assert spec["required"] == ["target", "message"]


def test_send_notification_to_injected_into_python_context(job_factory):
    """A Python-CLI job can call the system tool with an existing child."""
    sender = job_factory()
    target = job_factory()
    global_context = {}
    with _setup_execution_context(sender, global_context, {}):
        global_context["send_notification_to"](target, "from python")
    assert target._pending_notifications()[0].message == "from python"


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["job", "handler"])
async def test_send_notification_to_formal_call_binds_local_target(job_factory, target_kind):
    """Formal tool calls resolve local Job and handler names through the union type."""
    sender = job_factory()
    target = job_factory()
    local_target = target if target_kind == "job" else SubTaskHandler(job=target)
    sender.add_locals(child=local_target)
    call = CallSpec(
        id="notification", func_name="send_notification_to",
        kwargs={"target": "child", "message": "from formal tool"},
    )

    result, error = await exec_tool(call, sender)

    assert error is None
    assert result == ""
    assert len(target._pending_notifications()) == 1
    assert target._pending_notifications()[0].message == "from formal tool"
