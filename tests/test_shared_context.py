"""Tests for shared context data structures."""

from datetime import datetime, timedelta, timezone

import pytest

from statek.shared_context import ContextVar, SharedContext, SharedContextProxy


pytestmark = pytest.mark.usefixtures("db0_fixture")


def test_context_var_defaults_created_at_and_use_count():
    """ContextVar records creation time and starts unused."""
    var = ContextVar(category="preference", value="concise", description="tone")

    assert var.category == "preference"
    assert var.value == "concise"
    assert var.description == "tone"
    assert isinstance(var.created_at, datetime)
    assert var.created_at.tzinfo is timezone.utc
    assert var.use_count == 0


def test_set_and_get_var_returns_context_var():
    """SharedContext stores named variables and returns ContextVar objects."""
    context = SharedContext()

    context.set_var("preference", "tone", "concise", "Preferred response style")
    var = context.get_var("tone")

    assert var is not None
    assert var.category == "preference"
    assert var.value == "concise"
    assert var.description == "Preferred response style"
    assert var.use_count == 1


def test_get_missing_var_returns_none():
    """Missing variables return None without creating a context entry."""
    context = SharedContext()

    assert context.get_var("missing") is None
    assert "missing" not in context


def test_contains_reports_existing_key_without_incrementing_use_count():
    """Membership checks do not count as variable use."""
    context = SharedContext()
    context.set_var("entity", "client", "Acme", "Current client")

    assert "client" in context
    var = context.get_var("client")

    assert var is not None
    assert var.use_count == 1


def test_get_var_increments_use_count_each_time():
    """Successful lookups increment use_count on the stored variable."""
    context = SharedContext()
    context.set_var("vocabulary", "crm", "customer relationship manager", "Term expansion")

    first = context.get_var("crm")
    second = context.get_var("crm")

    assert first is second
    assert second is not None
    assert second.use_count == 2


def test_set_var_overwrites_existing_key_with_new_context_var():
    """Setting an existing key replaces value, metadata, timestamp, and use count."""
    context = SharedContext()
    context.set_var("preference", "tone", "concise", "Initial style")
    first = context.get_var("tone")
    assert first is not None

    context.set_var("preference", "tone", "detailed", "Updated style")
    second = context.get_var("tone")

    assert second is not None
    assert second is not first
    assert second.value == "detailed"
    assert second.description == "Updated style"
    assert second.use_count == 1
    assert second.created_at >= first.created_at


def test_context_instances_do_not_share_variables():
    """Each SharedContext instance owns an independent variable dictionary."""
    first = SharedContext()
    second = SharedContext()

    first.set_var("entity", "client", "Acme", "Current client")

    assert "client" in first
    assert "client" not in second
    assert second.get_var("client") is None


def _store_var_at(context: SharedContext, key: str, value: str, created_at: datetime) -> ContextVar:
    """Store a context variable with a deterministic timestamp for merge tests."""
    context.set_var("preference", key, value, f"{key} description")
    var = context.get_var(key)
    assert var is not None
    var.created_at = created_at
    var.use_count = 0
    return var


def test_proxy_set_var_forwards_to_writable_context():
    """SharedContextProxy writes only to its writable context."""
    writable = SharedContext()
    readable = SharedContext()
    proxy = SharedContextProxy(writable=writable, readables=readable)

    proxy.set_var("preference", "tone", "concise", "Preferred response style")

    assert "tone" in writable
    assert "tone" not in readable
    var = proxy.get_var("tone")
    assert var is not None
    assert var.value == "concise"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_writable_without_readables():
    """SharedContextProxy works with only a writable context."""
    writable = SharedContext()
    writable.set_var("preference", "tone", "concise", "Preferred response style")
    proxy = SharedContextProxy(writable=writable)

    var = proxy.get_var("tone")

    assert var is not None
    assert var.value == "concise"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_single_readable_context():
    """SharedContextProxy reads variables from its readable context."""
    writable = SharedContext()
    readable = SharedContext()
    readable.set_var("entity", "client", "Acme", "Current client")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("client")

    assert var is not None
    assert var.value == "Acme"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_multiple_readable_contexts():
    """SharedContextProxy merges reads from multiple readable contexts."""
    writable = SharedContext()
    first_readable = SharedContext()
    second_readable = SharedContext()
    first_readable.set_var("entity", "client", "Acme", "Current client")
    second_readable.set_var("vocabulary", "crm", "customer relationship manager", "Term")
    proxy = SharedContextProxy(writable=writable, readables=[first_readable, second_readable])

    client = proxy.get_var("client")
    crm = proxy.get_var("crm")

    assert client is not None
    assert client.value == "Acme"
    assert crm is not None
    assert crm.value == "customer relationship manager"


def test_proxy_get_var_returns_latest_duplicate_key():
    """When duplicate keys exist, SharedContextProxy returns the latest created_at."""
    older_time = datetime.now(timezone.utc)
    newer_time = older_time + timedelta(seconds=1)
    writable = SharedContext()
    readable = SharedContext()
    older = _store_var_at(writable, "tone", "concise", older_time)
    newer = _store_var_at(readable, "tone", "detailed", newer_time)
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("tone")

    assert var is newer
    assert var.value == "detailed"
    assert newer.use_count == 1
    assert older.use_count == 0


def test_proxy_get_var_prefers_writable_when_created_at_ties():
    """SharedContextProxy returns writable variable when timestamps tie."""
    created_at = datetime.now(timezone.utc)
    writable = SharedContext()
    readable = SharedContext()
    writable_var = _store_var_at(writable, "tone", "concise", created_at)
    readable_var = _store_var_at(readable, "tone", "detailed", created_at)
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("tone")

    assert var is writable_var
    assert var.value == "concise"
    assert writable_var.use_count == 1
    assert readable_var.use_count == 0


def test_proxy_contains_checks_all_contexts_without_incrementing_use_count():
    """SharedContextProxy membership checks do not count as variable use."""
    writable = SharedContext()
    readable = SharedContext()
    readable.set_var("entity", "client", "Acme", "Current client")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    assert "client" in proxy
    assert "missing" not in proxy
    var = proxy.get_var("client")
    assert var is not None
    assert var.use_count == 1


def test_proxy_contains_finds_key_in_writable_context():
    """SharedContextProxy membership checks include the writable context."""
    writable = SharedContext()
    readable = SharedContext()
    writable.set_var("preference", "tone", "concise", "Preferred response style")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    assert "tone" in proxy
    var = proxy.get_var("tone")
    assert var is not None
    assert var.use_count == 1
