"""Tests for shared context data structures."""

from datetime import datetime, timezone

import pytest

from statek.shared_context import ContextVar, SharedContext


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
