"""Tests for Statek ContextVar-backed execution context helpers."""

import asyncio

import pytest

from statek.utils import get_current_job, statek_ctx_get, statek_ctx_set, _statek_ctx_scope


def test_set_and_get():
    with _statek_ctx_scope({}):
        statek_ctx_set(foo="bar")
        assert statek_ctx_get("foo") == "bar"


def test_get_missing_key_raises():
    with _statek_ctx_scope({}):
        with pytest.raises(KeyError):
            statek_ctx_get("missing")


def test_get_missing_key_with_default():
    with _statek_ctx_scope({}):
        assert statek_ctx_get("missing", None) is None
        assert statek_ctx_get("missing", 42) == 42


def test_set_overwrites_existing():
    with _statek_ctx_scope({"x": 1}):
        statek_ctx_set(x=2)
        assert statek_ctx_get("x") == 2


def test_set_multiple_keys():
    with _statek_ctx_scope({}):
        statek_ctx_set(a=1, b=2)
        assert statek_ctx_get("a") == 1
        assert statek_ctx_get("b") == 2


def test_context_resets_after_scope():
    job = object()

    with _statek_ctx_scope({"job": job}):
        assert get_current_job() is job

    assert get_current_job() is None


def test_nested_context_scope_restores_parent():
    parent_job = object()
    child_job = object()

    with _statek_ctx_scope({"job": parent_job}):
        with _statek_ctx_scope({"job": child_job}):
            assert get_current_job() is child_job
        assert get_current_job() is parent_job


def test_set_without_context_raises_runtime_error():
    with pytest.raises(RuntimeError):
        statek_ctx_set(foo="bar")


@pytest.mark.asyncio
async def test_async_context_scopes_are_isolated_per_task():
    first_job = object()
    second_job = object()

    async def read_current_job(job):
        with _statek_ctx_scope({"job": job}):
            await asyncio.sleep(0)
            return get_current_job()

    first_result, second_result = await asyncio.gather(
        read_current_job(first_job),
        read_current_job(second_job),
    )

    assert first_result is first_job
    assert second_result is second_job
