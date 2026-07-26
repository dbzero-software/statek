"""Focused provider reasoning request and continuation tests."""

# pylint: disable=protected-access,no-member,unused-argument,arguments-differ

from unittest.mock import MagicMock, patch

import pytest

from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.llm_api import (
    ClaudeAI_API, DefaultLLM_API_Impl, LLM_API, LLM_API_Settings,
    LLM_Response, LLM_StepData, LLM_Stats, VertexAI_API,
)
from statek.executors.job import Job, JobStatus
from statek.provider_config import ProviderConfig


def _config(provider, family=None, model=None, payload=None):
    """Create a single reasoning-level mapping at an optional model path."""
    node = {"reasoning_level": [{"range": {"from": 1}, "payload": payload or {}}]}
    if model is not None:
        node = {family: {model: node}}
    elif family is not None:
        node = {family: node}
    return ProviderConfig({provider: node})


def _settings():
    return LLM_API_Settings(api_url="https://example.test", api_key="key")


def test_openrouter_reasoning_payload_is_merged_and_params_are_not_upstream_model(db0_fixture):
    api = DefaultLLM_API_Impl(_settings())
    config = _config("openrouter", "openai", "gpt-5.4", {"reasoning": {"effort": "high"}})

    payload = api.preview_request(
        model="openrouter/openai/gpt-5.4/rl=50",
        provider_config=config,
    )

    assert payload["model"] == "openai/gpt-5.4"
    assert payload["reasoning"] == {"effort": "high"}


def test_reasoning_alias_conflict_and_unmapped_positive_level_fail(db0_fixture):
    api = DefaultLLM_API_Impl(_settings())

    with pytest.raises(ValueError, match="conflicting"):
        api.preview_request(model="gpt-5/rl=10&reasoning_level=20")
    with pytest.raises(ValueError, match="provider configuration"):
        api.preview_request(model="openai//gpt-5/rl=10")


def test_zero_reasoning_does_not_require_or_send_mapping(db0_fixture):
    api = DefaultLLM_API_Impl(_settings())

    payload = api.preview_request(model="openai//gpt-5/reasoning_level=0")

    assert payload == {"model": "gpt-5", "messages": []}


def test_provider_payload_deep_merges_after_provider_defaults(db0_fixture):
    api = DefaultLLM_API_Impl(_settings(), reasoning={"effort": "low", "summary": "auto"})
    config = _config("openai", payload={"reasoning": {"effort": "high"}})

    payload = api.preview_request(model="openai//gpt-5/rl=1", provider_config=config)

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.asyncio
async def test_openai_reasoning_details_round_trip_without_entering_visible_text(db0_fixture):
    api = DefaultLLM_API_Impl(_settings())
    response = {
        "choices": [{"message": {
            "content": "visible",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        }}],
        "usage": {},
    }
    mock_response = MagicMock(content=b"response")
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response

    async def fake_post(*_args, **_kwargs):
        return mock_response

    with patch("httpx.AsyncClient.post", fake_post):
        result = await api._process_request(model="gpt-5")

    assert result.step_data.text == "visible"
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="visible",
        content_src=ContentSource.ASSISTANT,
        provider_reasoning_payload=result.step_data.reasoning_payload,
    )
    assert api.build_messages(chat_history=[item])[-1]["reasoning_details"] == (
        response["choices"][0]["message"]["reasoning_details"]
    )


def test_claude_thinking_blocks_replay_in_original_content_order(db0_fixture):
    api = ClaudeAI_API(_settings(), use_prompt_caching=False)
    payload = {"format": "claude", "content": [
        {"type": "thinking", "thinking": "opaque", "signature": "sig"},
        {"type": "text", "text": "visible"},
    ]}
    item = ChatHistoryItem(
        ChatRole.ASSISTANT,
        "visible",
        ContentSource.ASSISTANT,
        provider_reasoning_payload=payload,
    )

    assert api.build_messages([item]) == [{"role": "assistant", "content": payload["content"]}]


def test_vertex_thought_parts_replay_and_are_not_visible_text(db0_fixture):
    api = VertexAI_API(_settings())
    response = {"candidates": [{"content": {"parts": [
        {"text": "hidden", "thought": True, "thoughtSignature": "sig"},
        {"text": "visible"},
    ]}}]}

    text, calls, reasoning_payload = api._parse_response(response)
    item = ChatHistoryItem(
        ChatRole.ASSISTANT, text, ContentSource.ASSISTANT,
        provider_reasoning_payload=reasoning_payload,
    )

    assert text == "visible"
    assert calls is None
    assert api.build_contents([item]) == [{
        "role": "model",
        "parts": response["candidates"][0]["content"]["parts"],
    }]


def test_custom_provider_receives_resolved_reasoning_payload(db0_fixture):
    class CustomAPI(LLM_API):
        def _build_request_payload(self, **kwargs):
            return kwargs

        async def _process_request(self, **kwargs):
            return LLM_Response(LLM_StepData("", None), LLM_Stats(0, 0, None))

    api = CustomAPI()
    config = _config("custom", payload={"vendor_reasoning": {"level": 1}})

    request = api.preview_request(model="custom//model/rl=1", provider_config=config)

    assert request["model"] == "model"
    assert request["reasoning_payload"] == {"vendor_reasoning": {"level": 1}}


def test_job_persists_and_reconstructs_reasoning_payload(job_def_factory, db0_fixture):
    job = Job(job_def_factory(), job_status=JobStatus.STARTED)
    reasoning_payload = {"format": "openai", "fields": {"reasoning_details": [{"id": "opaque"}]}}
    response = LLM_Response(
        LLM_StepData("answer", None, reasoning_payload),
        LLM_Stats(0, 0, None),
    )

    job.append_chat_log({}, response)
    history = list(job.get_chat_history())

    assert job.chat_log[0].llm_reasoning_payload == reasoning_payload
    assert history[0].provider_reasoning_payload == reasoning_payload
