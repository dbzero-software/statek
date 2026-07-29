"""Focused provider reasoning request and continuation tests."""

# pylint: disable=protected-access,no-member,unused-argument,arguments-differ,too-many-arguments,too-many-positional-arguments

import json
from unittest.mock import MagicMock, patch

import pytest

from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.chat_style import ChatStyle
from statek.llm_api import (
    ClaudeAI_API, DefaultLLM_API_Impl, LLM_API, LLM_API_Settings, OpenAI_API,
    OpenRouter_API, LLM_Response, LLM_StepData, LLM_Stats, VertexAI_API,
)
from statek.executors.job import Job, JobStatus
from statek.provider_config import ProviderConfig


def _config(provider, family=None, model=None, payload=None, ignored_parameters=None):
    """Create a single reasoning-level mapping at an optional model path."""
    reasoning = {"reasoning_level": [{"range": {"from": 1}, "payload": payload or {}}]}
    if ignored_parameters is not None:
        reasoning["ignore_parameters"] = ignored_parameters
    node = {"reasoning": reasoning}
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


def test_metadata_provider_controls_model_formatting_and_reasoning_lookup(db0_fixture):
    """The effective provider formats a family/model selection and finds its mapping."""
    api = DefaultLLM_API_Impl(_settings())
    config = _config("openai", payload={"reasoning": {"effort": "high"}})

    payload = api.preview_request(
        model="openai/gpt-5/rl=50",
        metadata={"PROVIDER": "OPENAI"},
        provider_config=config,
    )

    assert payload["model"] == "gpt-5"
    assert payload["reasoning"] == {"effort": "high"}


def test_metadata_provider_resolves_model_specific_reasoning(db0_fixture):
    """Framework routing metadata selects a provider/model reasoning mapping."""
    config = ProviderConfig({
        "openai": {
            "gpt-5": {
                "reasoning": {
                    "reasoning_level": [{
                        "range": {"from": 1},
                        "payload": {"reasoning": {"effort": "high"}},
                    }],
                },
            },
        },
    })

    payload = OpenAI_API(_settings()).preview_request(
        model="openai/gpt-5/rl=50",
        metadata={"PROVIDER": "OPENAI"},
        provider_config=config,
    )

    assert payload["model"] == "gpt-5"
    assert payload["reasoning"] == {"effort": "high"}


def test_provider_payload_deep_merges_after_provider_defaults(db0_fixture):
    api = DefaultLLM_API_Impl(_settings(), reasoning={"effort": "low", "summary": "auto"})
    config = _config("openai", payload={"reasoning": {"effort": "high"}})

    payload = api.preview_request(model="openai//gpt-5/rl=1", provider_config=config)

    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.parametrize(
    ("api_class", "api_kwargs", "model", "expected_container"),
    [
        (DefaultLLM_API_Impl, {}, "openrouter/openai/gpt-5/rl=1", None),
        (ClaudeAI_API, {"use_prompt_caching": False}, "claudeai//claude-4/rl=1", None),
        (VertexAI_API, {}, "vertexai//gemini-3/rl=1", "generationConfig"),
    ],
)
def test_positive_reasoning_removes_configured_temperature_from_final_payload(
    db0_fixture,
    api_class,
    api_kwargs,
    model,
    expected_container,
):
    """Configured conflicts are removed after every provider payload source is merged."""
    api = api_class(_settings(), temperature=0.7, **api_kwargs)
    config = _config(
        model.split("/", 1)[0],
        payload={"temperature": 0.5},
        ignored_parameters=["temperature"],
    )

    payload = api.preview_request(
        model=model,
        temperature=0.3,
        provider_config=config,
    )

    if expected_container is None:
        assert "temperature" not in payload
    else:
        assert "temperature" not in payload.get(expected_container, {})


def test_positive_reasoning_with_empty_payload_still_ignores_temperature(db0_fixture):
    """An empty mapping is still an enabled reasoning configuration."""
    api = DefaultLLM_API_Impl(_settings())
    config = _config("openai", payload={}, ignored_parameters=["temperature"])

    payload = api.preview_request(
        model="openai//gpt-5/rl=1",
        temperature=0.3,
        provider_config=config,
    )

    assert "temperature" not in payload


def test_zero_reasoning_keeps_temperature_when_configuration_ignores_it(db0_fixture):
    """Ignoring applies only to a positive configured reasoning level."""
    api = DefaultLLM_API_Impl(_settings())
    config = _config("openai", ignored_parameters=["temperature"])

    payload = api.preview_request(
        model="openai//gpt-5/rl=0",
        temperature=0.3,
        provider_config=config,
    )

    assert payload["temperature"] == 0.3


@pytest.mark.asyncio
async def test_reasoning_preview_matches_filtered_execution_payload(db0_fixture):
    """Preview exposes the same post-merge parameter filtering as execution."""
    api = DefaultLLM_API_Impl(_settings(), temperature=0.7)
    config = _config(
        "openrouter",
        payload={"reasoning": {"effort": "high"}, "temperature": 0.5},
        ignored_parameters=["temperature"],
    )
    captured_payload = {}

    async def fake_post(*_args, **kwargs):
        captured_payload.update(kwargs["json"])
        response = MagicMock(content=b'{"choices":[{"message":{"content":"ok"}}]}')
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return response

    preview = api.preview_request(
        model="openrouter/openai/gpt-5/rl=1",
        temperature=0.3,
        provider_config=config,
    )

    with patch("httpx.AsyncClient.post", fake_post):
        await api.process_request(
            model="openrouter/openai/gpt-5/rl=1",
            temperature=0.3,
            provider_config=config,
        )

    assert preview == captured_payload
    assert "temperature" not in preview


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
        result = await api._process_request(model="openai//gpt-5")

    assert result.step_data.text == "visible"
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="visible",
        content_src=ContentSource.ASSISTANT,
        provider_reasoning_payload=result.step_data.reasoning_payload,
    )
    assert api.build_messages(chat_history=[item], provider="openai")[-1]["reasoning_details"] == (
        response["choices"][0]["message"]["reasoning_details"]
    )


def test_openai_reasoning_is_not_replayed_to_another_compatible_provider(db0_fixture):
    """OpenRouter continuation fields are not sent through an OpenAI formatter."""
    del db0_fixture
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="visible",
        content_src=ContentSource.ASSISTANT,
        provider_reasoning_payload={
            "provider": "openrouter",
            "format": "openai",
            "fields": {"reasoning_details": [{"data": "opaque"}]},
        },
    )

    api = DefaultLLM_API_Impl(_settings())
    openrouter_message = api.build_messages(chat_history=[item], provider="openrouter")[-1]
    openai_message = api.build_messages(chat_history=[item], provider="openai")[-1]

    assert openrouter_message["reasoning_details"] == [{"data": "opaque"}]
    assert "reasoning_details" not in openai_message


def test_opaque_custom_reasoning_payload_is_ignored_by_openai_formatter(db0_fixture):
    """A custom provider's bytes payload cannot crash an incompatible formatter."""
    del db0_fixture
    item = ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content="visible",
        content_src=ContentSource.ASSISTANT,
        provider_reasoning_payload=b"opaque",
    )

    message = DefaultLLM_API_Impl(_settings()).build_messages(
        chat_history=[item], chat_style=ChatStyle.DIRECT, provider="openai",
    )[-1]
    assert message == {
        "role": "assistant", "content": "visible",
    }


def test_claude_thinking_blocks_replay_in_original_content_order(db0_fixture):
    api = ClaudeAI_API(_settings(), use_prompt_caching=False)
    payload = {"provider": "claudeai", "format": "claude", "content": [
        {"type": "thinking", "thinking": "opaque", "signature": "sig"},
        {"type": "text", "text": "visible"},
    ]}
    item = ChatHistoryItem(
        ChatRole.ASSISTANT,
        "visible",
        ContentSource.ASSISTANT,
        provider_reasoning_payload=payload,
    )

    assert api.build_messages([item], provider="claudeai") == [
        {"role": "assistant", "content": payload["content"]},
    ]


def test_vertex_thought_parts_replay_and_are_not_visible_text(db0_fixture):
    api = VertexAI_API(_settings())
    response = {"candidates": [{"content": {"parts": [
        {"text": "hidden", "thought": True, "thoughtSignature": "sig"},
        {"text": "visible"},
    ]}}]}

    text, calls, reasoning_payload = api._parse_response(response)
    reasoning_payload["provider"] = "vertexai"
    item = ChatHistoryItem(
        ChatRole.ASSISTANT, text, ContentSource.ASSISTANT,
        provider_reasoning_payload=reasoning_payload,
    )

    assert text == "visible"
    assert calls is None
    assert api.build_contents([item], provider="vertexai") == [{
        "role": "model",
        "parts": response["candidates"][0]["content"]["parts"],
    }]


@pytest.mark.parametrize(
    ("provider", "reasoning_payload"),
    [
        ("openai", {
            "provider": "openai",
            "format": "openai",
            "fields": {"reasoning_details": [{"data": "opaque"}]},
        }),
        ("claudeai", {
            "provider": "claudeai",
            "format": "claude",
            "content": [{"type": "thinking", "thinking": "opaque", "signature": "sig"}],
        }),
        ("vertexai", {
            "provider": "vertexai",
            "format": "vertex",
            "parts": [{"text": "hidden", "thought": True, "thoughtSignature": "sig"}],
        }),
    ],
)
def test_reasoning_only_response_is_reconstructed_for_provider_history(
    job_def_factory,
    provider,
    reasoning_payload,
):
    """A response without text or tools retains its compatible continuation material."""
    job = Job(job_def_factory(), job_status=JobStatus.STARTED)
    job.append_chat_log({}, LLM_Response(
        LLM_StepData("", None, reasoning_payload),
        LLM_Stats(0, 0, None),
    ))

    history = list(job.get_chat_history())

    assert len(history) == 1
    assert history[0].role == ChatRole.ASSISTANT
    assert history[0].content is None
    assert history[0].provider_reasoning_payload == reasoning_payload
    if provider == "openai":
        assert DefaultLLM_API_Impl(_settings()).build_messages(
            chat_history=history, provider=provider,
        )[-1]["reasoning_details"] == [{"data": "opaque"}]
    elif provider == "claudeai":
        assert ClaudeAI_API(_settings(), use_prompt_caching=False).build_messages(
            history, provider=provider,
        ) == [{"role": "assistant", "content": reasoning_payload["content"]}]
    else:
        assert VertexAI_API(_settings()).build_contents(history, provider=provider) == [{
            "role": "model", "parts": reasoning_payload["parts"],
        }]


@pytest.mark.parametrize(
    ("api_class", "api_kwargs", "model", "provider", "reasoning_payload"),
    [
        (
            OpenRouter_API,
            {},
            "openrouter/openai/gpt-5",
            "openrouter",
            {
                "provider": "openrouter",
                "format": "openai",
                "fields": {"reasoning_details": [{"data": "opaque"}]},
            },
        ),
        (
            ClaudeAI_API,
            {"use_prompt_caching": False},
            "claudeai//claude-4",
            "claudeai",
            {
                "provider": "claudeai",
                "format": "claude",
                "content": [{"type": "thinking", "thinking": "opaque", "signature": "sig"}],
            },
        ),
        (
            VertexAI_API,
            {},
            "vertexai//gemini-2.5",
            "vertexai",
            {
                "provider": "vertexai",
                "format": "vertex",
                "parts": [{"text": "hidden", "thought": True, "thoughtSignature": "sig"}],
            },
        ),
    ],
)
def test_persisted_reasoning_payload_builds_json_serializable_request(
    job_def_factory,
    api_class,
    api_kwargs,
    model,
    provider,
    reasoning_payload,
):
    """Matching provider replay converts nested dbzero collections before JSON encoding."""
    job = Job(job_def_factory(), job_status=JobStatus.STARTED)
    job.append_chat_log({}, LLM_Response(
        LLM_StepData("visible", None, reasoning_payload),
        LLM_Stats(0, 0, None),
    ))

    payload = api_class(_settings(), **api_kwargs).preview_request(
        model=model,
        metadata={"PROVIDER": provider},
        chat_history=list(job.get_chat_history()),
    )

    json.dumps(payload)


def test_custom_provider_receives_documented_provider_config_contract(db0_fixture):
    """Custom providers receive the documented request arguments for model parameters."""
    class CustomAPI(LLM_API):
        def _build_request_payload(
            self, system_prompt=None, model=None, metadata=None, tools=None,
            chat_history=None, chat_style=None, temperature=None, provider_config=None,
        ):
            del system_prompt, metadata, tools, chat_history, chat_style, temperature
            return {"model": model, "provider_config": provider_config}

        async def _process_request(
            self, system_prompt=None, model=None, metadata=None, tools=None,
            chat_history=None, chat_style=None, temperature=None, provider_config=None,
        ):
            del (
                system_prompt, model, metadata, tools, chat_history, chat_style,
                temperature, provider_config,
            )
            return LLM_Response(LLM_StepData("", None), LLM_Stats(0, 0, None))

    api = CustomAPI()
    config = _config("custom", payload={"vendor_reasoning": {"level": 1}})

    request = api.preview_request(model="custom//model/rl=1", provider_config=config)

    assert request == {
        "model": "custom//model/rl=1",
        "provider_config": config,
    }


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
