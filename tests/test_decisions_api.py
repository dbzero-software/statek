"""Tests for the Decisions_API interface and single-question helpers."""

# pylint: disable=no-member,abstract-class-instantiated,unused-argument

from inspect import isabstract
import json

import httpx
import pytest

from statek import (
    BinaryCriteria, Decision, DecisionType, Decisions_API, Question,
)
from statek.settings import StatekSettings
import statek.settings as settings_module


class RecordingDecisionsAPI(Decisions_API):
    """Return a fixed decision while recording the structured request."""

    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    async def process_request(self, state, questions):
        self.calls.append((state, questions))
        return {key: self.decision for key in questions}


def test_decision_types_and_data_classes(db0_fixture):
    assert {str(value) for value in DecisionType.values()} == {
        "NOUL", "CHOICE", "SCORE"
    }
    criteria = BinaryCriteria(as_true="approved", as_false="denied")
    question = Question(type=DecisionType.NOUL, criteria=criteria)
    decision = Decision(choice=True, confidence=0.9)

    assert question.instructions is None
    assert question.criteria is criteria
    assert decision.probabilities is None
    assert decision.choice is True


def test_decisions_api_requires_process_request():
    assert isabstract(Decisions_API)
    with pytest.raises(TypeError, match="process_request"):
        Decisions_API()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper", "criteria", "expected_type"),
    [
        ("true_or_false", BinaryCriteria("yes", "no"), DecisionType.NOUL),
        ("choice", {"a": "first", "b": "second"}, DecisionType.CHOICE),
        ("score", ["low", "medium", "high"], DecisionType.SCORE),
    ],
)
async def test_single_question_helpers(helper, criteria, expected_type):
    decision = Decision(choice="a", confidence=0.75, probabilities={"a": 0.75})
    api = RecordingDecisionsAPI(decision)
    state = {"message": "hello"}

    result = await getattr(api, helper)(state, "Evaluate this", criteria)

    assert result is decision
    assert len(api.calls) == 1
    actual_state, questions = api.calls[0]
    assert actual_state is state
    assert len(questions) == 1
    question = next(iter(questions.values()))
    assert question.type == expected_type
    assert question.instructions == "Evaluate this"
    assert question.criteria is criteria


def test_get_selects_provider_model_and_caches(monkeypatch):
    Decisions_API.get.cache_clear()
    monkeypatch.setattr(settings_module, "get_statek_settings", lambda: StatekSettings(
        default_llm_api_provider="OPENAI"))
    first = Decisions_API.get(api_key="test-key")
    assert first.model == "gpt-6-luna"
    assert first is Decisions_API.get(api_key="test-key")
    assert Decisions_API.get("typesafe.ai", api_key="test-key").model == "jev-latest"
    assert Decisions_API.get("openrouter", api_key="test-key").model.startswith("typesafe/")
    with pytest.raises(ValueError, match="Unsupported decisions provider"):
        Decisions_API.get("unknown", api_key="test-key")
    with pytest.raises(ValueError, match="does not support"):
        Decisions_API.get("openai", model="gpt-5", api_key="test-key")
    Decisions_API.get.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,endpoint,model", [
    ("openrouter", "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13"),
    ("typesafe.ai", "https://api.typesafe.ai/v1/systemone", "jev-latest"),
])
async def test_system_one_provider_request_and_response(monkeypatch, provider, endpoint, model):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"answers": {
            "binary": {"type": "noul", "noul": 0.2},
            "route": {"type": "choice", "choice": "billing", "confidence": 0.7,
                      "probabilities": {"billing": 0.8, "other": 0.2}},
            "severity": {"type": "score", "score": 1.4, "confidence": 0.6,
                         "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}},
        }})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    api = Decisions_API.get(provider, api_key="test-key")
    questions = {
        "binary": Question(DecisionType.NOUL, BinaryCriteria("yes", "no"), "Is it true?"),
        "route": Question(DecisionType.CHOICE, {"billing": "payments", "other": "else"}, "Route?"),
        "severity": Question(DecisionType.SCORE, ["low", "medium", "high"], "Severity?"),
    }
    result = await api.process_request({"message": "hello"}, questions)
    await client.aclose()
    assert str(requests[0].url) == endpoint
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == model
    assert body["state"] == {"message": "hello"}
    assert body["questions"]["binary"]["criteria"] == {"true": "yes", "false": "no"}
    assert result["binary"] == Decision(False, 0.8, {"true": 0.2, "false": 0.8})
    assert result["route"] == Decision("billing", 0.7, {"billing": 0.8, "other": 0.2})
    assert result["severity"] == Decision(1.4, 0.6, {"0": 0.1, "1": 0.4, "2": 0.5})
    Decisions_API.get.cache_clear()


@pytest.mark.asyncio
async def test_openai_request_and_response(monkeypatch):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"answers": [
            {"type": "predicate", "name": "binary", "probability": 0.9},
            {"type": "choice", "name": "route", "choice": "billing", "confidence": 0.7,
             "probabilities": [{"value": "billing", "probability": 0.8},
                               {"value": "other", "probability": 0.2}]},
            {"type": "score", "name": "severity", "score": 1.4, "confidence": 0.6,
             "probabilities": [{"value": 0, "label": "low", "probability": 0.1},
                               {"value": 1, "label": "medium", "probability": 0.4},
                               {"value": 2, "label": "high", "probability": 0.5}]},
        ]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    api = Decisions_API.get("openai", api_key="test-key")
    questions = {
        "binary": Question(DecisionType.NOUL, BinaryCriteria("yes", "no"), "Is it true?"),
        "route": Question(DecisionType.CHOICE, {"billing": "payments", "other": "else"}, "Route?"),
        "severity": Question(DecisionType.SCORE, ["low", "medium", "high"], "Severity?"),
    }
    result = await api.process_request({"message": "hello"}, questions)
    await client.aclose()
    assert str(requests[0].url) == "https://api.openai.com/v1/decisions"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-6-luna"
    assert body["input"][0]["content"][0] == {"type": "input_json", "data": {"message": "hello"}}
    assert body["questions"][0]["type"] == "predicate"
    assert "yes" in body["questions"][0]["instructions"]
    assert body["questions"][1]["choices"] == [
        {"value": "billing", "description": "payments"},
        {"value": "other", "description": "else"}]
    assert body["questions"][2]["levels"] == [
        {"label": "low"}, {"label": "medium"}, {"label": "high"}]
    assert result["binary"].choice is True
    assert result["binary"].confidence == pytest.approx(0.9)
    assert result["binary"].probabilities == pytest.approx(
        {"true": 0.9, "false": 0.1})
    assert result["route"] == Decision("billing", 0.7, {"billing": 0.8, "other": 0.2})
    assert result["severity"] == Decision(1.4, 0.6, {"0": 0.1, "1": 0.4, "2": 0.5})
    Decisions_API.get.cache_clear()
