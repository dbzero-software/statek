# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured classification request and response interface."""

# pylint: disable=no-member

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Dict, List, Optional

import dbzero as db0
import httpx


@db0.enum(values=["NOUL", "CHOICE", "SCORE"])
class DecisionType:
    """Supported decision question types."""


@dataclass
class BinaryCriteria:
    """Labels for the true and false outcomes of a binary question."""

    as_true: str
    as_false: str


@dataclass
class Question:
    """A classification question with its decision criteria."""

    type: DecisionType
    criteria: Dict | List | BinaryCriteria
    instructions: Optional[str] = None


@dataclass
class Decision:
    """A selected outcome and its confidence."""

    choice: str | bool | float
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class Decisions_API(ABC):
    """Abstract interface for scored classification models."""

    @staticmethod
    @lru_cache
    def get(provider_name: str = None, **kwargs) -> "Decisions_API":
        """Return a cached provider client for a supported decisions model."""
        from .settings import get_statek_settings  # pylint: disable=import-outside-toplevel

        settings = get_statek_settings()
        provider = (provider_name or settings.default_llm_api_provider).strip().upper()
        provider = {"TYPESAFE.AI": "TYPESAFE"}.get(provider, provider)
        providers = {
            "OPENROUTER": OpenRouterDecisionsAPI,
            "OPENAI": OpenAIDecisionsAPI,
            "TYPESAFE": TypeSafeDecisionsAPI,
        }
        if provider not in providers:
            raise ValueError(f"Unsupported decisions provider: {provider_name}")
        provider_settings = settings.get_provider_settings(provider)
        api_key = kwargs.pop("api_key", None) or (
            provider_settings.api_key if provider_settings else None
        ) or os.environ.get(f"{provider}_API_KEY")
        return providers[provider](api_key=api_key, **kwargs)

    @abstractmethod
    async def process_request(
        self, state: Dict, questions: Dict[str, Question]
    ) -> Dict[str, Decision]:
        """Return a decision for each keyed question in the request."""
        raise NotImplementedError

    async def true_or_false(
        self, state: Dict, instructions: str, criteria: BinaryCriteria
    ) -> Decision:
        """Ask one binary question."""
        return await self._single_question(
            state, Question(DecisionType.NOUL, criteria, instructions)
        )

    async def choice(self, state: Dict, instructions: str, choices: Dict) -> Decision:
        """Ask one multiple-choice question."""
        return await self._single_question(
            state, Question(DecisionType.CHOICE, choices, instructions)
        )

    async def score(self, state: Dict, instructions: str, scores: List[str]) -> Decision:
        """Ask one scoring question."""
        return await self._single_question(
            state, Question(DecisionType.SCORE, scores, instructions)
        )

    async def _single_question(self, state: Dict, question: Question) -> Decision:
        return (await self.process_request(state, {"question": question}))["question"]


def _question_type(question: Question) -> str:
    value = str(question.type).lower()
    if value not in ("noul", "choice", "score"):
        raise ValueError(f"Unsupported decision type: {question.type}")
    return value


def _binary_decision(probability: float) -> Decision:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid binary decision probability: {probability}")
    return Decision(
        choice=probability >= 0.5,
        confidence=max(probability, 1.0 - probability),
        probabilities={"true": probability, "false": 1.0 - probability},
    )


def _parse_system_one_answers(data: dict, questions: Dict[str, Question]) -> Dict[str, Decision]:
    answers = data["answers"]
    result = {}
    for name, question in questions.items():
        answer = answers[name]
        expected_type = _question_type(question)
        if answer["type"] != expected_type:
            raise ValueError(f"Unexpected answer type for {name}: {answer['type']}")
        if expected_type == "noul":
            result[name] = _binary_decision(answer["noul"])
        else:
            value = answer["choice"] if expected_type == "choice" else answer["score"]
            result[name] = Decision(value, answer["confidence"], answer.get("probabilities"))
    return result


class _HTTPDecisionsAPI(Decisions_API):  # pylint: disable=abstract-method
    """Shared HTTP transport for native decisions endpoints."""

    DEFAULT_MODEL = ""
    API_URL = ""

    def __init__(self, api_key: Optional[str], model: Optional[str] = None,
                 api_url: Optional[str] = None, timeout: float = 60.0):
        self.model = model or self.DEFAULT_MODEL
        self._validate_model(self.model)
        self.api_key = api_key
        self.api_url = api_url or self.API_URL
        self.timeout = timeout

    @classmethod
    def _validate_model(cls, model: str) -> None:
        del cls, model

    async def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise ValueError("Decisions API key is required")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()


class _SystemOneDecisionsAPI(_HTTPDecisionsAPI):
    """Request and response mapping shared by TypeSafe and OpenRouter."""

    async def process_request(
        self, state: Dict, questions: Dict[str, Question]
    ) -> Dict[str, Decision]:
        if not questions:
            raise ValueError("At least one decision question is required")
        wire_questions = {}
        for name, question in questions.items():
            criteria = question.criteria
            if isinstance(criteria, BinaryCriteria):
                criteria = {"true": criteria.as_true, "false": criteria.as_false}
            wire_questions[name] = {
                "type": _question_type(question),
                "criteria": criteria,
            }
            if question.instructions is not None:
                wire_questions[name]["instructions"] = question.instructions
        data = await self._post({
            "model": self.model, "state": state, "questions": wire_questions,
        })
        return _parse_system_one_answers(data, questions)


class OpenRouterDecisionsAPI(_SystemOneDecisionsAPI):
    """OpenRouter's native Decisions API."""

    DEFAULT_MODEL = "typesafe/jev-1.13"
    API_URL = "https://openrouter.ai/api/alpha/decisions"

    @classmethod
    def _validate_model(cls, model: str) -> None:
        if model not in {"typesafe/jev-1.13", "typesafe/jev-1.14",
                         "~typesafe/jev-latest", "openai/gpt-6-luna"}:
            raise ValueError(f"OpenRouter decisions API does not support model: {model}")


class TypeSafeDecisionsAPI(_SystemOneDecisionsAPI):
    """TypeSafe System One API."""

    DEFAULT_MODEL = "jev-latest"
    API_URL = "https://api.typesafe.ai/v1/systemone"

    @classmethod
    def _validate_model(cls, model: str) -> None:
        if model not in {"jev-latest", "jev-preview", "jev-1.13.0", "jev-1.14.0"}:
            raise ValueError(f"TypeSafe decisions API does not support model: {model}")


class OpenAIDecisionsAPI(_HTTPDecisionsAPI):
    """OpenAI's native Decisions API."""

    DEFAULT_MODEL = "gpt-6-luna"
    API_URL = "https://api.openai.com/v1/decisions"

    @classmethod
    def _validate_model(cls, model: str) -> None:
        if model != "gpt-6-luna":
            raise ValueError(f"OpenAI decisions API does not support model: {model}")

    async def process_request(  # pylint: disable=too-many-branches
        self, state: Dict, questions: Dict[str, Question]
    ) -> Dict[str, Decision]:
        if not questions:
            raise ValueError("At least one decision question is required")
        wire_questions = []
        for name, question in questions.items():
            question_type = _question_type(question)
            wire = {"name": name, "type": "predicate" if question_type == "noul" else question_type}
            instructions = question.instructions or ""
            if question_type == "noul":
                if not isinstance(question.criteria, BinaryCriteria):
                    raise TypeError("Binary question requires BinaryCriteria")
                instructions = (
                    f"{instructions}\nTrue: {question.criteria.as_true}\n"
                    f"False: {question.criteria.as_false}"
                ).strip()
            elif question_type == "choice":
                wire["choices"] = [
                    {"value": value, "description": description}
                    for value, description in question.criteria.items()
                ]
            else:
                wire["levels"] = [{"label": label} for label in question.criteria]
            wire["instructions"] = instructions
            wire_questions.append(wire)
        data = await self._post({
            "model": self.model,
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": wire_questions,
        })
        result = {}
        for answer in data["answers"]:
            name = answer["name"]
            if name not in questions:
                raise ValueError(f"Unexpected decision answer: {name}")
            if answer["type"] == "refusal":
                raise ValueError(f"Decision refused for question: {name}")
            question_type = _question_type(questions[name])
            expected_type = "predicate" if question_type == "noul" else question_type
            if answer["type"] != expected_type:
                raise ValueError(f"Unexpected answer type for {name}: {answer['type']}")
            if question_type == "noul":
                result[name] = _binary_decision(answer["probability"])
            else:
                probabilities = {
                    str(item["value"]): item["probability"]
                    for item in answer.get("probabilities", [])
                }
                value = answer["choice"] if question_type == "choice" else answer["score"]
                result[name] = Decision(value, answer["confidence"], probabilities)
        if result.keys() != questions.keys():
            raise ValueError("Missing decision answers")
        return result
