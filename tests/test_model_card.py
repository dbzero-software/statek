"""Tests for persistent model cards."""

# pylint: disable=no-member,unused-argument

import json

import dbzero as db0
import pytest

from statek.agents.dialog_agent import RecurringReminder
from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, NotificationLogItem,
    PostProcessedItem, ReminderLogItem, SubTaskLogItem, SystemLogItem,
    ToolError, UserLogItem, WarmupLogItem,
)
from statek.executors.post_processor import FinalCheck
from statek.model_card import ModelCard, get_model_card, set_model_card
from statek.task import SubTaskHandler
from statek.task_difficulty import TaskDifficulty
from statek.utils import CallSpec, CodeBlock


def test_get_creates_default_card_with_normalized_tags(db0_fixture):
    card = get_model_card("Open AI", "gpt 4", model_family="GPT Family")

    assert card.max_context_length == 0
    assert card.token_to_bytes == 0.0
    assert card.total_parameter_count is None
    assert card.active_parameter_count is None
    assert card in list(db0.find(ModelCard, "OPENAI", "GPT4", "GPTFAMILY"))
    assert get_model_card("openai", "gpt4", "gptfamily") is card


def test_get_no_create_and_constructor_values(db0_fixture):
    assert get_model_card("openai", "gpt-4", no_create=True) is None

    card = get_model_card(
        "openai", "gpt-4", max_context_length=128000, token_to_bytes=2.5
    )
    assert card.max_context_length == 128000
    assert card.token_to_bytes == 2.5
    assert get_model_card("openai", "gpt-4", max_context_length=256000) is card


def test_set_initializes_default_fields_and_is_idempotent(db0_fixture):
    card = get_model_card("openai", "gpt-4")

    result = set_model_card(
        "openai", "gpt-4", max_context_length=128000,
        token_to_bytes=2.5, total_parameter_count=100,
    )
    assert result is card
    assert result.max_context_length == 128000
    assert result.token_to_bytes == 2.5
    assert result.total_parameter_count == 100
    assert result.active_parameter_count is None
    assert set_model_card("openai", "gpt-4", max_context_length=128000) is card
    assert list(db0.find(ModelCard, "OPENAI", "GPT-4")) == [card]


def test_set_creates_card_if_missing(db0_fixture):
    card = set_model_card(
        "openrouter", "llama", model_family="meta", active_parameter_count=10
    )

    assert card.max_context_length == 0
    assert card.active_parameter_count == 10
    assert card in list(db0.find(ModelCard, "OPENROUTER", "LLAMA", "META"))


def test_set_rejects_changed_initialized_value_atomically(db0_fixture):
    card = set_model_card("openai", "gpt-4", max_context_length=128000)

    with pytest.raises(ValueError, match="max_context_length"):
        set_model_card(
            "openai", "gpt-4", token_to_bytes=2.5, max_context_length=256000
        )

    assert card.max_context_length == 128000
    assert card.token_to_bytes == 0.0
    assert list(db0.find(ModelCard, "OPENAI", "GPT-4")) == [card]


def test_set_rejects_changed_optional_parameter_count(db0_fixture):
    card = set_model_card("openai", "gpt-4", total_parameter_count=100)

    with pytest.raises(ValueError, match="total_parameter_count"):
        set_model_card("openai", "gpt-4", total_parameter_count=200)

    assert card.total_parameter_count == 100


@pytest.mark.parametrize("payload,expected", [
    ("", (0, 0)), ("abcdefghij", (3, 5)), ("a", (0, 1)),
    ("żółw😀", (4, 5)), ({"text": "ż", "values": [1, None]}, (13, 15)),
])
def test_est_token_count_utf8_and_json(db0_fixture, payload, expected):
    card = ModelCard(token_to_bytes=2.5)

    assert card.est_token_count(payload) == expected


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_est_token_count_rejects_invalid_ratio(db0_fixture, ratio):
    with pytest.raises(ValueError, match="token_to_bytes"):
        ModelCard(token_to_bytes=ratio).est_token_count("")


@pytest.mark.parametrize("payload", [None, 3, b"text", ["text"], {"bad": object()}])
def test_est_token_count_rejects_unsupported_payload(db0_fixture, payload):
    with pytest.raises(TypeError):
        ModelCard(token_to_bytes=2.5).est_token_count(payload)


def test_est_token_count_all_log_types(db0_fixture, job_factory):
    card = ModelCard(token_to_bytes=2.5)
    handler = SubTaskHandler(job_factory(), id="child")
    handler.complete(result="not included")
    cases = [
        (ChatLogItem(0, tool_log=["abc", ToolError("ż")]), "abcż"),
        (WarmupLogItem(0, tool_log="result", warmup_block_num=999), "result"),
        (SystemLogItem(console_pos=0, message="system", tool_log="ignored"), "system"),
        (PostProcessedItem(console_pos=0, post_processor=FinalCheck(), message="review"),
         "review"),
        (ReminderLogItem(console_pos=0, reminder=RecurringReminder(text="remember")),
         "remember"),
        (SubTaskLogItem(console_pos=0, handler=handler, tool_log="result"),
         handler.get_log_message() + "result"),
        (NotificationLogItem(console_pos=0, message="notice"), "notice"),
        (NotificationLogItem(console_pos=0, message="notice", tool_log=ToolError("failed")),
         "failed"),
        (NotificationLogItem(console_pos=0, message="notice", tool_log=""), ""),
        (DifficultyChangeLogItem(console_pos=0, previous_difficulty=TaskDifficulty.low,
                                 difficulty=TaskDifficulty.medium, cumulative_token_usage=999,
                                 tool_log="ignored"), ""),
        (UserLogItem(message="hello"), "hello"),
        (LLM_LogItem(0, llm_resp="response", tool_log="result",
                     cumulative_token_usage=999),
         "responseresult"),
        (LLM_LogItem(0), ""),
        (WarmupLogItem(0), ""),
        (ChatLogItem(0, tool_log=ToolError("failed")), "failed"),
    ]
    for item, text in cases:
        assert card.est_token_count(item) == card.est_token_count(text), type(item).__name__


@pytest.mark.parametrize("reasoning", ["ż reasoning", b"opaque\xff", {"text": ["ż"]}])
def test_est_token_count_structured_response(db0_fixture, reasoning):
    card = ModelCard(token_to_bytes=2.5)
    call = CallSpec("call-1", "lookup", args=["ż"], kwargs={"filters": {"limit": [2]}})
    block = CodeBlock("code", tool_calls=[call], metadata={"ignored": "x" * 1000})
    item = LLM_LogItem(123, llm_resp=block, llm_reasoning_payload=reasoning,
                       tool_log=["result", ToolError("failure")])
    calls_text = json.dumps([{
        "id": "call-1", "name": "lookup", "args": ["ż"],
        "kwargs": {"filters": {"limit": [2]}},
    }], ensure_ascii=False)
    if isinstance(reasoning, bytes):
        reasoning_text = "x" * len(reasoning)
    elif isinstance(reasoning, str):
        reasoning_text = reasoning
    else:
        reasoning_text = json.dumps(reasoning, ensure_ascii=False)
    expected_text = "code" + calls_text + reasoning_text + "resultfailure"

    assert card.est_token_count(item) == card.est_token_count(expected_text)
    assert block.metadata == {"ignored": "x" * 1000}
    assert call.args == ["ż"]
    assert item.console_pos == 123
    assert item.tool_log[1].err_message == "failure"


def test_est_token_count_persistent_dict(db0_fixture):
    payload = db0.dict({"nested": ["ż", {"values": (1, 2)}]})
    card = ModelCard(token_to_bytes=2.5)

    assert card.est_token_count(payload) == card.est_token_count(
        {"nested": ["ż", {"values": (1, 2)}]}
    )


@pytest.mark.parametrize("margin,expected", [(0.05, (95, 105)), (0.2, (80, 120)),
                                           (0, (100, 100))])
def test_estimation_margin(db0_fixture, margin, expected):
    card = ModelCard(token_to_bytes=4, token_estimation_margin=margin)
    assert card.est_token_count("x" * 400) == expected


@pytest.mark.parametrize("margin", [-0.1, 1, float("inf"), float("nan")])
def test_invalid_estimation_margin(db0_fixture, margin):
    card = ModelCard(token_to_bytes=4, token_estimation_margin=margin)
    with pytest.raises(ValueError, match="token_estimation_margin"):
        card.est_token_count("hello")


@pytest.mark.parametrize("margin", [0, 0.1])
def test_set_initializes_margin_once(db0_fixture, margin):
    card = get_model_card("test", "margin")
    assert card.token_estimation_margin == 0.05
    assert set_model_card("test", "margin", token_estimation_margin=margin) is card
    assert set_model_card("test", "margin", token_estimation_margin=margin) is card
    with pytest.raises(ValueError, match="token_estimation_margin"):
        set_model_card("test", "margin", token_to_bytes=4, token_estimation_margin=0.2)
    assert card.token_to_bytes == 0


@pytest.mark.parametrize("margin,expected", [("", 0.05), ("0.15", 0.15),
                                           ("nan", None), ("-1", None), ("1", None)])
def test_load_estimation_margin(db0_fixture, tmp_path, margin, expected):
    from statek.model_card import init_model_cards  # pylint: disable=import-outside-toplevel
    (tmp_path / "cards.csv").write_text(
        "PROVIDER,MODEL,MAX_CONTEXT_LENGTH,TOKEN_TO_BYTES,TOKEN_ESTIMATION_MARGIN\n"
        f"test,margin,1000,4,{margin}\n", encoding="utf-8",
    )
    init_model_cards(str(tmp_path))
    card = get_model_card("test", "margin", no_create=True)
    if expected is None:
        assert card is None
    else:
        assert card.token_estimation_margin == expected


def test_legacy_card_uses_default_margin(db0_fixture):
    card = ModelCard(token_to_bytes=4)
    del card.token_estimation_margin
    assert card.est_token_count("x" * 400) == (95, 105)
