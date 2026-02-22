"""Tests for load_examples function."""

import os
import pytest

from statek.executors.example import load_examples


EXAMPLE_001 = """\
# seq_id: 0
# name: Answering a simple question
```python
user, message = fetch_next_message()
print(f"Message: {message.message}")
```
Message: What are my on-calls for next week?
```python
recent_history = chat_history(10)
if not recent_history:
    print("Chat history is empty")
else:
    print(f"Recent history: {recent_history}")
```
# ----------
```python
# To answer the user's question about their on-calls for next week, I will first retrieve their calendar using the `get_user_calendar()` function. Then, I will analyze the calendar to find the on-call schedule for the next week. Let's start by retrieving the user's calendar.
calendar = get_user_calendar()
calendar
```
get_user_calendar() log: user Piotr Janusz Lobowicki does not have any on-call assignments.
```python
# It seems that the user Piotr Janusz Lobowicki does not have any on-call assignments.
# I will answer his question and exit.
answer("You have no on-call assigments next week")
```
"""


@pytest.fixture(autouse=True)
def clear_cache():
    load_examples.cache_clear()
    yield
    load_examples.cache_clear()


def test_load_example(temp_dir):
    os.makedirs(os.path.join(temp_dir, "message_dispatcher"), exist_ok=True)
    path = os.path.join(temp_dir, "message_dispatcher", "example-001.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(EXAMPLE_001)

    result = load_examples(temp_dir)

    assert len(result) == 1
    ex = result[0]

    assert ex.example_metadata["seq_id"] == 0
    assert ex.example_metadata["name"] == "Answering a simple question"

    # Two warmup blocks: [code, console, code, console]
    assert len(ex.warmup_items) == 4
    assert "fetch_next_message()" in ex.warmup_items[0]
    assert ex.warmup_items[1] == "Message: What are my on-calls for next week?"
    assert "chat_history(10)" in ex.warmup_items[2]
    assert ex.warmup_items[3] == ""

    # Two LLM response blocks: [code, console, code, console]
    assert len(ex.example_items) == 4
    assert "get_user_calendar()" in ex.example_items[0]
    assert "does not have any on-call assignments" in ex.example_items[1]
    assert 'answer("You have no on-call assigments next week")' in ex.example_items[2]
    assert ex.example_items[3] == ""
