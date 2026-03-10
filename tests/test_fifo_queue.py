"""Tests for FiFoQueue implementation."""
# pylint: disable=unused-argument
import dbzero as db0
from statek.fifo_queue import FiFoQueue


def test_fifo_queue_can_be_created(db0_fixture):
    queue = FiFoQueue()
    assert queue is not None


def test_push_back_adds_element(db0_fixture):
    queue = FiFoQueue()
    queue.push_back(first=123, second="my item")
    result = queue.pop_front(1)
    assert len(result) == 1


def test_pop_front_returns_correct_kwargs(db0_fixture):
    queue = FiFoQueue()
    queue.push_back(first=123, second="my item")
    result = queue.pop_front(10)
    assert result == [{"first": 123, "second": "my item"}]


def test_pop_front_respects_count(db0_fixture):
    queue = FiFoQueue()
    for i in range(5):
        queue.push_back(value=i)
    result = queue.pop_front(3)
    assert len(result) == 3


def test_pop_front_returns_fifo_order(db0_fixture):
    queue = FiFoQueue()
    for i in range(5):
        queue.push_back(value=i)
    result = queue.pop_front(5)
    values = [r["value"] for r in result]
    assert values == [0, 1, 2, 3, 4]


def test_pop_front_removes_elements(db0_fixture):
    queue = FiFoQueue()
    queue.push_back(value=1)
    queue.push_back(value=2)
    result1 = queue.pop_front(1)
    assert result1 == [{"value": 1}]
    result2 = queue.pop_front(1)
    assert result2 == [{"value": 2}]


def test_pop_front_on_empty_queue_returns_empty_list(db0_fixture):
    queue = FiFoQueue()
    result = queue.pop_front(10)
    assert not result


def test_pop_front_with_count_exceeding_queue_size(db0_fixture):
    queue = FiFoQueue()
    queue.push_back(value=1)
    result = queue.pop_front(100)
    assert len(result) == 1


def test_pop_front_returns_memo_object_value(db0_fixture):
    """pop_front returns dict with accessible memo object values."""
    @db0.memo
    class SampleMemoObj:  # pylint: disable=too-few-public-methods
        def __init__(self, val):
            self.val = val

    obj = SampleMemoObj(99)
    queue = FiFoQueue()
    queue.push_back(item=obj)
    result = queue.pop_front(1)
    assert len(result) == 1
    assert result[0]["item"].val == 99


def test_multiple_pushes_after_pops(db0_fixture):
    """Keys keep incrementing; new pushes are appended at the back."""
    queue = FiFoQueue()
    queue.push_back(value=1)
    queue.push_back(value=2)
    queue.pop_front(2)
    queue.push_back(value=3)
    queue.push_back(value=4)
    result = queue.pop_front(10)
    values = [r["value"] for r in result]
    assert values == [3, 4]


def test_pop_front_leaves_remaining_elements(db0_fixture):
    """Elements not consumed by pop_front stay in the queue."""
    queue = FiFoQueue()
    for i in range(5):
        queue.push_back(value=i)
    queue.pop_front(2)
    result = queue.pop_front(10)
    values = [r["value"] for r in result]
    assert values == [2, 3, 4]


# --- filter parameter ---

def test_pop_front_filter_selects_matching_items(db0_fixture):
    """Only items passing the filter are returned."""
    queue = FiFoQueue()
    for i in range(5):
        queue.push_back(value=i)
    result = queue.pop_front(10, filter=lambda value: value % 2 == 0)
    values = [r["value"] for r in result]
    assert values == [0, 2, 4]


def test_pop_front_filter_non_matching_items_stay_in_queue(db0_fixture):
    """Items that do not pass the filter remain in the queue."""
    queue = FiFoQueue()
    for i in range(4):
        queue.push_back(value=i)
    queue.pop_front(10, filter=lambda value: value % 2 == 0)
    # Odd items (1, 3) should still be in the queue
    result = queue.pop_front(10)
    values = [r["value"] for r in result]
    assert values == [1, 3]


def test_pop_front_filter_respects_count(db0_fixture):
    """count limits how many passing items are returned."""
    queue = FiFoQueue()
    for i in range(6):
        queue.push_back(value=i)
    result = queue.pop_front(2, filter=lambda value: value % 2 == 0)
    values = [r["value"] for r in result]
    assert values == [0, 2]


def test_pop_front_filter_count_limited_items_stay_in_queue(db0_fixture):
    """Passing items not consumed due to count remain in the queue."""
    queue = FiFoQueue()
    for i in range(6):
        queue.push_back(value=i)
    queue.pop_front(2, filter=lambda value: value % 2 == 0)
    # 4 was passing but not consumed; 1, 3, 5 did not pass the filter
    result = queue.pop_front(10)
    values = [r["value"] for r in result]
    assert values == [1, 3, 4, 5]


def test_pop_front_filter_none_behaves_like_no_filter(db0_fixture):
    """Passing filter=None is identical to calling without a filter."""
    queue = FiFoQueue()
    for i in range(3):
        queue.push_back(value=i)
    result = queue.pop_front(10, filter=None)
    values = [r["value"] for r in result]
    assert values == [0, 1, 2]


def test_pop_front_filter_all_rejected_returns_empty(db0_fixture):
    """When every item is rejected by the filter the result is empty."""
    queue = FiFoQueue()
    for i in range(3):
        queue.push_back(value=i)
    result = queue.pop_front(10, filter=lambda value: False)
    assert not result


def test_pop_front_filter_all_rejected_queue_unchanged(db0_fixture):
    """When every item is rejected, the queue is left intact."""
    queue = FiFoQueue()
    for i in range(3):
        queue.push_back(value=i)
    queue.pop_front(10, filter=lambda value: False)
    result = queue.pop_front(10)
    values = [r["value"] for r in result]
    assert values == [0, 1, 2]


def test_pop_front_filter_receives_all_kwargs(db0_fixture):
    """The filter callable receives all kwargs of the item as keyword arguments."""
    queue = FiFoQueue()
    queue.push_back(a=1, b=2)
    queue.push_back(a=10, b=20)
    result = queue.pop_front(10, filter=lambda a, b: a + b > 10)
    assert len(result) == 1
    assert result[0] == {"a": 10, "b": 20}
