"""Task difficulty helpers for model selection."""

from typing import Optional

import dbzero as db0


@db0.enum(values=["low", "medium", "high"])
class TaskDifficulty:
    pass


TASK_DIFFICULTY_ORDER = [
    TaskDifficulty.low,  # pylint: disable=no-member
    TaskDifficulty.medium,  # pylint: disable=no-member
    TaskDifficulty.high,  # pylint: disable=no-member
]

_TASK_DIFFICULTY_BY_LABEL = {
    "L": TaskDifficulty.low,  # pylint: disable=no-member
    "LOW": TaskDifficulty.low,  # pylint: disable=no-member
    "M": TaskDifficulty.medium,  # pylint: disable=no-member
    "MEDIUM": TaskDifficulty.medium,  # pylint: disable=no-member
    "H": TaskDifficulty.high,  # pylint: disable=no-member
    "HIGH": TaskDifficulty.high,  # pylint: disable=no-member
}


def parse_task_difficulty(value) -> Optional[TaskDifficulty]:
    """Parse a difficulty label into ``TaskDifficulty``.

    Accepts the enum value itself plus the short labels L/M/H and full labels
    low/medium/high, case-insensitively.
    """
    if value is None:
        return None
    if value in TASK_DIFFICULTY_ORDER:
        return value
    label = str(value).strip().upper()
    try:
        return _TASK_DIFFICULTY_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Invalid task difficulty: {value!r}") from exc


def max_task_difficulty(left: TaskDifficulty, right: TaskDifficulty) -> TaskDifficulty:
    """Return the higher of two task difficulty values."""
    return (
        left
        if TASK_DIFFICULTY_ORDER.index(left) >= TASK_DIFFICULTY_ORDER.index(right)
        else right
    )
