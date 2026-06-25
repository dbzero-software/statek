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

"""Task difficulty helpers for model selection."""

from typing import Optional

import dbzero as db0


@db0.enum(values=["low", "medium", "high"])
class TaskDifficulty:
    pass


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
    if value in TaskDifficulty.values():  # pylint: disable=no-member
        return value
    label = str(value).strip().upper()
    try:
        return _TASK_DIFFICULTY_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Invalid task difficulty: {value!r}") from exc


def max_task_difficulty(left: TaskDifficulty, right: TaskDifficulty) -> TaskDifficulty:
    """Return the higher of two task difficulty values."""
    values = tuple(TaskDifficulty.values())  # pylint: disable=no-member
    return left if values.index(left) >= values.index(right) else right
