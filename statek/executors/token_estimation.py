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

"""Resolve model cards and maintain stored estimates for job chat-log items."""

# pylint: disable=protected-access

from typing import TYPE_CHECKING, Optional, Union

from statek.executors.chat_log_item import (
    ChatLogItem, DifficultyChangeLogItem, LLM_LogItem, UserLogItem,
)
from statek.model_card import get_model_card

if TYPE_CHECKING:
    from statek.executors.job import Job


def get_token_estimation_card(
    job: "Job", item: Optional[Union[ChatLogItem, UserLogItem]] = None, model=None, provider=None,
):
    """Resolve a card without creating one or changing the job's difficulty."""
    # Defer shared model-selection helpers until Job has finished loading.
    from statek.executors.job import (  # pylint: disable=import-outside-toplevel,cyclic-import
        _get_static_task_difficulty, _is_model_mapping,
    )

    metadata = job.job_def.metadata or {}
    if model is None:
        difficulty = None
        if _is_model_mapping(metadata.get("MODEL")):
            difficulty = (getattr(job, "_Job__last_difficulty", None)
                          or _get_static_task_difficulty(metadata))
            if isinstance(item, LLM_LogItem) and item.llm_resp is not None:
                difficulty = (item.request_difficulty
                              or _get_static_task_difficulty(metadata))
        model = job.get_current_model(difficulty=difficulty)
    if model is None:
        raise ValueError("Cannot estimate chat-log tokens: model is not configured")
    provider, model_id, family = job._model_identity(model, provider=provider)
    card = get_model_card(provider, model_id, family, no_create=True)
    if card is None:
        raise ValueError(f"Missing model card for {provider or 'UNKNOWN'}/{model_id}")
    return card


def refresh_token_usage(
    job: "Job", item: Union[ChatLogItem, UserLogItem], model=None, provider=None,
) -> None:
    """Store the high estimate using the currently resolved model card."""
    if isinstance(item, DifficultyChangeLogItem):
        item.est_token_usage = 0
        return
    card = get_token_estimation_card(job, item, model=model, provider=provider)
    item.est_token_usage = card.est_token_count(item)[1]
