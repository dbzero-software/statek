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

"""Post-processing contract for LLM step handling."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, Union

import dbzero as db0

from statek.llm_api import LLM_StepData

if TYPE_CHECKING:
    from statek.executors.job import Job


@db0.memo
class PostProcessor(ABC):
    """
    Base class for objects that can inspect or transform an LLM step.

    A post-processor runs after an LLM response is received and before the
    response is committed to job history. Implementations may return the input
    step, a replacement step, a system message, or a tuple combining those
    values.
    """

    @abstractmethod
    def process(
        self,
        llm_step: LLM_StepData,
        job: "Job",
    ) -> Union[LLM_StepData, str, Tuple[Any, ...]]:
        """
        Process a single LLM step.

        Args:
            llm_step: the LLM response (unprocessed, not included in the Job history yet)
            job: the job instance (for context)

        Returns either:
            - the LLM_StepData instance (possibly `llm_step`)
            - str: the message to be appended with the role = system
            - any combination of the above (as tuple)
        """


PostProcessingInput = Optional[Union[PostProcessor, Iterable[PostProcessor]]]
PostProcessorIdentity = Tuple[str, str, Any]


def _freeze_identity_value(value: Any) -> Any:
    """Return a stable comparable representation for simple processor state."""
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_identity_value(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_identity_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_identity_value(item) for item in value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def post_processor_identity(post_processor: PostProcessor) -> PostProcessorIdentity:
    """Return the value identity used for JobDef de-duplication."""
    state = getattr(post_processor, "__dict__", {})
    return (
        post_processor.__class__.__module__,
        post_processor.__class__.__qualname__,
        _freeze_identity_value(state),
    )


def normalize_post_processing(post_processing: PostProcessingInput) -> Tuple[PostProcessor, ...]:
    """Return post-processing configuration as an ordered tuple.

    Args:
        post_processing: None, a single post-processor, or an iterable of
            post-processors.

    Returns:
        Ordered tuple of post-processors.

    Raises:
        TypeError: If any provided value is not a PostProcessor.
    """
    if post_processing is None:
        return ()
    if isinstance(post_processing, PostProcessor):
        return (post_processing,)
    if isinstance(post_processing, (str, bytes)):
        raise TypeError("post_processing must contain PostProcessor instances")

    processors = tuple(post_processing)
    for processor in processors:
        if not isinstance(processor, PostProcessor):
            raise TypeError("post_processing must contain PostProcessor instances")
    return processors


def post_processing_identity(post_processing: PostProcessingInput) -> Tuple[PostProcessorIdentity, ...]:
    """Return stable value identity for a post-processing sequence."""
    return tuple(post_processor_identity(processor) for processor in normalize_post_processing(post_processing))


def stored_post_processing(post_processing: PostProcessingInput) -> Optional[list[PostProcessor]]:
    """Return normalized post-processing value for storage on JobDef."""
    processors = normalize_post_processing(post_processing)
    return list(processors) if processors else None
