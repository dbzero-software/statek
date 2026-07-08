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

import ast
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Tuple, Union

import dbzero as db0

from statek.executors.chat_log_item import PostProcessedItem
from statek.llm_api import CallParams, LLM_StepData

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


@db0.memo
@dataclass
class FinalCheck(PostProcessor):
    """Ask the LLM to review an early final answer before accepting it."""

    max_llm_actions: int = 3

    def process(
        self,
        llm_step: LLM_StepData,
        job: "Job",
    ) -> Union[LLM_StepData, str]:
        """Return a reflection prompt when an early final answer is detected.

        Args:
            llm_step: Uncommitted LLM response to inspect.
            job: Job whose history determines finality, action count, and prior
                post-processor activation.

        Returns:
            A system message string when activated, otherwise the original LLM
            step unchanged.
        """
        if not job.is_step_final(llm_step):
            return llm_step
        if job.count_llm_actions() >= self.max_llm_actions:
            return llm_step
        already_fired = any(
            isinstance(item, PostProcessedItem) and item.post_processor is self
            for item in job.chat_log
        )
        if already_fired:
            return llm_step

        return (
            "Final check: reflect on the user's latest request and your draft answer. "
            "Make sure it is correct, complete, and directly answers the user. "
            "Revise if needed; otherwise return it unchanged.\n\n"
            f"Draft answer:\n{llm_step.text}"
        )


PostProcessingInput = Optional[Union[PostProcessor, Iterable[PostProcessor]]]
PostProcessorIdentity = Tuple[str, str, Any]
_POST_PROCESSOR_REGISTRY: dict[str, type[PostProcessor]] = {
    "FinalCheck": FinalCheck,
}


def _parse_post_processor_call(call: ast.AST) -> CallParams:
    """Parse one post-processor constructor AST call into CallParams."""
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("Invalid post-processor metadata definition")
    if call.args:
        raise ValueError("Invalid post-processor metadata definition")

    kwargs: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("Invalid post-processor metadata definition")
        if keyword.arg in kwargs:
            raise ValueError("Invalid post-processor metadata definition")
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid post-processor metadata definition") from exc

    return CallParams(call_id=call.func.id, name=call.func.id, args=[], kwargs=kwargs)


def parse_post_processors(post_processors_def: str) -> list[CallParams]:
    """Parse comma-delimited post-processor metadata definitions.

    Args:
        post_processors_def: Comma-delimited constructor snippets.

    Returns:
        Parsed post-processor call parameters in metadata order.

    Raises:
        ValueError: If any definition is not a simple constructor call with literal
            keyword arguments.
    """
    if not post_processors_def.strip():
        return []

    try:
        expression = ast.parse(post_processors_def, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid post-processor metadata definition") from exc

    if isinstance(expression.body, ast.Tuple):
        calls = expression.body.elts
    else:
        calls = [expression.body]

    return [_parse_post_processor_call(call) for call in calls]


def _resolve_post_processor_call(call_params: CallParams) -> PostProcessor:
    """Resolve one parsed post-processor call into an instance."""
    processor_cls = _POST_PROCESSOR_REGISTRY.get(call_params.name)
    if processor_cls is None:
        raise ValueError(f"Unknown post-processor: {call_params.name}")
    if call_params.args:
        raise ValueError("Post-processor metadata must use keyword arguments only")

    kwargs = call_params.kwargs or {}
    expected_state = _post_processor_expected_state(processor_cls, kwargs)
    for existing_processor in db0.find(processor_cls):  # pylint: disable=no-member
        if _post_processor_matches_state(existing_processor, expected_state):
            return existing_processor

    processor = processor_cls(**kwargs)
    if not isinstance(processor, PostProcessor):
        raise TypeError("Resolved post-processor must be a PostProcessor instance")
    return processor


def _post_processor_expected_state(
    processor_cls: type[PostProcessor],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return constructor-bound state requested by metadata."""
    bound_kwargs = inspect.signature(processor_cls).bind(**kwargs)
    bound_kwargs.apply_defaults()
    return dict(bound_kwargs.arguments)


def _post_processor_matches_state(
    post_processor: PostProcessor,
    expected_state: dict[str, Any],
) -> bool:
    """Return whether an existing processor has the requested state."""
    missing = object()
    return all(
        getattr(post_processor, name, missing) == value
        for name, value in expected_state.items()
    )


def resolve_post_processors(call_params: list[CallParams]) -> list[PostProcessor]:
    """Resolve parsed post-processor metadata into executable processors.

    Args:
        call_params: Parsed post-processor definitions.

    Returns:
        Resolved post-processors in input order.

    Raises:
        ValueError: If a parsed processor name is unknown or uses positional args.
    """
    return [_resolve_post_processor_call(call_param) for call_param in call_params]


def resolve_post_processors_from_metadata(
    metadata: Optional[Mapping[str, str]],
) -> Optional[list[PostProcessor]]:
    """Resolve the POST_PROCESSORS metadata value into executable processors."""
    if not metadata:
        return None
    post_processors_def = metadata.get("POST_PROCESSORS")
    if post_processors_def is None or not post_processors_def.strip():
        return None
    return resolve_post_processors(parse_post_processors(post_processors_def))


def effective_post_processing(
    post_processing: PostProcessingInput,
    metadata: Optional[Mapping[str, str]],
) -> PostProcessingInput:
    """Return explicit post-processing or the metadata-derived default."""
    if post_processing is not None:
        return post_processing
    return resolve_post_processors_from_metadata(metadata)


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
