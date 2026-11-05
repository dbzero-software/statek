"""Parse the fraction of a model's context window that triggers compaction."""

import math
from typing import Optional


DEFAULT_COMPACTION_THRESHOLD = 0.65


def parse_compaction_threshold(value) -> Optional[float]:
    """Accept None to disable compaction, or a finite fraction in (0, 1]."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    message = "COMPACTION_THRESHOLD must be None or a finite number in (0, 1]"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError(message)
    return threshold
