"""Read model information rows from a directory of CSV files."""

import csv
import os
from typing import Iterator, Optional


def iter_model_info_rows(model_info_dir: str) -> Iterator[tuple[str, str, Optional[str], dict]]:
    """Yield model identities and rows from CSV and TXT files recursively."""
    for root, _dirs, files in os.walk(model_info_dir):
        for name in files:
            if not name.endswith((".csv", ".txt")):
                continue
            with open(os.path.join(root, name), newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    continue
                reader.fieldnames = [field.strip() for field in reader.fieldnames]
                for row in reader:
                    try:
                        provider = row["PROVIDER"].strip()
                        model = row["MODEL"].strip()
                    except (KeyError, AttributeError):
                        continue
                    if provider and model:
                        family = (row.get("MODEL_FAMILY") or "").strip() or None
                        yield provider, model, family, row
