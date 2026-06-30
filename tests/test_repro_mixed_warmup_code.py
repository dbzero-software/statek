"""Reproductions for Statek warmup handling bugs found by downstream integration."""

import pytest

from statek.executors.job import parse_warmup_code
from statek.utils import CallSpec, CodeBlock

pytestmark = pytest.mark.usefixtures("db0_fixture")

pytestmark = pytest.mark.usefixtures("db0_fixture")


def test_parse_warmup_code_accepts_mixed_string_and_codeblock_sequence():
    warmup = [
        "x = 1",
        CodeBlock(
            code="docstr('Tool')",
            tool_calls=[CallSpec(id="STATEK-001", func_name="docstr", args=["Tool"], kwargs={})],
        ),
    ]

    parsed = parse_warmup_code(warmup)

    assert parsed == warmup
