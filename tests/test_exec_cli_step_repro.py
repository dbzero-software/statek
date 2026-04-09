"""Reproducer for: python_cli results sometimes not evaluated.

Observed behaviour: the final statement of a python_cli block was a
parenthesized f-string expression, e.g.

    draft = create_draft_preference(...)
    est_cost = draft.calculate_est_budget()
    strength = draft.strength
    (f'Draft created with strength: {strength}, estimated cost: {est_cost} points')

The LLM chat log showed the tool result as ``<CONSOLE_OUTPUT>\n\n</CONSOLE_OUTPUT>``
— i.e. the console_append callback was never invoked, so the REPL-style
echo of the trailing expression produced no output at all.
"""

import pytest

from statek.exceptions import FutureError
from statek.executors.utils import exec_cli_step


DEFAULT_JOB_PARAMS = {"goal": "Test goal"}


def _raising_call(*_args, **_kwargs):
    raise RuntimeError("boom from calculate_est_budget")


def _return_none(*_args, **_kwargs):
    return None


def _raise_future(*_args, **_kwargs):
    # Simulate a temporal-function suspension: FutureError is reserved for
    # temporal functions whose result isn't ready yet.  When it propagates
    # out of a python_cli step, the job is SUSPENDED with nothing echoed.
    raise FutureError(future_result=None)


@pytest.mark.asyncio
async def test_parenthesized_fstring_expression(job_factory):
    """A parenthesized f-string as the final statement should be echoed.

    Reproduces the production log where a block ending in
    ``(f'Draft created with strength: {strength}, ...')`` produced an
    empty CONSOLE_OUTPUT.  The expression evaluates to a plain string
    and should be forwarded to the console_append callback.
    """
    job = job_factory(job_params=DEFAULT_JOB_PARAMS)
    outputs = []
    code = (
        "strength = 0.5\n"
        "est_cost = 10\n"
        "(f'Draft created with strength: {strength}, estimated cost: {est_cost} points')\n"
    )
    await exec_cli_step(code, job, outputs.append)

    # If the REPL echo is dropped, ``outputs`` is empty — reproducing the
    # production symptom (empty tool result for the python_cli call).
    assert outputs, (
        "Parenthesized f-string expression produced no console output — "
        "this reproduces the empty CONSOLE_OUTPUT seen in production."
    )
    assert "Draft created with strength: 0.5" in outputs[-1]
    assert "estimated cost: 10" in outputs[-1]


@pytest.mark.asyncio
async def test_plain_fstring_expression(job_factory):
    """Same code without parentheses — control case."""
    job = job_factory(job_params=DEFAULT_JOB_PARAMS)
    outputs = []
    code = (
        "strength = 0.5\n"
        "est_cost = 10\n"
        "f'Draft created with strength: {strength}, estimated cost: {est_cost} points'\n"
    )
    await exec_cli_step(code, job, outputs.append)
    assert outputs, "Bare f-string expression produced no console output"
    assert "Draft created with strength: 0.5" in outputs[-1]


@pytest.mark.asyncio
async def test_intermediate_statement_raises(job_factory):
    """An intermediate statement raising an exception — what reaches the console?

    This is the most likely root cause of the empty CONSOLE_OUTPUT in
    production: ``draft.calculate_est_budget()`` (or similar) raised, the
    REPL-style echo of the final f-string never ran, and something in the
    outer layers swallowed the error text.
    """
    job = job_factory(job_params=DEFAULT_JOB_PARAMS)
    job.py_env.local_state = {"calculate_est_budget": _raising_call}
    outputs = []
    # Flattened to plain values so intermediate state is dbzero-compatible —
    # the shape still mirrors production (4 statements, final = f-string echo).
    code = (
        "strength = 0.5\n"
        "est_cost = calculate_est_budget(unit_budget=2)\n"
        "(f'Draft created with strength: {strength}, estimated cost: {est_cost} points')\n"
    )
    # _exec_code_body re-raises after forwarding the error to error_fn, so we
    # expect exec_cli_step to propagate the exception here.  The important
    # question is whether the error text was appended to outputs first.
    with pytest.raises(RuntimeError):
        await exec_cli_step(code, job, outputs.append)

    # Document what actually happens — if outputs is empty here, the LLM
    # would see an empty CONSOLE_OUTPUT just like in the production log.
    assert outputs, (
        "Intermediate exception produced no console output at the "
        "exec_cli_step level — this would reach the LLM as empty CONSOLE_OUTPUT."
    )
    assert any("boom from calculate_est_budget" in o for o in outputs)


@pytest.mark.asyncio
async def test_intermediate_statement_returns_none_then_fstring(job_factory):
    """calculate_est_budget() legitimately returns None — does the f-string echo?

    The production ``calculate_est_budget`` can return ``None`` (e.g. when
    ``effective_unit_budget`` is ``None``).  If the final f-string still
    gets echoed, outputs should contain 'estimated cost: None points'.
    """
    job = job_factory(job_params=DEFAULT_JOB_PARAMS)
    job.py_env.local_state = {"calculate_est_budget": _return_none}
    outputs = []
    code = (
        "strength = 0.5\n"
        "est_cost = calculate_est_budget(unit_budget=2)\n"
        "(f'Draft created with strength: {strength}, estimated cost: {est_cost} points')\n"
    )
    await exec_cli_step(code, job, outputs.append)
    assert outputs, "Final f-string expression produced no console output"
    assert "estimated cost: None" in outputs[-1]


@pytest.mark.asyncio
async def test_intermediate_future_error_suspends_with_empty_output(job_factory):
    """Intermediate temporal call raises FutureError → empty tool result.

    This is the leading hypothesis for the production symptom.  When an
    intermediate statement raises ``FutureError`` (a temporal suspension —
    e.g. waiting on a future scheduling query), the final REPL-style echo
    never runs and *nothing* has been appended to the console_append
    callback, so the tool result ends up as empty CONSOLE_OUTPUT.

    ``_run_code_block``'s ``finally`` still pushes whatever is in
    ``cli_outputs[cli_idx]`` (an empty list → ``""``), which is exactly
    the ``<CONSOLE_OUTPUT>\\n\\n</CONSOLE_OUTPUT>`` the LLM receives.
    """
    job = job_factory(job_params=DEFAULT_JOB_PARAMS)
    job.py_env.local_state = {"calculate_est_budget": _raise_future}
    outputs = []
    code = (
        "strength = 0.5\n"
        "est_cost = calculate_est_budget(unit_budget=2)\n"
        "(f'Draft created with strength: {strength}, estimated cost: {est_cost} points')\n"
    )
    with pytest.raises(FutureError):
        await exec_cli_step(code, job, outputs.append)

    # Key observation: FutureError is re-raised from _exec_code_body
    # *without* calling error_fn (see _exec_code_body:351-354 — the
    # FutureError branch only sets instr_num, it does not call error_fn).
    # So outputs is empty — matching the production symptom.
    assert not outputs, (
        f"Expected empty outputs on FutureError suspension (matching the "
        f"production symptom), got: {outputs!r}"
    )
