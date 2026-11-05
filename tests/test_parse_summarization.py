"""Structured summarization response validation and persistence."""
# pylint: disable=no-member,protected-access
import json
from unittest.mock import patch

import dbzero as db0
import pytest

from statek.chat_style import ChatStyle
from statek.executors.chat_log_item import LLM_LogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import JobSummary, StateSummary, PyHandleDesc
from statek.executors.summarization import parse_summarization_result
from statek.llm_api import LLM_Response, LLM_StepData, LLM_Stats
from statek.utils import CodeBlock, CallSpec
from tests.conftest import DB0_DIR


def payload(names=()):
    return {'input_summary': 'User request', 'output_summary': 'Completed work',
            'system_summary': None, 'state': {'locals': [
                {'var_name': name, 'desc': 'A useful handle'} for name in names]}}


@pytest.mark.parametrize('style', ['CONSOLE', 'MARKDOWN', 'MD_DIALOG', 'DIRECT'])
@pytest.mark.parametrize('fence', [None, '', 'json'])
def test_parse_actual_response_in_all_styles(job_factory, style, fence):
    source = job_factory()
    source.job_def.set_chat_style(getattr(ChatStyle, style))
    job = ForkedJob(source, len(source.chat_log))
    job.add_locals(calendar=1)
    text = json.dumps(payload(['calendar']))
    if fence is not None:
        text = f'```{fence}\n{text}\n```'
    job.append_chat_log({}, LLM_Response(LLM_StepData(text, None), LLM_Stats(0, 0, None)))
    assert job.chat_log[-1].raw_response_text == text
    result = parse_summarization_result(job)
    assert result.input_summary == 'User request'
    assert result.output_summary == 'Completed work'
    assert result.system_summary is None
    assert [(x.var_name, x.desc) for x in result.state.locals] == [
        ('calendar', 'A useful handle')]


@pytest.mark.parametrize('response, error', [
    ('', 'empty'), ('{', 'JSON'), ('[]', 'object'),
    ('```python\n{}\n```', 'JSON'), ('prefix {}', 'JSON'),
    ('```json\n{}\n```\ncomment', 'JSON'),
    ('{"input_summary":1,"input_summary":2}', 'Duplicate'),
    ('{"input_summary":NaN}', 'NaN'),
])
def test_invalid_json(job_factory, response, error):
    job = job_factory()
    job.chat_log.append(LLM_LogItem(0, llm_resp=response))
    with pytest.raises(ValueError, match=error):
        parse_summarization_result(job)


@pytest.mark.parametrize('field, value, error', [
    ('input_summary', None, 'input_summary'),
    ('output_summary', 1, 'output_summary'),
    ('system_summary', [], 'system_summary'),
    ('state', None, 'state'), ('state', {'locals': {}}, 'state.locals'),
    ('state', {'locals': [None]}, r'locals\[0\]'),
    ('state', {'locals': [{'var_name': 1, 'desc': 'x'}]}, 'var_name'),
    ('state', {'locals': [{'var_name': 'x', 'desc': None}]}, 'desc'),
    ('state', {'locals': [{'var_name': 'x', 'desc': 'x'}]}, 'Unexpected'),
    ('state', {'locals': [{'var_name': 'x', 'desc': 'x'}] * 2}, 'Duplicate'),
    ('extra', 'x', 'extra'),
])
def test_invalid_schema_creates_no_summary_objects(job_factory, field, value, error):
    job = job_factory()
    data = payload()
    data[field] = value
    job.chat_log.append(LLM_LogItem(0, llm_resp=json.dumps(data)))
    before = [len(db0.find(cls)) for cls in (JobSummary, StateSummary, PyHandleDesc)]
    with pytest.raises(ValueError, match=error):
        parse_summarization_result(job)
    assert [len(db0.find(cls)) for cls in (JobSummary, StateSummary, PyHandleDesc)] == before


def test_missing_fields_names_and_response(job_factory):
    job = job_factory()
    with pytest.raises(ValueError, match='response'):
        parse_summarization_result(job)
    job.chat_log.append(LLM_LogItem(0, llm_resp='{}'))
    with pytest.raises(ValueError, match='Missing'):
        parse_summarization_result(job)
    job.chat_log[-1].llm_resp = json.dumps(payload())
    job.add_locals(missing=1)
    with pytest.raises(ValueError, match='missing'):
        parse_summarization_result(job)


def test_tools_rejected_even_with_valid_raw_text(job_factory):
    job = job_factory()
    job.chat_log.append(LLM_LogItem(0, llm_resp=CodeBlock(tool_calls=[CallSpec('1', 'tool')]),
                                    raw_response_text=json.dumps(payload())))
    with pytest.raises(ValueError, match='tool'):
        parse_summarization_result(job)


def test_legacy_codeblock_and_prefix_persistence(job_factory):
    job = job_factory()
    job.add_locals(handle=1)
    job.chat_log.append(LLM_LogItem(0, llm_resp=CodeBlock(code=json.dumps(payload(['handle'])))))
    job_prefix = db0.get_prefix_of(job).name
    db0.open('other-prefix', 'rw')
    result = parse_summarization_result(job)
    assert all(db0.get_prefix_of(obj).name == job_prefix
               for obj in [result, result.state, *result.state.locals])
    uuid = db0.uuid(result)
    # Retain the returned object as a caller would when applying compaction.
    job.saved_summary = result
    db0.close()
    db0.init(DB0_DIR, read_write=True)
    db0.open(job_prefix, 'rw')
    assert db0.fetch(uuid).input_summary == 'User request'


def test_raw_copy_and_database_errors(job_factory):
    job = job_factory()
    job.chat_log.append(LLM_LogItem(0, llm_resp='processed',
                                    raw_response_text=json.dumps(payload())))
    fork = ForkedJob(job, len(job.chat_log))
    assert fork.chat_log[-1].raw_response_text == job.chat_log[-1].raw_response_text
    with patch('statek.executors.summarization.db0.get_prefix_of',
               side_effect=RuntimeError('database failure')):
        with pytest.raises(RuntimeError, match='database failure'):
            parse_summarization_result(job)


def test_empty_summary_and_raw_text_precedence(job_factory):
    job = job_factory()
    data = payload()
    data.update(input_summary='', output_summary='', system_summary='A notification')
    job.chat_log.append(LLM_LogItem(0, llm_resp='not JSON', raw_response_text=json.dumps(data)))
    before = list(job.chat_log), job.status, dict(job.py_env.iter_locals())
    result = parse_summarization_result(job)
    assert result.input_summary == result.output_summary == ''
    assert result.system_summary == 'A notification'
    assert not result.state.locals
    assert (list(job.chat_log), job.status, dict(job.py_env.iter_locals())) == before
    job.chat_log[-1].llm_resp = json.dumps(data)
    job.chat_log[-1].raw_response_text = ''
    with pytest.raises(ValueError, match='empty'):
        parse_summarization_result(job)
