"""Tests for the web_ui job detail helper functions."""
# pylint: disable=unused-argument,no-member

from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch
from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.chat_style import ChatStyle
from statek.locale import StatekLocale, StatekLangCode, StatekCountryCode
from statek.pyenv import PyEnv
from statek.system import tool, docstr
from statek.task_difficulty import TaskDifficulty
from statek.utils import CodeBlock, CallSpec
from statek.executors.chat_log_item import LLM_LogItem, ToolError, WarmupLogItem
from web_ui.pages.job_detail import (
    _get_console_slice,
    _get_warmup_blocks,
    _get_warmup_console_ranges,
    _get_turn_console_ranges,
    _get_code_str,
    _get_tool_data_for_block,
    _get_exception_messages,
    _get_job_model,
    _get_job_provider,
    _get_job_temperature,
    _job_uses_reasoning,
    _get_locale_str,
    _get_difficulty_button_specs,
    _get_latency_samples,
    _summarize_latencies,
    _format_latency_seconds,
    _get_system_prompt,
    _strip_language_hint_suffix,
    _build_history_sections,
    _build_md_content,
    _build_raw_data,
    _build_raw_repr,
    _build_raw_html,
    _build_step_preview_data,
    _expand_json_viewer,
    _collapse_json_viewer,
    _json_viewer_expand_js,
    _json_viewer_collapse_js,
    _get_reported_tools,
)

# aliases used in _call_build_md helper
_warmup_blocks = _get_warmup_blocks
_warmup_ranges = _get_warmup_console_ranges
_turn_ranges = _get_turn_console_ranges



def _make_chat_log_item(console_pos: int, llm_resp, timestamp=None, tool_log=None):
    item = MagicMock()
    item.console_pos = console_pos
    item.llm_resp = llm_resp
    if timestamp is not None:
        item.timestamp = timestamp
    # Make it look like an LLM_LogItem for isinstance checks
    item.__class__ = LLM_LogItem
    # Add tool_log support
    if tool_log is not None:
        item.tool_log = tool_log
        # Add get_tool_result method
        def _get_tool_result(tool_call_id):
            if item.tool_log is None:
                raise KeyError(tool_call_id)
            if isinstance(item.tool_log, str):
                if tool_call_id != 0:
                    raise IndexError(f"tool_call_id {tool_call_id} out of range")
                return item.tool_log
            return item.tool_log[tool_call_id]
        item.get_tool_result = _get_tool_result
    else:
        item.tool_log = None
        def _get_tool_result_empty(tool_call_id):
            raise KeyError(tool_call_id)
        item.get_tool_result = _get_tool_result_empty
    return item


def _make_warmup_log_item(block_num, tool_log=None):
    """Create a mock WarmupLogItem."""
    item = MagicMock()
    item.warmup_block_num = block_num
    item.tool_log = tool_log
    item.__class__ = WarmupLogItem
    if tool_log is not None:
        def _get_tool_result(tool_call_id):
            if item.tool_log is None:
                raise KeyError(tool_call_id)
            if isinstance(item.tool_log, str):
                if tool_call_id != 0:
                    raise IndexError(f"tool_call_id {tool_call_id} out of range")
                return item.tool_log
            return item.tool_log[tool_call_id]
        item.get_tool_result = _get_tool_result
    else:
        def _get_tool_result_empty(tool_call_id):
            raise KeyError(tool_call_id)
        item.get_tool_result = _get_tool_result_empty
    return item


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _make_job(
    warmup_code=None,
    warmup_console_positions=None,
    chat_log=None,
    console=None,
    chat_style=None,
    metadata=None,
):
    job = MagicMock()
    job_def = MagicMock()
    job_def.warmup_code = warmup_code
    job_def.chat_style = chat_style
    job_def.model = None
    job_def.metadata = metadata or {}
    agent = MagicMock()
    agent._metadata = {}  # pylint: disable=protected-access
    job_def.agent = agent
    job.job_def = job_def
    positions = warmup_console_positions or []
    job._warmup_end_positions = MagicMock(return_value=positions)  # pylint: disable=protected-access
    job.chat_log = chat_log or []
    py_env = MagicMock()
    py_env.console = console
    py_env.exceptions = {}
    job.py_env = py_env
    job.get_chat_history = MagicMock(return_value=[])
    return job


class TestGetConsoleSlice:
    def test_returns_empty_string_when_console_is_none(self):
        assert _get_console_slice(None, 0, 5) == ''

    def test_returns_empty_string_when_empty_console(self):
        assert _get_console_slice([], 0, 5) == ''

    def test_joins_items_in_range(self):
        console = ['line1\n', 'line2\n', 'line3\n', 'line4\n']
        result = _get_console_slice(console, 1, 3)
        assert result == 'line2\nline3\n'

    def test_items_without_newline_get_one_appended(self):
        console = ['hello', 'world']
        assert _get_console_slice(console, 0, 2) == 'hello\nworld\n'

    def test_items_already_with_newline_not_doubled(self):
        console = ['hello\n', 'world\n']
        assert _get_console_slice(console, 0, 2) == 'hello\nworld\n'

    def test_full_range(self):
        console = ['a', 'b', 'c']
        assert _get_console_slice(console, 0, 3) == 'a\nb\nc\n'

    def test_empty_range(self):
        console = ['a', 'b', 'c']
        assert _get_console_slice(console, 2, 2) == ''

    def test_to_pos_beyond_list(self):
        console = ['a', 'b']
        assert _get_console_slice(console, 0, 100) == 'a\nb\n'


class TestGetLocaleStr:
    def test_returns_lang_and_country_when_locale_defined(self, db0_fixture):
        job = _make_job()
        job.job_def.locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )

        assert _get_locale_str(job) == 'PL-PL'

    def test_returns_empty_string_when_locale_missing(self):
        job = _make_job()
        job.job_def.locale = None

        assert _get_locale_str(job) == ''


class TestLatencyHelpers:
    def test_get_latency_samples_calls_job_getter(self):
        job = MagicMock()
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        job.get_response_times.return_value = [(t0, 5.5)]

        assert _get_latency_samples(job, 'get_response_times') == [(t0, 5.5)]

    def test_get_latency_samples_returns_empty_when_getter_missing(self):
        class _JobWithoutGetter:  # pylint: disable=too-few-public-methods
            pass
        job = _JobWithoutGetter()

        assert not _get_latency_samples(job, 'get_response_times')

    def test_get_latency_samples_returns_empty_on_error(self):
        job = MagicMock()
        job.get_response_times.side_effect = RuntimeError('boom')

        assert not _get_latency_samples(job, 'get_response_times')

    def test_summarize_latencies_counts_completed_and_pending(self):
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        t1 = datetime(2026, 1, 1, 12, 1, 0)
        t2 = datetime(2026, 1, 1, 12, 2, 0)

        summary = _summarize_latencies([
            (t0, 10.0),
            (t1, None),
            (t2, 20.0),
        ])

        assert summary.sample_count == 3
        assert summary.completed_count == 2
        assert summary.pending_count == 1
        assert summary.min_seconds == 10.0
        assert summary.avg_seconds == 15.0
        assert summary.max_seconds == 20.0

    def test_summarize_latencies_handles_empty_input(self):
        summary = _summarize_latencies([])

        assert summary.sample_count == 0
        assert summary.completed_count == 0
        assert summary.pending_count == 0
        assert summary.min_seconds is None
        assert summary.avg_seconds is None
        assert summary.max_seconds is None

    def test_format_latency_seconds_formats_pending(self):
        assert _format_latency_seconds(None) == 'Pending'

    def test_format_latency_seconds_formats_numeric_value(self):
        assert _format_latency_seconds(12.345) == '12.35s'

    def test_returns_empty_string_when_job_def_missing(self):
        job = MagicMock()
        job.job_def = None

        assert _get_locale_str(job) == ''


class TestGetJobTemperature:
    def test_reads_temperature_from_agent_metadata(self):
        job = _make_job(metadata={'TEMPERATURE': '0.3'})

        assert _get_job_temperature(job) == '0.3'

    def test_missing_temperature_returns_empty_string(self):
        job = _make_job()

        assert _get_job_temperature(job) == ''


class TestGetJobProvider:
    def test_reads_explicit_provider_from_job_metadata(self):
        job = _make_job(metadata={'PROVIDER': 'OPENROUTER'})

        assert _get_job_provider(job) == 'OPENROUTER'

    def test_missing_provider_returns_empty_string(self):
        job = _make_job()

        assert _get_job_provider(job) == ''


@tool
def _job_detail_app_tool(city: str, **kwargs) -> str:
    """Return a canned forecast."""
    del kwargs
    return city


@tool(target={ChatStyle.DIRECT})  # pylint: disable=no-member
def _job_detail_direct_only_tool(value: str, **kwargs) -> str:
    """Return a direct-only value."""
    del kwargs
    return value


class TestGetReportedTools:
    def test_returns_empty_list_when_scope_not_set(self):
        job = _make_job()
        job.job_def.agent.all_tools = [_job_detail_app_tool]

        assert not _get_reported_tools(job)

    def test_returns_selected_application_tools(self):
        job = _make_job(
            metadata={'LLM_TOOLS_SCOPE': 'APPLICATION'},
            chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
        )
        job.job_def.agent.all_tools = [_job_detail_app_tool, _job_detail_direct_only_tool, docstr]

        reported = _get_reported_tools(job)

        assert _job_detail_app_tool in reported
        assert _job_detail_direct_only_tool in reported
        assert docstr not in reported

    def test_includes_system_tools_when_scope_is_all(self):
        job = _make_job(
            metadata={'LLM_TOOLS_SCOPE': 'ALL'},
            chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
        )
        job.job_def.agent.all_tools = [_job_detail_app_tool]

        reported = _get_reported_tools(job)

        assert _job_detail_app_tool in reported
        assert docstr in reported

    def test_filters_tools_by_chat_style(self):
        job = _make_job(
            metadata={'LLM_TOOLS_SCOPE': 'APPLICATION'},
            chat_style=ChatStyle.MARKDOWN,  # pylint: disable=no-member
        )
        job.job_def.agent.all_tools = [_job_detail_app_tool, _job_detail_direct_only_tool]

        reported = _get_reported_tools(job)

        assert _job_detail_app_tool in reported
        assert _job_detail_direct_only_tool not in reported


class TestBuildStepPreviewData:
    def test_builds_preview_payload_from_historical_request(self):
        job = _make_job(metadata={'PROVIDER': 'OPENROUTER'})
        history = iter([
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'world'},
        ])
        job.get_request_data.return_value = {
            'system_prompt': 'sys',
            'model': 'test-model',
            'chat_history': history,
            'metadata': {'TEMPERATURE': '0.3'},
            'available_tools': ['tool-a'],
            'chat_style': ChatStyle.MARKDOWN,  # pylint: disable=no-member
            'temperature': 0.3,
            'enable_reasoning': True,
        }
        mock_api = MagicMock()
        mock_api.preview_request.return_value = {
            'provider_payload': {
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        }

        with patch('web_ui.pages.job_detail.LLM_API.get', return_value=mock_api) as mock_get:
            preview = _build_step_preview_data(job, 2)

        job.get_request_data.assert_called_once_with(2)
        mock_get.assert_called_once_with(provider_name='OPENROUTER')
        mock_api.preview_request.assert_called_once_with(
            system_prompt='sys',
            model='test-model',
            chat_history=[
                {'role': 'user', 'content': 'hello'},
                {'role': 'assistant', 'content': 'world'},
            ],
            metadata={'TEMPERATURE': '0.3'},
            available_tools=['tool-a'],
            chat_style=ChatStyle.MARKDOWN,  # pylint: disable=no-member
            temperature=0.3,
            enable_reasoning=True,
        )
        assert preview == {
            'provider_payload': {
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        }

    def test_uses_default_provider_when_job_provider_missing(self):
        job = _make_job()
        job.get_request_data.return_value = {'model': 'test-model'}
        mock_api = MagicMock()
        mock_api.preview_request.return_value = {'model': 'test-model'}

        with patch('web_ui.pages.job_detail.LLM_API.get', return_value=mock_api) as mock_get:
            preview = _build_step_preview_data(job, 0)

        mock_get.assert_called_once_with(provider_name=None)
        mock_api.preview_request.assert_called_once_with(model='test-model')
        assert preview == {'model': 'test-model'}

    def test_uses_provider_from_model_name_when_present(self):
        job = _make_job(metadata={'MODEL': 'openrouter/openai/gpt-5.4', 'PROVIDER': 'OPENAI'})
        job.get_request_data.return_value = {'model': 'openai/gpt-5.4'}
        mock_api = MagicMock()
        mock_api.preview_request.return_value = {'model': 'openai/gpt-5.4'}

        with patch('web_ui.pages.job_detail.LLM_API.get', return_value=mock_api) as mock_get:
            preview = _build_step_preview_data(job, 0)

        mock_get.assert_called_once_with(provider_name='openrouter')
        mock_api.preview_request.assert_called_once_with(model='openai/gpt-5.4')
        assert preview == {'model': 'openai/gpt-5.4'}

    def test_converts_db0_enum_values_to_display_safe_data(self):
        class _FakeEnumValue:  # pylint: disable=too-few-public-methods
            def __repr__(self):
                return 'FakeEnumValue(USER)'

        class _FakeHistoryItem:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self.role = _FakeEnumValue()
                self.content = 'hello'
                self.content_src = _FakeEnumValue()

        job = MagicMock()
        job.get_request_data.return_value = {'model': 'test-model'}
        mock_api = MagicMock()
        mock_api.preview_request.return_value = {
            'messages': [_FakeHistoryItem()],
        }

        with patch('web_ui.pages.job_detail.LLM_API.get', return_value=mock_api):
            preview = _build_step_preview_data(job, 0)

        assert preview['messages'][0]['__type__'] == '_FakeHistoryItem'
        assert preview['messages'][0]['role']['__type__'] == '_FakeEnumValue'
        assert preview['messages'][0]['content_src']['__type__'] == '_FakeEnumValue'


class TestJsonViewerControls:
    def test_expand_json_viewer_runs_expand_all(self):
        editor = MagicMock()

        _expand_json_viewer(editor)

        editor.run_editor_method.assert_called_once_with(':expand', [], '() => true')

    def test_collapse_json_viewer_runs_recursive_collapse(self):
        editor = MagicMock()

        _collapse_json_viewer(editor)

        editor.run_editor_method.assert_called_once_with('collapse', [], True)

    def test_builds_client_side_expand_handler(self):
        viewer = MagicMock()
        viewer.id = 123

        handler = _json_viewer_expand_js(viewer)

        assert 'getElement(123)' in handler
        assert '.jse-json-node:not(.jse-expanded)' in handler
        assert 'requestAnimationFrame(step)' in handler
        assert 'ctrlKey: true' in handler

    def test_builds_client_side_collapse_handler(self):
        viewer = MagicMock()
        viewer.id = 123

        handler = _json_viewer_collapse_js(viewer)

        assert 'getElement(123)' in handler
        assert '.jse-json-node.jse-root.jse-expanded' in handler
        assert 'dispatchEvent(new MouseEvent("click"' in handler
        assert 'ctrlKey: true' in handler


class TestJobUsesReasoning:
    def test_reads_reasoning_flag_from_agent_metadata(self):
        job = _make_job(metadata={'REASONING': 'true'})

        assert _job_uses_reasoning(job) is True

    def test_false_reasoning_flag_returns_false(self):
        job = _make_job(metadata={'REASONING': 'false'})

        assert _job_uses_reasoning(job) is False

    def test_missing_reasoning_flag_returns_false(self):
        job = _make_job()

        assert _job_uses_reasoning(job) is False


class TestGetWarmupBlocks:
    def test_no_warmup_returns_empty(self):
        job = _make_job(warmup_code=None)
        assert not _get_warmup_blocks(job)

    def test_single_string_block(self):
        job = _make_job(warmup_code='x = 1')
        blocks = _get_warmup_blocks(job)
        assert blocks == ['x = 1']

    def test_single_code_block(self, db0_fixture):
        cb = CodeBlock(code='x = 1')
        job = _make_job(warmup_code=cb)
        blocks = _get_warmup_blocks(job)
        assert blocks == [cb]

    def test_list_of_blocks(self, db0_fixture):
        cb1 = CodeBlock(code='a = 1')
        cb2 = 'b = 2'
        job = _make_job(warmup_code=[cb1, cb2])
        blocks = _get_warmup_blocks(job)
        assert blocks == [cb1, cb2]

    def test_tuple_of_blocks(self, db0_fixture):
        cb = CodeBlock(code='a = 1')
        job = _make_job(warmup_code=(cb, 'b = 2'))
        blocks = _get_warmup_blocks(job)
        assert blocks == [cb, 'b = 2']


class TestGetWarmupConsoleRanges:
    def test_no_warmup_returns_empty(self):
        job = _make_job(warmup_code=None)
        assert not _get_warmup_console_ranges(job)

    def test_single_block_no_positions_recorded(self):
        # Warmup hasn't finished yet — range starts at 0, ends at console length
        job = _make_job(
            warmup_code='x = 1',
            warmup_console_positions=[],
            console=['out1', 'out2'],
        )
        ranges = _get_warmup_console_ranges(job)
        assert ranges == [(0, 2)]

    def test_single_block_position_recorded(self):
        job = _make_job(
            warmup_code='x = 1',
            warmup_console_positions=[3],
            console=['a', 'b', 'c', 'd'],
        )
        ranges = _get_warmup_console_ranges(job)
        assert ranges == [(0, 3)]

    def test_two_blocks_both_positions_recorded(self):
        cb1 = 'block1'
        cb2 = 'block2'
        job = _make_job(
            warmup_code=[cb1, cb2],
            warmup_console_positions=[2, 5],
            console=['a', 'b', 'c', 'd', 'e'],
        )
        ranges = _get_warmup_console_ranges(job)
        assert ranges == [(0, 2), (2, 5)]

    def test_two_blocks_only_first_position_recorded(self):
        cb1 = 'block1'
        cb2 = 'block2'
        job = _make_job(
            warmup_code=[cb1, cb2],
            warmup_console_positions=[2],
            console=['a', 'b', 'c', 'd', 'e'],
        )
        ranges = _get_warmup_console_ranges(job)
        assert ranges == [(0, 2), (2, 5)]


class TestGetTurnConsoleRanges:
    def test_no_chat_log_returns_empty(self):
        job = _make_job(chat_log=[])
        assert not _get_turn_console_ranges(job)

    def test_single_turn_no_warmup(self):
        # console_pos=0 means this turn's output starts at index 0
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='code')
        job = _make_job(
            chat_log=[chat_item],
            warmup_console_positions=[],
            console=['a', 'b', 'c'],
        )
        ranges = _get_turn_console_ranges(job)
        assert ranges == [(0, 3)]

    def test_single_turn_with_warmup(self):
        # warmup consumed console[0:2]; turn starts at index 2
        chat_item = _make_chat_log_item(console_pos=2, llm_resp='code')
        job = _make_job(
            warmup_code='warmup',
            warmup_console_positions=[2],
            chat_log=[chat_item],
            console=['w1', 'w2', 'out1', 'out2', 'out3'],
        )
        ranges = _get_turn_console_ranges(job)
        assert ranges == [(2, 5)]

    def test_two_turns_no_warmup(self):
        # Turn 1 starts at 0, Turn 2 starts at 3; console has 7 items
        item1 = _make_chat_log_item(console_pos=0, llm_resp='code1')
        item2 = _make_chat_log_item(console_pos=3, llm_resp='code2')
        job = _make_job(
            chat_log=[item1, item2],
            warmup_console_positions=[],
            console=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        )
        ranges = _get_turn_console_ranges(job)
        assert ranges == [(0, 3), (3, 7)]

    def test_two_turns_with_two_warmup_blocks(self):
        # Two warmup blocks consumed console[0:2] and console[2:4];
        # Turn 1 starts at 4, Turn 2 starts at 6; console has 9 items
        item1 = _make_chat_log_item(console_pos=4, llm_resp='code1')
        item2 = _make_chat_log_item(console_pos=6, llm_resp='code2')
        job = _make_job(
            warmup_code=['w1', 'w2'],
            warmup_console_positions=[2, 4],
            chat_log=[item1, item2],
            console=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        )
        ranges = _get_turn_console_ranges(job)
        assert ranges == [(4, 6), (6, 9)]


class TestGetCodeStr:
    def test_plain_string(self):
        assert _get_code_str('print("hello")') == 'print("hello")'

    def test_code_block_with_code(self, db0_fixture):
        cb = CodeBlock(code='x = 1')
        assert _get_code_str(cb) == 'x = 1'

    def test_code_block_with_none_code(self, db0_fixture):
        cb = CodeBlock(code=None)
        assert _get_code_str(cb) == ''

    def test_none_returns_empty(self):
        assert _get_code_str(None) == ''

    def test_list_of_strings(self):
        assert _get_code_str(['x = 1\n', 'y = 2\n']) == 'x = 1\ny = 2\n'


class TestGetExceptionMessages:
    def test_excludes_tool_errors_from_tool_log(self, db0_fixture):
        """Tool errors in tool_log should NOT appear in exception messages."""
        item = _make_chat_log_item(console_pos=0, llm_resp='code',
                                   tool_log=[ToolError(err_message="tool err")])
        job = _make_job(chat_log=[item])
        job.py_env.exceptions = {0: "code err"}
        result = _get_exception_messages(job)
        assert "code err" in result
        assert "tool err" not in result


class TestGetToolDataForBlock:
    def test_plain_string_returns_empty(self, db0_fixture):
        item = _make_chat_log_item(console_pos=0, llm_resp='some code')
        assert not _get_tool_data_for_block('some code', item)

    def test_code_block_no_tool_calls_returns_empty(self, db0_fixture):
        cb = CodeBlock(code='x = 1', tool_calls=None)
        item = _make_chat_log_item(console_pos=0, llm_resp=cb)
        assert not _get_tool_data_for_block(cb, item)

    def test_code_block_empty_tool_calls_returns_empty(self, db0_fixture):
        cb = CodeBlock(code='x = 1', tool_calls=[])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb)
        assert not _get_tool_data_for_block(cb, item)

    def test_chat_log_item_none_returns_empty(self, db0_fixture):
        cs = CallSpec(id='T', func_name='search', args=['query'])
        cb = CodeBlock(tool_calls=[cs])
        assert not _get_tool_data_for_block(cb, None)

    def test_tool_log_missing_returns_empty(self, db0_fixture):
        cs = CallSpec(id='T', func_name='search', args=['query'])
        cb = CodeBlock(tool_calls=[cs])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb)
        assert not _get_tool_data_for_block(cb, item)

    def test_single_tool_call_with_string_result(self, db0_fixture):
        cs = CallSpec(id='T', func_name='search', args=['query'])
        cb = CodeBlock(tool_calls=[cs])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb, tool_log='result text')
        data = _get_tool_data_for_block(cb, item)
        assert data == [(cs, 'result text', None)]

    def test_single_tool_call_result_stored_as_list(self, db0_fixture):
        cs = CallSpec(id='T', func_name='fetch', kwargs={'url': 'http://x'})
        cb = CodeBlock(tool_calls=[cs])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb, tool_log=['fetched content'])
        data = _get_tool_data_for_block(cb, item)
        assert data == [(cs, 'fetched content', None)]

    def test_multiple_tool_calls(self, db0_fixture):
        cs1 = CallSpec(id='T', func_name='search', args=['q1'])
        cs2 = CallSpec(id='T', func_name='fetch', args=['url1'])
        cb = CodeBlock(tool_calls=[cs1, cs2])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb, tool_log=['res1', 'res2'])
        data = _get_tool_data_for_block(cb, item)
        assert data == [(cs1, 'res1', None), (cs2, 'res2', None)]

    def test_fewer_results_than_calls_uses_empty_string(self, db0_fixture):
        cs1 = CallSpec(id='T', func_name='a')
        cs2 = CallSpec(id='T', func_name='b')
        cb = CodeBlock(tool_calls=[cs1, cs2])
        item = _make_chat_log_item(console_pos=0, llm_resp=cb, tool_log=['only one result'])
        data = _get_tool_data_for_block(cb, item)
        assert data == [(cs1, 'only one result', None), (cs2, '', None)]


def _make_job_for_md(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    warmup_code=None,
    warmup_console_positions=None,
    chat_log=None,
    console=None,
    exceptions=None,
    chat_style=None,
):
    job = _make_job(warmup_code, warmup_console_positions, chat_log, console, chat_style)
    job.py_env = PyEnv(console=console, exceptions=exceptions or {})
    return job


def _call_build_md(job, **kwargs):
    """Helper: call _build_md_content with consistent defaults."""
    return _build_md_content(
        uuid_str=kwargs.get('uuid_str', 'test-uuid'),
        status_str=kwargs.get('status_str', 'completed'),
        agent_role=kwargs.get('agent_role', 'my-agent'),
        model=kwargs.get('model', 'claude-3'),
        total_cost=kwargs.get('total_cost', 0.0042),
        num_turns=kwargs.get('num_turns', 0),
        exception_count=kwargs.get('exception_count', 0),
        chat_style=kwargs.get('chat_style', ''),
        system_prompt=kwargs.get('system_prompt', ''),
        job=job,
    )


class TestBuildMdContentSummary:  # pylint: disable=too-many-public-methods
    def test_includes_title(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job)
        assert '# Job Detail' in md

    def test_includes_uuid_and_status(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, uuid_str='abc-123', status_str='running')
        assert 'abc-123' in md
        assert 'running' in md

    def test_includes_agent_model_cost_turns(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, agent_role='analyst', model='gpt-4', total_cost=1.5, num_turns=7)
        assert 'analyst' in md
        assert 'gpt-4' in md
        assert '1.5' in md
        assert '7' in md

    def test_includes_locale_when_present(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.locale = StatekLocale(
            lang_code=StatekLangCode.FR,
            country_code=StatekCountryCode.CA,
        )

        md = _call_build_md(job)

        assert 'Locale' in md
        assert 'FR-CA' in md

    def test_omits_locale_when_missing(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.locale = None

        md = _call_build_md(job)

        assert 'Locale' not in md

    def test_includes_temperature_when_present(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {'TEMPERATURE': '0.3'}

        md = _call_build_md(job)

        assert 'Temperature' in md
        assert '0.3' in md

    def test_omits_temperature_when_missing(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {}

        md = _call_build_md(job)

        assert 'Temperature' not in md

    def test_includes_provider_when_present(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {'PROVIDER': 'OPENAI'}

        md = _call_build_md(job)

        assert 'Provider' in md
        assert 'OPENAI' in md

    def test_includes_provider_from_model_name(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {'MODEL': 'openrouter/openai/gpt-5.4'}

        md = _call_build_md(job)

        assert 'Provider' in md
        assert 'openrouter' in md

    def test_omits_provider_when_missing(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {}

        md = _call_build_md(job)

        assert 'Provider' not in md

    def test_includes_reasoning_when_present(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {'REASONING': 'true'}

        md = _call_build_md(job)

        assert 'Reasoning' in md
        assert 'Enabled' in md

    def test_omits_reasoning_when_missing(self, db0_fixture):
        job = _make_job_for_md()
        job.job_def.metadata = {}

        md = _call_build_md(job)

        assert 'Reasoning' not in md

    def test_includes_chat_style_when_present(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, chat_style='MD_DIALOG')
        assert 'Chat Style' in md
        assert 'MD_DIALOG' in md

    def test_omits_chat_style_when_empty(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, chat_style='')
        assert 'Chat Style' not in md

    def test_includes_system_prompt_section_when_present(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, system_prompt='You are a helpful assistant.')
        assert 'System Prompt' in md
        assert 'You are a helpful assistant.' in md

    def test_omits_system_prompt_section_when_empty(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job, system_prompt='')
        assert 'System Prompt' not in md

    def test_omits_initial_prompt_history_when_present(self, db0_fixture):
        job = _make_job_for_md()
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.SYSTEM,
                content='Analyse this data.',
                content_src=ContentSource.SYSTEM,
            )
        ]
        md = _call_build_md(job)
        assert 'Initial Prompt' not in md
        assert 'Analyse this data.' not in md

    def test_omits_initial_prompt_when_empty(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job)
        assert 'Initial Prompt' not in md

    def test_includes_warmup_code(self, db0_fixture):
        job = _make_job_for_md(warmup_code='x = 1', warmup_console_positions=[1], console=['ok\n'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.SYSTEM,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='ok\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]
        md = _call_build_md(job)
        assert 'Warmup Code 1' in md
        assert 'x = 1' in md

    def test_includes_warmup_console_output(self, db0_fixture):
        job = _make_job_for_md(warmup_code='x = 1', warmup_console_positions=[1], console=['ok\n'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.SYSTEM,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='ok\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]
        md = _call_build_md(job)
        assert 'ok' in md

    def test_includes_llm_turn_section(self, db0_fixture):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='result = 42')
        job = _make_job_for_md(chat_log=[chat_item], console=['done\n'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='result = 42',
                content_src=ContentSource.ASSISTANT,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='done\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]
        md = _call_build_md(job)
        assert 'Turn 1' in md
        assert 'result = 42' in md

    def test_includes_turn_console_output(self, db0_fixture):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='x = 1')
        job = _make_job_for_md(chat_log=[chat_item], console=['output line\n'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.ASSISTANT,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='output line\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]
        md = _call_build_md(job)
        assert 'output line' in md

    def test_includes_tool_calls_in_turn(self, db0_fixture):
        cs = CallSpec(id='T', func_name='search', args=['query'])
        cb = CodeBlock(code='search("query")', tool_calls=[cs])
        chat_item = _make_chat_log_item(console_pos=0, llm_resp=cb, tool_log='result text')
        job = _make_job_for_md(chat_log=[chat_item])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='search("query")',
                content_src=ContentSource.ASSISTANT,
                tool_calls=[cs],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='result text',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs,
            ),
        ]
        md = _call_build_md(job)
        assert 'Tool Call' in md
        assert 'search' in md
        assert 'result text' in md

    def test_direct_llm_text_not_wrapped_as_python_code_block(self, db0_fixture):
        job = _make_job_for_md(chat_style=ChatStyle.DIRECT)  # pylint: disable=no-member
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='Oto Twoj grafik na kwiecien 2026 roku.',
                content_src=ContentSource.ASSISTANT,
            ),
        ]
        md = _call_build_md(job, chat_style='DIRECT')
        assert 'Oto Twoj grafik na kwiecien 2026 roku.' in md
        assert '```python\nOto Twoj grafik na kwiecien 2026 roku.\n```' not in md

    def test_direct_warmup_code_still_uses_python_code_block(self, db0_fixture):
        job = _make_job_for_md(
            warmup_code='x = 1',
            warmup_console_positions=[0],
            chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
        )
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.SYSTEM,
            ),
        ]
        md = _call_build_md(job, chat_style='DIRECT')
        assert '```python\nx = 1\n```' in md

    def test_includes_error_indicator_in_turn(self, db0_fixture):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='bad code')
        job = _make_job_for_md(chat_log=[chat_item], exceptions={0: 'NameError: x'})
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='bad code',
                content_src=ContentSource.ASSISTANT,
            )
        ]
        md = _call_build_md(job)
        assert 'NameError: x' in md

    def test_multiple_turns_all_included(self, db0_fixture):
        item1 = _make_chat_log_item(console_pos=0, llm_resp='step_one()')
        item2 = _make_chat_log_item(console_pos=1, llm_resp='step_two()')
        job = _make_job_for_md(chat_log=[item1, item2], console=['a\n', 'b\n'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='step_one()',
                content_src=ContentSource.ASSISTANT,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='a\n',
                content_src=ContentSource.CONSOLE,
            ),
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='step_two()',
                content_src=ContentSource.ASSISTANT,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='b\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]
        md = _call_build_md(job)
        assert 'Turn 1' in md
        assert 'Turn 2' in md
        assert 'step_one()' in md
        assert 'step_two()' in md

    def test_returns_string(self, db0_fixture):
        job = _make_job_for_md()
        md = _call_build_md(job)
        assert isinstance(md, str)
        assert len(md) > 0


class TestBuildMdContentHistory:

    def test_warmup_tool_calls_use_correct_key_per_block(self, db0_fixture):  # pylint: disable=too-many-locals
        """Each warmup block should look up tool results from its WarmupLogItem."""
        cs1 = CallSpec(id='T', func_name='list_of_examples')
        cs2 = CallSpec(id='T', func_name='docs', args=['get_user_calendar'])
        block1 = 'print("hello")'
        block2 = 'print("world")'
        block3 = 'print("setup")'
        block4 = CodeBlock(code=None, tool_calls=[cs1])
        block5 = CodeBlock(code=None, tool_calls=[cs2])
        warmup_code = [block1, block2, block3, block4, block5]
        console = ['hello\n', 'world\n', 'setup\n', 'tool_output\n']
        positions = [1, 2, 3, 4, 4]

        # Create WarmupLogItems for the tool-call blocks
        warmup_item_3 = _make_warmup_log_item(3, tool_log='examples_result')
        warmup_item_4 = _make_warmup_log_item(4, tool_log='docs_result')

        job = _make_job_for_md(
            warmup_code=warmup_code,
            warmup_console_positions=positions,
            console=console,
        )
        job.chat_log = [warmup_item_3, warmup_item_4]
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[cs1],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='examples_result',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs1,
            ),
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[cs2],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='docs_result',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs2,
            ),
        ]
        md = _call_build_md(job)
        assert 'examples_result' in md
        assert 'docs_result' in md

    def test_warmup_single_tool_block_at_nonzero_pos(self, db0_fixture):
        """A single warmup tool block after code blocks uses correct WarmupLogItem."""
        cs = CallSpec(id='T', func_name='my_tool')
        block1 = 'print("init")'
        block2 = CodeBlock(code=None, tool_calls=[cs])
        console = ['init\n']
        positions = [1, 1]

        warmup_item = _make_warmup_log_item(1, tool_log='tool_result_here')

        job = _make_job_for_md(
            warmup_code=[block1, block2],
            warmup_console_positions=positions,
            console=console,
        )
        job.chat_log = [warmup_item]
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='print("init")',
                content_src=ContentSource.SYSTEM,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='init\n',
                content_src=ContentSource.CONSOLE,
            ),
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[cs],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='tool_result_here',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs,
            ),
        ]
        md = _call_build_md(job)
        assert 'Warmup Code 2' in md
        assert 'tool_result_here' in md

    def test_creates_warmup_title_for_tool_only_block_without_content_src(self):
        """Tool-only warmup blocks should still render as warmup sections."""
        cs = MagicMock()
        cs.format.return_value = 'docstr("topic")'
        block = MagicMock()
        block.tool_calls = [cs]
        job = _make_job(warmup_code=[block])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[cs],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='docs result',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs,
            ),
        ]

        sections = _build_history_sections(job)

        assert len(sections) == 1
        assert sections[0].title == 'Warmup Code 1'
        assert sections[0].content_src == ContentSource.SYSTEM
        assert sections[0].warmup_num == 1
        assert sections[0].warmup_total == 1
        assert sections[0].tool_data == [(cs, 'docs result', None)]


def _make_job_with_agent(system_prompt_return=None, system_prompt_raises=None, raw_prompt=None,
                         no_agent=False, no_job_def=False):
    job = MagicMock()
    if no_job_def:
        job.job_def = None
        return job
    job_def = MagicMock()
    job_def.job_params = {}
    if no_agent:
        job_def.agent = None
    else:
        agent = MagicMock()
        agent._system_prompt = raw_prompt  # pylint: disable=protected-access
        if system_prompt_raises:
            type(job_def).system_prompt = PropertyMock(side_effect=system_prompt_raises)
            job.system_prompt.side_effect = system_prompt_raises
        else:
            type(job_def).system_prompt = PropertyMock(return_value=system_prompt_return)
            if system_prompt_return is not None:
                job.system_prompt.return_value = system_prompt_return
        job_def.agent = agent
    job.job_def = job_def
    return job


class TestGetSystemPrompt:
    def test_returns_formatted_system_prompt(self):
        job = _make_job_with_agent()
        job.system_prompt.return_value = 'You are an assistant.'

        assert _get_system_prompt(job) == 'You are an assistant.'
        job.system_prompt.assert_called_once_with()

    def test_uses_explicit_difficulty_when_provided(self):
        job = _make_job_with_agent()
        job.system_prompt.return_value = 'Low instructions.'

        assert _get_system_prompt(job, TaskDifficulty.low) == 'Low instructions.'
        job.system_prompt.assert_called_once_with(TaskDifficulty.low)

    def test_resolves_current_difficulty_through_job_system_prompt(self):
        job = _make_job_with_agent()
        job.system_prompt.side_effect = (
            lambda difficulty=None: 'High instructions.'
            if difficulty == TaskDifficulty.high else 'Current instructions.'
        )
        job.get_current_difficulty.return_value = TaskDifficulty.high

        assert _get_system_prompt(job) == 'Current instructions.'
        job.system_prompt.assert_called_once_with()

    def test_returns_empty_string_when_no_job_def(self):
        job = _make_job_with_agent(no_job_def=True)
        assert _get_system_prompt(job) == ''

    def test_returns_empty_string_when_no_agent(self):
        job = _make_job_with_agent(no_agent=True)
        assert _get_system_prompt(job) == ''

    def test_falls_back_to_raw_prompt_on_format_error(self):
        job = _make_job_with_agent(
            system_prompt_raises=KeyError('missing_key'),
            raw_prompt='Raw template with {missing_key}.',
        )
        job.system_prompt.side_effect = KeyError('missing_key')

        assert _get_system_prompt(job) == 'Raw template with {missing_key}.'

    def test_returns_empty_string_when_formatted_and_raw_both_none(self):
        job = _make_job_with_agent(
            system_prompt_raises=KeyError('x'),
            raw_prompt=None,
        )
        assert _get_system_prompt(job) == ''

    def test_returns_empty_string_on_unexpected_error(self):
        job = MagicMock()
        type(job).job_def = PropertyMock(side_effect=RuntimeError('db error'))
        assert _get_system_prompt(job) == ''


class TestDifficultyButtonSpecs:
    def test_returns_lmh_buttons_with_tooltips(self):
        specs = _get_difficulty_button_specs()

        assert [(spec['label'], spec['tooltip']) for spec in specs] == [
            ('L', 'Low difficulty'),
            ('M', 'Medium difficulty'),
            ('H', 'High difficulty'),
        ]

    def test_keys_match_task_difficulty_values(self):
        specs = _get_difficulty_button_specs()

        assert [spec['key'] for spec in specs] == [
            str(TaskDifficulty.low),
            str(TaskDifficulty.medium),
            str(TaskDifficulty.high),
        ]
        assert [spec['difficulty'] for spec in specs] == [
            TaskDifficulty.low,
            TaskDifficulty.medium,
            TaskDifficulty.high,
        ]


class TestBuildHistorySections:
    def test_strip_language_hint_suffix_removes_known_suffix(self):
        text = (
            'prosze o ustawienie negatywnej preferencji '
            '(PAMIĘTAJ: Odpowiedz wyłącznie po polsku)'
        )
        assert _strip_language_hint_suffix(text) == 'prosze o ustawienie negatywnej preferencji'

    def test_strip_language_hint_suffix_removes_suffix_per_line(self):
        text = (
            'pierwsza wiadomosc (PAMIĘTAJ: Odpowiedz wyłącznie po polsku)\n'
            'druga wiadomosc (PAMIĘTAJ: Odpowiedz wyłącznie po polsku)'
        )
        assert _strip_language_hint_suffix(text) == 'pierwsza wiadomosc\ndruga wiadomosc'

    def test_strip_language_hint_suffix_leaves_other_parenthetical_text(self):
        text = 'wiadomosc testowa (to ma zostac)'
        assert _strip_language_hint_suffix(text) == text

    def test_groups_assistant_tool_calls_and_console_followups(self, db0_fixture):
        cs = CallSpec(id='T', func_name='search', args=['query'])
        job = _make_job()
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.SYSTEM,
                content='Prompt text',
                content_src=ContentSource.SYSTEM,
            ),
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='search("query")',
                content_src=ContentSource.ASSISTANT,
                tool_calls=[cs],
            ),
            ChatHistoryItem(
                role=ChatRole.TOOL,
                content='tool result',
                content_src=ContentSource.CONSOLE,
                tool_calls=cs,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content='console output\n',
                content_src=ContentSource.CONSOLE,
            ),
        ]

        sections = _build_history_sections(job)

        assert len(sections) == 1
        assert sections[0].title == 'Turn 1'
        assert sections[0].tool_data == [(cs, 'tool result', None)]
        assert sections[0].followups[0].title == 'Console Output'
        assert sections[0].followups[0].content == 'console output\n'

    def test_ignores_system_history_items(self, db0_fixture):
        job = _make_job()
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.SYSTEM,
                content='Prompt text',
                content_src=ContentSource.SYSTEM,
            ),
        ]

        sections = _build_history_sections(job)

        assert not sections

    def test_strips_language_hint_from_user_followup_display(self, db0_fixture):
        job = _make_job()
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='assistant reply',
                content_src=ContentSource.ASSISTANT,
            ),
            ChatHistoryItem(
                role=ChatRole.USER,
                content=(
                    'prosze o ustawienie negatywnej preferencji '
                    '(PAMIĘTAJ: Odpowiedz wyłącznie po polsku)'
                ),
                content_src=ContentSource.USER,
            ),
        ]

        sections = _build_history_sections(job)

        assert len(sections) == 1
        assert sections[0].followups[0].title == 'User Message'
        assert sections[0].followups[0].content == 'prosze o ustawienie negatywnej preferencji'

    def test_creates_warmup_titles_from_system_assistant_items(self, db0_fixture):
        job = _make_job(warmup_code=['x = 1', 'y = 2'])
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.SYSTEM,
            ),
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='y = 2',
                content_src=ContentSource.SYSTEM,
            ),
        ]

        sections = _build_history_sections(job)

        assert [section.title for section in sections] == ['Warmup Code 1', 'Warmup Code 2']
        assert [
            (section.warmup_num, section.warmup_total) for section in sections
        ] == [(1, 2), (2, 2)]

    def test_direct_llm_assistant_section_marked_as_text(self, db0_fixture):
        job = _make_job(chat_style=ChatStyle.DIRECT)  # pylint: disable=no-member
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='Oto Twoj grafik na kwiecien 2026 roku.',
                content_src=ContentSource.ASSISTANT,
            ),
        ]

        sections = _build_history_sections(job)

        assert len(sections) == 1
        assert sections[0].title == 'Turn 1'
        assert sections[0].content == 'Oto Twoj grafik na kwiecien 2026 roku.'
        assert sections[0].render_as_code is False

    def test_direct_warmup_section_still_marked_as_code(self, db0_fixture):
        job = _make_job(
            warmup_code=['x = 1'],
            chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
        )
        job.get_chat_history.return_value = [
            ChatHistoryItem(
                role=ChatRole.ASSISTANT,
                content='x = 1',
                content_src=ContentSource.SYSTEM,
            ),
        ]

        sections = _build_history_sections(job)

        assert len(sections) == 1
        assert sections[0].title == 'Warmup Code 1'
        assert sections[0].render_as_code is True


class _StubPyEnv:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.console = ['line1\n', 'line2\n']
        self.exceptions = {0: 'NameError: x'}
        self.global_state = {'key': 'value'}
        self.local_state = {}


class _StubUsage:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.total_reported_cost = 0.0042
        self.context_bytes = 1024
        self.total_bytes_sent = 512
        self.total_bytes_received = 512

    @property
    def total_cost(self):
        return self.total_reported_cost


class _StubJob:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.job_def = MagicMock()
        self.job_def.model = 'claude-3-opus'
        self.job_def.model_family = 'claude'
        self.session_id = 'sess-abc'
        self.usage = _StubUsage()
        self.next_instr_num = None
        self.warmup_block_num = None
        self.chat_log = []
        self.py_env = _StubPyEnv()

    def _warmup_end_positions(self):
        return [2]


class TestBuildRawRepr:
    def test_build_raw_data_returns_dict(self):
        job = _StubJob()
        result = _build_raw_data(job)
        assert isinstance(result, dict)
        assert result['__type__'] == '_StubJob'

    def test_returns_string(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_model_field(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'model' in result
        assert 'claude-3-opus' in result

    def test_includes_model_family(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'model_family' in result
        assert 'claude' in result

    def test_includes_total_cost(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'total_reported_cost' in result

    def test_includes_nested_py_env_fields(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        # py_env and its fields should appear somewhere
        assert 'py_env' in result or 'PyEnv' in result or '_StubPyEnv' in result

    def test_includes_console_content(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'line1' in result

    def test_includes_exceptions(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'NameError' in result

    def test_handles_error_gracefully(self):
        # An object that raises on vars()
        class _BadJob:  # pylint: disable=too-few-public-methods
            @property
            def __dict__(self):
                raise RuntimeError('no vars')
        result = _build_raw_repr(_BadJob())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_chat_log(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'chat_log' in result

    def test_partial_error_shows_good_attrs_and_inline_error(self):
        """When one attribute fails, other attributes still render."""
        class _ExplodingList(list):
            """A list whose iteration raises, simulating a broken db0 object."""
            def __iter__(self):
                raise RuntimeError('Prefix: 999 not found')
        class _PartialJob:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self.good_field = 'hello'
                self.bad_field = _ExplodingList([1, 2, 3])
        result = _build_raw_repr(_PartialJob())
        assert 'hello' in result
        assert 'good_field' in result
        assert '(Error' in result
        assert 'Prefix: 999 not found' in result


class TestGetJobModel:
    def test_reads_bare_current_model_from_job(self):
        job = _make_job()
        job.get_current_model = MagicMock(return_value='deepseek/deepseek-v3.2')

        assert _get_job_model(job) == 'deepseek-v3.2'
        job.get_current_model.assert_called_once_with()

    def test_strips_provider_and_family_from_current_model(self):
        job = _make_job()
        job.get_current_model = MagicMock(return_value='openrouter/openai/gpt-5.4')

        assert _get_job_model(job) == 'gpt-5.4'
        job.get_current_model.assert_called_once_with()

    def test_blank_current_model_returns_dash(self):
        job = _make_job()
        job.get_current_model = MagicMock(return_value=None)
        assert _get_job_model(job) == '—'

    def test_missing_job_def_returns_dash(self):
        job = MagicMock()
        job.job_def = None
        assert _get_job_model(job) == '—'


class TestBuildRawHtml:
    def test_returns_string(self):
        job = _StubJob()
        result = _build_raw_html(job)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_model_field(self):
        job = _StubJob()
        result = _build_raw_html(job)
        assert 'model' in result
        assert 'claude-3-opus' in result

    def test_includes_console_content(self):
        job = _StubJob()
        result = _build_raw_html(job)
        assert 'line1' in result

    def test_list_items_have_alternating_backgrounds(self):
        job = _StubJob()
        result = _build_raw_html(job)
        assert '#f0f0f0' in result  # even bg
        assert '#e0e0e0' in result  # odd bg

    def test_includes_type_names(self):
        job = _StubJob()
        result = _build_raw_html(job)
        assert '_StubJob' in result

    def test_handles_error_gracefully(self):
        class _BadJob:  # pylint: disable=too-few-public-methods
            @property
            def __dict__(self):
                raise RuntimeError('no vars')
        result = _build_raw_html(_BadJob())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_errors_highlighted(self):
        class _ExplodingList(list):
            def __iter__(self):
                raise RuntimeError('Prefix: 999 not found')
        class _PartialJob:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self.good_field = 'hello'
                self.bad_field = _ExplodingList([1, 2, 3])
        result = _build_raw_html(_PartialJob())
        assert 'hello' in result
        assert 'Error' in result
        assert '#b71c1c' in result  # error color
