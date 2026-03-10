"""Tests for the web_ui job detail helper functions."""

from unittest.mock import MagicMock, PropertyMock
from web_ui.pages.job_detail import (
    _get_console_slice,
    _get_warmup_blocks,
    _get_warmup_console_ranges,
    _get_turn_console_ranges,
    _get_code_str,
    _get_tool_data_for_block,
    _get_system_prompt,
    _build_md_content,
    _build_raw_repr,
)

# aliases used in _call_build_md helper
_warmup_blocks = _get_warmup_blocks
_warmup_ranges = _get_warmup_console_ranges
_turn_ranges = _get_turn_console_ranges


class _FakeCodeBlock:  # pylint: disable=too-few-public-methods
    """Duck-typed stand-in for statek.utils.CodeBlock (avoids db0 initialization)."""
    def __init__(self, code=None, tool_calls=None):
        self.code = code
        self.tool_calls = tool_calls


class _FakeCallSpec:  # pylint: disable=too-few-public-methods
    """Duck-typed stand-in for statek.utils.CallSpec."""
    def __init__(self, func_name, args=None, kwargs=None):
        self.id = 'fake-id'
        self.func_name = func_name
        self.args = args or []
        self.kwargs = kwargs or {}

    def format(self) -> str:
        parts = [repr(a) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.func_name}({', '.join(parts)})"


def _make_chat_log_item(console_pos: int, llm_resp, timestamp=None):
    item = MagicMock()
    item.console_pos = console_pos
    item.llm_resp = llm_resp
    if timestamp is not None:
        item.timestamp = timestamp
    return item


def _make_job(
    warmup_code=None,
    warmup_console_positions=None,
    chat_log=None,
    console=None,
):
    job = MagicMock()
    job_def = MagicMock()
    job_def.warmup_code = warmup_code
    job.job_def = job_def
    job.warmup_console_positions = warmup_console_positions or []
    job.chat_log = chat_log or []
    py_env = MagicMock()
    py_env.console = console
    job.py_env = py_env
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


class TestGetWarmupBlocks:
    def test_no_warmup_returns_empty(self):
        job = _make_job(warmup_code=None)
        assert not _get_warmup_blocks(job)

    def test_single_string_block(self):
        job = _make_job(warmup_code='x = 1')
        blocks = _get_warmup_blocks(job)
        assert blocks == ['x = 1']

    def test_single_code_block(self):
        cb = _FakeCodeBlock(code='x = 1')
        job = _make_job(warmup_code=cb)
        blocks = _get_warmup_blocks(job)
        assert blocks == [cb]

    def test_list_of_blocks(self):
        cb1 = _FakeCodeBlock(code='a = 1')
        cb2 = 'b = 2'
        job = _make_job(warmup_code=[cb1, cb2])
        blocks = _get_warmup_blocks(job)
        assert blocks == [cb1, cb2]

    def test_tuple_of_blocks(self):
        cb = _FakeCodeBlock(code='a = 1')
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

    def test_code_block_with_code(self):
        cb = _FakeCodeBlock(code='x = 1')
        assert _get_code_str(cb) == 'x = 1'

    def test_code_block_with_none_code(self):
        cb = _FakeCodeBlock(code=None)
        assert _get_code_str(cb) == ''

    def test_none_returns_empty(self):
        assert _get_code_str(None) == ''

    def test_code_block_with_list_code(self):
        cb = _FakeCodeBlock(code=['line1\n', 'line2\n'])
        assert _get_code_str(cb) == 'line1\nline2\n'

    def test_list_of_strings(self):
        assert _get_code_str(['x = 1\n', 'y = 2\n']) == 'x = 1\ny = 2\n'


class TestGetToolDataForBlock:
    def test_plain_string_returns_empty(self):
        assert _get_tool_data_for_block('some code', {}, 0) == []

    def test_code_block_no_tool_calls_returns_empty(self):
        cb = _FakeCodeBlock(code='x = 1', tool_calls=None)
        assert _get_tool_data_for_block(cb, {}, 0) == []

    def test_code_block_empty_tool_calls_returns_empty(self):
        cb = _FakeCodeBlock(code='x = 1', tool_calls=[])
        assert _get_tool_data_for_block(cb, {}, 0) == []

    def test_tool_log_none_returns_empty(self):
        cs = _FakeCallSpec('search', args=['query'])
        cb = _FakeCodeBlock(tool_calls=[cs])
        assert _get_tool_data_for_block(cb, None, 0) == []

    def test_key_missing_from_tool_log_returns_empty(self):
        cs = _FakeCallSpec('search', args=['query'])
        cb = _FakeCodeBlock(tool_calls=[cs])
        assert _get_tool_data_for_block(cb, {}, 0) == []

    def test_single_tool_call_with_string_result(self):
        cs = _FakeCallSpec('search', args=['query'])
        cb = _FakeCodeBlock(tool_calls=[cs])
        tool_log = {1: 'result text'}
        data = _get_tool_data_for_block(cb, tool_log, 1)
        assert data == [(cs, 'result text')]

    def test_single_tool_call_result_stored_as_list(self):
        cs = _FakeCallSpec('fetch', kwargs={'url': 'http://x'})
        cb = _FakeCodeBlock(tool_calls=[cs])
        tool_log = {0: ['fetched content']}
        data = _get_tool_data_for_block(cb, tool_log, 0)
        assert data == [(cs, 'fetched content')]

    def test_multiple_tool_calls(self):
        cs1 = _FakeCallSpec('search', args=['q1'])
        cs2 = _FakeCallSpec('fetch', args=['url1'])
        cb = _FakeCodeBlock(tool_calls=[cs1, cs2])
        tool_log = {2: ['res1', 'res2']}
        data = _get_tool_data_for_block(cb, tool_log, 2)
        assert data == [(cs1, 'res1'), (cs2, 'res2')]

    def test_fewer_results_than_calls_uses_empty_string(self):
        cs1 = _FakeCallSpec('a')
        cs2 = _FakeCallSpec('b')
        cb = _FakeCodeBlock(tool_calls=[cs1, cs2])
        tool_log = {0: ['only one result']}
        data = _get_tool_data_for_block(cb, tool_log, 0)
        assert data == [(cs1, 'only one result'), (cs2, '')]


def _make_job_for_md(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    warmup_code=None,
    warmup_console_positions=None,
    chat_log=None,
    console=None,
    tool_log=None,
    exceptions=None,
):
    job = _make_job(warmup_code, warmup_console_positions, chat_log, console)
    job.py_env.tool_log = tool_log
    job.py_env.exceptions = exceptions or {}
    return job


def _call_build_md(job, **kwargs):
    """Helper: compute derived lists and call _build_md_content."""
    blocks = _warmup_blocks(job)
    ranges = _warmup_ranges(job)
    turns = _turn_ranges(job)
    return _build_md_content(
        uuid_str=kwargs.get('uuid_str', 'test-uuid'),
        status_str=kwargs.get('status_str', 'completed'),
        agent_role=kwargs.get('agent_role', 'my-agent'),
        model=kwargs.get('model', 'claude-3'),
        total_cost=kwargs.get('total_cost', 0.0042),
        num_turns=kwargs.get('num_turns', 0),
        exception_count=kwargs.get('exception_count', 0),
        system_prompt=kwargs.get('system_prompt', ''),
        initial_prompt=kwargs.get('initial_prompt', ''),
        job=job,
        warmup_blocks=blocks,
        warmup_ranges=ranges,
        turn_ranges=turns,
    )


class TestBuildMdContent:
    def test_includes_title(self):
        job = _make_job_for_md()
        md = _call_build_md(job)
        assert '# Job Detail' in md

    def test_includes_uuid_and_status(self):
        job = _make_job_for_md()
        md = _call_build_md(job, uuid_str='abc-123', status_str='running')
        assert 'abc-123' in md
        assert 'running' in md

    def test_includes_agent_model_cost_turns(self):
        job = _make_job_for_md()
        md = _call_build_md(job, agent_role='analyst', model='gpt-4', total_cost=1.5, num_turns=7)
        assert 'analyst' in md
        assert 'gpt-4' in md
        assert '1.5' in md
        assert '7' in md

    def test_includes_system_prompt_section_when_present(self):
        job = _make_job_for_md()
        md = _call_build_md(job, system_prompt='You are a helpful assistant.')
        assert 'System Prompt' in md
        assert 'You are a helpful assistant.' in md

    def test_omits_system_prompt_section_when_empty(self):
        job = _make_job_for_md()
        md = _call_build_md(job, system_prompt='')
        assert 'System Prompt' not in md

    def test_includes_initial_prompt_when_present(self):
        job = _make_job_for_md()
        md = _call_build_md(job, initial_prompt='Analyse this data.')
        assert 'Initial Prompt' in md
        assert 'Analyse this data.' in md

    def test_omits_initial_prompt_when_empty(self):
        job = _make_job_for_md()
        md = _call_build_md(job, initial_prompt='')
        assert 'Initial Prompt' not in md

    def test_includes_warmup_code(self):
        job = _make_job_for_md(warmup_code='x = 1', warmup_console_positions=[1], console=['ok\n'])
        md = _call_build_md(job)
        assert 'Warmup' in md
        assert 'x = 1' in md

    def test_includes_warmup_console_output(self):
        job = _make_job_for_md(warmup_code='x = 1', warmup_console_positions=[1], console=['ok\n'])
        md = _call_build_md(job)
        assert 'ok' in md

    def test_includes_llm_turn_section(self):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='result = 42')
        job = _make_job_for_md(chat_log=[chat_item], console=['done\n'])
        md = _call_build_md(job)
        assert 'Turn 1' in md
        assert 'result = 42' in md

    def test_includes_turn_console_output(self):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='x = 1')
        job = _make_job_for_md(chat_log=[chat_item], console=['output line\n'])
        md = _call_build_md(job)
        assert 'output line' in md

    def test_includes_tool_calls_in_turn(self):
        cs = _FakeCallSpec('search', args=['query'])
        cb = _FakeCodeBlock(code='search("query")', tool_calls=[cs])
        chat_item = _make_chat_log_item(console_pos=0, llm_resp=cb)
        job = _make_job_for_md(chat_log=[chat_item], tool_log={1: 'result text'})
        md = _call_build_md(job)
        assert 'Tool Call' in md
        assert 'search' in md
        assert 'result text' in md

    def test_includes_error_indicator_in_turn(self):
        chat_item = _make_chat_log_item(console_pos=0, llm_resp='bad code')
        job = _make_job_for_md(chat_log=[chat_item], exceptions={0: 'NameError: x'})
        md = _call_build_md(job)
        assert 'NameError: x' in md

    def test_multiple_turns_all_included(self):
        item1 = _make_chat_log_item(console_pos=0, llm_resp='step_one()')
        item2 = _make_chat_log_item(console_pos=1, llm_resp='step_two()')
        job = _make_job_for_md(chat_log=[item1, item2], console=['a\n', 'b\n'])
        md = _call_build_md(job)
        assert 'Turn 1' in md
        assert 'Turn 2' in md
        assert 'step_one()' in md
        assert 'step_two()' in md

    def test_returns_string(self):
        job = _make_job_for_md()
        md = _call_build_md(job)
        assert isinstance(md, str)
        assert len(md) > 0


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
            agent.system_prompt.side_effect = system_prompt_raises
        else:
            agent.system_prompt.return_value = system_prompt_return
        job_def.agent = agent
    job.job_def = job_def
    return job


class TestGetSystemPrompt:
    def test_returns_formatted_system_prompt(self):
        job = _make_job_with_agent(system_prompt_return='You are an assistant.')
        text, error = _get_system_prompt(job)
        assert text == 'You are an assistant.'
        assert error is None

    def test_returns_empty_string_when_no_job_def(self):
        job = _make_job_with_agent(no_job_def=True)
        text, error = _get_system_prompt(job)
        assert text == ''
        assert error is None

    def test_returns_empty_string_when_no_agent(self):
        job = _make_job_with_agent(no_agent=True)
        text, error = _get_system_prompt(job)
        assert text == ''
        assert error is None

    def test_falls_back_to_raw_prompt_on_format_error(self):
        job = _make_job_with_agent(
            system_prompt_raises=KeyError('missing_key'),
            raw_prompt='Raw template with {missing_key}.',
        )
        text, error = _get_system_prompt(job)
        assert text == 'Raw template with {missing_key}.'
        assert error is not None
        assert 'KeyError' in error

    def test_error_message_contains_exception_detail(self):
        job = _make_job_with_agent(
            system_prompt_raises=ValueError('bad format'),
            raw_prompt='template',
        )
        _, error = _get_system_prompt(job)
        assert 'ValueError' in error
        assert 'bad format' in error

    def test_returns_empty_string_when_formatted_and_raw_both_none(self):
        job = _make_job_with_agent(
            system_prompt_raises=KeyError('x'),
            raw_prompt=None,
        )
        text, error = _get_system_prompt(job)
        assert text == ''
        assert error is not None

    def test_returns_empty_string_on_unexpected_error(self):
        job = MagicMock()
        type(job).job_def = PropertyMock(side_effect=RuntimeError('db error'))
        text, error = _get_system_prompt(job)
        assert text == ''
        assert error is None


class _StubPyEnv:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.console = ['line1\n', 'line2\n']
        self.exceptions = {0: 'NameError: x'}
        self.tool_log = {1: 'result'}
        self.global_state = {'key': 'value'}
        self.local_state = {}


class _StubJob:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.model = 'claude-3-opus'
        self.model_family = 'claude'
        self.session_id = 'sess-abc'
        self.total_cost = 0.0042
        self.context_bytes = 1024
        self.total_bytes_sent = 512
        self.total_bytes_received = 512
        self.warmup_console_positions = [2]
        self.next_instr_num = None
        self.warmup_block_num = None
        self.chat_log = []
        self.py_env = _StubPyEnv()


class TestBuildRawRepr:
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
        assert 'total_cost' in result

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

    def test_includes_warmup_console_positions(self):
        job = _StubJob()
        result = _build_raw_repr(job)
        assert 'warmup_console_positions' in result
