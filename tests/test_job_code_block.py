"""Tests for Job CodeBlock handling."""

# pylint: disable=no-member

from tests.conftest import create_chat_log_item
from statek.executors.chat_log_item import LLM_LogItem
from statek.chat_history import ChatRole
from statek.llm_api import LLM_Response, LLM_Stats, CallParams
from statek.utils import CodeBlock, CallSpec


class TestChatLogItemCodeBlock:
    """Test that ChatLogItem.llm_resp accepts CodeBlock values."""

    def test_stores_code_block(self, db0_fixture):  # pylint: disable=unused-argument
        """LLM_LogItem.llm_resp can hold a CodeBlock."""
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        item = LLM_LogItem(console_pos=0, llm_resp=block)
        assert isinstance(item.llm_resp, CodeBlock)
        assert item.llm_resp.code == "x = 1"

    def test_still_accepts_str(self, db0_fixture):  # pylint: disable=unused-argument
        """LLM_LogItem.llm_resp still accepts a plain string (backwards compat)."""
        item = LLM_LogItem(console_pos=0, llm_resp="print('hello')")
        assert item.llm_resp == "print('hello')"


class TestGetChatHistoryCodeBlock:
    """Test get_chat_history() when llm_resp is a CodeBlock."""

    def test_code_block_llm_resp_uses_code_field(self, job_factory):
        """get_chat_history() uses .code when llm_resp is a CodeBlock."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())

        # ASSISTANT items with tool_calls carry the code in .content.
        asst = [h for h in history
                if h.role == ChatRole.ASSISTANT and h.tool_calls is not None]
        assert len(asst) == 1
        assert asst[0].content == "x = 1"

    def test_code_block_tool_calls_not_in_assistant_content(self, job_factory):
        """Tool call specs do not appear in the assistant content text."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())
        asst = next(h for h in history
                    if h.role == ChatRole.ASSISTANT and h.tool_calls is not None)
        assert "my_tool" not in (asst.content or "")

    def test_code_block_none_code_yields_no_assistant_content(self, job_factory):
        """When CodeBlock.code is None, the assistant item has no text content."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code=None, tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())
        asst = next(h for h in history
                    if h.role == ChatRole.ASSISTANT and h.tool_calls is not None)
        assert not asst.content

    def test_code_block_none_code_no_assistant_when_no_tool_calls(self, job_factory):
        """CodeBlock.code=None and no tool_calls produces no assistant item."""
        job = job_factory()
        block = CodeBlock(code=None, tool_calls=[])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())
        assert [h for h in history if h.role == ChatRole.ASSISTANT] == []


class TestAppendChatLogCodeBlock:
    """Test append_chat_log() with LLM_Response containing tool call requests."""

    def test_stores_code_block_when_call_requests_present(self, job_factory):
        """append_chat_log() stores a CodeBlock when call_requests is non-empty."""
        job = job_factory()
        call_params = CallParams(call_id="T-001", name="my_tool", args=[], kwargs={})
        llm_resp = LLM_Response(
            text="x = 1",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=[call_params],
        )

        request = job.get_next_request()
        job.append_chat_log(request, llm_resp)

        assert len(job.chat_log) == 1
        assert isinstance(job.chat_log[0].llm_resp, CodeBlock)
        assert job.chat_log[0].llm_resp.code == "x = 1"

    def test_code_block_tool_calls_mapped_from_call_requests(self, job_factory):
        """CallParams in call_requests are mapped to CallSpec in the stored CodeBlock."""
        job = job_factory()
        call_params = CallParams(call_id="T-001", name="my_tool", args=[], kwargs={"x": 1})
        llm_resp = LLM_Response(
            text="",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=[call_params],
        )

        request = job.get_next_request()
        job.append_chat_log(request, llm_resp)

        stored = job.chat_log[0].llm_resp
        assert isinstance(stored, CodeBlock)
        assert len(stored.tool_calls) == 1
        assert stored.tool_calls[0].func_name == "my_tool"
        assert stored.tool_calls[0].id == "T-001"

    def test_stores_str_when_no_call_requests(self, job_factory):
        """append_chat_log() stores a plain str when call_requests is None."""
        job = job_factory()
        llm_resp = LLM_Response(
            text="x = 1",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )

        request = job.get_next_request()
        job.append_chat_log(request, llm_resp)

        assert isinstance(job.chat_log[0].llm_resp, str)

    def test_console_pos_recorded_correctly_with_call_requests(self, job_factory):
        """console_pos is recorded correctly when llm_resp has call_requests."""
        job = job_factory()
        job.py_env.console_append("line 1")
        job.py_env.console_append("line 2")
        call_params = CallParams(call_id="T-001", name="my_tool", args=[], kwargs={})
        llm_resp = LLM_Response(
            text="x = 1",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=[call_params],
        )

        request = job.get_next_request()
        job.append_chat_log(request, llm_resp)

        assert job.chat_log[0].console_pos == 2


class TestGetToolCalls:
    """Tests for CodeBlock.get_tool_calls and convenience wrappers."""

    def test_get_tool_calls_returns_none_when_no_tool_calls(self, db0_fixture):  # pylint: disable=unused-argument
        block = CodeBlock(code="x = 1", tool_calls=None)
        assert block.get_tool_calls(lambda cs: True) is None

    def test_get_tool_calls_returns_matching_calls(self, db0_fixture):  # pylint: disable=unused-argument
        cs1 = CallSpec(id="1", func_name="alpha")
        cs2 = CallSpec(id="2", func_name="beta")
        cs3 = CallSpec(id="3", func_name="alpha")
        block = CodeBlock(tool_calls=[cs1, cs2, cs3])
        result = block.get_tool_calls(lambda c: c.func_name == "alpha")
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "3"

    def test_get_regular_tool_calls_excludes_python_cli(self, db0_fixture):  # pylint: disable=unused-argument
        cs1 = CallSpec(id="1", func_name="my_tool")
        cs2 = CallSpec(id="2", func_name="python_cli")
        block = CodeBlock(tool_calls=[cs1, cs2])
        result = block.get_regular_tool_calls()
        assert len(result) == 1
        assert result[0].func_name == "my_tool"

    def test_get_cli_tool_calls_returns_only_python_cli(self, db0_fixture):  # pylint: disable=unused-argument
        cs1 = CallSpec(id="1", func_name="my_tool")
        cs2 = CallSpec(id="2", func_name="python_cli")
        cs3 = CallSpec(id="3", func_name="python_cli", kwargs={"code": "y=2"})
        block = CodeBlock(tool_calls=[cs1, cs2, cs3])
        result = block.get_cli_tool_calls()
        assert len(result) == 2
        assert all(c.func_name == "python_cli" for c in result)
