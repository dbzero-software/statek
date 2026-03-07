"""Tests for Job CodeBlock handling."""

from unittest.mock import patch, MagicMock

from tests.conftest import create_chat_log_item
from statek.executors.chat_log_item import ChatLogItem
from statek.llm_api import LLM_Response, LLM_Stats, CallParams
from statek.utils import CodeBlock, CallSpec
from statek.settings import ChatStyle


class TestChatLogItemCodeBlock:
    """Test that ChatLogItem.llm_resp accepts CodeBlock values."""

    def test_stores_code_block(self, db0_fixture):  # pylint: disable=unused-argument
        """ChatLogItem.llm_resp can hold a CodeBlock."""
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        item = ChatLogItem(console_pos=0, llm_resp=block)
        assert isinstance(item.llm_resp, CodeBlock)
        assert item.llm_resp.code == "x = 1"

    def test_still_accepts_str(self, db0_fixture):  # pylint: disable=unused-argument
        """ChatLogItem.llm_resp still accepts a plain string (backwards compat)."""
        item = ChatLogItem(console_pos=0, llm_resp="print('hello')")
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

        assert len(history) == 2
        assert "x = 1" in history[1]

    def test_code_block_tool_calls_not_in_history(self, job_factory):
        """Tool call specs from CodeBlock do not appear in chat history text."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())

        assert "my_tool" not in history[1]

    def test_code_block_none_code_yields_empty(self, job_factory):
        """When CodeBlock.code is None, chat history yields empty string."""
        job = job_factory()
        block = CodeBlock(code=None, tool_calls=[])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        history = list(job.get_chat_history())

        assert history[1] == ""

    def test_code_block_none_code_yields_empty_in_markdown_mode(self, job_factory):
        """In MARKDOWN mode, CodeBlock.code=None yields '' not an empty fence."""
        job = job_factory()
        block = CodeBlock(code=None, tool_calls=[])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member
        mock_settings.get_xml_box_tags.return_value = None

        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings):
            history = list(job.get_chat_history())

        assert history[1] == ""


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
