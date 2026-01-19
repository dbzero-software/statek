"""Tests for Job class."""

from statek.executors.chat_log_item import ChatLogItem


class TestJob:
    """Test cases for Job class."""

    def test_last_response_empty_chat_log(self, simple_job):  # pylint: disable=redefined-outer-name
        """Test last_response returns None when chat_log is empty."""
        assert simple_job.chat_log == []
        assert simple_job.last_response is None

    def test_last_response_with_chat_log(self, simple_job):  # pylint: disable=redefined-outer-name
        """Test last_response returns the llm_resp from the last chat log item."""
        # Add chat log items
        simple_job.chat_log.append(ChatLogItem(
            console_pos=0,
            llm_resp="print('first response')"
        ))
        simple_job.chat_log.append(ChatLogItem(
            console_pos=1,
            llm_resp="print('second response')"
        ))

        assert simple_job.last_response == "print('second response')"
