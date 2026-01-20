"""Tests for Job class."""

import types
import dbzero as db0
from tests.conftest import create_chat_log_item
from statek.executors.job import Job, JobStatus


class TestJobDef:
    """Test cases for JobDef class."""

    def test_prompt_with_goal(self, job_def_factory):
        """Test prompt property formats description with goal."""
        job_def = job_def_factory(description="Complete the {goal} task", goal="analysis")
        assert job_def.prompt() == "Complete the analysis task"

    def test_prompt_without_goal(self, job_def_factory):
        """Test prompt property returns plain description when goal is None."""
        job_def = job_def_factory(description="Complete the task", goal=None)
        assert job_def.prompt() == "Complete the task"


class TestJob:
    """Test cases for Job class."""

    def test_get_next_prompt_first_prompt_empty_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console is empty."""
        job = job_factory(description="Analyze the data")
        result = job.get_next_prompt()

        # Should return just the job_def.prompt() since console is empty
        assert result == "Analyze the data"

    def test_get_next_prompt_first_prompt_with_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console has content."""
        job = job_factory(description="Process user data")

        # Add some console output
        job.py_env.console_append("Output line 1")
        job.py_env.console_append("Output line 2")

        result = job.get_next_prompt()

        # Should include the prompt and all console outputs from position 0
        expected = "Process user data\n> Output line 1\n> Output line 2"
        assert result == expected

    def test_get_next_prompt_subsequent_prompt_from_console_pos(self, job_factory):
        """Test get_next_prompt when chat_log has entries."""
        job = job_factory(description="Process data")

        # Setup console with multiple outputs
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")
        job.py_env.console_append("Output 3")

        # Add a chat log item that processed first 2 console entries
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Some LLM response"))

        result = job.get_next_prompt()

        # Should only include console outputs from position 2 onwards
        expected = "> Output 3"
        assert result == expected

    def test_get_next_prompt_subsequent_prompt_no_new_console(self, job_factory):
        """Test get_next_prompt when no new console output since last chat."""
        job = job_factory(description="Process data")

        # Setup console
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        # Add chat log item that already processed all console entries
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Response"))

        result = job.get_next_prompt()

        # Should return empty string as there's no new console output
        assert result == ""

    def test_get_next_prompt_multiple_chat_items(self, job_factory):
        """Test get_next_prompt uses the last chat log item's console_pos."""
        job = job_factory(description="Multi-step task")

        # Setup console with multiple outputs
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4", "Out5"]

        # Add multiple chat log items
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp2"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="resp3"))

        result = job.get_next_prompt()

        # Should use the last chat item's console_pos (4)
        expected = "> Out5"
        assert result == expected


class TestJobGetChatHistory:
    """Test cases for Job.get_chat_history method."""

    def test_get_chat_history_empty_chat_log(self, job_factory):
        """Test get_chat_history when chat_log is empty."""
        job = job_factory(description="Test task")

        # With empty chat_log, should yield nothing
        history = list(job.get_chat_history())
        assert not history

    def test_get_chat_history_single_chat_item(self, job_factory):
        """Test get_chat_history with one chat log item."""
        job = job_factory(description="Process data")

        # Setup console
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        # Add one chat log item
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="LLM response 1"))

        # Get history
        history = list(job.get_chat_history())

        # Should have 2 elements: [user_message, llm_response]
        assert len(history) == 2
        assert history[0] == "Process data\n> Output 1\n> Output 2"
        assert history[1] == "LLM response 1"

    def test_get_chat_history_multiple_chat_items(self, job_factory):
        """Test get_chat_history with multiple chat log items."""
        job = job_factory(description="Multi-step task")

        # Setup console with multiple outputs
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4", "Out5"]

        # Add multiple chat log items
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="resp2"))
        job.chat_log.append(create_chat_log_item(console_pos=5, llm_resp="resp3"))

        # Get history
        history = list(job.get_chat_history())

        # Should have 6 elements alternating user/assistant messages
        assert len(history) == 6

        # First user message: initial prompt + console from 0 to 2
        assert history[0] == "Multi-step task\n> Out1\n> Out2"
        # First assistant response
        assert history[1] == "resp1"
        # Second user message: console from 2 to 4
        assert history[2] == "> Out3\n> Out4"
        # Second assistant response
        assert history[3] == "resp2"
        # Third user message: console from 4 to 5
        assert history[4] == "> Out5"
        # Third assistant response
        assert history[5] == "resp3"


class TestJobGetNextRequest:
    """Test cases for Job.get_next_request method."""

    def test_get_next_request_first_request_no_history(self, job_factory):
        """Test get_next_request for the first request with no chat history."""
        job = job_factory(description="Analyze data")

        # Add some console output
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        request = job.get_next_request()

        # Verify all required keys are present
        assert "prompt" in request
        assert "chat_history" in request
        assert "system_prompt" in request

        # Verify prompt includes description and console
        assert request["prompt"] == "Analyze data\n> Output 1\n> Output 2"

        # Verify chat_history is empty generator (no history yet)
        assert not list(request["chat_history"])

        # Verify system_prompt is from agent
        assert request["system_prompt"] == "Test agent"

        # Verify session_id is not included when None
        assert "session_id" not in request

    def test_get_next_request_with_session_id(self, job_factory):
        """Test get_next_request includes session_id when set."""
        job = job_factory(description="Process data")
        job.session_id = "test-session-123"

        request = job.get_next_request()

        # Verify session_id is included
        assert "session_id" in request
        assert request["session_id"] == "test-session-123"

    def test_get_next_request_with_chat_history(self, job_factory):
        """Test get_next_request with existing chat history."""
        job = job_factory(description="Multi-step task")

        # Setup console
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4"]

        # Add chat log items
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Response 1"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="Response 2"))

        request = job.get_next_request()

        # Verify prompt is only new console output
        assert request["prompt"] == ""  # No new console after position 4

        # Verify chat_history contains alternating messages
        history = list(request["chat_history"])
        assert len(history) == 4
        assert history[0] == "Multi-step task\n> Out1\n> Out2"
        assert history[1] == "Response 1"
        assert history[2] == "> Out3\n> Out4"
        assert history[3] == "Response 2"

    def test_get_next_request_structure(self, job_factory):
        """Test that get_next_request returns a proper dictionary structure."""
        job = job_factory(description="Test")
        job.session_id = "session-abc"

        request = job.get_next_request()

        # Verify it's a dictionary
        assert isinstance(request, dict)

        # Verify types of values
        assert isinstance(request["prompt"], str)
        # chat_history should be a generator/iterable
        assert isinstance(request["chat_history"], types.GeneratorType)
        assert isinstance(request["system_prompt"], str)
        assert isinstance(request["session_id"], str)

    def test_get_next_request_empty_console_no_history(self, job_factory):
        """Test get_next_request with no console output and no history."""
        job = job_factory(description="Simple task")

        request = job.get_next_request()

        # Prompt should be just the description
        assert request["prompt"] == "Simple task"
        assert not list(request["chat_history"])
        assert "session_id" not in request

    def test_last_response_empty_chat_log(self, job_factory):
        """Test last_response returns None when chat_log is empty."""
        job = job_factory(description="Test job")
        assert job.chat_log == []
        assert job.last_response is None

    def test_last_response_with_chat_log(self, job_factory):
        """Test last_response returns the llm_resp from the last chat log item."""
        job = job_factory(description="Test job")

        # Add chat log items
        job.chat_log.append(create_chat_log_item(
            console_pos=0, llm_resp="print('first response')"
        ))
        job.chat_log.append(create_chat_log_item(
            console_pos=1, llm_resp="print('second response')"
        ))

        assert job.last_response == "print('second response')"


class TestJobSetStatus:  # pylint: disable=too-few-public-methods
    """Test cases for Job.set_status method."""

    def test_set_status_initial(self, job_factory):
        """Test setting initial job status."""

        job = job_factory(description="Test task")

        # Initial status should be READY
        assert job.status == JobStatus.READY  # pylint: disable=no-member
        assert len(db0.find(Job, JobStatus.READY)) == 1  # pylint: disable=no-member

        # Change status to STARTED
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member

        # Verify status is updated
        assert job.status == JobStatus.STARTED  # pylint: disable=no-member

        # Verify tags are updated
        assert len(db0.find(Job, JobStatus.READY)) == 0  # pylint: disable=no-member
        assert len(db0.find(Job, JobStatus.STARTED)) == 1  # pylint: disable=no-member


class TestJobAppendChatLog:
    """Test cases for Job.append_chat_log method."""

    def test_append_chat_log_empty_console(self, job_factory):
        """Test append_chat_log with empty console."""
        job = job_factory(description="Test task")

        # Create a request (doesn't matter for this test)
        request = job.get_next_request()

        # Append LLM response
        llm_resp = "print('hello')"
        job.append_chat_log(request, llm_resp)

        # Verify chat_log has one item
        assert len(job.chat_log) == 1
        assert job.chat_log[0].llm_resp == "print('hello')"
        assert job.chat_log[0].console_pos == 0  # Empty console length

    def test_append_chat_log_with_console_output(self, job_factory):
        """Test append_chat_log with console output."""
        job = job_factory(description="Process data")

        # Add console output
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")
        job.py_env.console_append("Output 3")

        # Get request and append response
        request = job.get_next_request()
        llm_resp = "x = 5"
        job.append_chat_log(request, llm_resp)

        # Verify console_pos is set to console length
        assert len(job.chat_log) == 1
        assert job.chat_log[0].console_pos == 3
        assert job.chat_log[0].llm_resp == "x = 5"

    def test_append_chat_log_multiple_times(self, job_factory):
        """Test append_chat_log called multiple times."""
        job = job_factory(description="Multi-step task")

        # First interaction
        job.py_env.console_append("Step 1 output")
        request1 = job.get_next_request()
        job.append_chat_log(request1, "code_block_1")

        # Second interaction
        job.py_env.console_append("Step 2 output")
        job.py_env.console_append("Step 2 more output")
        request2 = job.get_next_request()
        job.append_chat_log(request2, "code_block_2")

        # Third interaction
        job.py_env.console_append("Step 3 output")
        request3 = job.get_next_request()
        job.append_chat_log(request3, "code_block_3")

        # Verify all chat log items
        assert len(job.chat_log) == 3

        assert job.chat_log[0].console_pos == 1
        assert job.chat_log[0].llm_resp == "code_block_1"

        assert job.chat_log[1].console_pos == 3
        assert job.chat_log[1].llm_resp == "code_block_2"

        assert job.chat_log[2].console_pos == 4
        assert job.chat_log[2].llm_resp == "code_block_3"
