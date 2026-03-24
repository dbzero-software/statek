"""Tests for Job class."""

import types
from unittest.mock import patch, MagicMock
import dbzero as db0
from tests.conftest import create_chat_log_item, set_warmup_positions
from statek.executors.job import Job, JobDefError, JobStatus
from statek.llm_api import ChatStepData, LLM_Response, LLM_Stats
from statek.settings import ChatStyle
from statek.utils import CodeBlock, CallSpec


class TestJobDefError:
    """Test cases for JobDefError class."""

    def _make_raised_error(self, msg="something went wrong"):
        """Return an exception that has been raised (has a traceback)."""
        try:
            raise ValueError(msg)
        except ValueError as exc:
            return exc

    def test_error_message_is_set(self, db0_fixture):  # pylint: disable=unused-argument
        """error_message is set to the string representation of the exception."""
        error = self._make_raised_error("boom")
        jde = JobDefError(error)
        assert jde.error_message == "boom"

    def test_traceback_collected_by_default(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is a non-empty sequence of strings when collect_traceback=True."""
        error = self._make_raised_error("oops")
        jde = JobDefError(error)
        assert jde.traceback is not None
        assert len(jde.traceback) > 0
        assert all(isinstance(s, str) for s in jde.traceback)

    def test_traceback_not_collected_when_disabled(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is None when collect_traceback=False."""
        error = self._make_raised_error("oops")
        jde = JobDefError(error, collect_traceback=False)
        assert jde.traceback is None

    def test_traceback_none_when_no_traceback_on_exception(self, db0_fixture):  # pylint: disable=unused-argument
        """traceback is None when exception was never raised (no __traceback__)."""
        error = ValueError("never raised")
        jde = JobDefError(error)
        assert jde.traceback is None


class TestJobWithError:
    """Test cases for Job.error field."""

    def test_job_error_is_none_by_default(self, job_factory):
        """Job.error is None when created without error."""
        job = job_factory()
        assert job.error is None

    def test_job_error_can_be_set(self, job_factory, db0_fixture):  # pylint: disable=unused-argument
        """Job.error can be set to a JobDefError instance."""
        try:
            raise RuntimeError("job failed")
        except RuntimeError as exc:
            err = JobDefError(exc)
            job = job_factory()
            job.error = err
            assert job.error is err
            assert job.error.error_message == "job failed"


class TestJobDef:
    """Test cases for JobDef class."""

    def test_prompt_with_goal_in_job_params(self, agent_factory, job_def_factory):
        """Test prompt property formats template with goal in job_params."""
        agent = agent_factory(prompt_template="Complete the {goal} task")
        job_def = job_def_factory(job_params={"goal": "analysis"})
        # Need to use custom agent
        job_def.agent = agent
        assert job_def.prompt() == "Complete the analysis task"

    def test_prompt_without_goal(self, agent_factory, job_def_factory):
        """Test prompt property returns plain template when no job_params."""
        agent = agent_factory(prompt_template="Complete the task")
        job_def = job_def_factory(job_params=None)
        job_def.agent = agent
        assert job_def.prompt() == "Complete the task"

    def test_prompt_with_job_params(self, agent_factory, job_def_factory):
        """Test prompt method formats description with job_params."""
        agent = agent_factory(prompt_template="Process {data_type} for {user}")
        job_params = {"data_type": "transactions", "user": "John"}
        job_def = job_def_factory(job_params=job_params)
        job_def.agent = agent
        assert job_def.prompt() == "Process transactions for John"

    def test_prompt_with_job_params_in_jobdef(self, agent_factory, job_def_factory):
        """Test prompt method formats description with job_params stored in JobDef."""
        agent = agent_factory(prompt_template="Process {data_type} for {user}")
        job_params = {"data_type": "orders", "user": "Alice"}
        job_def = job_def_factory(job_params=job_params)
        job_def.agent = agent
        assert job_def.prompt() == "Process orders for Alice"

    def test_prompt_with_job_params_and_goal(self, agent_factory, job_def_factory):
        """Test prompt method with goal included in job_params."""
        agent = agent_factory(
            prompt_template="Process {data_type} with {status} for {user}. Complete the {goal}."
        )
        job_params = {"data_type": "orders", "status": "pending", "user": "Bob", "goal": "analysis"}
        job_def = job_def_factory(job_params=job_params)
        job_def.agent = agent
        result = job_def.prompt()
        assert result == "Process orders with pending for Bob. Complete the analysis."


class TestJob:
    """Test cases for Job class."""

    def test_get_next_prompt_first_prompt_empty_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console is empty."""
        job = job_factory()
        result = job.get_next_prompt()

        # Should return just the job_def.prompt() since console is empty
        assert result == "Test task"

    def test_get_next_prompt_first_prompt_with_console(self, job_factory):
        """Test get_next_prompt when chat_log is empty and console has content."""
        job = job_factory()

        # Add some console output
        job.py_env.console_append("Output line 1")
        job.py_env.console_append("Output line 2")

        result = job.get_next_prompt()

        # Should include the prompt and all console outputs from position 0
        expected = "Test task\n> Output line 1\n> Output line 2"
        assert result == expected

    def test_get_next_prompt_subsequent_prompt_from_console_pos(self, job_factory):
        """Test get_next_prompt when chat_log has entries."""
        job = job_factory()

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
        job = job_factory()

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
        job = job_factory()

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

    def test_get_next_prompt_push_log_none_no_change(self, job_factory):
        """No push_log → behaviour is unchanged."""
        job = job_factory()
        job.py_env.console_append("Out1")
        result = job.get_next_prompt()
        assert result == "Test task\n> Out1"

    def test_get_next_prompt_first_prompt_push_log_appended(self, job_factory):
        """First prompt: push_log message is appended after console output."""
        job = job_factory()
        job.py_env.console_append("Out1")
        job.push_to_console("user message")  # key=1
        result = job.get_next_prompt()
        assert "Out1" in result
        assert "user message" in result

    def test_get_next_prompt_first_prompt_push_log_order(self, job_factory):
        """First prompt: push_log message appears after console output."""
        job = job_factory()
        job.py_env.console_append("Out1")
        job.push_to_console("user message")
        result = job.get_next_prompt()
        assert result.index("Out1") < result.index("user message")

    def test_get_next_prompt_subsequent_prompt_push_log_at_from_pos(self, job_factory):
        """Subsequent prompt: push_log entry at from_pos is included."""
        job = job_factory()
        job.py_env.console = ["c1", "c2"]
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp"))
        job.push_to_console("pushed msg")  # key=2 (console len=2)
        result = job.get_next_prompt()
        assert "pushed msg" in result

    def test_get_next_prompt_subsequent_prompt_push_log_before_from_pos_excluded(self, job_factory):
        """Subsequent prompt: push_log entry with key < from_pos is excluded."""
        job = job_factory()
        job.py_env.console = ["c1", "c2"]
        job.push_to_console("early msg")     # key=2 (console len=2 at push time)
        job.py_env.console.append("c3")      # console grows to 3
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp"))
        result = job.get_next_prompt()
        assert "early msg" not in result

    def test_get_next_prompt_push_log_list_values_included(self, job_factory):
        """Multiple pushes at same position (stored as list) are all included."""
        job = job_factory()
        job.push_to_console("msg1")  # key=0
        job.push_to_console("msg2")  # key=0, becomes list
        result = job.get_next_prompt()
        assert "msg1" in result
        assert "msg2" in result


class TestJobGetChatHistory:
    """Test cases for Job.get_chat_history method."""

    def test_get_chat_history_empty_chat_log(self, job_factory):
        """Test get_chat_history when chat_log is empty."""
        job = job_factory()

        # With empty chat_log, should yield nothing
        history = list(job.get_chat_history())
        assert not history

    def test_get_chat_history_single_chat_item(self, job_factory):
        """Test get_chat_history with one chat log item."""
        job = job_factory()

        # Setup console
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        # Add one chat log item
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="LLM response 1"))

        # Get history
        history = list(job.get_chat_history())

        # Should have 2 elements: [user_message, llm_response]
        assert len(history) == 2
        assert history[0] == "Test task\n> Output 1\n> Output 2"
        assert history[1] == "LLM response 1"

    def test_get_chat_history_multiple_chat_items(self, job_factory):
        """Test get_chat_history with multiple chat log items."""
        job = job_factory()

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
        assert history[0] == "Test task\n> Out1\n> Out2"
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
        job = job_factory()

        # Add some console output
        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")

        request = job.get_next_request()

        # Verify required keys are present (no 'prompt' key — it's in chat_history)
        assert "prompt" not in request
        assert "chat_history" in request
        assert "system_prompt" in request

        # Verify chat_history contains only the current prompt (no prior history)
        history = list(request["chat_history"])
        expected = ChatStepData(code="", console_output="Test task\n> Output 1\n> Output 2")
        assert history == [expected]

        # Verify system_prompt is from agent
        assert request["system_prompt"] == "Test agent"

        # Verify session_id is not included when None
        assert "session_id" not in request

    def test_get_next_request_with_session_id(self, job_factory):
        """Test get_next_request includes session_id when set."""
        job = job_factory()
        job.session_id = "test-session-123"

        request = job.get_next_request()

        # Verify session_id is included
        assert "session_id" in request
        assert request["session_id"] == "test-session-123"

    def test_get_next_request_with_chat_history(self, job_factory):
        """Test get_next_request with existing chat history."""
        job = job_factory()

        # Setup console
        job.py_env.console = ["Out1", "Out2", "Out3", "Out4"]

        # Add chat log items
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="Response 1"))
        job.chat_log.append(create_chat_log_item(console_pos=4, llm_resp="Response 2"))

        request = job.get_next_request()

        # Verify no separate 'prompt' key — it's the last element of chat_history
        assert "prompt" not in request

        # Verify chat_history contains ChatStepData objects ending with the current prompt
        history = list(request["chat_history"])
        assert len(history) == 3
        assert history[0] == ChatStepData(code="", console_output="Test task\n> Out1\n> Out2")
        assert history[1] == ChatStepData(code="Response 1", console_output="> Out3\n> Out4")
        assert history[2] == ChatStepData(code="Response 2", console_output="")  # no new console

    def test_get_next_request_structure(self, job_factory):
        """Test that get_next_request returns a proper dictionary structure."""
        job = job_factory()
        job.session_id = "session-abc"

        request = job.get_next_request()

        # Verify it's a dictionary with no separate 'prompt' key
        assert isinstance(request, dict)
        assert "prompt" not in request

        # Verify types of values
        assert isinstance(request["chat_history"], types.GeneratorType)
        assert isinstance(request["system_prompt"], str)
        assert isinstance(request["session_id"], str)

    def test_get_next_request_empty_console_no_history(self, job_factory):
        """Test get_next_request with no console output and no history."""
        job = job_factory()

        request = job.get_next_request()

        # chat_history should contain only the current prompt (just the description)
        assert "prompt" not in request
        history = list(request["chat_history"])
        assert history == [ChatStepData(code="", console_output="Test task")]
        assert "session_id" not in request

    def test_last_response_empty_chat_log(self, job_factory):
        """Test last_response returns None when chat_log is empty."""
        job = job_factory()
        assert job.chat_log == []
        assert job.last_response is None

    def test_last_response_with_chat_log(self, job_factory):
        """Test last_response returns the llm_resp from the last chat log item."""
        job = job_factory()

        # Add chat log items
        job.chat_log.append(create_chat_log_item(
            console_pos=0, llm_resp="print('first response')"
        ))
        job.chat_log.append(create_chat_log_item(
            console_pos=1, llm_resp="print('second response')"
        ))

        assert job.last_response == "print('second response')"

    def test_last_response_returns_code_block(self, job_factory):
        """last_response returns a CodeBlock when llm_resp is a CodeBlock."""
        job = job_factory()
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        result = job.last_response

        assert isinstance(result, CodeBlock)
        assert result.code == "x = 1"
        assert result.tool_calls[0].func_name == "my_tool"


class TestJobGetNextCodeBlock:
    """Tests for Job.get_next_code_block — covering str and CodeBlock returns."""

    def test_returns_none_when_done(self, job_factory):
        """Returns None when job status is DONE."""
        job = job_factory()
        job.set_status(JobStatus.DONE)  # pylint: disable=no-member

        assert job.get_next_code_block() is None

    def test_returns_str_warmup_when_ready(self, job_factory):
        """Returns plain string warmup block when status is READY."""
        job = job_factory(warmup_code="x = 1")

        result = job.get_next_code_block()

        assert result == "x = 1"
        assert isinstance(result, str)

    def test_returns_code_block_warmup_when_ready(self, job_factory):
        """Returns CodeBlock warmup block when status is READY and warmup is a CodeBlock."""
        call_spec = CallSpec(id="W-001", func_name="setup_tool", args=[], kwargs={})
        block = CodeBlock(code="setup()", tool_calls=[call_spec])
        job = job_factory(warmup_code=block)

        result = job.get_next_code_block()

        assert isinstance(result, CodeBlock)
        assert result.code == "setup()"

    def test_returns_str_from_last_response(self, job_factory):
        """Returns plain string last_response when status is STARTED."""
        job = job_factory()
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 42"))

        result = job.get_next_code_block()

        assert result == "x = 42"
        assert isinstance(result, str)

    def test_returns_code_block_from_last_response(self, job_factory):
        """Returns CodeBlock last_response when status is STARTED and llm_resp is a CodeBlock."""
        job = job_factory()
        job.set_status(JobStatus.STARTED)  # pylint: disable=no-member
        call_spec = CallSpec(id="T-001", func_name="my_tool", args=[], kwargs={})
        block = CodeBlock(code="x = 1", tool_calls=[call_spec])
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp=block))

        result = job.get_next_code_block()

        assert isinstance(result, CodeBlock)
        assert result.code == "x = 1"
        assert result.tool_calls[0].func_name == "my_tool"


class TestJobGetNextPromptWithWarmup:
    """Test get_next_prompt when warmup was executed."""

    def test_no_warmup_behavior_unchanged(self, job_factory):
        """Without warmup_code the prompt is unchanged."""
        job = job_factory()
        job.py_env.console = ["line1", "line2"]

        result = job.get_next_prompt()

        assert result == "Test task\n> line1\n> line2"

    def test_warmup_prompt_is_last_block_console_output(self, job_factory):
        """With warmup, prompt is the console output OF the last warmup block."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["last block output"]
        set_warmup_positions(job, [1])  # block 0 produced 1 line

        result = job.get_next_prompt()

        assert "> last block output" in result

    def test_warmup_prompt_excludes_template_and_warmup_code(self, job_factory):
        """With warmup, template and code are NOT in the prompt (they're in chat_history)."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["last block output"]
        set_warmup_positions(job, [1])

        result = job.get_next_prompt()

        assert "Test task" not in result
        assert "x = 1" not in result

    def test_warmup_multi_block_prompt_is_last_block_output(self, job_factory):
        """With multiple warmup blocks, prompt is ONLY the last block's console output."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])  # block0->1 line, block1->1 line

        result = job.get_next_prompt()

        assert "> out2" in result

    def test_warmup_multi_block_prompt_excludes_earlier_blocks(self, job_factory):
        """Earlier warmup block outputs are NOT in the prompt (they're in chat_history)."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])

        result = job.get_next_prompt()

        assert "block1" not in result
        assert "block2" not in result
        assert "> out1" not in result


class TestJobGetChatHistoryWithWarmup:
    """Test get_chat_history formats warmup as assistant/user message pairs."""

    def test_warmup_no_chat_log_yields_history(self, job_factory):
        """With warmup but empty chat_log, get_chat_history is non-empty."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup output"]
        set_warmup_positions(job, [1])

        history = list(job.get_chat_history())

        assert len(history) > 0

    def test_warmup_no_chat_log_single_block_structure(self, job_factory):
        """Single warmup block, no chat_log: [template(user), warmup(asst)]."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup output", "remaining"]
        set_warmup_positions(job, [1])

        history = list(job.get_chat_history())

        # Exactly 2 items: template (user) and warmup code (assistant)
        assert len(history) == 2
        assert history[0] == "Test task"     # user: template
        assert "x = 1" in history[1]        # assistant: warmup code

    def test_warmup_no_chat_log_two_blocks_structure(self, job_factory):
        """Two warmup blocks, no chat_log: [template, w0, console0, w1]."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2", "remaining"]
        set_warmup_positions(job, [1, 2])

        history = list(job.get_chat_history())

        # 4 items: template, block1, console0, block2
        assert len(history) == 4
        assert history[0] == "Test task"      # user: template
        assert "block1" in history[1]         # assistant: warmup block 0
        assert "> out1" in history[2]         # user: console for block 0
        assert "block2" in history[3]         # assistant: warmup block 1

    def test_warmup_no_chat_log_last_item_is_assistant(self, job_factory):
        """Last item in chat_history is always the last warmup block (assistant)."""
        job = job_factory(warmup_code=["block1", "block2"])
        job.py_env.console = ["out1", "out2"]
        set_warmup_positions(job, [1, 2])

        history = list(job.get_chat_history())

        assert "block2" in history[-1]

    def test_warmup_with_chat_log_structure(self, job_factory):
        """With warmup and chat_log: [template, warmup, merged_console, first_resp, …]."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup out", "post-warmup out"]
        set_warmup_positions(job, [1])
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))

        history = list(job.get_chat_history())

        # [template(user), warmup(asst), merged_console(user), resp1(asst)]
        assert len(history) == 4
        assert history[0] == "Test task"             # user: template
        assert "x = 1" in history[1]                # assistant: warmup code
        assert "> warmup out" in history[2]          # user: merged console (warmup + remaining)
        assert "> post-warmup out" in history[2]     # user: merged console (both lines)
        assert history[3] == "resp1"                 # assistant: first LLM response

    def test_warmup_with_chat_log_merged_console_covers_remaining(self, job_factory):
        """The last warmup user message covers console up to the first LLM turn."""
        job = job_factory(warmup_code="x = 1")
        # warmup produced line 0, then extra line 1 before first LLM call
        job.py_env.console = ["warmup out", "extra before llm", "after llm"]
        set_warmup_positions(job, [1])   # warmup produced 1 line
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp2"))

        history = list(job.get_chat_history())

        # history[2] is the merged user message after warmup (covers console[0:2])
        assert "> warmup out" in history[2]
        assert "> extra before llm" in history[2]

    def test_warmup_not_in_subsequent_messages(self, job_factory):
        """Warmup code does not appear in subsequent user messages."""
        job = job_factory(warmup_code="x = 1")
        job.py_env.console = ["warmup out", "post-warmup out", "after-first-llm"]
        set_warmup_positions(job, [1])
        job.chat_log.append(create_chat_log_item(console_pos=2, llm_resp="resp1"))
        job.chat_log.append(create_chat_log_item(console_pos=3, llm_resp="resp2"))

        history = list(job.get_chat_history())

        # Structure: [template, warmup, merged(0:2), resp1, console(2:3), resp2]
        # history[4] is the second user message (console after first LLM turn)
        second_user_msg = history[4]
        assert "x = 1" not in second_user_msg
        assert "> after-first-llm" in second_user_msg


class TestJobSetStatus:  # pylint: disable=too-few-public-methods
    """Test cases for Job.set_status method."""

    def test_set_status_initial(self, job_factory):
        """Test setting initial job status."""
        job = job_factory()

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
        job = job_factory()

        request = job.get_next_request()
        llm_resp = LLM_Response(
            text="print('hello')",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )
        job.append_chat_log(request, llm_resp)

        assert len(job.chat_log) == 1
        assert job.chat_log[0].llm_resp == "print('hello')"
        assert job.chat_log[0].console_pos == 0

    def test_append_chat_log_with_console_output(self, job_factory):
        """Test append_chat_log with console output."""
        job = job_factory()

        job.py_env.console_append("Output 1")
        job.py_env.console_append("Output 2")
        job.py_env.console_append("Output 3")

        request = job.get_next_request()
        llm_resp = LLM_Response(
            text="x = 5",
            session_id=None,
            stats=LLM_Stats(total_bytes_sent=0, total_bytes_received=0, cost=None),
            call_requests=None,
        )
        job.append_chat_log(request, llm_resp)

        assert len(job.chat_log) == 1
        assert job.chat_log[0].console_pos == 3
        assert job.chat_log[0].llm_resp == "x = 5"

    def test_append_chat_log_multiple_times(self, job_factory):
        """Test append_chat_log called multiple times."""
        job = job_factory()

        job.py_env.console_append("Step 1 output")
        request1 = job.get_next_request()
        job.append_chat_log(request1, LLM_Response(
            text="code_block_1", session_id=None,
            stats=LLM_Stats(0, 0, None), call_requests=None,
        ))

        job.py_env.console_append("Step 2 output")
        job.py_env.console_append("Step 2 more output")
        request2 = job.get_next_request()
        job.append_chat_log(request2, LLM_Response(
            text="code_block_2", session_id=None,
            stats=LLM_Stats(0, 0, None), call_requests=None,
        ))

        job.py_env.console_append("Step 3 output")
        request3 = job.get_next_request()
        job.append_chat_log(request3, LLM_Response(
            text="code_block_3", session_id=None,
            stats=LLM_Stats(0, 0, None), call_requests=None,
        ))

        assert len(job.chat_log) == 3

        assert job.chat_log[0].console_pos == 1
        assert job.chat_log[0].llm_resp == "code_block_1"

        assert job.chat_log[1].console_pos == 3
        assert job.chat_log[1].llm_resp == "code_block_2"

        assert job.chat_log[2].console_pos == 4
        assert job.chat_log[2].llm_resp == "code_block_3"


class TestJobDefErrors:
    """Tests for JobDef.set_error, get_errors, has_errors."""

    def _make_raised_error(self, msg="something went wrong"):
        try:
            raise ValueError(msg)
        except ValueError as exc:
            return exc

    def test_has_errors_false_by_default(self, job_def_factory):
        """has_errors returns False when no errors have been set."""
        job_def = job_def_factory()
        assert job_def.has_errors() is False

    def test_get_errors_empty_by_default(self, job_def_factory):
        """get_errors yields nothing when no errors have been set."""
        job_def = job_def_factory()
        assert not list(job_def.get_errors())

    def test_set_error_creates_job_def_error(self, job_def_factory):
        """set_error creates a JobDefError associated with the job definition."""
        job_def = job_def_factory()
        error = self._make_raised_error("boom")
        job_def.set_error(error)
        errors = list(job_def.get_errors())
        assert len(errors) == 1
        assert isinstance(errors[0], JobDefError)
        assert errors[0].error_message == "boom"

    def test_set_error_has_errors_true(self, job_def_factory):
        """has_errors returns True after set_error is called."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("oops"))
        assert job_def.has_errors() is True

    def test_set_error_collects_traceback_by_default(self, job_def_factory):
        """set_error collects traceback by default."""
        job_def = job_def_factory()
        error = self._make_raised_error("traceback test")
        job_def.set_error(error)
        errors = list(job_def.get_errors())
        assert errors[0].traceback is not None
        assert len(errors[0].traceback) > 0

    def test_set_error_no_traceback_when_disabled(self, job_def_factory):
        """set_error does not collect traceback when collect_traceback=False."""
        job_def = job_def_factory()
        error = self._make_raised_error("no tb")
        job_def.set_error(error, collect_traceback=False)
        errors = list(job_def.get_errors())
        assert errors[0].traceback is None

    def test_set_error_multiple_errors(self, job_def_factory):
        """set_error can be called multiple times; all errors are retrievable."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("first"))
        job_def.set_error(self._make_raised_error("second"))
        errors = list(job_def.get_errors())
        assert len(errors) == 2
        messages = {e.error_message for e in errors}
        assert messages == {"first", "second"}

    def test_errors_isolated_between_job_defs(self, job_def_factory):
        """Errors set on one JobDef are not visible from another."""
        job_def1 = job_def_factory()
        job_def2 = job_def_factory()
        job_def1.set_error(self._make_raised_error("only for def1"))
        assert not list(job_def2.get_errors())
        assert job_def2.has_errors() is False

    def test_clear_errors_on_empty_job_def_does_not_raise(self, job_def_factory):
        """clear_errors does nothing and does not raise when there are no errors."""
        job_def = job_def_factory()
        job_def.clear_errors()
        assert job_def.has_errors() is False

    def test_clear_errors_removes_single_error(self, job_def_factory):
        """clear_errors removes a single error so has_errors returns False."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("boom"))
        job_def.clear_errors()
        assert job_def.has_errors() is False
        assert not list(job_def.get_errors())

    def test_clear_errors_removes_multiple_errors(self, job_def_factory):
        """clear_errors removes all errors when multiple errors were set."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("first"))
        job_def.set_error(self._make_raised_error("second"))
        job_def.set_error(self._make_raised_error("third"))
        job_def.clear_errors()
        assert not list(job_def.get_errors())

    def test_clear_errors_is_idempotent(self, job_def_factory):
        """clear_errors can be called twice without raising."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("once"))
        job_def.clear_errors()
        job_def.clear_errors()
        assert job_def.has_errors() is False

    def test_clear_errors_does_not_affect_other_job_def(self, job_def_factory):
        """clear_errors on one JobDef does not remove errors from another."""
        job_def1 = job_def_factory()
        job_def2 = job_def_factory()
        job_def1.set_error(self._make_raised_error("def1 error"))
        job_def2.set_error(self._make_raised_error("def2 error"))
        job_def1.clear_errors()
        assert job_def1.has_errors() is False
        assert job_def2.has_errors() is True

    def test_set_error_after_clear_errors(self, job_def_factory):
        """set_error can be used again after clear_errors."""
        job_def = job_def_factory()
        job_def.set_error(self._make_raised_error("old"))
        job_def.clear_errors()
        job_def.set_error(self._make_raised_error("new"))
        errors = list(job_def.get_errors())
        assert len(errors) == 1
        assert errors[0].error_message == "new"


class TestJobDefUpdateWarmupCode:
    """Tests for JobDef.update_warmup_code."""

    def test_update_applies_new_value(self, job_def_factory):
        """warmup_code is updated when the parsed new value differs from current."""
        job_def = job_def_factory(warmup_code=None)
        job_def.update_warmup_code("x = 1")
        assert job_def.warmup_code == "x = 1"

    def test_update_none_clears_existing(self, job_def_factory):
        """Passing None clears an existing warmup_code value."""
        job_def = job_def_factory(warmup_code="x = 1")
        job_def.update_warmup_code(None)
        assert job_def.warmup_code is None

    def test_no_update_when_value_identical(self, job_def_factory):
        """warmup_code is not reassigned when the parsed value equals the current one."""
        job_def = job_def_factory(warmup_code="x = 1")
        with patch('statek.executors.job.statek_log') as mock_log:
            job_def.update_warmup_code("x = 1")
        update_calls = [c for c in mock_log.call_args_list
                        if "updating warmup code" in str(c)]
        assert update_calls == []
        assert job_def.warmup_code == "x = 1"

    def test_no_update_when_none_stays_none(self, job_def_factory):
        """Calling update_warmup_code(None) on a None field does nothing."""
        job_def = job_def_factory(warmup_code=None)
        with patch('statek.executors.job.statek_log') as mock_log:
            job_def.update_warmup_code(None)
        update_calls = [c for c in mock_log.call_args_list
                        if "updating warmup code" in str(c)]
        assert update_calls == []
        assert job_def.warmup_code is None

    def test_update_logs_debug_when_changed(self, job_def_factory):
        """A debug log is emitted exactly once when the value actually changes."""
        job_def = job_def_factory(warmup_code=None)
        with patch('statek.executors.job.statek_log') as mock_log:
            job_def.update_warmup_code("x = 1")
        update_calls = [c for c in mock_log.call_args_list
                        if "updating warmup code" in str(c)]
        assert len(update_calls) == 1

    def test_update_sequence_to_list(self, job_def_factory):
        """A sequence of two blocks is stored as a two-element sequence."""
        job_def = job_def_factory(warmup_code=None)
        job_def.update_warmup_code(["a = 1", "b = 2"])
        warmup = job_def.warmup_code
        assert len(warmup) == 2
        assert warmup[0] == "a = 1"
        assert warmup[1] == "b = 2"


class TestJobDefChatStyle:
    """Tests for JobDef._chat_style field and chat_style property."""

    def test_chat_style_defaults_to_none(self, job_def_factory):
        """_chat_style is None by default."""
        job_def = job_def_factory()
        assert job_def._chat_style is None  # pylint: disable=protected-access

    def test_chat_style_property_returns_job_level_when_set(self, job_def_factory):
        """chat_style property returns the job-level value when explicitly set."""
        job_def = job_def_factory()
        job_def.set_chat_style(ChatStyle.CONSOLE)  # pylint: disable=no-member
        assert job_def.chat_style == ChatStyle.CONSOLE  # pylint: disable=no-member

    def test_chat_style_property_falls_back_to_settings(self, job_def_factory):
        """chat_style property returns StatekSettings.chat_style when _chat_style is None."""
        job_def = job_def_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = ChatStyle.MARKDOWN  # pylint: disable=no-member
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings):
            assert job_def.chat_style == ChatStyle.MARKDOWN  # pylint: disable=no-member

    def test_chat_style_property_returns_none_when_both_unset(self, job_def_factory):
        """chat_style property returns None when neither job-level nor settings are set."""
        job_def = job_def_factory()
        mock_settings = MagicMock()
        mock_settings.chat_style = None
        with patch('statek.executors.job.get_statek_settings', return_value=mock_settings):
            assert job_def.chat_style is None




class TestJobWithUserLogItem:
    """Tests for Job integration with UserLogItem and str in chat_log."""

    def test_chat_log_accepts_str(self, job_factory):
        """A plain str can be appended to chat_log."""
        job = job_factory()
        job.chat_log.append("user message")
        assert job.chat_log[-1] == "user message"

    def test_chat_log_accepts_user_log_item(self, job_factory):
        """A UserLogItem can be appended to chat_log."""
        job = job_factory()
        item = UserLogItem(message="Hello")
        job.chat_log.append(item)
        assert job.chat_log[-1] is item

    def test_num_turns_ignores_user_log_item(self, job_factory):
        """num_turns only counts LLM_LogItem, not UserLogItem or str."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp1"))
        job.chat_log.append(UserLogItem(message="follow-up"))
        assert job.num_turns == 1

    def test_last_response_skips_user_log_item(self, job_factory):
        """last_response returns the last LLM response, not a UserLogItem."""
        job = job_factory()
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="resp1"))
        job.chat_log.append(UserLogItem(message="follow-up"))
        assert job.last_response == "resp1"

    def test_get_next_prompt_after_user_log_item(self, job_factory):
        """get_next_prompt after a UserLogItem includes the user message."""
        job = job_factory()
        job.py_env.console_append("output1")
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code1"))
        job.chat_log.append(UserLogItem(message="Please continue"))
        result = job.get_next_prompt()
        assert "Please continue" in result

    def test_get_next_prompt_after_str_message(self, job_factory):
        """get_next_prompt after a str message includes it."""
        job = job_factory()
        job.py_env.console_append("output1")
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code1"))
        job.chat_log.append("Please continue")
        result = job.get_next_prompt()
        assert "Please continue" in result

    def test_get_chat_history_includes_user_log_item(self, job_factory):
        """get_chat_history yields UserLogItem message as a user turn."""
        job = job_factory()
        job.py_env.console_append("output1")
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code1"))
        job.chat_log.append(UserLogItem(message="next question"))
        history = list(job.get_chat_history())
        # The user message should appear in the history
        assert any("next question" in h for h in history)

    def test_get_chat_history_includes_str_message(self, job_factory):
        """get_chat_history yields str message as a user turn."""
        job = job_factory()
        job.py_env.console_append("output1")
        job.chat_log.append(create_chat_log_item(console_pos=1, llm_resp="code1"))
        job.chat_log.append("next question")
        history = list(job.get_chat_history())
        assert any("next question" in h for h in history)
