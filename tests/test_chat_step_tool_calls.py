# pylint: disable=unused-argument
"""Tests that get_next_request populates tool_calls in ChatStepAssistantData."""

from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.chat_log_item import WarmupLogItem
from statek.utils import CodeBlock, CallSpec
from tests.conftest import create_chat_log_item


def _make_agent(role):
    return Agent(role=role, _system_prompt="Sys",
                 _metadata={"prompt_template": "Task"}, _tools=[])


def _make_job(agent, warmup_code=None, started=False):
    job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
    status = JobStatus.STARTED if started else JobStatus.READY  # pylint: disable=no-member
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=status)


def _add_warmup_log_item(job, block_num, tool_log=None, console_pos=0):
    """Helper to add a WarmupLogItem with optional tool_log."""
    item = WarmupLogItem(
        console_pos=console_pos,
        warmup_block_num=block_num,
        tool_log=tool_log,
    )
    job.chat_log.append(item)
    return item


class TestGetNextRequestToolCalls:
    """_full_history in get_next_request populates tool_calls in ChatStepAssistantData."""

    def test_warmup_code_block_with_tool_calls_populates_tool_calls(self, db0_fixture):
        """Warmup CodeBlock with tool calls gives populated tool_calls in ChatStepAssistantData."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={"x": 1})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc1"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log="'result_value'")

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [s for s in history if s.tool_calls is not None]
        assert len(steps_with_tools) == 1
        step = steps_with_tools[0]
        assert len(step.tool_calls) == 1
        key = next(iter(step.tool_calls))
        assert key.id == "STATEK-001"
        assert key.name == "my_tool"
        assert step.tool_calls[key] == "'result_value'"

    def test_warmup_str_block_has_no_tool_calls(self, db0_fixture):
        """A plain-string warmup block produces tool_calls=None in ChatStepAssistantData."""
        job = _make_job(_make_agent("tc2"), warmup_code="x = 1")

        history = list(job.get_next_request()["chat_history"])

        for step in history:
            assert step.tool_calls is None

    def test_warmup_tool_calls_key_args_and_kwargs_correct(self, db0_fixture):
        """CallParams key preserves id, name, args, and kwargs from CallSpec."""
        cs = CallSpec(id="STATEK-007", func_name="fetch_data",
                      args=["arg1"], kwargs={"limit": 10})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc3"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log="fetched")

        history = list(job.get_next_request()["chat_history"])
        step = next(s for s in history if s.tool_calls)

        key = next(iter(step.tool_calls))
        assert key.id == "STATEK-007"
        assert key.name == "fetch_data"
        assert key.args == ["arg1"]
        assert key.kwargs == {"limit": 10}

    def test_warmup_multiple_tool_calls_all_in_dict(self, db0_fixture):
        """Multiple tool calls in a warmup block are all included in tool_calls dict."""
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs1, cs2])
        job = _make_job(_make_agent("tc4"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log=["'alpha'", "'beta'"])

        history = list(job.get_next_request()["chat_history"])
        step = next(s for s in history if s.tool_calls)

        assert len(step.tool_calls) == 2
        names = {k.name: v for k, v in step.tool_calls.items()}
        assert names["tool_a"] == "'alpha'"
        assert names["tool_b"] == "'beta'"

    def test_warmup_tool_calls_none_when_tool_log_missing(self, db0_fixture):
        """tool_calls is None when no WarmupLogItem exists (tool wasn't executed)."""
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc5"), warmup_code=warmup)
        # No WarmupLogItem added

        history = list(job.get_next_request()["chat_history"])

        for step in history:
            assert step.tool_calls is None

    def test_llm_turn_with_tool_calls_populates_chat_step_data(self, db0_fixture):
        """An LLM response with tool calls populates tool_calls in the ChatStepAssistantData."""
        job = _make_job(_make_agent("tc6"), started=True)
        cs = CallSpec(id="T-001", func_name="do_thing", args=[], kwargs={"n": 5})
        block = CodeBlock(code="x = 1", tool_calls=[cs])
        item = create_chat_log_item(console_pos=0, llm_resp=block)
        item.tool_log = "'thing_done'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [s for s in history if s.tool_calls is not None]
        assert len(steps_with_tools) == 1
        step = steps_with_tools[0]
        key = next(iter(step.tool_calls))
        assert key.id == "T-001"
        assert key.name == "do_thing"
        assert step.tool_calls[key] == "'thing_done'"

    def test_llm_turn_without_tool_calls_has_none(self, db0_fixture):
        """An LLM response without tool calls produces tool_calls=None."""
        job = _make_job(_make_agent("tc7"), started=True)
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 42"))

        history = list(job.get_next_request()["chat_history"])

        for step in history:
            assert step.tool_calls is None

    def test_warmup_and_llm_turn_both_get_correct_tool_calls(self, db0_fixture):
        """Warmup and LLM turn tool calls are each assigned the correct log entry."""
        cs_warmup = CallSpec(id="STATEK-001", func_name="warmup_tool", args=[], kwargs={})
        warmup = CodeBlock(code="setup()", tool_calls=[cs_warmup])
        job = _make_job(_make_agent("tc8"), warmup_code=warmup, started=True)

        job.py_env.console = ["warmup_line", "llm_line"]
        # Add WarmupLogItem with tool result
        _add_warmup_log_item(job, 0, tool_log="'warmup_result'")
        # Add LLM turn with tool result
        cs_llm = CallSpec(id="T-001", func_name="llm_tool", args=[], kwargs={})
        block = CodeBlock(code="run()", tool_calls=[cs_llm])
        item = create_chat_log_item(console_pos=2, llm_resp=block)
        item.tool_log = "'llm_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [(i, s) for i, s in enumerate(history) if s.tool_calls is not None]
        assert len(steps_with_tools) == 2

        warmup_step = next(s for _, s in steps_with_tools
                           if any(k.name == "warmup_tool" for k in s.tool_calls))
        wkey = next(k for k in warmup_step.tool_calls if k.name == "warmup_tool")
        assert warmup_step.tool_calls[wkey] == "'warmup_result'"

        llm_step = next(s for _, s in steps_with_tools
                        if any(k.name == "llm_tool" for k in s.tool_calls))
        lkey = next(k for k in llm_step.tool_calls if k.name == "llm_tool")
        assert llm_step.tool_calls[lkey] == "'llm_result'"

    def test_second_llm_turn_tool_calls_assigned_correctly(self, db0_fixture):
        """Second LLM turn's tool_calls use the item's own tool_log."""
        job = _make_job(_make_agent("tc9"), started=True)

        # Turn 0: plain string (no tool calls), console_pos=0
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 1"))
        # Turn 1: CodeBlock with tool call, console_pos=2
        cs = CallSpec(id="T-002", func_name="second_tool", args=[], kwargs={})
        block = CodeBlock(code="y = 2", tool_calls=[cs])
        item = create_chat_log_item(console_pos=2, llm_resp=block)
        item.tool_log = "'second_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [s for s in history if s.tool_calls is not None]
        assert len(steps_with_tools) == 1
        key = next(iter(steps_with_tools[0].tool_calls))
        assert key.name == "second_tool"
        assert steps_with_tools[0].tool_calls[key] == "'second_result'"

    def test_two_warmup_blocks_each_get_own_tool_calls(self, db0_fixture):
        """Two warmup CodeBlocks with tool calls each populate their own tool_calls."""
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        block1 = CodeBlock(code="", tool_calls=[cs1])
        block2 = CodeBlock(code="", tool_calls=[cs2])
        job = _make_job(_make_agent("tc_multi_wu"), warmup_code=[block1, block2])
        # Add WarmupLogItems with tool results
        _add_warmup_log_item(job, 0, tool_log="'alpha'")
        _add_warmup_log_item(job, 1, tool_log="'beta'", console_pos=1)
        job.py_env.console = ["tool_a_output", "tool_b_output"]
        job.warmup_block_num = 1

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [s for s in history if s.tool_calls is not None]
        assert len(steps_with_tools) == 2
        names = {}
        for step in steps_with_tools:
            for k, v in step.tool_calls.items():
                names[k.name] = v
        assert names["tool_a"] == "'alpha'"
        assert names["tool_b"] == "'beta'"

    def test_two_warmup_blocks_and_llm_turn_all_correct(self, db0_fixture):  # pylint: disable=too-many-locals
        """Two warmup blocks + LLM turn all get correct tool_calls from tool_log."""
        cs1 = CallSpec(id="STATEK-001", func_name="warmup_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="warmup_b", args=[], kwargs={})
        block1 = CodeBlock(code="", tool_calls=[cs1])
        block2 = CodeBlock(code="", tool_calls=[cs2])
        job = _make_job(_make_agent("tc_multi_wu_llm"), warmup_code=[block1, block2],
                        started=True)
        job.warmup_block_num = 1
        job.py_env.console = ["wu_a_out", "wu_b_out", "llm_line"]
        # Add WarmupLogItems — each warmup block added 1 console entry
        _add_warmup_log_item(job, 0, tool_log="'wu_alpha'")
        _add_warmup_log_item(job, 1, tool_log="'wu_beta'", console_pos=1)
        # Add LLM turn
        cs_llm = CallSpec(id="T-001", func_name="llm_tool", args=[], kwargs={})
        llm_block = CodeBlock(code="run()", tool_calls=[cs_llm])
        item = create_chat_log_item(console_pos=2, llm_resp=llm_block)
        item.tool_log = "'llm_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])

        steps_with_tools = [s for s in history if s.tool_calls is not None]
        assert len(steps_with_tools) == 3
        names = {}
        for step in steps_with_tools:
            for k, v in step.tool_calls.items():
                names[k.name] = v
        assert names["warmup_a"] == "'wu_alpha'"
        assert names["warmup_b"] == "'wu_beta'"
        assert names["llm_tool"] == "'llm_result'"
