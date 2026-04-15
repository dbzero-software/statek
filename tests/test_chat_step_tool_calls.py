# pylint: disable=unused-argument,no-member
"""Tests that get_next_request emits ChatHistoryItem objects with tool calls."""

from statek.agents.agent import Agent
from statek.executors.job import Job, JobDef, JobStatus
from statek.executors.chat_log_item import WarmupLogItem
from statek.chat_history import ChatRole
from statek.utils import CodeBlock, CallSpec
from tests.conftest import create_chat_log_item


def _make_agent(role):
    return Agent(role=role, _system_prompt="Sys",
                 _metadata={"prompt_template": "Task", "MODEL": "test-model"}, _tools=[])


def _make_job(agent, warmup_code=None, started=False):
    job_def = JobDef(agent=agent, job_params=None, warmup_code=warmup_code)
    status = JobStatus.STARTED if started else JobStatus.READY  # pylint: disable=no-member
    return Job(job_def=job_def, model_family="test", model="test-model",
               job_status=status)


def _add_warmup_log_item(job, block_num, tool_log=None, console_pos=0):
    item = WarmupLogItem(
        console_pos=console_pos,
        warmup_block_num=block_num,
        tool_log=tool_log,
    )
    job.chat_log.append(item)
    return item


def _asst_with_tools(history):
    """Return ChatHistoryItems that are ASSISTANT messages carrying tool_calls."""
    return [
        h for h in history
        if h.role == ChatRole.ASSISTANT and h.tool_calls is not None
    ]


def _tool_results(history):
    """Return ChatHistoryItems with role TOOL, indexed by tool_call id."""
    return {h.tool_calls.id: h for h in history if h.role == ChatRole.TOOL}


class TestGetNextRequestToolCalls:
    """get_next_request emits ChatHistoryItem ASSISTANT(tool_calls) + TOOL items."""

    def test_warmup_code_block_with_tool_calls_populates_tool_calls(self, db0_fixture):
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={"x": 1})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc1"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log="'result_value'")

        history = list(job.get_next_request()["chat_history"])

        asst = _asst_with_tools(history)
        assert len(asst) == 1
        tcs = asst[0].tool_calls
        assert len(tcs) == 1
        assert tcs[0].id == "STATEK-001"
        assert tcs[0].func_name == "my_tool"
        results = _tool_results(history)
        assert results["STATEK-001"].content == "'result_value'"

    def test_warmup_str_block_has_no_tool_calls(self, db0_fixture):
        job = _make_job(_make_agent("tc2"), warmup_code="x = 1")
        history = list(job.get_next_request()["chat_history"])
        assert _asst_with_tools(history) == []

    def test_warmup_tool_calls_key_args_and_kwargs_correct(self, db0_fixture):
        cs = CallSpec(id="STATEK-007", func_name="fetch_data",
                      args=["arg1"], kwargs={"limit": 10})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc3"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log="fetched")

        history = list(job.get_next_request()["chat_history"])
        asst = _asst_with_tools(history)[0]
        cs_emitted = asst.tool_calls[0]
        assert cs_emitted.id == "STATEK-007"
        assert cs_emitted.func_name == "fetch_data"
        assert cs_emitted.args == ["arg1"]
        assert cs_emitted.kwargs == {"limit": 10}

    def test_warmup_multiple_tool_calls_all_in_dict(self, db0_fixture):
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs1, cs2])
        job = _make_job(_make_agent("tc4"), warmup_code=warmup)
        _add_warmup_log_item(job, 0, tool_log=["'alpha'", "'beta'"])

        history = list(job.get_next_request()["chat_history"])
        asst = _asst_with_tools(history)[0]
        assert [tc.func_name for tc in asst.tool_calls] == ["tool_a", "tool_b"]
        results = _tool_results(history)
        assert results["STATEK-001"].content == "'alpha'"
        assert results["STATEK-002"].content == "'beta'"

    def test_warmup_tool_calls_none_when_tool_log_missing(self, db0_fixture):
        cs = CallSpec(id="STATEK-001", func_name="my_tool", args=[], kwargs={})
        warmup = CodeBlock(code='exit("ok")', tool_calls=[cs])
        job = _make_job(_make_agent("tc5"), warmup_code=warmup)
        # No WarmupLogItem added — no assistant tool-call item is emitted at all.
        history = list(job.get_next_request()["chat_history"])
        assert _asst_with_tools(history) == []

    def test_llm_turn_with_tool_calls_populates_chat_step_data(self, db0_fixture):
        job = _make_job(_make_agent("tc6"), started=True)
        cs = CallSpec(id="T-001", func_name="do_thing", args=[], kwargs={"n": 5})
        block = CodeBlock(code="x = 1", tool_calls=[cs])
        item = create_chat_log_item(console_pos=0, llm_resp=block)
        item.tool_log = "'thing_done'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])
        asst = _asst_with_tools(history)
        assert len(asst) == 1
        assert asst[0].tool_calls[0].id == "T-001"
        assert _tool_results(history)["T-001"].content == "'thing_done'"

    def test_llm_turn_without_tool_calls_has_none(self, db0_fixture):
        job = _make_job(_make_agent("tc7"), started=True)
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 42"))

        history = list(job.get_next_request()["chat_history"])
        assert _asst_with_tools(history) == []

    def test_warmup_and_llm_turn_both_get_correct_tool_calls(self, db0_fixture):
        cs_warmup = CallSpec(id="STATEK-001", func_name="warmup_tool", args=[], kwargs={})
        warmup = CodeBlock(code="setup()", tool_calls=[cs_warmup])
        job = _make_job(_make_agent("tc8"), warmup_code=warmup, started=True)

        job.py_env.console = ["warmup_line", "llm_line"]
        _add_warmup_log_item(job, 0, tool_log="'warmup_result'")
        cs_llm = CallSpec(id="T-001", func_name="llm_tool", args=[], kwargs={})
        block = CodeBlock(code="run()", tool_calls=[cs_llm])
        item = create_chat_log_item(console_pos=2, llm_resp=block)
        item.tool_log = "'llm_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])

        asst = _asst_with_tools(history)
        assert len(asst) == 2
        names = [a.tool_calls[0].func_name for a in asst]
        assert set(names) == {"warmup_tool", "llm_tool"}
        results = _tool_results(history)
        assert results["STATEK-001"].content == "'warmup_result'"
        assert results["T-001"].content == "'llm_result'"

    def test_second_llm_turn_tool_calls_assigned_correctly(self, db0_fixture):
        job = _make_job(_make_agent("tc9"), started=True)
        job.chat_log.append(create_chat_log_item(console_pos=0, llm_resp="x = 1"))
        cs = CallSpec(id="T-002", func_name="second_tool", args=[], kwargs={})
        block = CodeBlock(code="y = 2", tool_calls=[cs])
        item = create_chat_log_item(console_pos=2, llm_resp=block)
        item.tool_log = "'second_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])
        asst = _asst_with_tools(history)
        assert len(asst) == 1
        assert asst[0].tool_calls[0].func_name == "second_tool"
        assert _tool_results(history)["T-002"].content == "'second_result'"

    def test_two_warmup_blocks_each_get_own_tool_calls(self, db0_fixture):
        cs1 = CallSpec(id="STATEK-001", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="tool_b", args=[], kwargs={})
        block1 = CodeBlock(code="", tool_calls=[cs1])
        block2 = CodeBlock(code="", tool_calls=[cs2])
        job = _make_job(_make_agent("tc_multi_wu"), warmup_code=[block1, block2])
        _add_warmup_log_item(job, 0, tool_log="'alpha'")
        _add_warmup_log_item(job, 1, tool_log="'beta'", console_pos=1)
        job.py_env.console = ["tool_a_output", "tool_b_output"]
        job.warmup_block_num = 1

        history = list(job.get_next_request()["chat_history"])
        results = _tool_results(history)
        assert results["STATEK-001"].content == "'alpha'"
        assert results["STATEK-002"].content == "'beta'"

    def test_two_warmup_blocks_and_llm_turn_all_correct(self, db0_fixture):
        cs1 = CallSpec(id="STATEK-001", func_name="warmup_a", args=[], kwargs={})
        cs2 = CallSpec(id="STATEK-002", func_name="warmup_b", args=[], kwargs={})
        block1 = CodeBlock(code="", tool_calls=[cs1])
        block2 = CodeBlock(code="", tool_calls=[cs2])
        job = _make_job(_make_agent("tc_multi_wu_llm"), warmup_code=[block1, block2],
                        started=True)
        job.warmup_block_num = 1
        job.py_env.console = ["wu_a_out", "wu_b_out", "llm_line"]
        _add_warmup_log_item(job, 0, tool_log="'wu_alpha'")
        _add_warmup_log_item(job, 1, tool_log="'wu_beta'", console_pos=1)
        cs_llm = CallSpec(id="T-001", func_name="llm_tool", args=[], kwargs={})
        llm_block = CodeBlock(code="run()", tool_calls=[cs_llm])
        item = create_chat_log_item(console_pos=2, llm_resp=llm_block)
        item.tool_log = "'llm_result'"
        job.chat_log.append(item)

        history = list(job.get_next_request()["chat_history"])
        results = _tool_results(history)
        assert results["STATEK-001"].content == "'wu_alpha'"
        assert results["STATEK-002"].content == "'wu_beta'"
        assert results["T-001"].content == "'llm_result'"
