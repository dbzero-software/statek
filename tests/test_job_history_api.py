"""The extracted history API preserves the existing Job interface."""
# pylint: disable=no-member,protected-access,import-outside-toplevel

import ast
from pathlib import Path

from statek.executors import job as job_module
from statek.executors.chat_log_item import LLM_LogItem, UserLogItem


def test_history_api_matches_existing_methods(job_factory):
    from statek.executors.job_history import JobHistory

    job = job_factory()
    job.chat_log = [UserLogItem("hello"), LLM_LogItem(console_pos=0, llm_resp="reply")]
    job.console_append("output")
    history = job.history
    assert isinstance(history, JobHistory)
    assert list(history.get_chat_history()) == list(job.get_chat_history())
    assert list(history.iter_history_steps()) == list(job.iter_history_steps())
    assert list(history.get_dialog()) == list(job.get_dialog())
    assert list(history.get_chat_responses()) == list(job.get_chat_responses())
    assert list(history.get_llm_response_times()) == list(job.get_llm_response_times())
    assert list(history.get_llm_step_sizes()) == list(job.get_llm_step_sizes())
    assert history.num_turns == job.num_turns == 1
    assert history.last_response == job.last_response == "reply"
    assert history.exception_count == job.exception_count == 0
    job.chat_log.append(LLM_LogItem(console_pos=1, llm_resp="later"))
    assert history.num_turns == job.num_turns == 2


def test_existing_models_stay_defined_in_job():
    from statek.executors import job_history

    module = ast.parse(Path(job_module.__file__).read_text(encoding="utf-8"))
    definitions = {
        node.name: node for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    for name in ("PyHandleDesc", "StateSummary", "JobSummary", "JobCompactionResult"):
        assert name in definitions
        assert [ast.unparse(decorator) for decorator in definitions[name].decorator_list] == [
            "db0.memo", "dataclass",
        ]
        assert getattr(job_module, name).__module__ == "statek.executors.job"
    assert job_history.HistoryStep is job_module.HistoryStep
    assert job_module.DialogItem.__module__ == "statek.executors.job"
    assert job_history.format_job_summary is job_module.format_job_summary
