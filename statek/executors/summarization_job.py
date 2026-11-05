# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forked jobs that apply structured summaries to a target job's history."""
# pylint: disable=no-member
from dataclasses import dataclass

import dbzero as db0

from statek.executors.chat_log_item import SystemLogItem
from statek.executors.forked_job import ForkedJob
from statek.executors.job import Job, JobStatus
from statek.executors.summarization import parse_summarization_result


@db0.memo
@dataclass(init=False, eq=False, repr=False)
class SummarizationJob(ForkedJob):
    """Summarize a source continuation and compact an explicit target range.

    The target range uses raw chat-log indices independently of ``forked_at``.
    Call ``push_summarization_instr`` separately before starting execution.
    """
    __target_job: Job
    __chat_log_pos: int
    __chat_log_len: int

    def __init__(self, src_job: Job, forked_at: int, *, target_job: Job,
                 chat_log_pos: int, chat_log_len: int):
        if not isinstance(target_job, Job):
            raise TypeError('target_job must be a Job')
        if (not isinstance(chat_log_pos, int) or isinstance(chat_log_pos, bool)
                or not isinstance(chat_log_len, int) or isinstance(chat_log_len, bool)):
            raise TypeError('chat_log_pos and chat_log_len must be integers')
        if (chat_log_pos < 0 or chat_log_len <= 0
                or chat_log_pos + chat_log_len > len(target_job.chat_log)):
            raise ValueError('Invalid target chat-log range')
        super().__init__(src_job, forked_at)
        self.__target_job = target_job
        self.__chat_log_pos = chat_log_pos
        self.__chat_log_len = chat_log_len
        db0.tags(self).add(db0.as_tag(target_job))
        if src_job != target_job:
            db0.tags(self).add(db0.as_tag(src_job))

    def post_turn_hook(self) -> None:
        """Apply a valid summary or request a correction for invalid model output."""
        try:
            summary = parse_summarization_result(self)
        except ValueError as error:
            self.chat_log.append(SystemLogItem(
                console_pos=self.py_env.console_len,
                message=(f'{type(error).__name__}: {error}\n'
                         'Return a corrected summarization response following the '
                         'original JSON instructions.'),
            ))
            self.py_env.exit_status = None
            self.set_status(JobStatus.STARTED)
            return

        self.__target_job.compact(self.__chat_log_pos, self.__chat_log_len, summary)
        if self.__target_job.status == JobStatus.SUSPENDED:
            self.__target_job.set_status(JobStatus.STARTED)
        self.set_status(JobStatus.DONE)
