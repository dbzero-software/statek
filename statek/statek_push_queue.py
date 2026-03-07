"""StatekPushQueue - singleton wrapper over FiFoQueue(s) for inter-process messaging."""
# pylint: disable=no-member
from typing import List, Optional, Tuple

import db0_rpc
import dbzero as db0

from statek.executors.job import Job
from statek.fifo_queue import FiFoQueue


@db0.memo(singleton=True)
class StatekPushQueue:
    """Singleton wrapper over one or more FiFoQueues for specific use-cases.

    Manages a dedicated queue per channel (currently: job_console).
    Both push and pop are decorated with @db0_rpc.remote so they can be
    invoked across prefix boundaries.
    """

    def __init__(self):
        self.__job_console_queue = FiFoQueue()

    @db0_rpc.remote
    def push_to_job_console(self, job: Job, message: str):
        """Append a (job, message) pair to the job-console queue.

        The job is a memo object and may come from a different prefix;
        FiFoQueue stores it automatically as a weak_ref.

        Args:
            job: The Job instance associated with the message.
            message: The console message text to push.
        """
        self.__job_console_queue.push_back(job=job, message=message)

    @db0_rpc.remote
    def pop_from_job_console(
        self,
        count: int,
        prefix: Optional[str | int] = None,  # pylint: disable=unused-argument
    ) -> List[Tuple[Job, str]]:
        """Retrieve and remove up to *count* earliest job-console entries.

        Intended for remote use from the Statek process.

        Args:
            count: Maximum number of entries to retrieve.
            prefix: Optional job-selection criteria (not yet handled).

        Returns:
            List of (Job, message) tuples in FIFO order.
        """
        job_filter = None
        if prefix is not None:
            if isinstance(prefix, str):
                def _filter_by_name(**kwargs):
                    return db0.get_prefix_of(kwargs['job']).name == prefix
                job_filter = _filter_by_name
            else:
                def _filter_by_uuid(**kwargs):
                    return db0.get_prefix_of(kwargs['job']).uuid == prefix
                job_filter = _filter_by_uuid
        items = self.__job_console_queue.pop_front(count, filter=job_filter)
        return [(item["job"], item["message"]) for item in items]
