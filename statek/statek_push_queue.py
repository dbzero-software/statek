"""StatekPushQueue - singleton wrapper over FiFoQueue(s) for inter-process messaging."""
# pylint: disable=no-member
from typing import List, Optional, Tuple

import dbzero as db0

from statek.fifo_queue import FiFoQueue
from statek.rpc_integration import rpc as db0_rpc

@db0.memo(singleton=True)
class StatekPushQueue:
    """Singleton wrapper over one or more FiFoQueues for specific use-cases.

    Manages a dedicated queue per channel (currently: job_console).
    Both push and pop are decorated with @db0_rpc.remote so they can be
    invoked across prefix boundaries.
    """

    def __init__(self):
        self.__job_console_queue = FiFoQueue()

    def is_empty(self) -> bool:
        return self.__job_console_queue.is_empty()

    @db0_rpc.remote
    def push_to_job_console(self, job_uuid: str, message: str):
        """Append a (job_uuid, message) pair to the job-console queue.

        Args:
            job_uuid: The job object's UUID (may be from a different prefix).
            message: The console message text to push.
        """
        self.__job_console_queue.push_back(job_uuid=job_uuid, message=message)

    @db0_rpc.remote
    def pop_from_job_console(
        self,
        count: int,
        prefix: Optional[str | int] = None,  # pylint: disable=unused-argument
    ) -> List[Tuple[str, str]]:
        """Retrieve and remove up to *count* earliest job-console entries.

        Intended for remote use from the Statek process.

        Args:
            count: Maximum number of entries to retrieve.
            prefix: Optional job-selection criteria — prefix name (str) or
                    prefix UUID (int).

        Returns:
            List of (job_uuid, message) tuples in FIFO order.
        """
        job_filter = None
        if prefix is not None:
            if isinstance(prefix, str):
                def _filter_by_name(**kwargs):
                    return db0.get_prefix_of(kwargs['job_uuid']).name == prefix
                job_filter = _filter_by_name
            else:
                def _filter_by_uuid(**kwargs):
                    return db0.get_prefix_of(kwargs['job_uuid']).uuid == prefix
                job_filter = _filter_by_uuid
        items = self.__job_console_queue.pop_front(count, filter=job_filter)
        return [(item["job_uuid"], item["message"]) for item in items]
