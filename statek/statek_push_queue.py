"""StatekPushQueue - singleton wrapper over FiFoQueue(s) for inter-process messaging."""
# pylint: disable=no-member
from typing import Any, List, Optional, Tuple

import dbzero as db0

from statek.agents.agent import SupervisedAgent
from statek.fifo_queue import FiFoQueue
from statek.rpc_integration import rpc as db0_rpc

@db0.memo(singleton=True)
class StatekPushQueue:
    """Singleton wrapper over one or more FiFoQueues for specific use-cases.

    Manages a dedicated queue per channel (currently: job_console) and a
    dedicated event queue per supervised agent.
    Both push and pop are decorated with @db0_rpc.remote so they can be
    invoked across prefix boundaries.
    """

    def __init__(self):
        self.__job_console_queue = FiFoQueue()
        self.__agent_queues = {}

    def is_empty(self) -> bool:
        return self.__job_console_queue.is_empty()

    @db0_rpc.remote
    def push_to_job_console(self, job_uuid: str, message: Any):
        """Append a (job_uuid, message) pair to the job-console queue.

        Args:
            job_uuid: The job object's UUID (may be from a different prefix).
            message: Arbitrary string-convertible object to push.
        """
        self.__job_console_queue.push_back(job_uuid=job_uuid, message=message)

    def push_to_agent_queue(self, agent: SupervisedAgent, event: Any):
        """Append an event to the queue for a specific supervised agent.

        Args:
            agent: Destination agent instance.
            event: Agent/application-specific event object.
        """
        self.__get_agent_queue(agent).push_back(event=event)

    def has_agent_events(
        self,
        agent: SupervisedAgent,
    ) -> bool:
        """Check whether *agent* has a non-empty event queue."""
        queue = self.__get_agent_queue(agent, create=False)
        return queue is not None and not queue.is_empty()

    def has_job(self, job_uuid: str, max_scan: int = 100) -> Optional[bool]:
        """Check whether a job UUID is present in the job-console queue.

        Scanning uses the queue's underlying index iteration order. Returns
        ``None`` when answering would require scanning more than *max_scan*
        queued elements.
        """
        return self.__job_console_queue.has_item(
            filter=lambda **kwargs: kwargs.get("job_uuid") == job_uuid,
            max_scan=max_scan,
        )

    @db0_rpc.remote
    def pop_from_job_console(
        self,
        count: int,
        prefix: Optional[str | int] = None,  # pylint: disable=unused-argument
    ) -> List[Tuple[str, Any]]:
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

    @db0_rpc.remote
    def pop_from_agent_queue(
        self,
        agent: SupervisedAgent,
        count: int,
    ) -> List[Any]:
        """Retrieve and remove up to *count* events queued for *agent*.

        Args:
            agent: Destination agent instance.
            count: Maximum number of event objects to retrieve.

        Returns:
            List of event objects in FIFO order.
        """
        queue = self.__get_agent_queue(agent, create=False)
        if queue is None:
            return []
        items = queue.pop_front(count)
        return [item["event"] for item in items]

    def __get_agent_queue(
        self,
        agent: SupervisedAgent,
        create: bool = True,
    ) -> Optional[FiFoQueue]:
        """Return the dedicated queue for *agent*, creating it when requested."""
        agent_uuid = db0.uuid(agent)
        if agent_uuid not in self.__agent_queues:
            if not create:
                return None
            self.__agent_queues[agent_uuid] = FiFoQueue()

        return self.__agent_queues[agent_uuid]
