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

"""High-level Statek worker startup helpers."""

import asyncio
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import dbzero as db0

from statek.agents.agent import Agent, SupervisedAgent, update_warmup_defs
from statek.executors.utils import AgentLoopDef, run_agentic_fleet, run_agentic_loop
from statek.prompt_config import update_prompt_config
from statek.settings import StatekSettings, get_statek_settings
from statek.statek_client_api import StatekClientAPI
from statek.statek_push_queue import StatekPushQueue


def _dedupe_agents(agents: Sequence[Agent]) -> list[Agent]:
    """Deduplicate agent objects while preserving caller order."""
    result = []
    seen = set()
    for agent in agents:
        marker = id(agent)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(agent)
    return result


def _active_agents(agents: Optional[Sequence[Agent]]) -> list[Agent]:
    if agents is None:
        return _dedupe_agents(list(db0.find(Agent)))  # pylint: disable=no-member
    return _dedupe_agents(list(agents))


def _queue_prefixes(push_queues: Optional[Sequence[StatekPushQueue]]) -> list[Union[str, int]]:
    if push_queues is None:
        return [db0.get_current_prefix().name]

    prefixes = []
    seen = set()
    for queue in push_queues:
        prefix = db0.get_prefix_of(queue)
        queue_prefix = prefix.name
        if queue_prefix in seen:
            continue
        seen.add(queue_prefix)
        prefixes.append(queue_prefix)
    return prefixes


def _agent_warmup_code(agent: Agent):
    if isinstance(agent, SupervisedAgent) and agent.warmup_def is not None:
        return agent.warmup_def.warmup_code
    return None


def _no_queued_tasks() -> int:
    """Return the default pull queue size for push-only agents."""
    return 0


def _task_queue_size_func(
    agent: Agent,
    task_queue_size_funcs: Optional[Mapping[Agent, Callable[[], int]]],
) -> Callable[[], int]:
    """Return the configured pull callback or the push-only default for an agent."""
    if task_queue_size_funcs is None:
        return _no_queued_tasks
    return task_queue_size_funcs.get(agent, _no_queued_tasks)


def _prepare_statek(agents: list[Agent], settings: StatekSettings) -> None:
    import statek  # pylint: disable=import-outside-toplevel,cyclic-import

    statek.init(settings)
    StatekClientAPI()

    if settings.prompt_defs:
        update_prompt_config(settings.prompt_defs, agents=agents)

    if settings.warmup_defs_dir:
        warmup_path = Path(settings.warmup_defs_dir)
        if warmup_path.is_dir():
            supervised_agents = [
                agent for agent in agents if isinstance(agent, SupervisedAgent)
            ]
            update_warmup_defs(warmup_path, agents=supervised_agents)


async def start_statek_async(
    agents: Optional[Sequence[Agent]] = None,
    push_queues: Optional[Sequence[StatekPushQueue]] = None,
    settings: Optional[StatekSettings] = None,
    max_concurrency: int = 100,
    provider: str = None,
    task_queue_size_funcs: Optional[Mapping[Agent, Callable[[], int]]] = None,
):
    """Prepare Statek agents and run the job processing loop.

    Args:
        agents: Optional sequence of agents to activate.
        push_queues: Optional Statek queues to monitor for events and notifications.
        settings: Optional worker settings.
        max_concurrency: Maximum number of jobs processed concurrently.
        provider: Optional default LLM provider.
        task_queue_size_funcs: Optional per-agent callbacks for legacy pull queues.
            Agents absent from the mapping remain push-only.
    """
    resolved_settings = settings or get_statek_settings()
    active_agents = _active_agents(agents)
    _prepare_statek(active_agents, resolved_settings)
    queue_prefixes = _queue_prefixes(push_queues)

    loop_defs = [
        AgentLoopDef(
            agent=agent,
            warmup_code=_agent_warmup_code(agent),
            task_queue_size_func=_task_queue_size_func(agent, task_queue_size_funcs),
        )
        for agent in active_agents
    ]

    if len(loop_defs) == 1:
        loop_def = loop_defs[0]
        await run_agentic_loop(
            agent=loop_def.agent,
            warmup_code=loop_def.warmup_code,
            task_queue_size_func=loop_def.task_queue_size_func,
            queue_prefixes=queue_prefixes,
            max_concurrency=max_concurrency,
            provider=provider,
        )
        return

    await run_agentic_fleet(
        agent_loop_defs=loop_defs,
        queue_prefixes=queue_prefixes,
        max_concurrency=max_concurrency,
        provider=provider,
    )


def start_statek(
    agents: Optional[Sequence[Agent]] = None,
    push_queues: Optional[Sequence[StatekPushQueue]] = None,
    settings: Optional[StatekSettings] = None,
    max_concurrency: int = 100,
    provider: str = None,
    task_queue_size_funcs: Optional[Mapping[Agent, Callable[[], int]]] = None,
):
    """Blocking wrapper for :func:`start_statek_async`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            start_statek_async(
                agents=agents,
                push_queues=push_queues,
                settings=settings,
                max_concurrency=max_concurrency,
                provider=provider,
                task_queue_size_funcs=task_queue_size_funcs,
            )
        )
    raise RuntimeError(
        "start_statek() cannot run inside an active event loop; "
        "use await start_statek_async(...) instead"
    )
