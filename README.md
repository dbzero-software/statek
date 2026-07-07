# STATEK

**STATEK** (**Stateful Agent Execution Kit**) is a Python orchestration package
for code-first AI agents with durable workflows.

STATEK lets LLM agents be embedded into a Python codebase and execute Python
inside it. A job can start with application objects already in scope, run
model-written code against those objects, persist its local variables and
history, pause, and resume later from the same execution state.

The framework splits the resources of a single Python process across many
parallel agentic jobs, potentially tens of thousands of waiting or active jobs
per process, with dependencies and continuation state managed as durable
workflow data.

Read the full documentation at
[docs.dbzero.io/statek](https://docs.dbzero.io/statek).

## Where STATEK Fits

STATEK is designed for systems where agents need to operate inside a real
application state model instead of only calling JSON-shaped tools.

It is a strong fit for:

- data-intensive agents working in complex domains
- large-scale deployments where efficient orchestration matters
- complex workflows such as human-in-the-loop systems and agent fleets
- multi-model orchestration across different LLM providers
- durable jobs that may wait for minutes, hours, or much longer before resuming

## Why Code-First Agents?

A code-first agent is an agent that can choose, at each step, to:

1. respond or report to a user
2. call a tool
3. execute a chunk of ad-hoc Python code

STATEK focuses on the third option without removing the first two. The Python
execution is stateful, so variables created in previous steps remain available.
It is also durable, so execution can pause and resume in the future.

Code-first agents are useful because:

- Python is often more expressive than plain JSON request payloads.
- Agents can use handles to real application objects instead of juggling large
  sets of opaque IDs.
- Agents can work against durable, mutable state that remains inspectable and
  auditable.
- LLMs are already strong Python generators, so the orchestration layer can use
  that capability directly.
- Fewer integration layers are needed because agents, services, and application
  code can share the same Python object model.

See [STATEK Core Concepts](https://docs.dbzero.io/statek/concepts) for the
full mental model.

## How It Works

A STATEK job is a durable Python workspace for one unit of agent work.
Application code, a dispatcher, or another agent can create a job with useful
locals already available:

```python
user
message
calendar
today
timestamp
```

The agent can then run Python against those values:

```python
events = calendar.events_for(today)
meeting = calendar.find_meeting("Weekly planning", day=today)
empty_slot = calendar.find_empty_slot(
    after=meeting.ends_at,
    duration=meeting.duration,
)
meeting.move_to(empty_slot)
```

STATEK persists the execution state that matters:

- Python locals
- console output
- chat history
- tool calls and results
- job status and continuation state
- errors and model usage where available
- references to durable application objects

The worker loop can run many jobs independently in one process while preserving
per-job Python locals and persisted history.

## Quick Start

Install STATEK with [dbzero](https://docs.dbzero.io) support:

```bash
pip install "statek[dbzero]"
```

For deployments that need advanced
[dbzero-pro](https://docs.dbzero.io/dbzero-pro) storage and security features,
install:

```bash
pip install "statek[dbzero-pro]"
```

Initialize dbzero and open a prefix before constructing persisted STATEK
objects:

```python
import dbzero as db0
import statek

DBZERO_ROOT = "./statek-data"
PREFIX = "/acme/triage/dev/jobs"


def open_statek_store():
    db0.init(dbzero_root=DBZERO_ROOT, read_write=True)
    statek.init()
    statek.open_prefix(PREFIX, "rw")
```

Define a supervised agent:

```python
from statek.agents.agent import SupervisedAgent
from statek.prompt_config import make_system_prompt


def create_triage_agent():
    agent = SupervisedAgent(
        role="triage_agent",
        _system_prompt=make_system_prompt(
            "You are a concise triage agent. Inspect the event and decide the next action."
        ),
        _tools=[],
        _metadata={"MODEL": "openrouter/openai/gpt-5-mini"},
    )

    agent.update_warmup_def(
        """
print("received event:", event)
"""
    )
    return agent
```

Start a blocking worker:

```python
from statek import start_statek
from statek.statek_push_queue import StatekPushQueue


open_statek_store()

agent = create_triage_agent()
queue = StatekPushQueue(prefix=PREFIX)

start_statek(
    agents=[agent],
    push_queues=[queue],
    max_concurrency=10,
)
```

Submit work from a host service, script, webhook handler, or RPC endpoint:

```python
from statek.statek_client_api import StatekClientAPI


open_statek_store()

agent = create_triage_agent()
incoming_event = {
    "type": "support_ticket",
    "subject": "Invoice total looks wrong",
    "priority": "normal",
}

job = StatekClientAPI().submit_new_job(
    agent=agent,
    shared_vars={"event": incoming_event},
    locale="EN-US",
    source="quickstart",
)

print("submitted job:", job)
```

For the complete walkthrough, see the
[STATEK Quickstart](https://docs.dbzero.io/statek/quickstart).

## dbzero and dbzero-pro

STATEK can be used as a regular orchestration framework with dialogs and tool
calls. To get the full durable, auditable code-first model, application state
should live in [dbzero](https://docs.dbzero.io) objects.

With dbzero, durable application objects are normal Python classes decorated
with `@db0.memo`:

```python
import dbzero as db0


@db0.memo
class Meeting:
    def __init__(self, title, starts_at, ends_at):
        self.title = title
        self.starts_at = starts_at
        self.ends_at = ends_at

    def move_to(self, slot):
        self.starts_at = slot.starts_at
        self.ends_at = slot.ends_at
```

From the agent's point of view, this remains ordinary Python:

```python
meeting.move_to(empty_slot)
```

From the application's point of view, the mutation is durable state. The same
object graph can be inspected by application code, workers, dashboards, and
future agent steps.

Use [dbzero-pro](https://docs.dbzero.io/dbzero-pro) when deployments need
advanced isolation features such as object-level security, protected fields,
field masks, or data filtering policies.

## Security Model

STATEK runs model-written Python in restricted mode by default. Restricted mode
limits imports, builtins, reflection paths, hidden tool calls, source size, and
AST size before code execution.

Keep restricted mode on for normal use:

```python
import statek

statek.init()
```

Disable it only for trusted local development or when another isolation layer is
responsible for the execution boundary:

```python
import statek

statek.init(restricted=False)
```

Restricted mode is a defense layer, not a complete production security model.
The host application still owns:

- authorization and tenant boundaries
- secrets and provider credentials
- filesystem, network, CPU, memory, and process limits
- permissions around tools and application objects
- idempotency and policy for external side effects
- deployment monitoring and recovery procedures

Read [STATEK Security & Sandboxing](https://docs.dbzero.io/statek/security)
before exposing workers to real users or production data.

## Documentation

- [STATEK overview](https://docs.dbzero.io/statek)
- [Quickstart](https://docs.dbzero.io/statek/quickstart)
- [Core concepts](https://docs.dbzero.io/statek/concepts)
- [Agents](https://docs.dbzero.io/statek/agents)
- [Jobs](https://docs.dbzero.io/statek/jobs)
- [Tools](https://docs.dbzero.io/statek/tools)
- [Futures and temporal tools](https://docs.dbzero.io/statek/futures)
- [Model providers](https://docs.dbzero.io/statek/providers)
- [Operations](https://docs.dbzero.io/statek/operations)
- [Security](https://docs.dbzero.io/statek/security)
- [API reference](https://docs.dbzero.io/statek/api-reference)
- [dbzero documentation](https://docs.dbzero.io)
- [dbzero-pro documentation](https://docs.dbzero.io/dbzero-pro)

## Development

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run lint and tests inside the Codex repository environment:

```bash
./scripts/run_lint_codex.sh
./scripts/run_tests_codex.sh
```

Outside Codex, use the standard wrappers:

```bash
./scripts/run_lint.sh
./scripts/run_tests.sh
```

Build the package:

```bash
./scripts/build.sh
```

## License

Statek is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for the full license text and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party dependency
attributions.
