# STATEK

**STATEK** (**Stateful Agent Execution Kit**) gives an LLM agent a durable
Python workspace inside your application.

Instead of treating an agent as something outside the codebase that can only
send JSON-shaped tool calls, STATEK lets the agent work with Python directly.
A job can start with application objects already in scope, run model-written
code against those objects, keep the variables it creates, pause, and resume
later from the same execution state.

That makes it possible to embed agents into a Python system in a very literal
way. The agent can use the same object model as the rest of the application,
while STATEK keeps the job's local state, console output, chat history, tool
calls, status, errors, and continuation state durable and inspectable.

One worker process can host many independent agent jobs, including large fleets
of jobs that wait, resume, depend on one another, or run under bounded
concurrency. In practical deployments this is meant to support anything from a
few long-running workflows to tens of thousands of waiting or active jobs in a
single process.

Read the full documentation at
[docs.dbzero.io/statek](https://docs.dbzero.io/statek).

## Where STATEK Fits

STATEK makes the most sense when an agent needs to work inside real application
state, not just around it.

Use it for systems such as:

- data-intensive agents working in complex domains
- large-scale deployments where efficient orchestration matters
- human-in-the-loop workflows and long-running approval flows
- coordinated agent fleets with parent and child jobs
- workflows that route work across multiple LLM models or providers
- jobs that may wait for minutes, hours, or much longer before continuing

If your agent only needs to answer a single prompt or call a small set of
stateless tools, STATEK may be more machinery than you need. It is most useful
when the agent's work has memory, durable state, dependencies, or meaningful
application-side effects.

## Why Code-First Agents?

A code-first agent can choose what to do at each step:

1. answer or report something to the user
2. call a tool
3. execute a chunk of Python code

STATEK is built around the third option, while still supporting the first two.
The important difference is that the Python execution is stateful and durable.
Variables created in one step can still be used later, and a job can pause and
resume without losing its working context.

That shape has some practical advantages:

- Python is often a better fit than plain JSON for expressing non-trivial work.
- Agents can use handles to real application objects instead of passing around
  large sets of opaque IDs.
- Durable, mutable state can be inspected and audited while the workflow is
  still in progress.
- LLMs are already strong Python generators, so it makes sense to let them use
  that skill directly.
- Agents, services, and application code can share one codebase and object
  model instead of meeting only through narrow integration layers.

See [STATEK Core Concepts](https://docs.dbzero.io/statek/concepts) for the
full mental model.

## How It Works

A STATEK job is a persisted Python session for one unit of agent work.
Application code, a dispatcher, or another agent can create a job with useful
locals already available:

```python
user
message
calendar
today
timestamp
```

Those names are not special framework APIs. They are just Python variables in
the job's workspace. They can point to application objects, dbzero objects,
service adapters, functions, or other controlled values your application
chooses to expose.

The agent can then write ordinary Python against them:

```python
events = calendar.events_for(today)
meeting = calendar.find_meeting("Weekly planning", day=today)
empty_slot = calendar.find_empty_slot(
    after=meeting.ends_at,
    duration=meeting.duration,
)
meeting.move_to(empty_slot)
```

STATEK persists the execution record that matters:

- Python locals
- console output
- chat history
- tool calls and results
- job status and continuation state
- errors and model usage where available
- references to durable application objects

Later, the same job can continue with the variables it created earlier. Other
jobs can run in the same process without their local Python state colliding.

## Quick Start

This is the minimal shape of a STATEK worker: install the package, open dbzero
state, define an agent, start a worker, and submit work. The full walkthrough is
in the [STATEK Quickstart](https://docs.dbzero.io/statek/quickstart).

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

## dbzero and dbzero-pro

STATEK can run ordinary orchestration flows with dialogs and tool calls. The
code-first model becomes much more powerful when application state lives in
[dbzero](https://docs.dbzero.io), because both your application and the agent
can work with the same durable Python object graph.

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

From the agent's point of view, this is still just Python:

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

Restricted mode is a useful defense layer, but it is not a complete production
security model. The host application still owns:

- authorization and tenant boundaries
- secrets and provider credentials
- filesystem, network, CPU, memory, and process limits
- permissions around tools and application objects
- idempotency and policy for external side effects
- deployment monitoring and recovery procedures

Read [STATEK Security & Sandboxing](https://docs.dbzero.io/statek/security)
before exposing workers to real users or production data.

## Documentation

STATEK guides:

- [Overview](https://docs.dbzero.io/statek)
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

dbzero:

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
