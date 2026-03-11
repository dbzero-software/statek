"""Specialized agent implementations."""

from .coordinator import Coordinator
from .researcher import Researcher
from .agent import Agent, SupervisedAgent, WarmupDef, update_warmup_defs
from .message_dispatcher import MessageDispatcher
from .list_of_examples import list_of_examples, show_example

__all__ = ["Agent", "Coordinator", "Researcher", "SupervisedAgent", "WarmupDef",
           "MessageDispatcher", "list_of_examples", "show_example", "update_warmup_defs"]
