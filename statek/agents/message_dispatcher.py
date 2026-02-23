"""Message Dispatcher agent implementation for analyzing and routing messages."""

from typing import Callable, Dict, Optional
from dataclasses import dataclass
import dbzero as db0
from statek.agents.agent import SupervisedAgent
from statek.system import create_tool, docs


@db0.memo
@dataclass
class MessageDispatcher(SupervisedAgent):
    """
    Message Dispatcher agent for analyzing and routing incoming messages.
    
    The dispatcher's role is to analyze a new incoming user's message, reviewing
    the most recent communication history and classify the message as a continuation
    of a specific past thread (possibly an answer to agent's question) - or as a
    starter of a new thread.
    
    Args:
        chat_history: Callable for retrieving the chat communication history
        start_new_thread: Callable for starting a new chat thread
        dispatch_to: Callable for dispatching the analyzed message to a specific thread
    """

    chat_history: Callable = None
    start_new_thread: Callable = None
    dispatch_to: Callable = None

    def __init__(self, chat_history: Callable, start_new_thread: Callable,
                 dispatch_to: Callable):
        """
        Initialize the MessageDispatcher agent.
        
        Args:
            chat_history: Callable for retrieving the chat communication history
            start_new_thread: Callable for starting a new chat thread
            dispatch_to: Callable for dispatching the analyzed message to a specific thread
        """
        # Store the callables
        self.chat_history = chat_history
        self.start_new_thread = start_new_thread
        self.dispatch_to = dispatch_to

        # Initialize with basic tools (docs)
        basic_tools = [docs, chat_history, start_new_thread, dispatch_to]
        # add insnces to _x_context
        x_context = {
            'chat_history': chat_history,
            'start_new_thread': start_new_thread,
            'dispatch_to': dispatch_to
        }
        # Call parent constructor
        super().__init__(
            role="message_dispatcher",
            _system_prompt=None, # Prompt is readed in StatekSetings
            _tools=basic_tools,
            _X__context=x_context
        )

    @property
    def context(self) -> Optional[Dict]:
        """
        Get dispatcher's private context with dynamically created tools.
        
        When accessed, verifies if the dispatcher-specific tools are available.
        If not, creates them dynamically within the private context.
        
        Returns:
            Dictionary containing private tools and context
        """
        # Initialize context if it doesn't exist
        if self._X__context is None:
            self._X__context = {}

        # Check if tools are already in context
        if 'chat_history' not in self._X__context:
            self._create_chat_history_tool()

        if 'start_new_chat_thread' not in self._X__context:
            self._create_start_new_thread_tool()

        if 'dispatch_to' not in self._X__context:
            self._create_dispatch_to_tool()

        return self._X__context

    def _create_chat_history_tool(self):
        """Create the chat_history tool dynamically."""
        docstring = """Retrieve the chat communication history.
        
        Use this to review recent conversations and understand the context of
        ongoing threads before making a routing decision.
        
        Returns:
            The chat communication history
        """

        def chat_history_impl():
            return self.chat_history()

        create_tool(
            tool_name='chat_history',
            callable=chat_history_impl,
            docstring=docstring,
            context=self._X__context
        )

    def _create_start_new_thread_tool(self):
        """Create the start_new_chat_thread tool dynamically."""
        docstring = """Start a new chat thread for a new conversation.
        
        Use this when the incoming message introduces a new topic, request,
        or question that doesn't relate to any existing thread.
            
        Returns:
            The newly created thread identifier
        """

        def start_new_thread_impl():
            return self.start_new_thread()

        create_tool(
            tool_name='start_new_chat_thread',
            callable=start_new_thread_impl,
            docstring=docstring,
            context=self._X__context
        )

    def _create_dispatch_to_tool(self):
        """Create the dispatch_to tool dynamically."""
        docstring = """Dispatch the analyzed message to a specific thread.
        
        Use this when the incoming message is a continuation of an existing
        conversation thread (e.g., answering a question, providing requested
        information, or following up on a previous topic).
            
        Returns:
            Confirmation of successful dispatch
        """

        def dispatch_to_impl():
            return self.dispatch_to()

        create_tool(
            tool_name='dispatch_to',
            callable=dispatch_to_impl,
            docstring=docstring,
            context=self._X__context
        )
