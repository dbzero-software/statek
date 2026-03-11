"""Tests for MessageDispatcher agent."""

import pytest

from statek.agents.message_dispatcher import MessageDispatcher
from statek.agents.agent import SupervisedAgent


# Mock data
MOCK_HISTORY_DATA = [
    {"thread_id": "1", "message": "Hello"},
    {"thread_id": "1", "message": "How are you?"}
]


# Global mock functions to avoid nested function issues with dbzero @memo decorator

def mock_chat_history():
    """Returns empty chat history by default."""
    return []


def mock_chat_history_with_data():
    """Returns predefined chat history data."""
    return MOCK_HISTORY_DATA


def mock_start_new_thread(message):
    """Generic mock that returns a simple thread identifier."""
    return f"thread_{message}"


def mock_dispatch_to(thread_id, message):  # pylint: disable=unused-argument
    """Generic mock that returns a dispatch confirmation."""
    return "dispatched"


# Fixtures for tracking state across tests

@pytest.fixture
def created_threads():
    """Fixture to track created threads in tests."""
    threads = []
    yield threads
    threads.clear()


@pytest.fixture
def dispatched_messages():
    """Fixture to track dispatched messages in tests."""
    messages = []
    yield messages
    messages.clear()


# Fixture for creating dispatcher instances

@pytest.fixture
def create_dispatcher():
    """Factory fixture for creating MessageDispatcher instances."""
    def _create(chat_history_fn=None, start_new_thread_fn=None, dispatch_to_fn=None):
        return MessageDispatcher(
            chat_history=chat_history_fn or mock_chat_history,
            start_new_thread=start_new_thread_fn or mock_start_new_thread,
            dispatch_to=dispatch_to_fn or mock_dispatch_to
        )
    return _create


class TestMessageDispatcher:
    """Test cases for MessageDispatcher class."""

    def test_message_dispatcher_initialization(self, db0_fixture, create_dispatcher):  # pylint: disable=unused-argument,redefined-outer-name
        """Test MessageDispatcher initialization with required callables."""
        dispatcher = create_dispatcher()

        # Verify instance
        assert dispatcher is not None
        assert isinstance(dispatcher, SupervisedAgent)
        assert dispatcher.role == "message_dispatcher"

    def test_message_dispatcher_context_tools(self, db0_fixture, create_dispatcher):  # pylint: disable=unused-argument,redefined-outer-name
        """Test that MessageDispatcher context contains all required tools."""
        dispatcher = create_dispatcher()
        context = dispatcher.context

        # Verify all tools are present
        assert 'chat_history' in context
        assert 'start_new_chat_thread' in context
        assert 'dispatch_to' in context
        assert 'docs' not in context  # docs is in _tools, not context

    def test_message_dispatcher_chat_history_tool(self, db0_fixture, create_dispatcher):  # pylint: disable=unused-argument,redefined-outer-name
        """Test that chat_history tool works correctly."""
        dispatcher = create_dispatcher(chat_history_fn=mock_chat_history_with_data)
        chat_history_tool = dispatcher.context['chat_history']

        result = chat_history_tool()

        # Verify result
        assert result == MOCK_HISTORY_DATA
        assert len(result) == 2

    def test_message_dispatcher_default_role(self, db0_fixture, create_dispatcher):  # pylint: disable=unused-argument,redefined-outer-name
        """Test MessageDispatcher uses 'message_dispatcher' as default role."""
        dispatcher = create_dispatcher()
        assert dispatcher.role == "message_dispatcher"

    def test_message_dispatcher_custom_role(self, db0_fixture):  # pylint: disable=unused-argument
        """Test MessageDispatcher accepts a custom role name."""
        dispatcher = MessageDispatcher(
            chat_history=mock_chat_history,
            start_new_thread=mock_start_new_thread,
            dispatch_to=mock_dispatch_to,
            role="email_router"
        )
        assert dispatcher.role == "email_router"

    def test_message_dispatcher_custom_role_retains_tools(self, db0_fixture):  # pylint: disable=unused-argument
        """Test MessageDispatcher with custom role retains all context tools."""
        dispatcher = MessageDispatcher(
            chat_history=mock_chat_history,
            start_new_thread=mock_start_new_thread,
            dispatch_to=mock_dispatch_to,
            role="custom_dispatcher"
        )
        assert dispatcher.role == "custom_dispatcher"
        context = dispatcher.context
        assert 'chat_history' in context
        assert 'start_new_chat_thread' in context
        assert 'dispatch_to' in context
