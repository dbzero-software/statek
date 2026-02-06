"""Tests for Researcher agent class."""

from statek.agents.researcher import Researcher
from statek.future import FutureResult, temporal
from statek.system import docs


# Mock send_message functions for testing

def sync_send_message(message: str) -> str:
    """Synchronous send_message mock."""
    return f"Response to: {message}"


async def async_send_message(message: str) -> str:
    """Asynchronous send_message mock."""
    return f"Async response to: {message}"


# Mock complement and condition for temporal function
def fetch_message_result(_future: FutureResult) -> str:
    """Fetch result from temporal send_message."""
    return "Temporal response"


def check_message_ready(_future: FutureResult) -> bool:
    """Check if temporal message is ready."""
    return True


@temporal(complement=fetch_message_result, condition=check_message_ready)
def temporal_send_message(_message: str) -> FutureResult:
    """Temporal send_message mock."""
    # In real implementation, this would return a FutureResult
    # For testing, we'll create a mock FutureResult
    return FutureResult(deps=None, state_num=0)


# Mock tools for testing
def mock_tool():
    """Mock tool."""
    return "mock"


def tool1():
    """Tool 1."""
    return "tool1"


def tool2():
    """Tool 2."""
    return "tool2"


def tool3():
    """Tool 3."""
    return "tool3"


class TestResearcher:
    """Test cases for Researcher agent class."""

    def test_researcher_initialization_sync(self, db0_fixture):  # pylint: disable=unused-argument
        """Test Researcher initialization with synchronous send_message."""
        researcher = Researcher(
            send_message=sync_send_message,
            tools=[mock_tool]
        )

        assert researcher.role == "researcher"
        assert researcher.send_message is sync_send_message
        assert mock_tool in researcher._tools  # pylint: disable=protected-access
        assert docs in researcher._tools  # pylint: disable=protected-access
        assert len(researcher._tools) == 2  # pylint: disable=protected-access

    def test_researcher_initialization_async(self, db0_fixture):  # pylint: disable=unused-argument
        """Test Researcher initialization with asynchronous send_message."""
        researcher = Researcher(
            send_message=async_send_message,
            tools=[]
        )

        assert researcher.role == "researcher"
        assert researcher.send_message is async_send_message
        assert len(researcher._tools) == 1  # pylint: disable=protected-access

    def test_researcher_initialization_temporal(self, db0_fixture):  # pylint: disable=unused-argument
        """Test Researcher initialization with temporal send_message."""
        researcher = Researcher(
            send_message=temporal_send_message,
            tools=[]
        )

        assert researcher.role == "researcher"
        assert researcher.send_message is temporal_send_message
        assert getattr(temporal_send_message, '__is_temporal__', False) is True

    def test_researcher_context_creates_ask_tool_sync(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates ask tool for sync send_message."""
        researcher = Researcher(send_message=sync_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'ask' in context
        assert callable(context['ask'])
        assert 'question' in context['ask'].__doc__.lower()

    def test_researcher_context_creates_answer_tool_sync(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates answer tool for sync send_message."""
        researcher = Researcher(send_message=sync_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'answer' in context
        assert callable(context['answer'])
        assert 'content' in context['answer'].__doc__.lower()

    def test_researcher_context_creates_ask_tool_async(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates ask tool for async send_message."""
        researcher = Researcher(send_message=async_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'ask' in context
        assert callable(context['ask'])
        # ask is a normal function that calls send_message
        # Python handles async transparently

    def test_researcher_context_creates_answer_tool_async(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates answer tool for async send_message."""
        researcher = Researcher(send_message=async_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'answer' in context
        assert callable(context['answer'])
        # answer is a normal function that calls send_message
        # Python handles async transparently

    def test_researcher_context_creates_ask_tool_temporal(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates ask tool for temporal send_message."""
        researcher = Researcher(send_message=temporal_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'ask' in context
        assert callable(context['ask'])
        # ask is a normal function that calls send_message
        # Python handles temporal transparently

    def test_researcher_context_creates_answer_tool_temporal(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context property creates answer tool for temporal send_message."""
        researcher = Researcher(send_message=temporal_send_message)

        # Access context to trigger tool creation
        context = researcher.context

        assert 'answer' in context
        assert callable(context['answer'])
        # answer is a normal function that calls send_message
        # Python handles temporal transparently

    def test_researcher_context_persists(self, db0_fixture):  # pylint: disable=unused-argument
        """Test that context is created once and persists."""
        researcher = Researcher(send_message=sync_send_message)

        # Access context twice
        context1 = researcher.context
        context2 = researcher.context

        # Should be the same object
        assert context1 is context2
        assert 'ask' in context1
        assert 'answer' in context1


    def test_researcher_ask_tool_has_required_parameter_in_docstring(
        self, db0_fixture  # pylint: disable=unused-argument
    ):
        """Test that ask tool mentions required parameter in docstring."""
        researcher = Researcher(send_message=sync_send_message)
        context = researcher.context

        ask_docstring = context['ask'].__doc__
        assert 'question' in ask_docstring.lower()
        assert 'required' in ask_docstring.lower()

    def test_researcher_answer_tool_has_required_parameter_in_docstring(
        self, db0_fixture  # pylint: disable=unused-argument
    ):
        """Test that answer tool mentions required parameter in docstring."""
        researcher = Researcher(send_message=sync_send_message)
        context = researcher.context

        answer_docstring = context['answer'].__doc__
        assert 'content' in answer_docstring.lower()
        assert 'required' in answer_docstring.lower()

    def test_researcher_no_tools_initialization(self, db0_fixture):  # pylint: disable=unused-argument
        """Test Researcher initialization with no additional tools."""
        researcher = Researcher(send_message=sync_send_message)

        # Should only have docs tool
        assert len(researcher._tools) == 1  # pylint: disable=protected-access
        assert docs in researcher._tools  # pylint: disable=protected-access

    def test_researcher_multiple_tools_initialization(self, db0_fixture):  # pylint: disable=unused-argument
        """Test Researcher initialization with multiple additional tools."""
        researcher = Researcher(
            send_message=sync_send_message,
            tools=[tool1, tool2, tool3]
        )

        # Should have docs + 3 additional tools
        assert len(researcher._tools) == 4  # pylint: disable=protected-access
        assert docs in researcher._tools  # pylint: disable=protected-access
        assert tool1 in researcher._tools  # pylint: disable=protected-access
        assert tool2 in researcher._tools  # pylint: disable=protected-access
        assert tool3 in researcher._tools  # pylint: disable=protected-access
