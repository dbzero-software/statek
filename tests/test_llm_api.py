# pylint: disable=unused-argument,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-lines
"""Tests for LLM_API process_request available_tools / LLM_TOOLS_SCOPE integration."""

from unittest.mock import patch, MagicMock

import pytest

from statek.llm_api import (
    LLM_Response, LLM_Stats, OpenRouter_API, Claude_API, CallParams, extract_call_params,
    ChatStepAssistantData, ChatStepUserData
)
from statek.exceptions import InvalidFormat
from statek.system import tool
from statek.settings import LLM_API_Settings
from statek.utils import format_tool_spec


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_stats():
    return LLM_Stats(total_bytes_sent=10, total_bytes_received=20, cost=None)


def _make_response(text="ok"):
    return LLM_Response(text=text, session_id=None, stats=_make_stats(), call_requests=None)


@pytest.fixture()
def app_tool():
    @tool
    def my_app_tool(x: str, **kwargs):  # pylint: disable=unused-argument
        """An application-level tool.

        Args:
            x: The input string.
        """
        return x
    return my_app_tool


@pytest.fixture()
def sys_tool():
    @tool(system=True)
    def my_sys_tool(n: int, **kwargs):  # pylint: disable=unused-argument
        """A system-level tool.

        Args:
            n: A number.
        """
        return n
    return my_sys_tool


@pytest.fixture()
def openrouter_api():
    settings = LLM_API_Settings(
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key="test-key",
        default_model="gpt-4o",
    )
    return OpenRouter_API(settings=settings, model="gpt-4o")


@pytest.fixture()
def claude_api():
    settings = LLM_API_Settings(
        api_url="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        default_model="claude-3-5-sonnet-20241022",
    )
    return Claude_API(settings=settings, model="claude-3-5-sonnet-20241022",
                      use_prompt_caching=False)


# ---------------------------------------------------------------------------
# process_request: tool-scope selection
# ---------------------------------------------------------------------------

class TestProcessRequestToolScope:
    """Tests for the LLM_TOOLS_SCOPE metadata handling in process_request."""

    @pytest.mark.asyncio
    async def test_no_tools_scope_passes_none_to_process_request(
            self, openrouter_api, app_tool, sys_tool):
        """Without LLM_TOOLS_SCOPE in metadata, tools=None is passed."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[app_tool, sys_tool],
                metadata={"MODEL": "gpt-4o"},
            )

        assert captured["tools"] is None

    @pytest.mark.asyncio
    async def test_system_scope_filters_to_system_tools(
            self, openrouter_api, app_tool, sys_tool):
        """LLM_TOOLS_SCOPE=SYSTEM passes only system tools to _process_request."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[app_tool, sys_tool],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
            )

        tool_names = [t.__name__ for t in captured["tools"]]
        assert sys_tool.__name__ in tool_names
        assert app_tool.__name__ not in tool_names
        # Registry system tools (e.g. python_cli) are merged in
        assert "python_cli" in tool_names

    @pytest.mark.asyncio
    async def test_application_scope_filters_to_app_tools(
            self, openrouter_api, app_tool, sys_tool):
        """LLM_TOOLS_SCOPE=APPLICATION passes only non-system tools."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[app_tool, sys_tool],
                metadata={"LLM_TOOLS_SCOPE": "APPLICATION"},
            )

        assert captured["tools"] == [app_tool]

    @pytest.mark.asyncio
    async def test_all_scope_passes_all_tools(
            self, openrouter_api, app_tool, sys_tool):
        """LLM_TOOLS_SCOPE=ALL passes every available tool."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[app_tool, sys_tool],
                metadata={"LLM_TOOLS_SCOPE": "ALL"},
            )

        tool_names = [t.__name__ for t in captured["tools"]]
        assert app_tool.__name__ in tool_names
        assert sys_tool.__name__ in tool_names
        # Registry system tools are merged in for ALL scope
        assert "python_cli" in tool_names

    @pytest.mark.asyncio
    async def test_tools_scope_without_available_tools_passes_none(
            self, openrouter_api):
        """When LLM_TOOLS_SCOPE is set but available_tools is None, tools=None."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=None,
                metadata={"LLM_TOOLS_SCOPE": "ALL"},
            )

        assert captured["tools"] is None

    @pytest.mark.asyncio
    async def test_no_metadata_passes_tools_none(self, openrouter_api, app_tool):
        """When metadata is None entirely, tools=None is passed."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[app_tool],
                metadata=None,
            )

        assert captured["tools"] is None

    @pytest.mark.asyncio
    async def test_chat_style_parameter_filters_tools_by_target(self, openrouter_api):
        """chat_style parameter filters tools by their tool_target."""
        from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

        @tool(system=True, target=ChatStyle.DIRECT)  # pylint: disable=no-member
        def sys_direct_only(n: int, **kwargs):
            """Direct-only system tool.

            Args:
                n: A number.
            """
            return n

        @tool(system=True, target=ChatStyle.CONSOLE)  # pylint: disable=no-member
        def sys_console_only(n: int, **kwargs):
            """Console-only system tool.

            Args:
                n: A number.
            """
            return n

        @tool(system=True)
        def sys_universal(n: int, **kwargs):
            """Universal system tool.

            Args:
                n: A number.
            """
            return n

        captured = {}

        async def fake_process(self_, *, system_prompt=None, metadata=None,
                               tools=None, chat_history=None, session_id=None):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                available_tools=[sys_direct_only, sys_console_only, sys_universal],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
                chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
            )

        assert sys_direct_only in captured["tools"]
        assert sys_universal in captured["tools"]
        assert sys_console_only not in captured["tools"]


# ---------------------------------------------------------------------------
# OpenRouter_API: tools in payload
# ---------------------------------------------------------------------------

class TestOpenRouterToolsPayload:
    """Tests that OpenRouter_API injects tools into the HTTP payload."""

    @pytest.mark.asyncio
    async def test_tools_added_to_payload(self, openrouter_api, app_tool):
        """When tools are provided, the payload includes 'tools' in OpenAI format."""
        captured_payload = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"choices":[{"message":{"content":"ok"}}]}'
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            await openrouter_api._process_request(tools=[app_tool])

        assert "tools" in captured_payload
        tool_spec = captured_payload["tools"][0]
        assert tool_spec["type"] == "function"
        assert tool_spec["function"]["name"] == "my_app_tool"

    @pytest.mark.asyncio
    async def test_no_tools_key_when_tools_none(self, openrouter_api):
        """When tools=None, the 'tools' key is absent from the payload."""
        captured_payload = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"choices":[{"message":{"content":"ok"}}]}'
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            await openrouter_api._process_request(tools=None)

        assert "tools" not in captured_payload

    @pytest.mark.asyncio
    async def test_multiple_tools_all_in_payload(
            self, openrouter_api, app_tool, sys_tool):
        """All provided tools are included as separate entries."""
        captured_payload = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"choices":[{"message":{"content":"ok"}}]}'
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            await openrouter_api._process_request(tools=[app_tool, sys_tool])

        names = {t["function"]["name"] for t in captured_payload["tools"]}
        assert names == {"my_app_tool", "my_sys_tool"}


# ---------------------------------------------------------------------------
# Claude_API: tools in Anthropic format
# ---------------------------------------------------------------------------

class TestClaudeToolConversion:
    """Tests for Claude_API Anthropic-format tool conversion."""

    def test_to_anthropic_tool_structure(self, app_tool):
        """_to_anthropic_tool converts OpenAI spec to Anthropic format."""
        openai_spec = format_tool_spec(app_tool)
        anthropic_spec = Claude_API._to_anthropic_tool(openai_spec)

        assert "name" in anthropic_spec
        assert "description" in anthropic_spec
        assert "input_schema" in anthropic_spec
        # OpenAI uses "parameters"; Anthropic uses "input_schema"
        assert "parameters" not in anthropic_spec
        assert anthropic_spec["name"] == "my_app_tool"
        assert anthropic_spec["input_schema"]["type"] == "object"

    def test_to_anthropic_tool_preserves_properties(self, app_tool):
        """input_schema contains the same properties as the original parameters."""
        openai_spec = format_tool_spec(app_tool)
        anthropic_spec = Claude_API._to_anthropic_tool(openai_spec)

        assert "x" in anthropic_spec["input_schema"]["properties"]
        assert anthropic_spec["input_schema"]["properties"]["x"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_claude_tools_in_anthropic_format(self, claude_api, app_tool):
        """Claude_API sends tools in Anthropic format (input_schema, not parameters)."""
        captured_payload = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"content":[{"type":"text","text":"ok"}]}'
            mock_resp.json.return_value = {
                "content": [{"type": "text", "text": "ok"}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            await claude_api._process_request(tools=[app_tool])

        assert "tools" in captured_payload
        tool_entry = captured_payload["tools"][0]
        assert "name" in tool_entry
        assert "description" in tool_entry
        assert "input_schema" in tool_entry
        assert "parameters" not in tool_entry
        assert "type" not in tool_entry  # no "type": "function" wrapper

    @pytest.mark.asyncio
    async def test_claude_no_tools_key_when_none(self, claude_api):
        """Claude_API omits 'tools' key when tools=None."""
        captured_payload = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured_payload.update(json)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"content":[{"type":"text","text":"ok"}]}'
            mock_resp.json.return_value = {
                "content": [{"type": "text", "text": "ok"}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            await claude_api._process_request(tools=None)

        assert "tools" not in captured_payload


# ---------------------------------------------------------------------------
# OpenRouter_API: call_requests parsed from tool_calls in response
# ---------------------------------------------------------------------------

class TestOpenRouterCallRequests:
    """Tests that OpenRouter_API parses tool_calls from the HTTP response."""

    @pytest.mark.asyncio
    async def test_tool_call_response_populates_call_requests(
            self, openrouter_api, app_tool):
        """When the response includes tool_calls, call_requests is a list of CallParams."""
        tool_calls_payload = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "my_app_tool", "arguments": '{"x": "hello"}'}
            }
        ]

        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            body = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_payload
                    }
                }]
            }
            mock_resp.content = b"..."
            mock_resp.json.return_value = body
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await openrouter_api._process_request(tools=[app_tool])

        assert result.call_requests is not None
        assert len(result.call_requests) == 1
        cp = result.call_requests[0]
        assert cp.id == "call_abc"
        assert cp.name == "my_app_tool"
        assert cp.args == []
        assert cp.kwargs == {"x": "hello"}

    @pytest.mark.asyncio
    async def test_no_tool_calls_gives_none_call_requests(self, openrouter_api):
        """When the response has no tool_calls, call_requests is None."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "just text"}}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await openrouter_api._process_request()

        assert result.call_requests is None
        assert result.text == "just text"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_all_parsed(
            self, openrouter_api, app_tool, sys_tool):
        """Multiple tool_calls entries are all parsed into call_requests."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "my_app_tool", "arguments": '{"x": "a"}'}},
                            {"id": "c2", "type": "function",
                             "function": {"name": "my_sys_tool", "arguments": '{"n": 7}'}},
                        ]
                    }
                }]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await openrouter_api._process_request(tools=[app_tool, sys_tool])

        assert len(result.call_requests) == 2
        names = {cp.name for cp in result.call_requests}
        assert names == {"my_app_tool", "my_sys_tool"}

    @pytest.mark.asyncio
    async def test_text_empty_when_only_tool_calls(self, openrouter_api):
        """response.text is '' when content is null and tool_calls are present."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "foo", "arguments": "{}"}}
                        ]
                    }
                }]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await openrouter_api._process_request()

        assert result.text == ""
        assert result.call_requests is not None


# ---------------------------------------------------------------------------
# Claude_API: call_requests parsed from tool_use blocks in response
# ---------------------------------------------------------------------------

class TestClaudeCallRequests:
    """Tests that Claude_API parses tool_use content blocks into call_requests."""

    @pytest.mark.asyncio
    async def test_tool_use_block_populates_call_requests(
            self, claude_api, app_tool):
        """A tool_use block in Claude's response is parsed into call_requests."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "my_app_tool",
                        "input": {"x": "world"}
                    }
                ]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await claude_api._process_request(tools=[app_tool])

        assert result.call_requests is not None
        assert len(result.call_requests) == 1
        cp = result.call_requests[0]
        assert cp.id == "toolu_01"
        assert cp.name == "my_app_tool"
        assert cp.args == []
        assert cp.kwargs == {"x": "world"}

    @pytest.mark.asyncio
    async def test_no_tool_use_gives_none_call_requests(self, claude_api):
        """When no tool_use blocks are present, call_requests is None."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "content": [{"type": "text", "text": "hello"}]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await claude_api._process_request()

        assert result.call_requests is None
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_use_blocks(self, claude_api, app_tool):
        """Text and tool_use blocks co-existing: text is concatenated, tool call parsed."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "content": [
                    {"type": "text", "text": "I will call "},
                    {"type": "tool_use", "id": "toolu_02", "name": "my_app_tool",
                     "input": {"x": "hi"}},
                ]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await claude_api._process_request(tools=[app_tool])

        assert result.text == "I will call "
        assert len(result.call_requests) == 1
        assert result.call_requests[0].name == "my_app_tool"

    @pytest.mark.asyncio
    async def test_multiple_tool_use_blocks(self, claude_api, app_tool, sys_tool):
        """Multiple tool_use blocks are all parsed into call_requests."""
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "my_app_tool",
                     "input": {"x": "a"}},
                    {"type": "tool_use", "id": "t2", "name": "my_sys_tool",
                     "input": {"n": 42}},
                ]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await claude_api._process_request(tools=[app_tool, sys_tool])

        assert len(result.call_requests) == 2
        names = {cp.name for cp in result.call_requests}
        assert names == {"my_app_tool", "my_sys_tool"}

    @pytest.mark.asyncio
    async def test_tool_use_input_is_kwargs_not_string(self, claude_api, app_tool):
        """Claude's input dict is passed directly as kwargs (not JSON-decoded)."""
        nested = {"key": [1, 2], "nested": {"a": True}}

        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "my_app_tool",
                     "input": nested}
                ]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await claude_api._process_request(tools=[app_tool])

        assert result.call_requests[0].kwargs == nested


# ---------------------------------------------------------------------------
# Agent.all_tools and Job.get_next_request integration
# ---------------------------------------------------------------------------

class TestAgentAllTools:
    """Tests for Agent.all_tools property."""

    def test_all_tools_excludes_system_registry_tools(self, agent):
        """all_tools does not contain system tools from the registry."""
        tools = agent.all_tools
        assert isinstance(tools, list)
        # System tools are in the registry, not on the agent
        tool_names = [t.__name__ for t in tools]
        assert "list_of_examples" not in tool_names
        assert "show_example" not in tool_names

    def test_all_tools_returns_copy(self, agent):
        """Mutating the returned list does not affect the agent."""
        @tool
        def _copy_test_tool(**kwargs):
            """Dummy tool."""
        agent.append_tool(_copy_test_tool)
        tools1 = agent.all_tools
        tools1.clear()
        tools2 = agent.all_tools
        assert len(tools2) > 0

    def test_all_tools_includes_both_system_and_app(self, agent):
        """all_tools includes both system and application tools added via append_tool."""
        from statek.system import find_tools  # pylint: disable=import-outside-toplevel
        # Use two already-registered tools known to have different system flags
        sys_t = next(t for t in find_tools("SYSTEM") if t.__name__ == "docs")
        app_t = next(t for t in find_tools("APPLICATION") if t.__name__ == "get_any")
        agent.append_tool(sys_t)
        agent.append_tool(app_t)
        all_t = agent.all_tools
        assert sys_t in all_t
        assert app_t in all_t


class TestAppendTool:
    """Tests for Agent.append_tool method."""

    def test_append_callable_with_arguments(self, agent):
        """append_tool accepts a callable tool that has parameters."""
        @tool
        def greet(name: str, greeting: str = "hello", **kwargs):
            """Greet someone.

            Args:
                name: The person's name.
                greeting: The greeting word.
            """
            return f"{greeting} {name}"

        agent.append_tool(greet)
        assert greet in agent.all_tools

    def test_appended_tool_with_args_callable(self, agent):
        """A tool with arguments appended via append_tool remains callable."""
        @tool
        def add(a: int, b: int, **kwargs):
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        agent.append_tool(add)
        assert add(a=2, b=3) == 5

    def test_append_by_name(self, agent):
        """append_tool with a string name adds it to _tools_by_name."""
        agent.append_tool("some_tool")
        assert "some_tool" in agent._tools_by_name

    def test_append_by_name_resolved_via_context(self, agent):
        """A tool added by name is resolved from context in all_tools."""
        @tool
        def ctx_tool(x: int, **kwargs):
            """A context tool.

            Args:
                x: A number.
            """
            return x

        agent.append_tool("ctx_tool")
        agent.context["ctx_tool"] = ctx_tool
        assert ctx_tool in agent.all_tools

    def test_append_internal_callable(self, agent):
        """Callable with '_' prefix name is available in all_tools."""
        @tool
        def _secret(x: str, **kwargs):
            """Internal tool.

            Args:
                x: Input.
            """
            return x

        agent.append_tool(_secret)
        assert _secret in agent.all_tools

    def test_append_internal_by_name(self, agent):
        """String name starting with '_' resolves from context into all_tools."""
        @tool
        def _hidden(**kwargs):
            """Hidden tool."""
            return "hidden"

        agent.context["_hidden"] = _hidden
        agent.append_tool("_hidden")
        assert _hidden in agent.all_tools

    def test_append_system_tool_with_args(self, agent):
        """A system tool with arguments is included in all_tools after append."""
        @tool(system=True)
        def sys_info(detail: str, **kwargs):
            """System info tool.

            Args:
                detail: Detail level.
            """
            return detail

        agent.append_tool(sys_info)
        assert sys_info in agent.all_tools


class TestGetNextRequestAvailableTools:
    """Tests that get_next_request includes available_tools."""

    def test_get_next_request_has_available_tools(self, job_factory):
        """get_next_request dict includes 'available_tools' key."""
        job = job_factory()
        request = job.get_next_request()
        assert "available_tools" in request

    def test_get_next_request_available_tools_is_list(self, job_factory):
        """available_tools value is a list of callables."""
        job = job_factory()
        request = job.get_next_request()
        assert isinstance(request["available_tools"], list)

    def test_get_next_request_available_tools_matches_agent(self, job_factory, agent):
        """available_tools in the request matches the agent's all_tools."""
        job = job_factory()
        request = job.get_next_request()
        assert set(request["available_tools"]) == set(job.job_def.agent.all_tools)


# ---------------------------------------------------------------------------
# extract_call_params
# ---------------------------------------------------------------------------

class TestExtractCallParams:  # pylint: disable=too-many-public-methods
    """Tests for extract_call_params parsing of OpenAI-format tool_call objects."""

    def _valid_call(self, *, call_id="call_abc123", name="my_tool", arguments='{"x": 1}'):
        return {
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}
        }

    def test_returns_call_params_namedtuple(self):
        """Return value is a CallParams instance."""
        result = extract_call_params(self._valid_call())
        assert isinstance(result, CallParams)

    def test_id_extracted(self):
        """The 'id' field is taken from the top-level 'id' key."""
        result = extract_call_params(self._valid_call(call_id="call_xyz"))
        assert result.id == "call_xyz"

    def test_name_extracted(self):
        """The 'name' field matches function.name."""
        result = extract_call_params(self._valid_call(name="do_something"))
        assert result.name == "do_something"

    def test_args_always_empty_list(self):
        """args is always an empty list (OpenAI protocol is kwargs-only)."""
        result = extract_call_params(self._valid_call())
        assert result.args == []

    def test_kwargs_parsed_from_arguments(self):
        """kwargs is the decoded dict from function.arguments JSON string."""
        result = extract_call_params(self._valid_call(arguments='{"x": 1, "y": "hello"}'))
        assert result.kwargs == {"x": 1, "y": "hello"}

    def test_missing_id_defaults_to_empty_string(self):
        """When 'id' is absent, id defaults to ''."""
        call = {"type": "function", "function": {"name": "foo", "arguments": "{}"}}
        result = extract_call_params(call)
        assert result.id == ""

    def test_missing_arguments_defaults_to_empty_kwargs(self):
        """When 'arguments' is absent, kwargs is an empty dict."""
        call = {"id": "call_1", "type": "function", "function": {"name": "foo"}}
        result = extract_call_params(call)
        assert result.kwargs == {}

    def test_type_none_is_accepted(self):
        """When 'type' key is absent entirely, no error is raised."""
        call = {"id": "call_1", "function": {"name": "foo", "arguments": "{}"}}
        result = extract_call_params(call)
        assert result.name == "foo"

    def test_wrong_type_raises_invalid_format(self):
        """A 'type' value other than 'function' raises InvalidFormat."""
        call = {"id": "call_1", "type": "retrieval", "function": {"name": "foo", "arguments": "{}"}}
        with pytest.raises(InvalidFormat):
            extract_call_params(call)

    def test_missing_function_key_raises_invalid_format(self):
        """Missing 'function' key raises InvalidFormat."""
        with pytest.raises(InvalidFormat):
            extract_call_params({"id": "call_1", "type": "function"})

    def test_missing_name_raises_invalid_format(self):
        """Missing 'name' inside 'function' raises InvalidFormat."""
        with pytest.raises(InvalidFormat):
            extract_call_params(
                {"id": "call_1", "type": "function", "function": {"arguments": "{}"}}
            )

    def test_invalid_json_in_arguments_raises_invalid_format(self):
        """Malformed JSON in 'arguments' raises InvalidFormat."""
        call = self._valid_call(arguments="{not valid json}")
        with pytest.raises(InvalidFormat):
            extract_call_params(call)

    def test_arguments_non_dict_raises_invalid_format(self):
        """When 'arguments' decodes to a non-dict (e.g. list), raises InvalidFormat."""
        call = self._valid_call(arguments='[1, 2, 3]')
        with pytest.raises(InvalidFormat):
            extract_call_params(call)

    def test_name_non_string_raises_invalid_format(self):
        """When 'name' is not a string, raises InvalidFormat."""
        call = {"id": "call_1", "type": "function", "function": {"name": 42, "arguments": "{}"}}
        with pytest.raises(InvalidFormat):
            extract_call_params(call)

    def test_complex_nested_kwargs(self):
        """Deeply nested kwargs are parsed correctly."""
        args = '{"config": {"level": 3, "tags": ["a", "b"]}, "enabled": true}'
        result = extract_call_params(self._valid_call(arguments=args))
        assert result.kwargs == {"config": {"level": 3, "tags": ["a", "b"]}, "enabled": True}

    def test_raises_invalid_format_not_generic_exception(self):
        """The raised exception is specifically InvalidFormat, not a generic Exception."""
        call = {
            "id": "call_1", "type": "function",
            "function": {"name": "foo", "arguments": "BROKEN"}
        }
        exc = None
        try:
            extract_call_params(call)
        except Exception as e:  # pylint: disable=broad-except
            exc = e
        assert isinstance(exc, InvalidFormat)

    # -- Anthropic / Claude tool_use format ------------------------------------

    def test_anthropic_format_returns_call_params(self):
        """Anthropic tool_use block is parsed into CallParams."""
        block = {"type": "tool_use", "id": "toolu_01", "name": "get_weather",
                 "input": {"city": "Boston"}}
        result = extract_call_params(block)
        assert isinstance(result, CallParams)
        assert result.id == "toolu_01"
        assert result.name == "get_weather"
        assert result.args == []
        assert result.kwargs == {"city": "Boston"}

    def test_anthropic_format_missing_input_defaults_to_empty(self):
        """Anthropic tool_use block without 'input' gives empty kwargs."""
        block = {"type": "tool_use", "id": "toolu_02", "name": "no_args"}
        result = extract_call_params(block)
        assert result.kwargs == {}

    def test_anthropic_format_missing_name_raises_invalid_format(self):
        """Anthropic tool_use block without 'name' raises InvalidFormat."""
        with pytest.raises(InvalidFormat):
            extract_call_params({"type": "tool_use", "id": "toolu_03", "input": {}})

    def test_anthropic_format_non_dict_input_raises_invalid_format(self):
        """Anthropic tool_use block with a non-dict 'input' raises InvalidFormat."""
        with pytest.raises(InvalidFormat):
            extract_call_params({"type": "tool_use", "id": "t", "name": "foo",
                                 "input": ["not", "a", "dict"]})

    def test_anthropic_format_nested_input(self):
        """Anthropic tool_use input dict is passed through as-is (no JSON decoding)."""
        nested = {"config": {"level": 3, "tags": ["a", "b"]}}
        block = {"type": "tool_use", "id": "t1", "name": "configure", "input": nested}
        result = extract_call_params(block)
        assert result.kwargs == nested

    def test_can_be_used_as_dict_key(self):
        """CallParams instances can be stored and retrieved as dict keys."""
        cp = extract_call_params(self._valid_call(call_id="call_1"))
        d = {cp: "result_1"}
        assert d[cp] == "result_1"

    def test_dict_lookup_by_equal_id(self):
        """Two CallParams with the same id resolve to the same dict entry."""
        cp1 = extract_call_params(self._valid_call(call_id="call_1", name="tool_a"))
        cp2 = extract_call_params(self._valid_call(call_id="call_1", name="tool_b"))
        d = {cp1: "result_1"}
        assert d[cp2] == "result_1"

    def test_dict_keys_distinct_for_different_ids(self):
        """CallParams with different ids are treated as distinct dict keys."""
        cp1 = extract_call_params(self._valid_call(call_id="call_1"))
        cp2 = extract_call_params(self._valid_call(call_id="call_2"))
        d = {cp1: "result_1", cp2: "result_2"}
        assert d[cp1] == "result_1"
        assert d[cp2] == "result_2"


# ---------------------------------------------------------------------------
# ChatStepAssistantData
# ---------------------------------------------------------------------------

class TestChatStepAssistantData:
    """Tests for the ChatStepAssistantData dataclass."""

    def test_create_with_required_fields(self):
        """ChatStepAssistantData can be created with code and console_output."""
        step = ChatStepAssistantData(code="x = 1", console_output="> ok")
        assert step.code == "x = 1"
        assert step.console_output == "> ok"

    def test_tool_calls_defaults_to_none(self):
        """tool_calls defaults to None."""
        step = ChatStepAssistantData(code="x = 1", console_output="> ok")
        assert step.tool_calls is None

    def test_create_with_tool_calls(self):
        """ChatStepAssistantData can be created with tool_calls dict."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={})
        step = ChatStepAssistantData(code="x = 1", console_output="> ok", tool_calls={cp: "result"})
        assert step.tool_calls == {cp: "result"}

    def test_empty_code_and_console_output(self):
        """ChatStepAssistantData allows empty strings for code and console_output."""
        step = ChatStepAssistantData(code="", console_output="")
        assert step.code == ""
        assert step.console_output == ""


# ---------------------------------------------------------------------------
# OpenRouter_API.build_messages with ChatStepAssistantData
# ---------------------------------------------------------------------------

class TestOpenRouterBuildMessages:
    """Tests for OpenRouter_API.build_messages accepting ChatStepAssistantData objects."""

    def test_none_chat_history_returns_empty(self, openrouter_api):
        """With no chat_history, build_messages returns empty list."""
        msgs = openrouter_api.build_messages(chat_history=None)
        assert msgs == []

    def test_system_prompt_only(self, openrouter_api):
        """System prompt alone produces a single system message."""
        msgs = openrouter_api.build_messages(system_prompt="sys", chat_history=None)
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_step_code_empty_emits_only_user_message(self, openrouter_api):
        """A step with code='' only produces a user message (no assistant message)."""
        step = ChatStepAssistantData(code="", console_output="hello")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_step_with_code_and_console_output(self, openrouter_api):
        """A step with both fields produces an assistant then a user message."""
        step = ChatStepAssistantData(code="x = 1", console_output="> ok")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert len(msgs) == 2
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_multiple_steps_ordered_correctly(self, openrouter_api):
        """Multiple steps produce messages in the correct alternating order."""
        history = [
            ChatStepAssistantData(code="", console_output="initial"),
            ChatStepAssistantData(code="code1", console_output="console1"),
            ChatStepAssistantData(code="code2", console_output="current"),
        ]
        msgs = openrouter_api.build_messages(chat_history=history)
        assert len(msgs) == 5
        assert msgs[0] == {"role": "user", "content": "initial"}
        assert msgs[1] == {"role": "assistant", "content": "code1"}
        assert msgs[2] == {"role": "user", "content": "console1"}
        assert msgs[3] == {"role": "assistant", "content": "code2"}
        assert msgs[4] == {"role": "user", "content": "current"}

    def test_system_prompt_appears_before_chat_history(self, openrouter_api):
        """System prompt is prepended before all chat history messages."""
        step = ChatStepAssistantData(code="", console_output="hello")
        msgs = openrouter_api.build_messages(system_prompt="sys", chat_history=[step])
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_empty_console_output_skipped(self, openrouter_api):
        """A step with console_output='' does not add a user message."""
        step = ChatStepAssistantData(code="x = 1", console_output="")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert len(msgs) == 1
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}

    def test_step_with_tool_calls_emits_openai_format(self, openrouter_api):
        """A step with tool_calls emits an assistant message with tool_calls and tool messages."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={"x": 1})
        step = ChatStepAssistantData(code="", console_output="> ok", tool_calls={cp: "tool_output"})
        msgs = openrouter_api.build_messages(chat_history=[step])
        # assistant (with tool_calls), tool result, user (console)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "foo", "arguments": '{"x": 1}'}}
        ]
        assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "tool_output"}
        assert msgs[2] == {"role": "user", "content": "> ok"}

    def test_step_with_tool_calls_and_code(self, openrouter_api):
        """A step with both code and tool_calls includes code as content."""
        cp = CallParams(call_id="c1", name="bar", args=[], kwargs={})
        step = ChatStepAssistantData(code="x = 1", console_output="> ok", tool_calls={cp: "res"})
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "x = 1"
        assert "tool_calls" in msgs[0]

    def test_step_with_multiple_tool_calls(self, openrouter_api):
        """Multiple tool calls emit multiple tool messages in order."""
        cp1 = CallParams(call_id="c1", name="tool_a", args=[], kwargs={})
        cp2 = CallParams(call_id="c2", name="tool_b", args=[], kwargs={"y": 2})
        step = ChatStepAssistantData(code="", console_output="> out",
                            tool_calls={cp1: "alpha", cp2: "beta"})
        msgs = openrouter_api.build_messages(chat_history=[step])
        # assistant + 2 tool messages + user
        assert len(msgs) == 4
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0] == {"role": "tool", "tool_call_id": "c1", "content": "alpha"}
        assert tool_msgs[1] == {"role": "tool", "tool_call_id": "c2", "content": "beta"}

    def test_step_with_tool_calls_no_console_output(self, openrouter_api):
        """A step with tool_calls but no console_output does not emit a trailing user message."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={})
        step = ChatStepAssistantData(code="", console_output="", tool_calls={cp: "res"})
        msgs = openrouter_api.build_messages(chat_history=[step])
        # only assistant + tool message, no user message
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"


# ---------------------------------------------------------------------------
# Claude_API.build_messages with ChatStepAssistantData
# ---------------------------------------------------------------------------

class TestClaudeBuildMessages:
    """Tests for Claude_API.build_messages accepting ChatStepAssistantData objects."""

    def test_step_produces_assistant_then_user(self, claude_api):
        """A full step produces assistant (code) then user (console_output)."""
        step = ChatStepAssistantData(code="x = 1", console_output="> ok")
        msgs = claude_api.build_messages(chat_history=[step])
        assert len(msgs) == 2
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_multiple_steps_correct_order(self, claude_api):
        """Multiple steps expand into the correct alternating sequence."""
        history = [
            ChatStepAssistantData(code="", console_output="initial"),
            ChatStepAssistantData(code="code1", console_output="current"),
        ]
        msgs = claude_api.build_messages(chat_history=history)
        assert len(msgs) == 3
        assert msgs[0] == {"role": "user", "content": "initial"}
        assert msgs[1] == {"role": "assistant", "content": "code1"}
        assert msgs[2] == {"role": "user", "content": "current"}

    def test_prompt_caching_adds_cache_control_to_last_code(self):
        """With prompt caching enabled, cache_control is on the last step's code."""
        settings = LLM_API_Settings(
            api_url="https://api.anthropic.com/v1/messages",
            api_key="test-key",
            default_model="claude-3-5-sonnet-20241022",
            use_prompt_caching=True,
        )
        api = Claude_API(settings=settings, model="claude-3-5-sonnet-20241022")
        history = [
            ChatStepAssistantData(code="", console_output="initial"),
            ChatStepAssistantData(code="code1", console_output="current"),
        ]
        msgs = api.build_messages(chat_history=history)
        # msgs: [user: initial, asst: code1 (with cache), user: current]
        asst_msg = msgs[1]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["type"] == "text"
        assert asst_msg["content"][0]["text"] == "code1"
        assert asst_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_when_prompt_caching_disabled(self, claude_api):
        """Without prompt caching, assistant content is a plain string."""
        step = ChatStepAssistantData(code="code1", console_output="current")
        msgs = claude_api.build_messages(chat_history=[step])
        asst_msg = msgs[0]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], str)

    def test_cache_control_not_added_when_last_step_has_no_code(self):
        """Cache control is not added if the last step has no code."""
        settings = LLM_API_Settings(
            api_url="https://api.anthropic.com/v1/messages",
            api_key="test-key",
            default_model="claude-3-5-sonnet-20241022",
            use_prompt_caching=True,
        )
        api = Claude_API(settings=settings, model="claude-3-5-sonnet-20241022")
        # Last step has no code — no assistant message should be emitted at all
        history = [ChatStepAssistantData(code="", console_output="only user")]
        msgs = api.build_messages(chat_history=history)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "only user"}

    def test_step_with_tool_calls_emits_anthropic_format(self, claude_api):
        """A step with tool_calls emits assistant tool_use block and user tool_result block."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={"x": 1})
        step = ChatStepAssistantData(code="", console_output="> ok", tool_calls={cp: "tool_output"})
        msgs = claude_api.build_messages(chat_history=[step])
        # assistant (tool_use) + user (tool_result + text)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == [
            {"type": "tool_use", "id": "c1", "name": "foo", "input": {"x": 1}}
        ]
        assert msgs[1]["role"] == "user"
        assert {"type": "tool_result", "tool_use_id": "c1", "content": "tool_output"} \
            in msgs[1]["content"]
        assert {"type": "text", "text": "> ok"} in msgs[1]["content"]

    def test_step_with_tool_calls_and_code_claude(self, claude_api):
        """A step with both code and tool_calls includes a text block in assistant content."""
        cp = CallParams(call_id="c1", name="bar", args=[], kwargs={})
        step = ChatStepAssistantData(code="x = 1", console_output="", tool_calls={cp: "res"})
        msgs = claude_api.build_messages(chat_history=[step])
        assert msgs[0]["role"] == "assistant"
        types_in_content = [b["type"] for b in msgs[0]["content"]]
        assert "text" in types_in_content
        assert "tool_use" in types_in_content
        text_block = next(b for b in msgs[0]["content"] if b["type"] == "text")
        assert text_block["text"] == "x = 1"

    def test_step_with_multiple_tool_calls_claude(self, claude_api):
        """Multiple tool calls emit multiple tool_use blocks and tool_result blocks."""
        cp1 = CallParams(call_id="c1", name="tool_a", args=[], kwargs={})
        cp2 = CallParams(call_id="c2", name="tool_b", args=[], kwargs={"n": 3})
        step = ChatStepAssistantData(
            code="", console_output="", tool_calls={cp1: "alpha", cp2: "beta"})
        msgs = claude_api.build_messages(chat_history=[step])
        # assistant content: 2 tool_use blocks
        tool_use_blocks = [b for b in msgs[0]["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 2
        # user content: 2 tool_result blocks
        tool_result_blocks = [b for b in msgs[1]["content"] if b["type"] == "tool_result"]
        assert len(tool_result_blocks) == 2
        results_by_id = {b["tool_use_id"]: b["content"] for b in tool_result_blocks}
        assert results_by_id["c1"] == "alpha"
        assert results_by_id["c2"] == "beta"


# ---------------------------------------------------------------------------
# ChatStepUserData
# ---------------------------------------------------------------------------

class TestChatStepUserData:
    """Tests for the ChatStepUserData dataclass."""

    def test_create_with_message(self):
        """ChatStepUserData can be created with a message."""
        step = ChatStepUserData(message="hello")
        assert step.message == "hello"

    def test_empty_message(self):
        """ChatStepUserData allows empty string for message."""
        step = ChatStepUserData(message="")
        assert step.message == ""


# ---------------------------------------------------------------------------
# OpenRouter_API.build_messages with ChatStepUserData
# ---------------------------------------------------------------------------

class TestOpenRouterBuildMessagesUserData:
    """Tests for OpenRouter_API.build_messages handling ChatStepUserData objects."""

    def test_user_data_emits_user_message(self, openrouter_api):
        """A ChatStepUserData produces a single user message."""
        step = ChatStepUserData(message="user says hi")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert msgs == [{"role": "user", "content": "user says hi"}]

    def test_user_data_empty_message_skipped(self, openrouter_api):
        """A ChatStepUserData with empty message produces no messages."""
        step = ChatStepUserData(message="")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert msgs == []

    def test_mixed_user_and_assistant_data(self, openrouter_api):
        """ChatStepUserData and ChatStepAssistantData can be mixed in chat_history."""
        history = [
            ChatStepUserData(message="initial question"),
            ChatStepAssistantData(code="x = 1", console_output="> ok"),
            ChatStepUserData(message="follow-up"),
        ]
        msgs = openrouter_api.build_messages(chat_history=history)
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "initial question"}
        assert msgs[1] == {"role": "assistant", "content": "x = 1"}
        assert msgs[2] == {"role": "user", "content": "> ok"}
        assert msgs[3] == {"role": "user", "content": "follow-up"}

    def test_user_data_with_system_prompt(self, openrouter_api):
        """System prompt appears before ChatStepUserData messages."""
        step = ChatStepUserData(message="hello")
        msgs = openrouter_api.build_messages(system_prompt="sys", chat_history=[step])
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# Claude_API.build_messages with ChatStepUserData
# ---------------------------------------------------------------------------

class TestClaudeBuildMessagesUserData:
    """Tests for Claude_API.build_messages handling ChatStepUserData objects."""

    def test_user_data_emits_user_message(self, claude_api):
        """A ChatStepUserData produces a single user message."""
        step = ChatStepUserData(message="user says hi")
        msgs = claude_api.build_messages(chat_history=[step])
        assert msgs == [{"role": "user", "content": "user says hi"}]

    def test_user_data_empty_message_skipped(self, claude_api):
        """A ChatStepUserData with empty message produces no messages."""
        step = ChatStepUserData(message="")
        msgs = claude_api.build_messages(chat_history=[step])
        assert msgs == []

    def test_mixed_user_and_assistant_data(self, claude_api):
        """ChatStepUserData and ChatStepAssistantData can be mixed in chat_history."""
        history = [
            ChatStepUserData(message="initial question"),
            ChatStepAssistantData(code="x = 1", console_output="> ok"),
            ChatStepUserData(message="follow-up"),
        ]
        msgs = claude_api.build_messages(chat_history=history)
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "initial question"}
        assert msgs[1] == {"role": "assistant", "content": "x = 1"}
        assert msgs[2] == {"role": "user", "content": "> ok"}
        assert msgs[3] == {"role": "user", "content": "follow-up"}
