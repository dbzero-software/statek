# pylint: disable=unused-argument,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Tests for LLM_API process_request available_tools / LLM_TOOLS_SCOPE integration."""

from unittest.mock import patch, MagicMock

import pytest

from statek.llm_api import (
    LLM_Response, LLM_Stats, OpenRouter_API, Claude_API, CallParams, extract_call_params,
    ChatStepData
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

        assert captured["tools"] == [sys_tool]

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

        assert set(captured["tools"]) == {app_tool, sys_tool}

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

    def test_all_tools_includes_list_tools(self, agent):
        """all_tools contains the tools in _tools."""
        tools = agent.all_tools
        assert isinstance(tools, list)
        # Agent fixture has list_of_examples and show_example appended in __post_init__
        tool_names = [t.__name__ for t in tools]
        assert "list_of_examples" in tool_names
        assert "show_example" in tool_names

    def test_all_tools_returns_copy(self, agent):
        """Mutating the returned list does not affect the agent."""
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
# ChatStepData
# ---------------------------------------------------------------------------

class TestChatStepData:
    """Tests for the ChatStepData dataclass."""

    def test_create_with_required_fields(self):
        """ChatStepData can be created with code and console_output."""
        step = ChatStepData(code="x = 1", console_output="> ok")
        assert step.code == "x = 1"
        assert step.console_output == "> ok"

    def test_tool_calls_defaults_to_none(self):
        """tool_calls defaults to None."""
        step = ChatStepData(code="x = 1", console_output="> ok")
        assert step.tool_calls is None

    def test_create_with_tool_calls(self):
        """ChatStepData can be created with tool_calls dict."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={})
        step = ChatStepData(code="x = 1", console_output="> ok", tool_calls={cp: "result"})
        assert step.tool_calls == {cp: "result"}

    def test_empty_code_and_console_output(self):
        """ChatStepData allows empty strings for code and console_output."""
        step = ChatStepData(code="", console_output="")
        assert step.code == ""
        assert step.console_output == ""


# ---------------------------------------------------------------------------
# OpenRouter_API.build_messages with ChatStepData
# ---------------------------------------------------------------------------

class TestOpenRouterBuildMessages:
    """Tests for OpenRouter_API.build_messages accepting ChatStepData objects."""

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
        step = ChatStepData(code="", console_output="hello")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_step_with_code_and_console_output(self, openrouter_api):
        """A step with both fields produces an assistant then a user message."""
        step = ChatStepData(code="x = 1", console_output="> ok")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert len(msgs) == 2
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_multiple_steps_ordered_correctly(self, openrouter_api):
        """Multiple steps produce messages in the correct alternating order."""
        history = [
            ChatStepData(code="", console_output="initial"),
            ChatStepData(code="code1", console_output="console1"),
            ChatStepData(code="code2", console_output="current"),
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
        step = ChatStepData(code="", console_output="hello")
        msgs = openrouter_api.build_messages(system_prompt="sys", chat_history=[step])
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_empty_console_output_skipped(self, openrouter_api):
        """A step with console_output='' does not add a user message."""
        step = ChatStepData(code="x = 1", console_output="")
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert len(msgs) == 1
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}

    def test_tool_calls_ignored(self, openrouter_api):
        """tool_calls field is not yet reflected in messages (plain code blocks only)."""
        cp = CallParams(call_id="c1", name="foo", args=[], kwargs={})
        step = ChatStepData(code="x = 1", console_output="> ok", tool_calls={cp: "result"})
        msgs = openrouter_api.build_messages(chat_history=[step])
        assert len(msgs) == 2  # only assistant + user, no tool messages


# ---------------------------------------------------------------------------
# Claude_API.build_messages with ChatStepData
# ---------------------------------------------------------------------------

class TestClaudeBuildMessages:
    """Tests for Claude_API.build_messages accepting ChatStepData objects."""

    def test_step_produces_assistant_then_user(self, claude_api):
        """A full step produces assistant (code) then user (console_output)."""
        step = ChatStepData(code="x = 1", console_output="> ok")
        msgs = claude_api.build_messages(chat_history=[step])
        assert len(msgs) == 2
        assert msgs[0] == {"role": "assistant", "content": "x = 1"}
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_multiple_steps_correct_order(self, claude_api):
        """Multiple steps expand into the correct alternating sequence."""
        history = [
            ChatStepData(code="", console_output="initial"),
            ChatStepData(code="code1", console_output="current"),
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
            ChatStepData(code="", console_output="initial"),
            ChatStepData(code="code1", console_output="current"),
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
        step = ChatStepData(code="code1", console_output="current")
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
        history = [ChatStepData(code="", console_output="only user")]
        msgs = api.build_messages(chat_history=history)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "only user"}
