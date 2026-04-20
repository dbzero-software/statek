# pylint: disable=unused-argument,redefined-outer-name,protected-access
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-lines,no-member
"""Tests for LLM_API process_request available_tools / LLM_TOOLS_SCOPE integration."""

from unittest.mock import patch, MagicMock

import pytest

from statek.llm_api import (
    LLM_API, LLM_Response, LLM_Stats, OpenRouter_API, OpenAI_API, VertexAI_API,
    ClaudeAI_API, Claude_API, CallParams, extract_call_params,
)
from statek.chat_history import ChatHistoryItem, ChatRole, ContentSource
from statek.exceptions import InvalidFormat
from statek.system import tool
from statek.settings import LLM_API_Settings
from statek.utils import format_tool_spec, CallSpec


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_stats():
    return LLM_Stats(total_bytes_sent=10, total_bytes_received=20, cost=None)


def _make_response(text="ok"):
    return LLM_Response(text=text, stats=_make_stats(), call_requests=None)


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
    )
    return OpenRouter_API(settings=settings)


@pytest.fixture()
def openai_api():
    settings = LLM_API_Settings(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="test-key",
    )
    return OpenAI_API(settings=settings)


@pytest.fixture()
def vertexai_api():
    settings = LLM_API_Settings(
        api_url="https://aiplatform.googleapis.com/v1",
        api_key="test-token",
    )
    return VertexAI_API(settings=settings, project="p1", location="us-central1")


@pytest.fixture()
def claude_api():
    settings = LLM_API_Settings(
        api_url="https://api.anthropic.com/v1/messages",
        api_key="test-key",
    )
    return Claude_API(settings=settings, use_prompt_caching=False)


# ---------------------------------------------------------------------------
# LLM_API.get factory
# ---------------------------------------------------------------------------

class TestLLMAPIGetFactory:
    """Tests for provider factory selection and caching."""

    def setup_method(self):
        LLM_API.get.cache_clear()

    def teardown_method(self):
        LLM_API.get.cache_clear()

    def test_get_openai_provider(self):
        settings = LLM_API_Settings(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key="key",
        )
        with patch("statek.llm_api.get_provider_settings", return_value=settings):
            api = LLM_API.get(provider_name="OPENAI")

        assert isinstance(api, OpenAI_API)

    def test_get_vertexai_provider(self):
        settings = LLM_API_Settings(
            api_url="https://aiplatform.googleapis.com/v1",
            api_key="token",
        )
        with patch("statek.llm_api.get_provider_settings", return_value=settings):
            api = LLM_API.get(
                provider_name="VERTEXAI", project="p1", location="us-central1")

        assert isinstance(api, VertexAI_API)
        assert api.kwargs["project"] == "p1"

    def test_get_claudeai_provider_and_legacy_alias(self):
        settings = LLM_API_Settings(
            api_url="https://api.anthropic.com/v1/messages",
            api_key="key",
        )
        with patch("statek.llm_api.get_provider_settings", return_value=settings):
            api = LLM_API.get(provider_name="CLAUDEAI")

        assert isinstance(api, ClaudeAI_API)
        assert Claude_API is ClaudeAI_API

    def test_get_caches_same_provider_and_kwargs(self):
        settings = LLM_API_Settings(
            api_url="https://api.openai.com/v1/chat/completions",
            api_key="key",
        )
        with patch("statek.llm_api.get_provider_settings", return_value=settings):
            first = LLM_API.get(provider_name="OPENAI", timeout=30)
            second = LLM_API.get(provider_name="OPENAI", timeout=30)

        assert first is second


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

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[app_tool, sys_tool],
                metadata={},
            )

        assert captured["tools"] is None

    @pytest.mark.asyncio
    async def test_system_scope_filters_to_system_tools(
            self, openrouter_api, app_tool, sys_tool):
        """LLM_TOOLS_SCOPE=SYSTEM passes only system tools to _process_request."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
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

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[app_tool, sys_tool],
                metadata={"LLM_TOOLS_SCOPE": "APPLICATION"},
            )

        assert captured["tools"] == [app_tool]

    @pytest.mark.asyncio
    async def test_all_scope_passes_all_tools(
            self, openrouter_api, app_tool, sys_tool):
        """LLM_TOOLS_SCOPE=ALL passes every available tool."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
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

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=None,
                metadata={"LLM_TOOLS_SCOPE": "ALL"},
            )

        assert captured["tools"] is None

    @pytest.mark.asyncio
    async def test_no_model_raises(self, openrouter_api, app_tool):
        """process_request fails fast when model is omitted."""
        with pytest.raises(ValueError, match="model"):
            await openrouter_api.process_request(
                available_tools=[app_tool],
                metadata=None,
            )

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

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[sys_direct_only, sys_console_only, sys_universal],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
                chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
            )

        assert sys_direct_only in captured["tools"]
        assert sys_universal in captured["tools"]
        assert sys_console_only not in captured["tools"]

    @pytest.mark.asyncio
    async def test_python_cli_included_only_for_direct_style(self, openrouter_api):
        """python_cli (target=DIRECT) appears when chat_style=DIRECT, not for other styles."""
        from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

        captured = {}

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        # DIRECT style — python_cli must be present
        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
                chat_style=ChatStyle.DIRECT,
            )
        tool_names = [t.__name__ for t in captured["tools"]]
        assert "python_cli" in tool_names

        # MARKDOWN style — python_cli must NOT be present
        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
                chat_style=ChatStyle.MARKDOWN,
            )
        tool_names = [t.__name__ for t in captured["tools"]]
        assert "python_cli" not in tool_names

    @pytest.mark.asyncio
    async def test_system_scope_deduplicates_tools_by_name(self, openrouter_api, sys_tool):
        """Registry tools with the same name are not duplicated in the merged list."""
        captured = {}

        async def fake_process(self_, *, system_prompt=None, model=None, metadata=None,
                               tools=None, chat_history=None, chat_style=None,
                               temperature=None, enable_reasoning=False):
            captured["tools"] = tools
            return _make_response()

        with patch.object(OpenRouter_API, "_process_request", fake_process):
            await openrouter_api.process_request(
                model="gpt-4o",
                available_tools=[sys_tool],
                metadata={"LLM_TOOLS_SCOPE": "SYSTEM"},
            )

        tool_names = [t.__name__ for t in captured["tools"]]
        duplicates = [n for n in set(tool_names) if tool_names.count(n) > 1]
        assert duplicates == [], f"Duplicate tools in merged list: {duplicates}"


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
            await openrouter_api._process_request(
                model="gpt-4o", tools=[app_tool], metadata={})

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
            await openrouter_api._process_request(
                model="gpt-4o", tools=None, metadata={})

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
            await openrouter_api._process_request(
                model="gpt-4o", tools=[app_tool, sys_tool], metadata={})

        names = {t["function"]["name"] for t in captured_payload["tools"]}
        assert names == {"my_app_tool", "my_sys_tool"}


# ---------------------------------------------------------------------------
# TEMPERATURE metadata handling
# ---------------------------------------------------------------------------

class TestOpenRouterTemperature:
    """Tests that OpenRouter_API passes TEMPERATURE from metadata to payload."""

    @pytest.mark.asyncio
    async def test_temperature_added_to_payload(self, openrouter_api):
        """TEMPERATURE in metadata is forwarded as 'temperature' in the payload."""
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
            await openrouter_api._process_request(
                model="gpt-4o", temperature=0.3, metadata={})

        assert captured_payload["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_no_temperature_when_absent(self, openrouter_api):
        """When TEMPERATURE is not in metadata, 'temperature' is absent from payload."""
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
            await openrouter_api._process_request(model="gpt-4o", metadata={})

        assert "temperature" not in captured_payload

    @pytest.mark.asyncio
    async def test_temperature_out_of_range_raises(self, openrouter_api):
        """TEMPERATURE outside 0.0-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="TEMPERATURE"):
            LLM_API.parse_temperature("1.5")

    @pytest.mark.asyncio
    async def test_model_required_in_metadata(self, openrouter_api):
        """MODEL is mandatory on every request."""
        with pytest.raises(ValueError, match="model"):
            await openrouter_api._process_request(metadata=None)

    @pytest.mark.asyncio
    async def test_reasoning_added_to_payload(self, openrouter_api):
        """REASONING in metadata is forwarded as 'reasoning' in the payload."""
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
            await openrouter_api._process_request(
                model="gpt-4o", enable_reasoning=True, metadata={})

        assert captured_payload["reasoning"] == {}


class TestClaudeTemperature:
    """Tests that Claude_API  passes TEMPERATURE from metadata to payload."""

    @pytest.mark.asyncio
    async def test_temperature_added_to_payload(self, claude_api):
        """TEMPERATURE in metadata is forwarded as 'temperature' in the payload."""
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
            await claude_api._process_request(
                model="claude-3", temperature=0.7, metadata={})

        assert captured_payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_no_temperature_when_absent(self, claude_api):
        """When TEMPERATURE is not in metadata, 'temperature' is absent from payload."""
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
            await claude_api._process_request(model="claude-3", metadata={})

        assert "temperature" not in captured_payload

    @pytest.mark.asyncio
    async def test_temperature_out_of_range_raises(self, claude_api):
        """TEMPERATURE outside 0.0-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="TEMPERATURE"):
            LLM_API.parse_temperature("-0.1")

    @pytest.mark.asyncio
    async def test_model_required_in_metadata(self, claude_api):
        """MODEL is mandatory on every request."""
        with pytest.raises(ValueError, match="model"):
            await claude_api._process_request(metadata=None)

    @pytest.mark.asyncio
    async def test_reasoning_added_to_payload(self, claude_api):
        """REASONING in metadata is forwarded as 'thinking' in the payload."""
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
            await claude_api._process_request(
                model="claude-3", enable_reasoning=True, metadata={})

        assert captured_payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}


# ---------------------------------------------------------------------------
# Claude_API : tools in Anthropic format
# ---------------------------------------------------------------------------

class TestClaudeToolConversion:
    """Tests for Claude_API  Anthropic-format tool conversion."""

    def test_to_anthropic_tool_structure(self, app_tool):
        """_to_anthropic_tool converts OpenAI spec to Anthropic format."""
        openai_spec = format_tool_spec(app_tool)
        anthropic_spec = Claude_API ._to_anthropic_tool(openai_spec)

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
        anthropic_spec = Claude_API ._to_anthropic_tool(openai_spec)

        assert "x" in anthropic_spec["input_schema"]["properties"]
        assert anthropic_spec["input_schema"]["properties"]["x"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_claude_tools_in_anthropic_format(self, claude_api, app_tool):
        """Claude_API  sends tools in Anthropic format (input_schema, not parameters)."""
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
            await claude_api._process_request(
                model="claude-3", tools=[app_tool], metadata={})

        assert "tools" in captured_payload
        tool_entry = captured_payload["tools"][0]
        assert "name" in tool_entry
        assert "description" in tool_entry
        assert "input_schema" in tool_entry
        assert "parameters" not in tool_entry
        assert "type" not in tool_entry  # no "type": "function" wrapper

    @pytest.mark.asyncio
    async def test_claude_no_tools_key_when_none(self, claude_api):
        """Claude_API  omits 'tools' key when tools=None."""
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
            await claude_api._process_request(
                model="claude-3", tools=None, metadata={})

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
            result = await openrouter_api._process_request(
                model="gpt-4o", tools=[app_tool], metadata={})

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
            result = await openrouter_api._process_request(model="gpt-4o", metadata={})

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
            result = await openrouter_api._process_request(
                model="gpt-4o", tools=[app_tool, sys_tool], metadata={})

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
            result = await openrouter_api._process_request(model="gpt-4o", metadata={})

        assert result.text == ""
        assert result.call_requests is not None


# ---------------------------------------------------------------------------
# Claude_API : call_requests parsed from tool_use blocks in response
# ---------------------------------------------------------------------------

class TestClaudeCallRequests:
    """Tests that Claude_API  parses tool_use content blocks into call_requests."""

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
            result = await claude_api._process_request(
                model="claude-3", tools=[app_tool], metadata={})

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
            result = await claude_api._process_request(model="claude-3", metadata={})

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
            result = await claude_api._process_request(
                model="claude-3", tools=[app_tool], metadata={})

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
            result = await claude_api._process_request(
                model="claude-3", tools=[app_tool, sys_tool], metadata={})

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
            result = await claude_api._process_request(
                model="claude-3", tools=[app_tool], metadata={})

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
        sys_t = next(t for t in find_tools("SYSTEM") if t.__name__ == "docstr")
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
# Helpers — build ChatHistoryItem objects in test code
# ---------------------------------------------------------------------------

def _sys(text="sys"):
    return ChatHistoryItem(
        role=ChatRole.SYSTEM, content=text, content_src=ContentSource.SYSTEM)


def _user(text):
    return ChatHistoryItem(
        role=ChatRole.USER, content=text, content_src=ContentSource.USER)


def _console(text):
    return ChatHistoryItem(
        role=ChatRole.USER, content=text, content_src=ContentSource.CONSOLE)


def _asst_code(code, src=ContentSource.ASSISTANT):
    return ChatHistoryItem(
        role=ChatRole.ASSISTANT, content=code, content_src=src)


def _asst_tools(tool_calls, code=None, src=ContentSource.ASSISTANT):
    return ChatHistoryItem(
        role=ChatRole.ASSISTANT,
        content=code,
        content_src=src if code else None,
        tool_calls=tool_calls,
    )


def _tool_result(content, call_spec):
    return ChatHistoryItem(
        role=ChatRole.TOOL,
        content=content,
        content_src=ContentSource.CONSOLE,
        tool_calls=call_spec,
    )


# Use MARKDOWN style throughout these tests so console-source content is
# emitted verbatim (CONSOLE style would add ``> `` line prefixes).
def _MD():  # pylint: disable=invalid-name
    from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel
    return ChatStyle.MARKDOWN  # pylint: disable=no-member


# ---------------------------------------------------------------------------
# OpenRouter_API.build_messages with ChatHistoryItem
# ---------------------------------------------------------------------------

class TestOpenRouterBuildMessages:
    """Tests for OpenRouter_API.build_messages accepting ChatHistoryItem objects."""

    def test_none_chat_history_returns_empty(self, openrouter_api, db0_fixture):
        msgs = openrouter_api.build_messages(chat_history=None, chat_style=_MD())
        assert msgs == []

    def test_system_item_emits_system_message(self, openrouter_api, db0_fixture):
        msgs = openrouter_api.build_messages(chat_history=[_sys("sys")], chat_style=_MD())
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_user_console_only_emits_user_message(self, openrouter_api, db0_fixture):
        msgs = openrouter_api.build_messages(chat_history=[_console("hello")], chat_style=_MD())
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_assistant_code_then_console(self, openrouter_api, db0_fixture):
        history = [_asst_code("x = 1"), _console("> ok")]
        msgs = openrouter_api.build_messages(chat_history=history, chat_style=_MD())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert "x = 1" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_full_alternating_sequence(self, openrouter_api, db0_fixture):
        history = [
            _sys("sys"),
            _user("initial"),
            _asst_code("code1"),
            _console("console1"),
            _asst_code("code2"),
            _console("current"),
        ]
        msgs = openrouter_api.build_messages(chat_history=history, chat_style=_MD())
        assert len(msgs) == 6
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "initial"}
        assert msgs[2]["role"] == "assistant"
        assert msgs[3] == {"role": "user", "content": "console1"}
        assert msgs[4]["role"] == "assistant"
        assert msgs[5] == {"role": "user", "content": "current"}

    def test_assistant_tool_calls_emits_openai_format(self, openrouter_api, db0_fixture):
        cs = CallSpec(id="c1", func_name="foo", args=[], kwargs={"x": 1})
        history = [
            _asst_tools(cs),
            _tool_result("tool_output", cs),
            _console("> ok"),
        ]
        msgs = openrouter_api.build_messages(chat_history=history, chat_style=_MD())
        assert len(msgs) == 3
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"] == [
            {"id": "c1", "type": "function",
             "function": {"name": "foo", "arguments": '{"x": 1}'}}
        ]
        assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "tool_output"}
        assert msgs[2] == {"role": "user", "content": "> ok"}

    def test_multiple_tool_calls_in_one_assistant(self, openrouter_api, db0_fixture):
        cs1 = CallSpec(id="c1", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="c2", func_name="tool_b", args=[], kwargs={"y": 2})
        history = [
            _asst_tools([cs1, cs2]),
            _tool_result("alpha", cs1),
            _tool_result("beta", cs2),
        ]
        msgs = openrouter_api.build_messages(chat_history=history, chat_style=_MD())
        assert msgs[0]["role"] == "assistant"
        ids = [tc["id"] for tc in msgs[0]["tool_calls"]]
        assert ids == ["c1", "c2"]
        assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "alpha"}
        assert msgs[2] == {"role": "tool", "tool_call_id": "c2", "content": "beta"}


# ---------------------------------------------------------------------------
# Claude_API .build_messages with ChatHistoryItem
# ---------------------------------------------------------------------------

class TestClaudeBuildMessages:
    """Tests for Claude_API .build_messages accepting ChatHistoryItem objects."""

    def test_none_chat_history_returns_empty(self, claude_api, db0_fixture):
        assert claude_api.build_messages(chat_history=None, chat_style=_MD()) == []

    def test_system_item_skipped(self, claude_api, db0_fixture):
        """SYSTEM items are not emitted by Claude build_messages — handled at top level."""
        msgs = claude_api.build_messages(chat_history=[_sys("sys")], chat_style=_MD())
        assert msgs == []

    def test_extract_system_prompt_pulls_first_system(self, claude_api, db0_fixture):
        history = [_sys("sysprompt"), _user("hello")]
        sys_text, rest = Claude_API .extract_system_prompt(history)
        assert sys_text == "sysprompt"
        assert len(rest) == 1
        assert rest[0].role == ChatRole.USER

    def test_openrouter_build_messages_includes_dedicated_system_prompt(
        self, openrouter_api, db0_fixture
    ):
        msgs = openrouter_api.build_messages(
            system_prompt="sysprompt",
            chat_history=[_user("hello")],
            chat_style=_MD(),
        )
        assert msgs[0] == {"role": "system", "content": "sysprompt"}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_assistant_then_user_console(self, claude_api, db0_fixture):
        history = [_asst_code("x = 1"), _console("> ok")]
        msgs = claude_api.build_messages(chat_history=history, chat_style=_MD())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1] == {"role": "user", "content": "> ok"}

    def test_prompt_caching_marks_last_assistant(self, db0_fixture):
        settings = LLM_API_Settings(
            api_url="https://api.anthropic.com/v1/messages",
            api_key="test-key",
            use_prompt_caching=True,
        )
        api = Claude_API(settings=settings)
        history = [_user("initial"), _asst_code("code1"), _console("current")]
        msgs = api.build_messages(chat_history=history, chat_style=_MD())
        asst_msg = msgs[1]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_when_disabled(self, claude_api, db0_fixture):
        history = [_asst_code("code1"), _console("ok")]
        msgs = claude_api.build_messages(chat_history=history, chat_style=_MD())
        assert isinstance(msgs[0]["content"], str)

    def test_assistant_tool_calls_emits_anthropic_format(self, claude_api, db0_fixture):
        cs = CallSpec(id="c1", func_name="foo", args=[], kwargs={"x": 1})
        history = [
            _asst_tools(cs),
            _tool_result("tool_output", cs),
            _console("> ok"),
        ]
        msgs = claude_api.build_messages(chat_history=history, chat_style=_MD())
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == [
            {"type": "tool_use", "id": "c1", "name": "foo", "input": {"x": 1}}
        ]
        assert msgs[1]["role"] == "user"
        assert {"type": "tool_result", "tool_use_id": "c1", "content": "tool_output"} \
            in msgs[1]["content"]
        assert {"type": "text", "text": "> ok"} in msgs[1]["content"]

    def test_assistant_tool_calls_with_text_content(self, claude_api, db0_fixture):
        cs = CallSpec(id="c1", func_name="bar", args=[], kwargs={})
        history = [_asst_tools(cs, code="x = 1"), _tool_result("res", cs)]
        msgs = claude_api.build_messages(chat_history=history, chat_style=_MD())
        assert msgs[0]["role"] == "assistant"
        types_in_content = [b["type"] for b in msgs[0]["content"]]
        assert "text" in types_in_content
        assert "tool_use" in types_in_content
        text_block = next(b for b in msgs[0]["content"] if b["type"] == "text")
        assert text_block["text"] == "x = 1"

    def test_multiple_tool_calls_anthropic(self, claude_api, db0_fixture):
        cs1 = CallSpec(id="c1", func_name="tool_a", args=[], kwargs={})
        cs2 = CallSpec(id="c2", func_name="tool_b", args=[], kwargs={"n": 3})
        history = [
            _asst_tools([cs1, cs2]),
            _tool_result("alpha", cs1),
            _tool_result("beta", cs2),
        ]
        msgs = claude_api.build_messages(chat_history=history, chat_style=_MD())
        tool_use_blocks = [b for b in msgs[0]["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 2
        tool_result_blocks = [b for b in msgs[1]["content"] if b["type"] == "tool_result"]
        results_by_id = {b["tool_use_id"]: b["content"] for b in tool_result_blocks}
        assert results_by_id == {"c1": "alpha", "c2": "beta"}


# ---------------------------------------------------------------------------
# VertexAI_API Gemini payload / response handling
# ---------------------------------------------------------------------------

class TestVertexAIBuildContents:
    """Tests for VertexAI_API Gemini content conversion."""

    def test_user_and_assistant_messages_use_gemini_roles(self, vertexai_api, db0_fixture):
        history = [_user("hello"), _asst_code("x = 1")]
        contents = vertexai_api.build_contents(history, chat_style=_MD())

        assert contents[0] == {"role": "user", "parts": [{"text": "hello"}]}
        assert contents[1]["role"] == "model"
        assert "x = 1" in contents[1]["parts"][0]["text"]

    def test_assistant_tool_call_and_result_use_function_parts(self, vertexai_api, db0_fixture):
        cs = CallSpec(id="c1", func_name="foo", args=[], kwargs={"x": 1})
        history = [_asst_tools(cs), _tool_result("done", cs)]
        contents = vertexai_api.build_contents(history, chat_style=_MD())

        assert contents[0] == {
            "role": "model",
            "parts": [{"functionCall": {"name": "foo", "args": {"x": 1}}}],
        }
        assert contents[1] == {
            "role": "function",
            "parts": [{
                "functionResponse": {
                    "name": "foo",
                    "response": {"content": "done"},
                }
            }],
        }


class TestVertexAIRequest:
    """Tests that VertexAI_API sends Gemini GenerateContent payloads."""

    @pytest.mark.asyncio
    async def test_payload_headers_and_url(self, vertexai_api, app_tool, db0_fixture):
        captured = {}

        async def fake_post(self_, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b'{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}'
            mock_resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await vertexai_api._process_request(
                system_prompt="sys",
                model="gemini-2.5-flash",
                tools=[app_tool],
                chat_history=[_user("hello")],
                temperature=0.2,
            )

        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/projects/p1/locations/"
            "us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        )
        assert captured["headers"]["Authorization"] == "Bearer test-token"
        assert captured["json"]["systemInstruction"] == {"parts": [{"text": "sys"}]}
        assert captured["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "hello"}]}
        ]
        assert captured["json"]["generationConfig"]["temperature"] == 0.2
        fn_decl = captured["json"]["tools"][0]["functionDeclarations"][0]
        assert fn_decl["name"] == "my_app_tool"
        assert fn_decl["parameters"]["type"] == "object"
        assert result.text == "ok"

    @pytest.mark.asyncio
    async def test_function_call_response_populates_call_requests(self, vertexai_api):
        async def fake_post(self_, url, json=None, headers=None):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"..."
            mock_resp.json.return_value = {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "name": "my_app_tool",
                                "args": {"x": "hi"},
                            }
                        }]
                    }
                }]
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", fake_post):
            result = await vertexai_api._process_request(model="gemini-2.5-flash")

        assert result.text == ""
        assert len(result.call_requests) == 1
        assert result.call_requests[0].name == "my_app_tool"
        assert result.call_requests[0].kwargs == {"x": "hi"}


# ---------------------------------------------------------------------------
# process_request: DIRECT preserves assistant content
# ---------------------------------------------------------------------------

class TestProcessRequestDirectChatStyle:
    """Tests that process_request preserves assistant content in DIRECT style."""

    @pytest.mark.asyncio
    async def test_direct_style_preserves_assistant_content(self, openrouter_api, db0_fixture):
        from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

        captured = {}

        async def fake_process_request(self, **kwargs):
            captured["chat_history"] = list(kwargs.get("chat_history", []))
            return LLM_Response(
                text="ok", stats=_make_stats(), call_requests=None)

        history = [
            _user("prompt"),
            _asst_code("```python\nwarmup_code()\n```"),
            _console("> result"),
        ]

        with patch.object(OpenRouter_API, "_process_request", fake_process_request):
            await openrouter_api.process_request(
                model="gpt-4o",
                chat_history=iter(history),
                metadata={},
                chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
            )

        items = captured["chat_history"]
        assert len(items) == 3
        assert items[0].role == ChatRole.USER
        assert items[1].role == ChatRole.ASSISTANT
        assert items[1].content == "```python\nwarmup_code()\n```"
        assert items[1].tool_calls is None
        assert items[2].role == ChatRole.USER
        assert items[2].content == "> result"

    @pytest.mark.asyncio
    async def test_non_direct_style_does_not_wrap(self, openrouter_api, db0_fixture):
        from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

        captured = {}

        async def fake_process_request(self, **kwargs):
            captured["chat_history"] = list(kwargs.get("chat_history", []))
            return LLM_Response(
                text="ok", stats=_make_stats(), call_requests=None)

        history = [_asst_code("warmup_code()"), _console("> result")]

        with patch.object(OpenRouter_API, "_process_request", fake_process_request):
            await openrouter_api.process_request(
                model="gpt-4o",
                chat_history=iter(history),
                metadata={},
                chat_style=ChatStyle.MD_DIALOG,  # pylint: disable=no-member
            )

        items = captured["chat_history"]
        assert items[0].role == ChatRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_direct_style_preserves_plain_assistant_text(
        self, openrouter_api, db0_fixture
    ):
        from statek.chat_style import ChatStyle  # pylint: disable=import-outside-toplevel

        captured = {}

        async def fake_process_request(self, **kwargs):
            captured["chat_history"] = list(kwargs.get("chat_history", []))
            return LLM_Response(
                text="ok", stats=_make_stats(), call_requests=None)

        history = [
            _user("prompt"),
            _asst_code("Oto Twoj grafik na kwiecien 2026 roku."),
        ]

        with patch.object(OpenRouter_API, "_process_request", fake_process_request):
            await openrouter_api.process_request(
                model="gpt-4o",
                chat_history=iter(history),
                metadata={},
                chat_style=ChatStyle.DIRECT,  # pylint: disable=no-member
            )

        items = captured["chat_history"]
        assert len(items) == 2
        assert items[1].role == ChatRole.ASSISTANT
        assert items[1].content == "Oto Twoj grafik na kwiecien 2026 roku."
        assert items[1].tool_calls is None
        assert items[0].tool_calls is None
