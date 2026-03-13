# pylint: disable=unused-argument, E1101
"""Tests for statek.utils.format_tool_spec."""

from typing import Iterable, Union, List, Dict, Optional
import dbzero as db0
from statek.utils import format_tool_spec


# ---------------------------------------------------------------------------
# format_tool_spec tests
# ---------------------------------------------------------------------------

def test_format_tool_spec_output_structure():
    """Verify the top-level structure of the tool spec."""
    def send_message(message: str) -> str:
        """Send a message.

        Args:
            message: The text to send.

        Returns:
            str: Confirmation.
        """
        return message

    spec = format_tool_spec(send_message)
    assert spec["type"] == "function"
    assert "function" in spec
    fn = spec["function"]
    assert fn["name"] == "send_message"
    assert "description" in fn
    assert "parameters" in fn
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "properties" in params


def test_format_tool_spec_simple_string_param():
    """Single required string parameter produces correct type and required list."""
    def execute_python(code: str):
        """Execute Python code.

        Args:
            code: The Python code to execute.
        """

    spec = format_tool_spec(execute_python)
    fn = spec["function"]
    assert fn["name"] == "execute_python"
    props = fn["parameters"]["properties"]
    assert "code" in props
    assert props["code"]["type"] == "string"
    assert fn["parameters"]["required"] == ["code"]


def test_format_tool_spec_description_from_docstring():
    """Brief description is extracted from the docstring."""
    def search(query: str) -> str:
        """Search the knowledge base.

        Args:
            query: The search string.

        Returns:
            str: Matching results.
        """
        return query

    spec = format_tool_spec(search)
    assert spec["function"]["description"] == "Search the knowledge base."


def test_format_tool_spec_param_descriptions_from_docstring():
    """Parameter descriptions are extracted from the Args section."""
    def find_user(pattern: str, max_result: int = 10) -> str:
        """Find matching users.

        Args:
            pattern: Glob pattern to match user names.
            max_result: Maximum number of results to return.

        Returns:
            str: Serialized user list.
        """
        return pattern

    spec = format_tool_spec(find_user)
    props = spec["function"]["parameters"]["properties"]
    assert props["pattern"]["description"] == "Glob pattern to match user names."
    assert props["max_result"]["description"] == "Maximum number of results to return."


def test_format_tool_spec_required_vs_optional_params():
    """Only parameters without default values appear in 'required'."""
    def process(name: str, count: int = 1, active: bool = False) -> None:
        """Process something.

        Args:
            name: The item name.
            count: How many times.
            active: Whether active.
        """

    spec = format_tool_spec(process)
    params = spec["function"]["parameters"]
    assert params["required"] == ["name"]
    assert "count" not in params["required"]
    assert "active" not in params["required"]


def test_format_tool_spec_no_required_when_all_have_defaults():
    """When every param has a default, the 'required' key is absent."""
    def greet(name: str = "World") -> str:
        """Greet someone.

        Args:
            name: The name to greet.

        Returns:
            str: Greeting.
        """
        return name

    spec = format_tool_spec(greet)
    assert "required" not in spec["function"]["parameters"]


def test_format_tool_spec_no_params():
    """Function with no parameters yields empty properties and no required key."""
    def ping() -> str:
        """Ping the server."""

    spec = format_tool_spec(ping)
    params = spec["function"]["parameters"]
    assert not params["properties"]
    assert "required" not in params


def test_format_tool_spec_int_param():
    """int type hint maps to JSON schema 'integer'."""
    def set_limit(limit: int) -> None:
        """Set a numeric limit.

        Args:
            limit: The maximum value.
        """

    spec = format_tool_spec(set_limit)
    assert spec["function"]["parameters"]["properties"]["limit"]["type"] == "integer"


def test_format_tool_spec_float_param():
    """float type hint maps to JSON schema 'number'."""
    def set_threshold(threshold: float) -> None:
        """Set a threshold.

        Args:
            threshold: The threshold value.
        """

    spec = format_tool_spec(set_threshold)
    assert spec["function"]["parameters"]["properties"]["threshold"]["type"] == "number"


def test_format_tool_spec_bool_param():
    """bool type hint maps to JSON schema 'boolean'."""
    def toggle(enabled: bool) -> None:
        """Toggle a feature.

        Args:
            enabled: Whether to enable.
        """

    spec = format_tool_spec(toggle)
    assert spec["function"]["parameters"]["properties"]["enabled"]["type"] == "boolean"


def test_format_tool_spec_list_param():
    """List[str] maps to JSON schema 'array'."""
    def process_items(items: List[str]) -> None:
        """Process a list of items.

        Args:
            items: The items to process.
        """

    spec = format_tool_spec(process_items)
    assert spec["function"]["parameters"]["properties"]["items"]["type"] == "array"


def test_format_tool_spec_plain_list_param():
    """Plain list (no type argument) maps to JSON schema 'array'."""
    def process_raw(values: list) -> None:
        """Process raw values.

        Args:
            values: Raw list of values.
        """

    spec = format_tool_spec(process_raw)
    assert spec["function"]["parameters"]["properties"]["values"]["type"] == "array"


def test_format_tool_spec_dict_param():
    """Dict[str, int] maps to JSON schema 'object'."""
    def set_mapping(data: Dict[str, int]) -> None:
        """Set a key-value mapping.

        Args:
            data: The mapping to store.
        """

    spec = format_tool_spec(set_mapping)
    assert spec["function"]["parameters"]["properties"]["data"]["type"] == "object"


def test_format_tool_spec_optional_type_maps_inner():
    """Optional[str] resolves to 'string' (the inner type)."""
    def lookup(key: str, default: Optional[str] = None) -> Optional[str]:
        """Look up a value.

        Args:
            key: The lookup key.
            default: Fallback value.

        Returns:
            str: Found value or default.
        """
        return default

    spec = format_tool_spec(lookup)
    props = spec["function"]["parameters"]["properties"]
    assert props["key"]["type"] == "string"
    assert props["default"]["type"] == "string"


def test_format_tool_spec_union_type_uses_first_non_none():
    """Union[str, int] resolves to the first non-None type."""
    def accept(value: Union[str, int]) -> None:
        """Accept a value.

        Args:
            value: The value to accept.
        """

    spec = format_tool_spec(accept)
    assert spec["function"]["parameters"]["properties"]["value"]["type"] == "string"


def test_format_tool_spec_iterable_param():
    """Iterable[str] maps to JSON schema 'array'."""
    def batch_send(messages: Iterable[str]) -> None:
        """Send messages in batch.

        Args:
            messages: Messages to send.
        """

    spec = format_tool_spec(batch_send)
    assert spec["function"]["parameters"]["properties"]["messages"]["type"] == "array"


def test_format_tool_spec_untyped_param_defaults_to_string():
    """Parameters without a type annotation default to JSON schema 'string'."""
    def raw_call(payload):  # no annotation
        """Make a raw call.

        Args:
            payload: The raw payload.
        """

    spec = format_tool_spec(raw_call)
    assert spec["function"]["parameters"]["properties"]["payload"]["type"] == "string"


def test_format_tool_spec_without_docstring():
    """Function without a docstring produces an empty description."""
    def no_doc(x: int) -> int:
        return x

    spec = format_tool_spec(no_doc)
    assert spec["function"]["description"] == ""
    # param should still be present with correct type
    assert spec["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_format_tool_spec_partial_docstring_no_args_section():
    """Docstring without Args section falls back gracefully; no param descriptions."""
    def partial(x: str) -> str:
        """Just a brief description without an args section."""
        return x

    spec = format_tool_spec(partial)
    assert spec["function"]["description"] == "Just a brief description without an args section."
    prop = spec["function"]["parameters"]["properties"]["x"]
    assert prop["type"] == "string"
    assert "description" not in prop


def test_format_tool_spec_enum_param_shown_as_string(db0_fixture):
    """db0 enum parameter type maps to JSON schema 'string'."""
    Severity = db0.enum("Severity", ["LOW", "MEDIUM", "HIGH"])

    def report(level: Severity) -> None:
        """Report an issue.

        Args:
            level: Severity level of the issue.
        """

    spec = format_tool_spec(report)
    assert spec["function"]["parameters"]["properties"]["level"]["type"] == "string"


def test_format_tool_spec_mixed_types():
    """Multiple parameters of different types are all handled correctly."""
    def create_task(name: str, priority: int, weight: float,
                    active: bool, tags: List[str]) -> None:
        """Create a task.

        Args:
            name: Task name.
            priority: Numeric priority.
            weight: Task weight.
            active: Whether the task is active.
            tags: Labels for the task.
        """

    spec = format_tool_spec(create_task)
    props = spec["function"]["parameters"]["properties"]
    assert props["name"]["type"] == "string"
    assert props["priority"]["type"] == "integer"
    assert props["weight"]["type"] == "number"
    assert props["active"]["type"] == "boolean"
    assert props["tags"]["type"] == "array"
    # all required (no defaults)
    required = spec["function"]["parameters"]["required"]
    assert set(required) == {"name", "priority", "weight", "active", "tags"}


def test_format_tool_spec_function_name():
    """The spec 'name' field matches the actual function name."""
    def my_special_tool(x: str) -> str:
        """A special tool.

        Args:
            x: Input.

        Returns:
            str: Output.
        """
        return x

    spec = format_tool_spec(my_special_tool)
    assert spec["function"]["name"] == "my_special_tool"


def test_format_tool_spec_skips_kwargs():
    """**kwargs is not included in the tool spec properties."""
    def flexible(name: str, **kwargs) -> None:
        """Flexible tool.

        Args:
            name: The name parameter.
        """

    spec = format_tool_spec(flexible)
    props = spec["function"]["parameters"]["properties"]
    assert "kwargs" not in props
    assert "name" in props


def test_format_tool_spec_skips_internal_params():
    """Parameters starting with '_' are excluded from the spec."""
    def internal_func(name: str, _context: dict = None) -> None:
        """Function with internal param.

        Args:
            name: The public name.
        """

    spec = format_tool_spec(internal_func)
    props = spec["function"]["parameters"]["properties"]
    assert "_context" not in props
    assert "name" in props
