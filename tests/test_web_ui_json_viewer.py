"""Tests for the reusable JSON viewer helpers."""

import pytest

from StatekWebUI.components.json_viewer import (
    _coerce_json_value,
    _build_json_editor_props,
)


class TestCoerceJsonValue:
    def test_keeps_python_objects_unchanged(self):
        value = {'a': [1, 2, 3]}
        assert _coerce_json_value(value) == value

    def test_parses_json_string(self):
        assert _coerce_json_value('{"a": 1, "b": true}') == {'a': 1, 'b': True}

    def test_leaves_plain_string_as_is(self):
        assert _coerce_json_value('not json') == 'not json'


class TestBuildJsonEditorProps:
    def test_builds_tree_mode_read_only_props(self):
        props = _build_json_editor_props({'a': 1})

        assert props['content'] == {'json': {'a': 1}}
        assert props['mode'] == 'tree'
        assert props['readOnly'] is True
        assert props['mainMenuBar'] is False
        assert props['navigationBar'] is True
        assert props['statusBar'] is False

    def test_accepts_json_string_input(self):
        props = _build_json_editor_props('{"items": [1, 2]}', mode='view')
        assert props['content'] == {'json': {'items': [1, 2]}}
        assert props['mode'] == 'view'

    def test_rejects_empty_mode(self):
        with pytest.raises(ValueError):
            _build_json_editor_props({'a': 1}, mode='')
