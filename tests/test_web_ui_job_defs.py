"""Tests for the web_ui job definitions page helper functions."""

from web_ui.pages.job_defs import (
    _paginate, _job_def_has_errors, _job_def_get_errors, _format_traceback
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeError:  # pylint: disable=too-few-public-methods
    def __init__(self, message='something went wrong', traceback=None):
        self.error_message = message
        self.traceback = traceback


class _FakeJobDef:  # pylint: disable=too-few-public-methods
    def __init__(self, errors=None):
        self._errors = errors or []

    def has_errors(self):
        return bool(self._errors)

    def get_errors(self):
        return iter(self._errors)


class _FakeJobDefRaises:  # pylint: disable=too-few-public-methods
    def has_errors(self):
        raise RuntimeError('db error')

    def get_errors(self):
        raise RuntimeError('db error')


# ---------------------------------------------------------------------------
# _paginate
# ---------------------------------------------------------------------------

class TestPaginate:
    def test_first_page_full(self):
        items = list(range(25))
        assert _paginate(items, 1, 10) == list(range(10))

    def test_second_page_full(self):
        items = list(range(25))
        assert _paginate(items, 2, 10) == list(range(10, 20))

    def test_last_partial_page(self):
        items = list(range(25))
        assert _paginate(items, 3, 10) == list(range(20, 25))

    def test_page_beyond_end_returns_empty(self):
        assert not _paginate(list(range(5)), 2, 10)

    def test_empty_list_returns_empty(self):
        assert not _paginate([], 1, 10)


# ---------------------------------------------------------------------------
# _job_def_has_errors
# ---------------------------------------------------------------------------

class TestJobDefHasErrors:
    def test_returns_true_when_errors_present(self):
        jd = _FakeJobDef(errors=[_FakeError()])
        assert _job_def_has_errors(jd) is True

    def test_returns_false_when_no_errors(self):
        jd = _FakeJobDef(errors=[])
        assert _job_def_has_errors(jd) is False

    def test_returns_false_on_exception(self):
        assert _job_def_has_errors(_FakeJobDefRaises()) is False


# ---------------------------------------------------------------------------
# _job_def_get_errors
# ---------------------------------------------------------------------------

class TestJobDefGetErrors:
    def test_returns_error_list(self):
        e = _FakeError('oops')
        jd = _FakeJobDef(errors=[e])
        assert _job_def_get_errors(jd) == [e]

    def test_returns_empty_list_when_no_errors(self):
        jd = _FakeJobDef(errors=[])
        assert not _job_def_get_errors(jd)

    def test_returns_empty_list_on_exception(self):
        assert not _job_def_get_errors(_FakeJobDefRaises())


# ---------------------------------------------------------------------------
# _format_traceback
# ---------------------------------------------------------------------------

class TestFormatTraceback:
    def test_none_returns_empty_string(self):
        assert _format_traceback(None) == ''

    def test_list_of_strings_joined(self):
        tb = ['  File "a.py", line 1\n', '    x = 1\n']
        result = _format_traceback(tb)
        assert 'a.py' in result
        assert 'x = 1' in result

    def test_single_string_returned_as_is(self):
        tb = ['single frame\n']
        assert _format_traceback(tb) == 'single frame\n'

    def test_empty_list_returns_empty_string(self):
        assert _format_traceback([]) == ''
