"""Tests for the web_ui jobs page helper functions."""

from web_ui.pages.jobs import _paginate, _job_agent_role, _job_status_str, _job_uuid, _job_model


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeAgent:  # pylint: disable=too-few-public-methods
    def __init__(self, role='researcher'):
        self.role = role


class _FakeJobDef:  # pylint: disable=too-few-public-methods
    def __init__(self, agent=None, model='test-model'):
        self.agent = agent
        self.model = model


class _FakeJob:  # pylint: disable=too-few-public-methods
    def __init__(self, agent_role='researcher', status='DONE', model='test-model'):
        self.job_def = _FakeJobDef(agent=_FakeAgent(agent_role), model=model)
        self.status = status


class _FakeJobNoAgent:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.job_def = _FakeJobDef(agent=None)
        self.status = 'READY'


class _FakeJobBadAttr:  # pylint: disable=too-few-public-methods
    @property
    def job_def(self):
        raise AttributeError('no job_def')

    @property
    def status(self):
        raise AttributeError('no status')


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
        items = list(range(5))
        assert not _paginate(items, 2, 10)

    def test_empty_list_returns_empty(self):
        assert not _paginate([], 1, 10)

    def test_single_item_single_page(self):
        assert _paginate(['only'], 1, 10) == ['only']

    def test_exact_page_boundary(self):
        items = list(range(20))
        assert _paginate(items, 2, 10) == list(range(10, 20))
        assert not _paginate(items, 3, 10)


# ---------------------------------------------------------------------------
# _job_agent_role
# ---------------------------------------------------------------------------

class TestJobAgentRole:
    def test_returns_agent_role(self):
        job = _FakeJob(agent_role='coordinator')
        assert _job_agent_role(job) == 'coordinator'

    def test_no_agent_returns_dash(self):
        job = _FakeJobNoAgent()
        assert _job_agent_role(job) == '—'

    def test_no_job_def_returns_dash(self):
        class _J:  # pylint: disable=too-few-public-methods
            job_def = None
        assert _job_agent_role(_J()) == '—'

    def test_exception_returns_dash(self):
        assert _job_agent_role(_FakeJobBadAttr()) == '—'


# ---------------------------------------------------------------------------
# _job_status_str
# ---------------------------------------------------------------------------

class TestJobStatusStr:
    def test_returns_status_string(self):
        job = _FakeJob(status='DONE')
        assert _job_status_str(job) == 'DONE'

    def test_none_status_returns_dash(self):
        class _J:  # pylint: disable=too-few-public-methods
            status = None
        assert _job_status_str(_J()) == '—'

    def test_exception_returns_dash(self):
        assert _job_status_str(_FakeJobBadAttr()) == '—'


# ---------------------------------------------------------------------------
# _job_uuid
# ---------------------------------------------------------------------------

class TestJobUuid:  # pylint: disable=too-few-public-methods
    def test_exception_returns_dash(self):
        # Plain Python objects are not tracked by db0, so uuid() raises.
        assert _job_uuid(object()) == '—'


class TestJobModel:
    def test_returns_job_def_model(self):
        job = _FakeJob(model='deepseek/deepseek-v3.2')
        assert _job_model(job) == 'deepseek/deepseek-v3.2'

    def test_missing_job_def_model_returns_dash(self):
        class _J:  # pylint: disable=too-few-public-methods
            job_def = _FakeJobDef(model=None)
        assert _job_model(_J()) == '—'

    def test_exception_returns_dash(self):
        assert _job_model(_FakeJobBadAttr()) == '—'
