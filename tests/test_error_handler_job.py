"""Tests for ErrorHandler and Job.add_error_handler."""

from statek.system import error_handler
from statek.executors.job import ErrorHandler, Job


# Module-level capture list — not managed by db0, so mutations are always visible
_call_log = []


@error_handler
def _capture(context, error=None):  # pylint: disable=unused-argument
    """Append received arguments to the module-level capture list."""
    _call_log.append((context, error))


@error_handler
def _raises(context, error=None):  # pylint: disable=unused-argument
    """Always raises to verify exception suppression."""
    raise RuntimeError("handler exploded")


class TestErrorHandler:
    """Tests for ErrorHandler.__call__ and Job.add_error_handler."""

    def test_call_forwards_error_to_handler(self, db0_fixture):  # pylint: disable=unused-argument
        """__call__ invokes the underlying handler with the supplied error."""
        _call_log.clear()
        exc = ValueError("boom")
        handler = ErrorHandler(error_handler=_capture, context="ctx")

        handler(error=exc)

        assert len(_call_log) == 1
        assert _call_log[0][1] is exc

    def test_call_suppresses_exceptions_from_handler(self, db0_fixture):  # pylint: disable=unused-argument
        """__call__ silently suppresses any exception raised by the handler."""
        handler = ErrorHandler(error_handler=_raises, context=None)

        handler(error=None)  # must not propagate RuntimeError

    def test_add_error_handler_registers_handler_on_job(self, job_factory):
        """add_error_handler appends an ErrorHandler wrapping the given callable."""
        job: Job = job_factory()

        job.add_error_handler(_capture, "ctx")

        assert len(job.error_handlers) == 1
        registered = job.error_handlers[0]
        assert isinstance(registered, ErrorHandler)
        assert registered.error_handler is _capture

    def test_add_error_handler_multiple_handlers(self, job_factory):
        """add_error_handler can be called several times; all handlers are retained."""
        job: Job = job_factory()

        job.add_error_handler(_capture, "ctx1")
        job.add_error_handler(_raises, "ctx2")

        assert len(job.error_handlers) == 2

    def test_notify_handlers_calls_all_and_clears_list(self, job_factory):
        """notify_handlers invokes every registered handler and empties error_handlers."""
        _call_log.clear()
        job: Job = job_factory()
        exc = RuntimeError("fatal")
        job.add_error_handler(_capture, "ctx1")
        job.add_error_handler(_capture, "ctx2")

        job.notify_handlers(error=exc)

        assert len(_call_log) == 2
        assert all(entry[1] is exc for entry in _call_log)
        assert len(job.error_handlers) == 0
