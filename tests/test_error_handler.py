"""Tests for error_handler decorator and is_valid_error_handler."""

from statek.system import error_handler, is_valid_error_handler


class TestIsValidErrorHandler:
    """Tests for is_valid_error_handler."""

    def test_returns_true_for_valid_handler(self):
        """A correctly decorated handler with a '_' name and proper signature is valid."""
        @error_handler
        def _terminate(context, error=None):  # pylint: disable=unused-argument
            pass

        assert is_valid_error_handler(_terminate) is True

    def test_returns_false_without_decorator(self):
        """A function missing the @error_handler decorator is not valid."""
        def _terminate(context, error=None):  # pylint: disable=unused-argument
            pass

        assert is_valid_error_handler(_terminate) is False
