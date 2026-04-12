"""Tests for StatekClientAPI external submission methods."""

# pylint: disable=no-member,unused-argument

from unittest.mock import Mock, patch

from statek.locale import StatekCountryCode, StatekLangCode, StatekLocale
from statek.statek_client_api import StatekClientAPI


class TestStatekClientAPI:
    """Tests for StatekClientAPI."""

    def test_submit_new_job_forwards_locale_object(self, db0_fixture, supervised_agent):
        """submit_new_job forwards an existing StatekLocale object to the task helper."""
        locale = StatekLocale(
            lang_code=StatekLangCode.IT,
            country_code=StatekCountryCode.IT,
        )
        expected_job = Mock()

        with patch(
            "statek.statek_client_api._submit_new_job",
            return_value=expected_job,
        ) as submit_new_job:
            result = StatekClientAPI().submit_new_job(
                supervised_agent,
                shared_vars={"user_id": 1},
                locale=locale,
                kind="invoice",
            )

        assert result is expected_job
        submit_new_job.assert_called_once_with(
            supervised_agent,
            shared_vars={"user_id": 1},
            locale=locale,
            kind="invoice",
        )

    def test_submit_new_job_resolves_string_locale(self, db0_fixture, supervised_agent):
        """submit_new_job resolves a locale string to a StatekLocale before forwarding."""
        expected_job = Mock()
        resolved_locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )

        with patch(
            "statek.statek_client_api._submit_new_job",
            return_value=expected_job,
        ) as submit_new_job, patch(
            "statek.statek_client_api.resolve_locale",
            return_value=resolved_locale,
        ) as mock_resolve:
            result = StatekClientAPI().submit_new_job(
                supervised_agent,
                shared_vars={"user_id": 1},
                locale="PL-PL",
                kind="invoice",
            )

        assert result is expected_job
        mock_resolve.assert_called_once_with("PL-PL")
        submit_new_job.assert_called_once_with(
            supervised_agent,
            shared_vars={"user_id": 1},
            locale=resolved_locale,
            kind="invoice",
        )

    def test_submit_new_jobs_batch_forwards_locale_object(
        self, db0_fixture, supervised_agent
    ):
        """submit_new_jobs_batch forwards an existing StatekLocale object to the task helper."""
        locale = StatekLocale(
            lang_code=StatekLangCode.ES,
            country_code=StatekCountryCode.ES,
        )
        expected_jobs = [Mock(), Mock()]

        with patch(
            "statek.statek_client_api._submit_new_jobs_batch",
            return_value=expected_jobs,
        ) as submit_new_jobs_batch:
            result = StatekClientAPI().submit_new_jobs_batch(
                supervised_agent,
                shared_vars_batch=[{"user_id": 1}, {"user_id": 2}],
                locale=locale,
                kind="invoice",
            )

        assert result is expected_jobs
        submit_new_jobs_batch.assert_called_once_with(
            supervised_agent,
            shared_vars_batch=[{"user_id": 1}, {"user_id": 2}],
            locale=locale,
            kind="invoice",
        )

    def test_submit_new_jobs_batch_resolves_string_locale(
        self, db0_fixture, supervised_agent
    ):
        """submit_new_jobs_batch resolves a locale string before forwarding."""
        expected_jobs = [Mock(), Mock()]
        resolved_locale = StatekLocale(
            lang_code=StatekLangCode.ES,
            country_code=StatekCountryCode.ES,
        )

        with patch(
            "statek.statek_client_api._submit_new_jobs_batch",
            return_value=expected_jobs,
        ) as submit_new_jobs_batch, patch(
            "statek.statek_client_api.resolve_locale",
            return_value=resolved_locale,
        ) as mock_resolve:
            result = StatekClientAPI().submit_new_jobs_batch(
                supervised_agent,
                shared_vars_batch=[{"user_id": 1}, {"user_id": 2}],
                locale="ES-ES",
                kind="invoice",
            )

        assert result is expected_jobs
        mock_resolve.assert_called_once_with("ES-ES")
        submit_new_jobs_batch.assert_called_once_with(
            supervised_agent,
            shared_vars_batch=[{"user_id": 1}, {"user_id": 2}],
            locale=resolved_locale,
            kind="invoice",
        )
