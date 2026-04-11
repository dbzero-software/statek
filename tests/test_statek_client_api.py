"""Tests for StatekClientAPI external submission methods."""

# pylint: disable=no-member,unused-argument

from unittest.mock import Mock, patch

from statek.locale import StatekCountryCode, StatekLangCode, StatekLocale
from statek.statek_client_api import StatekClientAPI


class TestStatekClientAPI:
    """Tests for StatekClientAPI."""

    def test_submit_new_job_forwards_locale(self, db0_fixture, supervised_agent):
        """submit_new_job forwards locale to the task helper."""
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

    def test_submit_new_jobs_batch_forwards_locale(
        self, db0_fixture, supervised_agent
    ):
        """submit_new_jobs_batch forwards locale to the task helper."""
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
