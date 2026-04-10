"""Tests for statek.locale model classes."""

# pylint: disable=no-member,unused-argument

import dbzero as db0

from statek.locale import (
    StatekLangCode, StatekCountryCode, StatekLocale, get_language_rule,
)


class TestStatekLangCode:
    """Tests for the StatekLangCode enum."""

    def test_known_language_codes_exist(self):
        """Core language codes referenced in the design must be present."""
        assert StatekLangCode.EN is not None
        assert StatekLangCode.PL is not None

    def test_additional_language_codes_exist(self):
        """Common ISO 639-1 language codes should be registered."""
        for code in ("DE", "FR", "ES", "IT", "PT", "NL", "RU", "UK", "CS",
                     "JA", "ZH", "KO", "AR", "HI", "TR", "SV", "NO", "DA",
                     "FI", "EL", "HE", "TH", "VI", "RO", "HU", "SK", "BG",
                     "HR", "SR", "SL", "LT", "LV", "ET"):
            assert getattr(StatekLangCode, code) is not None, (
                f"Language code {code} missing from StatekLangCode"
            )


class TestStatekCountryCode:
    """Tests for the StatekCountryCode enum."""

    def test_known_country_codes_exist(self):
        """Country codes explicitly listed in the design must be present."""
        assert StatekCountryCode.US is not None
        assert StatekCountryCode.PL is not None
        assert StatekCountryCode.DE is not None
    def test_additional_country_codes_exist(self):
        """Common ISO 3166-1 alpha-2 country codes should be registered."""
        for code in ("GB", "FR", "ES", "IT", "PT", "NL", "BE", "AT", "CH",
                     "CZ", "SK", "RO", "HU", "BG", "HR", "RS", "SI", "LT",
                     "LV", "EE", "SE", "NO", "DK", "FI", "IE", "RU", "UA",
                     "TR", "GR", "IL", "SA", "AE", "IN", "CN", "JP", "KR",
                     "AU", "NZ", "CA", "MX", "BR", "AR", "CL", "CO",
                     "ZA", "EG", "TH", "VN", "PH", "ID", "MY", "SG"):
            assert getattr(StatekCountryCode, code) is not None, (
                f"Country code {code} missing from StatekCountryCode"
            )


class TestStatekLocale:
    """Tests for the StatekLocale memo class."""

    def test_create_locale(self, db0_fixture):
        """A StatekLocale can be instantiated with lang and country codes."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        assert locale.lang_code == StatekLangCode.EN
        assert locale.country_code == StatekCountryCode.US

    def test_create_polish_locale(self, db0_fixture):
        """Verify the PL locale from the design examples works."""
        locale = StatekLocale(
            lang_code=StatekLangCode.PL,
            country_code=StatekCountryCode.PL,
        )
        assert locale.lang_code == StatekLangCode.PL
        assert locale.country_code == StatekCountryCode.PL

    def test_locale_is_persistent(self, db0_fixture):
        """StatekLocale instances should have a db0 UUID (memo object)."""
        locale = StatekLocale(
            lang_code=StatekLangCode.EN,
            country_code=StatekCountryCode.US,
        )
        assert db0.uuid(locale) is not None


class TestGetLanguageRule:
    """Tests for the get_language_rule helper."""

    def test_returns_none_for_english(self):
        """English should not have a language rule."""
        assert get_language_rule(StatekLangCode.EN) is None

    def test_returns_rule_for_polish(self):
        """Polish must have a hardcoded language rule."""
        rule = get_language_rule(StatekLangCode.PL)
        assert rule is not None
        assert len(rule) > 0

    def test_returns_rule_for_popular_languages(self):
        """A few popular non-EN languages should have rules."""
        for code in (StatekLangCode.DE, StatekLangCode.FR,
                     StatekLangCode.ES, StatekLangCode.IT):
            rule = get_language_rule(code)
            assert rule is not None, f"Missing language rule for {code}"
            assert len(rule) > 0

    def test_returns_none_for_unsupported_language(self):
        """Languages without an explicit rule return None."""
        # Pick a language unlikely to have a hardcoded rule
        rule = get_language_rule(StatekLangCode.ET)
        assert rule is None
