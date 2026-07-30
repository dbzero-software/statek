"""Tests for statek.locale model classes."""

# pylint: disable=no-member,unused-argument

import dbzero as db0

from statek.locale import (
    StatekLangCode, StatekCountryCode, StatekLocale,
    get_language_rule, get_language_hint, resolve_locale,
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

    def test_returns_rule_for_english(self):
        """An explicit English locale must enforce English-only responses."""
        rule = get_language_rule(StatekLangCode.EN)
        assert rule is not None
        assert "exclusively in English" in rule

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


class TestResolveLocale:
    """Tests for the resolve_locale helper."""

    def setup_method(self):
        """Clear the module-level cache between tests."""
        # pylint: disable=import-outside-toplevel
        import statek.locale as _locale_mod
        _locale_mod._LOCALE_CACHE.clear()  # pylint: disable=protected-access

    def test_creates_new_locale_from_string(self, db0_fixture):
        """resolve_locale creates a StatekLocale when none exists yet."""
        locale = resolve_locale("PL-PL")
        assert locale.lang_code == StatekLangCode.PL
        assert locale.country_code == StatekCountryCode.PL

    def test_returns_same_instance_for_repeated_calls(self, db0_fixture):
        """A second call with the same string returns the cached instance."""
        first = resolve_locale("DE-DE")
        second = resolve_locale("DE-DE")
        assert first is second

    def test_different_locale_strings_produce_distinct_instances(self, db0_fixture):
        """Different locale strings must produce distinct StatekLocale objects."""
        pl = resolve_locale("PL-PL")
        en = resolve_locale("EN-GB")
        assert pl is not en

    def test_locale_string_is_case_insensitive(self, db0_fixture):
        """resolve_locale should handle lower-case or mixed-case input."""
        lower = resolve_locale("fr-fr")
        upper = resolve_locale("FR-FR")
        assert lower is upper


class TestGetLanguageHint:
    """Tests for the get_language_hint helper."""

    def test_returns_hint_for_english(self):
        """An explicit English locale must provide an English-only hint."""
        hint = get_language_hint(StatekLangCode.EN)
        assert hint == "REMEMBER: Respond exclusively in English"

    def test_returns_hint_for_polish(self):
        """Polish must have a hardcoded language hint."""
        hint = get_language_hint(StatekLangCode.PL)
        assert hint is not None
        assert len(hint) > 0

    def test_returns_hint_for_popular_languages(self):
        """A few popular non-EN languages should have hints."""
        for code in (StatekLangCode.DE, StatekLangCode.FR,
                     StatekLangCode.ES, StatekLangCode.IT):
            hint = get_language_hint(code)
            assert hint is not None, f"Missing language hint for {code}"
            assert len(hint) > 0

    def test_returns_none_for_unsupported_language(self):
        """Languages without an explicit hint return None."""
        hint = get_language_hint(StatekLangCode.ET)
        assert hint is None
