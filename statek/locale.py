"""Locale model classes for Statek.

Defines language codes, country codes, and the locale object that
controls language-specific behaviour such as the language rule
appended to system prompts and the language hint appended to user
messages.
"""

# pylint: disable=no-member

from dataclasses import dataclass

import dbzero as db0


@db0.enum(values=[
    "AR",  # Arabic
    "BG",  # Bulgarian
    "CS",  # Czech
    "DA",  # Danish
    "DE",  # German
    "EL",  # Greek
    "EN",  # English
    "ES",  # Spanish
    "ET",  # Estonian
    "FI",  # Finnish
    "FR",  # French
    "HE",  # Hebrew
    "HI",  # Hindi
    "HR",  # Croatian
    "HU",  # Hungarian
    "IT",  # Italian
    "JA",  # Japanese
    "KO",  # Korean
    "LT",  # Lithuanian
    "LV",  # Latvian
    "NL",  # Dutch
    "NO",  # Norwegian
    "PL",  # Polish
    "PT",  # Portuguese
    "RO",  # Romanian
    "RU",  # Russian
    "SK",  # Slovak
    "SL",  # Slovenian
    "SR",  # Serbian
    "SV",  # Swedish
    "TH",  # Thai
    "TR",  # Turkish
    "UK",  # Ukrainian
    "VI",  # Vietnamese
    "ZH",  # Chinese
])
class StatekLangCode:
    """ISO 639-1 language codes supported by Statek."""


@db0.enum(values=[
    "AE",  # United Arab Emirates
    "AR",  # Argentina
    "AT",  # Austria
    "AU",  # Australia
    "BE",  # Belgium
    "BG",  # Bulgaria
    "BR",  # Brazil
    "CA",  # Canada
    "CH",  # Switzerland
    "CL",  # Chile
    "CN",  # China
    "CO",  # Colombia
    "CZ",  # Czech Republic
    "DE",  # Germany
    "DK",  # Denmark
    "EE",  # Estonia
    "EG",  # Egypt
    "ES",  # Spain
    "FI",  # Finland
    "FR",  # France
    "GB",  # United Kingdom (ISO 3166-1)
    "GR",  # Greece
    "HR",  # Croatia
    "HU",  # Hungary
    "ID",  # Indonesia
    "IE",  # Ireland
    "IL",  # Israel
    "IN",  # India
    "IT",  # Italy
    "JP",  # Japan
    "KR",  # South Korea
    "LT",  # Lithuania
    "LV",  # Latvia
    "MX",  # Mexico
    "MY",  # Malaysia
    "NL",  # Netherlands
    "NO",  # Norway
    "NZ",  # New Zealand
    "PH",  # Philippines
    "PL",  # Poland
    "PT",  # Portugal
    "RO",  # Romania
    "RS",  # Serbia
    "RU",  # Russia
    "SA",  # Saudi Arabia
    "SE",  # Sweden
    "SG",  # Singapore
    "SI",  # Slovenia
    "SK",  # Slovakia
    "TH",  # Thailand
    "TR",  # Turkey
    "UA",  # Ukraine
    "US",  # United States
    "VN",  # Vietnam
    "ZA",  # South Africa
])
class StatekCountryCode:
    """ISO 3166-1 alpha-2 country codes supported by Statek."""


@db0.memo(no_default_tags=True)
@dataclass
class StatekLocale:
    """Locale settings that control language-specific LLM behaviour.

    Combines a language code and country code (e.g. ``EN`` + ``US``)
    to determine:
    - the language rule appended to system prompts
    - the language hint appended to user messages
    """
    lang_code: StatekLangCode
    country_code: StatekCountryCode
