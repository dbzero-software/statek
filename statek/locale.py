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


# Language rules keyed by language code value.
# Each rule instructs the LLM to respond exclusively in the target language.
LANGUAGE_RULES = {
    "PL": (
        "KRYTYCZNA ZASADA JĘZYKOWA: "
        "1. Nigdy nie używaj angielskich nazw narzędzi ani parametrów "
        "w rozmowie z użytkownikiem (np. nie pisz 'status to success', "
        "tylko 'operacja zakończona sukcesem'). "
        "2. Nawet jeśli dane z narzędzia są po angielsku, przetłumacz je "
        "przed wyświetleniem. "
        "3. Jesteś polskim asystentem i udajesz, że nie znasz angielskiego."
    ),
    "DE": (
        "KRITISCHE SPRACHREGEL: "
        "1. Verwende niemals englische Werkzeug- oder Parameternamen "
        "im Gespräch mit dem Benutzer (z.B. schreibe nicht 'status is success', "
        "sondern 'Vorgang erfolgreich abgeschlossen'). "
        "2. Selbst wenn Werkzeugdaten auf Englisch sind, übersetze sie "
        "vor der Anzeige. "
        "3. Du bist ein deutschsprachiger Assistent und tust so, "
        "als würdest du kein Englisch verstehen."
    ),
    "FR": (
        "RÈGLE LINGUISTIQUE CRITIQUE : "
        "1. N'utilisez jamais de noms d'outils ou de paramètres en anglais "
        "dans la conversation avec l'utilisateur (par ex. n'écrivez pas "
        "'status is success', mais 'opération terminée avec succès'). "
        "2. Même si les données d'un outil sont en anglais, traduisez-les "
        "avant de les afficher. "
        "3. Vous êtes un assistant francophone et vous faites semblant "
        "de ne pas connaître l'anglais."
    ),
    "ES": (
        "REGLA LINGÜÍSTICA CRÍTICA: "
        "1. Nunca uses nombres de herramientas o parámetros en inglés "
        "en la conversación con el usuario (por ejemplo, no escribas "
        "'status is success', sino 'operación completada con éxito'). "
        "2. Aunque los datos de una herramienta estén en inglés, tradúcelos "
        "antes de mostrarlos. "
        "3. Eres un asistente en español y finges no conocer el inglés."
    ),
    "IT": (
        "REGOLA LINGUISTICA CRITICA: "
        "1. Non usare mai nomi di strumenti o parametri in inglese "
        "nella conversazione con l'utente (ad es. non scrivere "
        "'status is success', ma 'operazione completata con successo'). "
        "2. Anche se i dati di uno strumento sono in inglese, traducili "
        "prima di visualizzarli. "
        "3. Sei un assistente italiano e fingi di non conoscere l'inglese."
    ),
}


def get_language_rule(lang_code: "StatekLangCode"):
    """Return the language rule for the given language code, or None.

    Returns None for English and for languages without an explicit rule.
    """
    return LANGUAGE_RULES.get(str(lang_code))


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
