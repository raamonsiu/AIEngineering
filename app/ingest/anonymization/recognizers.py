"""Domain-specific PII recognizers and the Spanish analyzer.

Presidio ships recognizers for the universal categories (email, phone,
IBAN, card numbers, people, organisations). It knows nothing about this
company's identifiers, and two of those matter here:

- ``BUDGET_ID`` (``BUDGET-YYYY-NNNN``) — not personal data in the strict
  sense, but it leaks commercial structure: how many projects closed, in
  what year, under what internal numbering.
- ``CLIENT_CODE`` (``CLI-NNNN``) — maps one-to-one to a real client, so
  it identifies as surely as the name does.

Two operational decisions, both consequences of Presidio being markedly
weaker in Spanish than in English:

**Confidence threshold raised to 0.7.** The Spanish spaCy model tags
common nouns as PERSON with irritating frequency — "Mar", "Sol", "Cruz",
"Alba" are ordinary words and ordinary first names. The default 0.5 keeps
too many of them. Raising it trades some false negatives for far fewer
false positives.

**An explicit blacklist.** Threshold alone cannot fix a word the model is
confidently wrong about. Short, auditable, and preferred over training a
custom NER model, which is not worth it at this corpus size.
"""

from __future__ import annotations

import functools

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

SPANISH_MODEL = "es_core_news_md"
DEFAULT_SCORE_THRESHOLD = 0.7

# Spanish words the NER model tags as PERSON or LOCATION although they are
# ordinary vocabulary in these documents. Compared lowercased.
FALSE_POSITIVE_BLACKLIST: frozenset[str] = frozenset(
    {
        "mar", "sol", "cruz", "alba", "rosa", "olivia", "angel", "ángel",
        "alcance", "entregables", "cronograma", "equipo", "presupuesto",
        "resumen", "proyecto", "cliente", "fase", "fases", "contrato",
        "cierre", "cuenta", "cuentas", "cargo", "cargos", "cuota",
    }
)

BUDGET_ID_RECOGNIZER = PatternRecognizer(
    supported_entity="BUDGET_ID",
    name="budget_id_recognizer",
    patterns=[Pattern(name="budget_id_canonical", regex=r"\bBUDGET-\d{4}-\d{4}\b", score=0.95)],
    supported_language="es",
)

CLIENT_CODE_RECOGNIZER = PatternRecognizer(
    supported_entity="CLIENT_CODE",
    name="client_code_recognizer",
    patterns=[Pattern(name="client_code_internal", regex=r"\b(?:CLI|CLT-INT)-[A-Z0-9]{3,8}\b", score=0.9)],
    supported_language="es",
)

# Presidio's built-in PHONE_NUMBER recognizer leans on `phonenumbers`
# with a region hint it does not get here, and it misses the spacings
# Spanish documents actually use (+34 612 345 678, 91 234 56 78). Phone
# numbers are declared PII for this corpus, so a miss is a leak, not an
# inconvenience — hence an explicit pattern rather than trusting the
# default.
ES_PHONE_RECOGNIZER = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    name="es_phone_recognizer",
    patterns=[
        Pattern(
            name="es_phone_intl",
            regex=r"(?:\+34[\s.-]?)?(?:[6789]\d{2})[\s.-]?\d{2,3}[\s.-]?\d{2,3}[\s.-]?\d{0,2}\b",
            score=0.8,
        ),
    ],
    supported_language="es",
)

# Entity types the pipeline acts on. Listed explicitly rather than taking
# everything Presidio can find: DATE_TIME in particular is detected
# everywhere and pseudonymising dates would destroy the temporal signal
# the estimation system depends on.
TARGET_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "LOCATION",
    "ORGANIZATION",
    "BUDGET_ID",
    "CLIENT_CODE",
)


@functools.lru_cache(maxsize=1)
def build_analyzer(model: str = SPANISH_MODEL) -> AnalyzerEngine:
    """Build the Spanish analyzer once per process.

    Cached because loading a spaCy model costs seconds and the offline
    pipeline calls this per document. Spanish is configured explicitly;
    Presidio's default is English-only, and running an English pipeline
    over Spanish text misses names a human spots instantly.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": model}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["es"])
    analyzer.registry.add_recognizer(BUDGET_ID_RECOGNIZER)
    analyzer.registry.add_recognizer(CLIENT_CODE_RECOGNIZER)
    analyzer.registry.add_recognizer(ES_PHONE_RECOGNIZER)
    return analyzer


def is_blacklisted(text: str) -> bool:
    return text.strip().lower() in FALSE_POSITIVE_BLACKLIST
