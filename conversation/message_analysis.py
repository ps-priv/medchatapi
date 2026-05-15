"""Analiza wypowiedzi przedstawiciela — wykrywanie sygnałów etycznych i jakościowych."""

import re
from typing import Dict, List, Optional, Set

from .claims import extract_drug_keywords
from .constants import (
    BRIBERY_KEYWORDS,
    CLINICAL_STUDY_PHRASES,
    DISRESPECTFUL_LANGUAGE_KEYWORDS,
    EMPTY_PRAISE_PHRASES,
    ENGLISH_MARKETING_WORDS,
    EVIDENCE_PHRASES,
    FEMALE_ADDRESS_FORMS,
    INAPPROPRIATE_PROPOSAL_KEYWORDS,
    MALE_ADDRESS_FORMS,
    MARKETING_BUZZWORDS,
    OFF_TOPIC_KEYWORDS,
)


def limit_sentences(text: str, max_sentences: int) -> str:
    """Przycina tekst do max_sentences zdań, zachowując pełne pierwsze zdania.

    Używane przez policy_postprocess_message do skracania odpowiedzi lekarza
    zgodnie z dyrektywami stylu (np. max 1–2 zdania przy wysokiej presji czasu).
    """
    cleaned = text.strip()
    if max_sentences <= 0 or not cleaned:
        return cleaned

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(parts) <= max_sentences:
        return cleaned
    return " ".join(parts[:max_sentences])


def normalize_gender(value: str) -> str:
    """Mapuje różne warianty zapisu płci (pl/en) na kanoniczne female/male/unknown."""
    normalized = str(value or "").strip().lower()
    if normalized in {"female", "f", "kobieta", "kobieta_lekarz", "żeńska", "zenska"}:
        return "female"
    if normalized in {"male", "m", "mężczyzna", "mezczyzna", "meska", "męska"}:
        return "male"
    return "unknown"


def expected_address_form(gender: str) -> str:
    """Zwraca poprawną formę grzecznościową (pani/pan doktor) zależną od płci lekarza."""
    if gender == "female":
        return "pani doktor"
    if gender == "male":
        return "pan doktor"
    return "pani doktor / pan doktor"


def detect_gender_addressing(message: str, doctor_gender: str) -> Dict:
    """Wykrywa użycie formy grzecznościowej niezgodnej z płcią lekarza; zwraca listę trafionych form i oczekiwany wzorzec."""
    text = message.lower()
    male_hits = [form for form in MALE_ADDRESS_FORMS if form in text]
    female_hits = [form for form in FEMALE_ADDRESS_FORMS if form in text]
    generic_pan = bool(re.search(r"\bpan\b|\bpanie\b", text))
    generic_pani = bool(re.search(r"\bpani\b", text))

    mismatch_hits: List[str] = []
    if doctor_gender == "female":
        if male_hits:
            mismatch_hits.extend(male_hits)
        if generic_pan and "pani" not in text:
            mismatch_hits.append("pan")
    elif doctor_gender == "male":
        if female_hits:
            mismatch_hits.extend(female_hits)
        if generic_pani and "pan" not in text:
            mismatch_hits.append("pani")

    return {
        "gender_mismatch_hits": list(dict.fromkeys(mismatch_hits)),
        "expected_address": expected_address_form(doctor_gender),
    }


def analyze_message(
    message: str,
    drug_info: Dict,
    doctor_profile: Dict,
    focus_keywords: Optional[Set[str]] = None,
) -> Dict:
    """Skanuje wiadomość pod kątem korupcji, off-topic, buzzwordów, anglicyzmów, niestosowności, braku szacunku, formy grzecznościowej i wzmianek o leku."""
    text = message.lower()
    drug_keywords = focus_keywords if focus_keywords is not None else extract_drug_keywords(drug_info)
    doctor_gender = normalize_gender(doctor_profile.get("gender", "unknown"))
    gender_info = detect_gender_addressing(message, doctor_gender)

    bribery_hits = [keyword for keyword in BRIBERY_KEYWORDS if keyword in text]
    off_topic_hits = [keyword for keyword in OFF_TOPIC_KEYWORDS if keyword in text]
    marketing_hits = [keyword for keyword in MARKETING_BUZZWORDS if keyword in text]
    english_hits = [keyword for keyword in ENGLISH_MARKETING_WORDS if keyword in text]
    inappropriate_hits = [keyword for keyword in INAPPROPRIATE_PROPOSAL_KEYWORDS if keyword in text]
    disrespect_hits = [keyword for keyword in DISRESPECTFUL_LANGUAGE_KEYWORDS if keyword in text]
    empty_praise_hits = [phrase for phrase in EMPTY_PRAISE_PHRASES if phrase in text]
    evidence_hits = [phrase for phrase in EVIDENCE_PHRASES if phrase in text]
    clinical_study_hits = [phrase for phrase in CLINICAL_STUDY_PHRASES if phrase in text]
    drug_mentions = [keyword for keyword in drug_keywords if keyword in text]

    return {
        "bribery_hits": bribery_hits,
        "off_topic_hits": off_topic_hits,
        "marketing_hits": marketing_hits,
        "english_hits": english_hits,
        "inappropriate_hits": inappropriate_hits,
        "disrespect_hits": disrespect_hits,
        "empty_praise_hits": empty_praise_hits,
        "evidence_hits": evidence_hits,
        "clinical_study_hits": clinical_study_hits,
        "gender_mismatch_hits": gender_info["gender_mismatch_hits"],
        "expected_address": gender_info["expected_address"],
        "doctor_gender": doctor_gender,
        "has_drug_focus": len(drug_mentions) > 0,
        "drug_mentions_sample": sorted(drug_mentions)[:8],
    }
