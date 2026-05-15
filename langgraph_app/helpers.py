"""Wykrywanie intencji i leku w wypowiedzi przedstawiciela + drobne utilitki tury."""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from .state import ConversationState

INTENT_SIGNAL_KEYWORDS = (
    "lek", "leku", "preparat", "terapi", "leczen", "wskazan",
    "dawkowan", "profil bezpieczenstwa", "substanc", "refundac",
    "farmakoterapi", "skuteczn",
)

_DRUG_CONTEXT_WORDS = {"lek", "leku", "leki", "preparatu", "preparat", "produkt", "produktu", "tabletki", "kapsulki"}


def normalize_text(text: str) -> str:
    """Normalizuje tekst do porównań: małe litery, bez diakrytyków, ASCII."""
    nfkd = unicodedata.normalize("NFKD", str(text).lower())
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_drug_intro_keywords(drug_info: Dict) -> List[str]:
    raw = " ".join([str(drug_info.get("id", "")), str(drug_info.get("nazwa", ""))])
    text = normalize_text(raw)
    return list(dict.fromkeys(t for t in re.findall(r"[a-z0-9-]{4,}", text)))


def detect_drug_introduction(message: str, intro_keywords: List[str]) -> List[str]:
    text = normalize_text(message)
    return list(dict.fromkeys(kw for kw in intro_keywords if kw and kw in text))


def detect_conversation_intent(message: str) -> List[str]:
    text = normalize_text(message)
    return list(dict.fromkeys(kw for kw in INTENT_SIGNAL_KEYWORDS if kw in text))


def detect_wrong_drug(message: str, drug_intro_keywords: List[str]) -> bool:
    """Zwraca True gdy wiadomość zawiera nazwę leku innego niż lek sesji."""
    brand_candidates = re.findall(r"\b[A-ZŁŚĄŹĆÓĘ][a-ząćęłńóśźż]{2,}\b", message)
    if not brand_candidates:
        return False

    text_lower = normalize_text(message)
    has_drug_context = any(w in text_lower for w in _DRUG_CONTEXT_WORDS)
    if not has_drug_context:
        return False

    session_keywords = set(drug_intro_keywords)
    for candidate in brand_candidates:
        normalized = normalize_text(candidate)
        if normalized not in session_keywords and len(normalized) >= 4:
            return True

    return False


def build_turn_metrics_payload(turn_metrics: Dict, frustration_score: float) -> Dict:
    return {
        "topic_adherence": float(turn_metrics.get("topic_adherence", 0.0)),
        "clinical_precision": float(turn_metrics.get("clinical_precision", 0.0)),
        "ethics": float(turn_metrics.get("ethics", 0.0)),
        "language_quality": float(turn_metrics.get("language_quality", 0.0)),
        "critical_claim_coverage": float(turn_metrics.get("critical_claim_coverage", 0.0)),
        "frustration": round(float(frustration_score), 2),
    }


def check_termination(state: ConversationState, message_analysis: Dict) -> Tuple[bool, Optional[str]]:
    """Decyduje o zakończeniu rozmowy po aktualizacji cech lekarza."""
    turn_index = int(state.get("turn_index", 0))
    traits = state.get("traits", {})
    max_turns = int(state.get("max_turns", 10))

    if turn_index >= max_turns and (float(traits.get("time_pressure", 0.5)) >= 0.7 or state.get("phase") == "close"):
        return True, f"Lekarz zakończył rozmowę po osiągnięciu limitu tur ({max_turns})."

    if turn_index < 5:
        if message_analysis.get("bribery_hits"):
            return True, "Lekarz zakończył rozmowę z powodu naruszenia etyki."
        return False, None

    should_terminate = (
        float(traits.get("patience", 0.5)) <= 0.1
        or float(traits.get("time_pressure", 0.5)) >= 0.8
        or float(state.get("frustration_score", 0.0)) >= float(state.get("termination_frustration_threshold", 8.0))
        or (
            (message_analysis.get("inappropriate_hits") or message_analysis.get("disrespect_hits"))
            and float(state.get("frustration_score", 0.0)) >= 7.0
        )
    )

    if not should_terminate:
        return False, None

    if message_analysis.get("inappropriate_hits") or message_analysis.get("disrespect_hits"):
        return True, "Lekarz zakończył rozmowę z powodu nieprofesjonalnego tonu."
    return True, "Lekarz stracił cierpliwość i przerwał spotkanie."
