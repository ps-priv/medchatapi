"""Planowanie trybu tury i wyznaczanie decyzji lekarza z conviction."""

from typing import Dict

from .random_events import deterministic_probability
from .state import ConversationState


def plan_turn_mode(state: ConversationState) -> str:
    """Wyznacza tryb tury heurystycznie — bez wywołania LLM.

    Tryby: REACT | PROBE | SHARE | CHALLENGE | DRIFT | CLOSE
    Kolejność reguł jest hierarchiczna: hard rules mają priorytet.
    """
    turn = int(state.get("turn_index", 0))
    frustration = float(state.get("frustration_score", 0.0))
    analysis = state.get("current_analysis", {})
    claim_check = state.get("current_claim_check", {})
    agenda = state.get("doctor_agenda", [])
    conviction = state.get("conviction", {})
    intent_revealed = bool(state.get("intent_revealed", False))
    session_id = str(state.get("session_id", ""))

    # Hard rules — blokujące
    if frustration > 7.0 or turn >= int(state.get("max_turns", 10)):
        return "CLOSE"
    if not intent_revealed:
        return "PROBE"

    # Konfrontacja gdy fałszywe lub nieodpowiednie zachowanie
    has_false_claims = bool(claim_check.get("false_claims"))
    has_violations = bool(analysis.get("inappropriate_hits") or analysis.get("disrespect_hits"))
    has_marketing = bool(analysis.get("marketing_hits") or analysis.get("english_hits"))
    if has_false_claims or has_violations or has_marketing:
        return "CHALLENGE"

    # Co 4 tury — SHARE jeśli jest niezużyty wątek patient_case i lekarz jest zainteresowany
    unused_cases = [a for a in agenda if a.get("kind") == "patient_case" and not a.get("used")]
    interest = float(conviction.get("interest_level", 0.3))
    if turn % 4 == 3 and unused_cases and interest > 0.4:
        return "SHARE"

    # ~10% szans na DRIFT po turze 3 (deterministyczne z seed sesji)
    if turn >= 3 and deterministic_probability(f"{session_id}|{turn}|drift") < 0.10:
        return "DRIFT"

    return "REACT"


def derive_decision_from_conviction(conviction: Dict[str, float], state: ConversationState) -> str:
    """Wyznacza decyzję lekarza na podstawie 5-wymiarowego stanu przekonań."""
    trust = float(conviction.get("trust_in_rep", 0.3))
    interest = float(conviction.get("interest_level", 0.3))
    clinical = float(conviction.get("clinical_confidence", 0.2))
    fit = float(conviction.get("perceived_fit", 0.2))
    readiness = float(conviction.get("decision_readiness", 0.0))

    # Hard reject — brak zaufania
    if trust < 0.25:
        return "reject"

    avg_positive = (interest + trust + clinical + fit) / 4

    if avg_positive >= 0.60 and readiness >= 0.65:
        return "will_prescribe"

    if trust >= 0.70 and clinical >= 0.65 and readiness >= 0.60:
        return "recommend"

    if avg_positive >= 0.50 and readiness >= 0.50:
        return "trial_use"

    return "undecided"
