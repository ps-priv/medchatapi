"""Budowanie stanu początkowego sesji i ewaluacja celu rozmowy."""

from typing import Dict, List, Optional

from conversation.claims import build_claim_catalog
from conversation.doctor_traits import (
    clamp_traits,
    derive_turn_limit,
    difficulty_profile,
    extract_preferred_strategies,
)
from conversation.schemas import ConversationGoal, SessionConfig

from .helpers import extract_drug_intro_keywords
from .state import ConversationState

DECISION_MAP = {
    "trial": "trial_use", "trial_use": "trial_use", "try": "trial_use",
    "will_prescribe": "will_prescribe", "prescribe": "will_prescribe",
    "recommend": "recommend",
    "reject": "reject", "decline": "reject",
    "undecided": "undecided", "neutral": "undecided",
}

POSITIVE_DECISIONS = {"trial_use", "will_prescribe", "recommend"}
NEGATIVE_DECISIONS = {"reject"}


def normalize_doctor_decision(value: str) -> str:
    """Mapuje dowolny zapis decyzji lekarza (pl/en, aliasy) na kanoniczne wartości DECISION_MAP."""
    return DECISION_MAP.get(str(value or "").strip().lower(), "undecided")


def infer_decision_from_message(doctor_message: str) -> str:
    """Wnioskuje decyzję z treści wypowiedzi gdy LLM nie podał jej jawnie."""
    text = doctor_message.lower()
    if any(p in text for p in ("odrzucam", "nie przepis", "nie wdroż", "nie zamierzam", "kończę rozmowę")):
        return "reject"
    if any(p in text for p in ("będę przepisy", "włączę do terapii", "zacznę stosować", "decyduję się na")):
        return "will_prescribe"
    if any(p in text for p in ("zarekomenduj", "zalecę", "polecę")):
        return "recommend"
    if any(p in text for p in ("rozważę", "przetestuję", "spróbuję", "u wybranych pacjent")):
        return "trial_use"
    return "undecided"


def evaluate_conversation_goal(
    state: ConversationState,
    turn_metrics: Dict,
    claim_check: Dict,
    coverage_summary: Dict,
    evidence_requirements: Dict,
    doctor_decision: str,
    doctor_attitude: str,
) -> Dict:
    """Ocenia realizację celu rozmowy na bazie metryk bieżącej tury."""
    decision = normalize_doctor_decision(doctor_decision)
    drug_revealed = bool(state.get("drug_revealed", False))

    ethical_ok = not (
        state.get("current_analysis", {}).get("bribery_hits")
        or state.get("current_analysis", {}).get("inappropriate_hits")
        or state.get("current_analysis", {}).get("disrespect_hits")
    )
    clinical_ok = len(claim_check.get("false_claims", [])) == 0 and float(turn_metrics.get("clinical_precision", 0)) >= 0.7
    coverage_ratio = float(coverage_summary.get("coverage_ratio", 1.0))
    coverage_ok = coverage_summary.get("total_critical", 0) == 0 or coverage_ratio >= 0.8
    evidence_ok = not evidence_requirements.get("require_verification", False)

    if evidence_requirements.get("require_probe", False) and coverage_ratio < 0.8:
        evidence_ok = False

    frustration = float(state.get("frustration_score", 0.0))
    doctor_satisfied = (
        doctor_attitude in {"happy", "neutral"}
        and frustration <= 6.0
        and float(turn_metrics.get("topic_adherence", 0)) >= 0.6
    )

    decision_positive = decision in POSITIVE_DECISIONS
    decision_negative = decision in NEGATIVE_DECISIONS

    relationship_score = max(0.0, min(1.0, 1.0 - frustration / 10.0))
    if doctor_attitude == "happy":
        relationship_score = min(1.0, relationship_score + 0.15)
    elif doctor_attitude == "sad":
        relationship_score = max(0.0, relationship_score - 0.2)

    evidence_score = 1.0 if evidence_ok else 0.35
    base_score = (
        0.35 * float(turn_metrics.get("clinical_precision", 0))
        + 0.25 * coverage_ratio
        + 0.2 * float(turn_metrics.get("ethics", 1.0))
        + 0.1 * float(turn_metrics.get("language_quality", 1.0))
        + 0.1 * relationship_score
    )
    if decision_positive:
        base_score += 0.12
    elif decision_negative:
        base_score -= 0.1
    if evidence_score < 1.0:
        base_score -= 0.12

    score = int(round(max(0.0, min(1.0, base_score)) * 100))
    achieved = drug_revealed and decision_positive and ethical_ok and clinical_ok and coverage_ok and evidence_ok and doctor_satisfied and score >= 75

    if achieved:
        status = "achieved"
    elif (not decision_negative) and ethical_ok and score >= 55:
        status = "partial"
    else:
        status = "not_achieved"

    reasons: List[str] = []
    missing: List[str] = []
    if decision_positive:
        reasons.append(f"Lekarz zadeklarował decyzję: {decision}.")
    elif decision == "undecided":
        missing.append("Brakuje jasnej pozytywnej decyzji lekarza.")
    else:
        missing.append("Lekarz odrzucił propozycję leku.")
    if coverage_ok:
        reasons.append(f"Pokrycie claimów krytycznych wystarczające ({coverage_summary.get('covered_critical', 0)}/{coverage_summary.get('total_critical', 0)}).")
    else:
        missing.append("Niewystarczające pokrycie claimów krytycznych (<80%).")
    if clinical_ok:
        reasons.append("Brak istotnych sprzeczności merytorycznych.")
    else:
        missing.append("Wystąpiły błędy merytoryczne lub zbyt niska precyzja kliniczna.")
    if not ethical_ok:
        missing.append("Naruszono standard etyczny rozmowy.")
    if not evidence_ok:
        missing.append("Nie spełniono wymagań evidence-first.")
    if not doctor_satisfied:
        missing.append("Niski poziom satysfakcji lekarza.")
    if not drug_revealed:
        missing.append("Lek nie został jeszcze przedstawiony lekarzowi.")

    return ConversationGoal(
        achieved=achieved,
        status=status,  # type: ignore[arg-type]
        score=score,
        doctor_decision=decision,  # type: ignore[arg-type]
        doctor_satisfied=doctor_satisfied,
        reasons=list(dict.fromkeys(reasons))[:5],
        missing=list(dict.fromkeys(missing))[:6],
    ).model_dump()


def forced_goal_payload(state: ConversationState, doctor_decision: str, reason: str) -> Dict:
    """Buduje wynik celu dla wymuszonego zakończenia."""
    decision = normalize_doctor_decision(doctor_decision)
    return ConversationGoal(
        achieved=False,
        status="not_achieved",
        score=max(0, int(state.get("goal_score", 0))),
        doctor_decision=decision,  # type: ignore[arg-type]
        doctor_satisfied=False,
        reasons=["Rozmowa zakończona wymuszenie przez reguły bezpieczeństwa/czasu."],
        missing=[reason],
    ).model_dump()


def build_initial_state(
    doctor_profile: Dict,
    drug_info: Dict,
    session_id: str,
    session_config: Optional[SessionConfig] = None,
) -> ConversationState:
    """Tworzy pełny stan początkowy sesji."""
    initial_traits = clamp_traits(doctor_profile.get("traits", {}))
    preferred_strategies = extract_preferred_strategies(doctor_profile)
    difficulty_cfg = difficulty_profile(doctor_profile.get("difficulty", "medium"))
    claim_catalog = build_claim_catalog(drug_info)
    max_turns = derive_turn_limit(initial_traits, difficulty_cfg)

    if session_config is None:
        session_config = SessionConfig()

    _trust_start = {"first_meeting": 0.20, "acquainted": 0.45, "familiar": 0.65}.get(
        session_config.familiarity.value, 0.30
    )
    _openness = float(initial_traits.get("openness", 0.5))
    initial_conviction: Dict[str, float] = {
        "interest_level": round(max(0.10, min(0.50, 0.20 + _openness * 0.15)), 2),
        "trust_in_rep": _trust_start,
        "clinical_confidence": 0.2,
        "perceived_fit": 0.2,
        "decision_readiness": 0.0,
    }

    return ConversationState(
        messages=[],
        session_id=session_id,
        doctor_profile=doctor_profile,
        drug_info=drug_info,
        familiarity=session_config.familiarity.value,
        register=session_config.register.value,
        warmth=session_config.warmth.value,
        rep_name=session_config.rep_name,
        rep_company=session_config.rep_company,
        prior_visits_summary=session_config.prior_visits_summary,
        traits=initial_traits,
        conviction=initial_conviction,
        doctor_agenda=[],
        turn_index=0,
        max_turns=max_turns,
        phase="opening",
        last_phase_reason="initial_state",
        frustration_score=0.0,
        difficulty=difficulty_cfg["difficulty"],
        preferred_strategies=preferred_strategies,
        close_phase_threshold=difficulty_cfg["close_phase_threshold"],
        termination_frustration_threshold=difficulty_cfg["termination_frustration_threshold"],
        intent_revealed=False,
        intent_revealed_turn=0,
        drug_revealed=False,
        drug_revealed_turn=0,
        drug_intro_keywords=extract_drug_intro_keywords(drug_info),
        claim_index=claim_catalog["claim_index"],
        critical_claim_ids=claim_catalog["critical_claim_ids"],
        critical_claim_labels=claim_catalog["critical_claim_labels"],
        seen_claim_ids=[],
        covered_critical_claim_ids=[],
        goal_achieved=False,
        goal_status="not_achieved",
        goal_score=0,
        goal_achieved_turn=0,
        latest_goal={},
        turn_metrics_history=[],
        turn_modes_history=[],
        critical_flags=[],
        random_events_history=[],
        last_random_event_turn=-999,
        is_terminated=False,
        current_turn_mode="REACT",
        current_user_message="",
        current_analysis={},
        current_claim_check={},
        current_coverage_summary={},
        current_coverage_update={},
        current_evidence_requirements={},
        current_pre_policy={},
        current_turn_metrics={},
        current_frustration_update={},
        current_random_event=None,
        current_behavior_directives=[],
        current_style_runtime={},
        current_doctor_response="",
        current_doctor_attitude="neutral",
        current_doctor_decision="undecided",
        current_reasoning="",
        current_detected_errors=[],
        current_termination_reason=None,
        current_conversation_goal={},
        current_turn_metrics_payload={},
    )
