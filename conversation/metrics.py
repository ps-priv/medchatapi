"""Metryki tury i frustracja lekarza — obliczenia deterministyczne."""

from typing import Dict, List

from .claims import critical_coverage_summary
from .constants import CLAIM_SEVERITY_WEIGHTS, PHASES


def compute_turn_metrics(message_analysis: Dict, claim_check: Dict, state: Dict) -> Dict:
    """Oblicza 5 metryk jakości wypowiedzi przedstawiciela w bieżącej turze.

    Zwraca: topic_adherence, clinical_precision, ethics, language_quality,
    critical_claim_coverage — wszystkie w zakresie [0.0, 1.0].
    """
    coverage = critical_coverage_summary(state)
    turn_index = int(state.get("turn_index", 0))
    max_turns = max(1, int(state.get("max_turns", 7)))
    progress_ratio = min(1.0, turn_index / max_turns)

    topic_adherence = 0.3
    if message_analysis["has_drug_focus"]:
        topic_adherence += 0.5
    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        topic_adherence -= 0.35

    clinical_precision = 0.8
    clinical_precision -= min(0.25 * len(claim_check["false_claims"]), 0.75)
    clinical_precision -= min(0.12 * len(claim_check["unsupported_claims"]), 0.35)
    clinical_precision -= min(0.07 * len(message_analysis["marketing_hits"]), 0.2)

    drug_revealed = bool(state.get("drug_revealed", True))
    if drug_revealed and coverage["total_critical"] > 0 and turn_index >= 2:
        missing_ratio = coverage["missing_critical"] / max(1, coverage["total_critical"])
        coverage_penalty = min((0.25 * missing_ratio) + (0.2 * progress_ratio), 0.45)
        clinical_precision -= coverage_penalty

    ethics = 1.0 if not message_analysis["bribery_hits"] else 0.0
    ethics -= min(0.3 * len(message_analysis["inappropriate_hits"]), 0.7)
    ethics -= min(0.2 * len(message_analysis["disrespect_hits"]), 0.6)
    ethics -= min(0.15 * len(message_analysis["gender_mismatch_hits"]), 0.4)

    language_quality = 1.0
    language_quality -= min(0.12 * len(message_analysis["english_hits"]), 0.5)
    language_quality -= min(0.08 * len(message_analysis["marketing_hits"]), 0.3)
    language_quality -= min(0.2 * len(message_analysis["disrespect_hits"]), 0.6)
    language_quality -= min(0.1 * len(message_analysis["gender_mismatch_hits"]), 0.3)

    return {
        "topic_adherence": round(max(0.0, min(1.0, topic_adherence)), 2),
        "clinical_precision": round(max(0.0, min(1.0, clinical_precision)), 2),
        "ethics": round(max(0.0, min(1.0, ethics)), 2),
        "language_quality": round(max(0.0, min(1.0, language_quality)), 2),
        "critical_claim_coverage": coverage["coverage_ratio"],
    }


def compute_frustration(
    state: Dict,
    message_analysis: Dict,
    claim_check: Dict,
    metrics: Dict,
    difficulty_cfg: Dict,
    preferred_strategies: List[str],
) -> Dict:
    """Oblicza deltę i nową wartość frustracji lekarza — uwzględnia claimy, etykę, off-topic, strategie i postęp rozmowy."""
    delta = 0.0
    severity_counts = claim_check.get("severity_counts", {})
    critical_count = int(severity_counts.get("critical", 0))
    major_count = int(severity_counts.get("major", 0))
    minor_count = int(severity_counts.get("minor", 0))

    coverage = critical_coverage_summary(state)
    turn_index = int(state.get("turn_index", 0))
    max_turns = max(1, int(state.get("max_turns", 7)))
    progress_ratio = min(1.0, turn_index / max_turns)

    if message_analysis["bribery_hits"]:
        delta += 8.0
    delta += 2.2 * len(message_analysis["inappropriate_hits"])
    delta += 1.5 * len(message_analysis["disrespect_hits"])
    delta += 0.9 * len(message_analysis["gender_mismatch_hits"])
    delta += CLAIM_SEVERITY_WEIGHTS["critical"] * critical_count
    delta += CLAIM_SEVERITY_WEIGHTS["major"] * major_count
    delta += CLAIM_SEVERITY_WEIGHTS["minor"] * minor_count
    delta += 1.4 if (message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]) else 0.0
    delta += 0.45 * len(message_analysis["marketing_hits"])
    delta += 0.35 * len(message_analysis["english_hits"])

    drug_revealed = bool(state.get("drug_revealed", True))
    if drug_revealed and coverage["total_critical"] > 0 and turn_index >= 2:
        missing_ratio = coverage["missing_critical"] / max(1, coverage["total_critical"])
        delta += missing_ratio * (0.6 + 1.1 * progress_ratio)
        if progress_ratio >= 0.75 and coverage["missing_critical"] > 0:
            delta += 0.5

    if "skeptical" in preferred_strategies:
        delta += 0.35 * len(claim_check["unsupported_claims"])
        delta += 0.45 * len(claim_check["false_claims"])

    if "confrontational" in preferred_strategies and (claim_check["false_claims"] or claim_check["unsupported_claims"]):
        delta += 0.5

    if "transactional" in preferred_strategies and (
        message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]
    ):
        delta += 0.7

    if "disengaging" in preferred_strategies:
        if not message_analysis["has_drug_focus"] and state.get("intent_revealed", False):
            delta += 0.6
        if int(state.get("turn_index", 0)) >= int(state.get("max_turns", 7)) - 1:
            delta += 0.5

    if "exploratory" in preferred_strategies:
        if (
            message_analysis["has_drug_focus"]
            and not claim_check["false_claims"]
            and not claim_check["unsupported_claims"]
        ):
            delta -= 0.35

    if metrics["topic_adherence"] >= 0.7:
        delta -= 0.4
    if metrics["clinical_precision"] >= 0.8:
        delta -= 0.2
    if claim_check.get("supported_claims"):
        delta -= min(0.15 * len(claim_check["supported_claims"]), 0.5)
    if turn_index > max_turns:
        delta += 2.0

    if turn_index >= 2:
        delta += float(difficulty_cfg.get("frustration_bias", 0.0))
    delta = max(-1.0, min(8.0, delta))
    total = max(0.0, min(10.0, float(state.get("frustration_score", 0.0)) + delta))
    return {"delta": round(delta, 2), "total": round(total, 2)}


def advance_phase(current_phase: str, state: Dict, message_analysis: Dict, metrics: Dict) -> Dict:
    """Przełącza fazę rozmowy zgodnie z regułami state-machine.

    Fazy: opening → needs → objection → evidence → close.
    Naruszenie etyki lub przekroczenie limitu/frustracji wymusza natychmiastowe przejście do close.
    Zwraca dict z kluczami 'phase' i 'reason'.
    """
    phase = current_phase if current_phase in PHASES else "opening"
    turn_index = int(state.get("turn_index", 0))
    max_turns = int(state.get("max_turns", 7))
    frustration = float(state.get("frustration_score", 0.0))
    close_phase_threshold = float(state.get("close_phase_threshold", 7.0))

    if message_analysis["bribery_hits"]:
        return {"phase": "close", "reason": "ethical_breach"}
    if turn_index >= 2 and (frustration >= close_phase_threshold or turn_index >= max_turns):
        return {"phase": "close", "reason": "time_or_frustration_limit"}

    if phase == "opening":
        if message_analysis["has_drug_focus"]:
            return {"phase": "needs", "reason": "drug_focus_established"}
        return {"phase": "opening", "reason": "awaiting_relevance"}

    if phase == "needs":
        if metrics["clinical_precision"] >= 0.6:
            return {"phase": "objection", "reason": "objection_phase_started"}
        return {"phase": "needs", "reason": "needs_clarification"}

    if phase == "objection":
        if metrics["clinical_precision"] >= 0.7 and not message_analysis["marketing_hits"]:
            return {"phase": "evidence", "reason": "evidence_requested"}
        return {"phase": "objection", "reason": "objection_not_resolved"}

    if phase == "evidence":
        if turn_index >= max_turns - 1 or frustration >= 8.5:
            return {"phase": "close", "reason": "closing_window"}
        return {"phase": "evidence", "reason": "continue_evidence"}

    return {"phase": "close", "reason": "already_closing"}
