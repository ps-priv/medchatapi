"""Aktualizacja 5-wymiarowego stanu przekonań lekarza (DoctorConviction)."""

from typing import Dict, List, Optional


def update_conviction(
    conviction: Dict[str, float],
    claim_check: Dict,
    turn_metrics: Dict,
    message_analysis: Dict,
    familiarity: str,
    frustration_score: float = 0.0,
    doctor_traits: Optional[Dict[str, float]] = None,
    preferred_strategies: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Aktualizuje 5-wymiarowy stan przekonań lekarza na podstawie zachowania przedstawiciela.

    Delty są modyfikowane przez cechy i strategie konkretnego lekarza — sceptyk trudniej
    uwierzy w dane kliniczne, lekarz eksploracyjny chętniej podniesie zainteresowanie.
    """
    c = {
        "interest_level": float(conviction.get("interest_level", 0.3)),
        "trust_in_rep": float(conviction.get("trust_in_rep", 0.3)),
        "clinical_confidence": float(conviction.get("clinical_confidence", 0.2)),
        "perceived_fit": float(conviction.get("perceived_fit", 0.2)),
        "decision_readiness": float(conviction.get("decision_readiness", 0.0)),
    }
    traits = doctor_traits or {}
    strategies = preferred_strategies or []

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    # --- Mnożniki z cech i strategii lekarza ---
    skepticism = float(traits.get("skepticism", 0.5))
    openness = float(traits.get("openness", 0.5))

    clinical_gain_mult = 1.0 - max(0.0, (skepticism - 0.5) * 0.6)
    interest_gain_mult = 0.85 + openness * 0.3

    trust_gain_mult = {"acquainted": 1.15, "familiar": 1.30}.get(familiarity, 1.0)
    trust_loss_mult = {"familiar": 1.20}.get(familiarity, 1.0)
    if "confrontational" in strategies:
        trust_gain_mult *= 0.80

    # Malejące zwroty dla clinical_confidence
    cc_factor = max(0.40, 1.0 - c["clinical_confidence"] * 0.70)

    # --- Claimy potwierdzone ---
    supported = len(claim_check.get("supported_claims", []))
    if supported:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.08 * min(supported, 3) * clinical_gain_mult * cc_factor)
        readiness_gain = 0.07 * min(supported, 3)
        if "transactional" in strategies:
            readiness_gain *= 1.25
        c["decision_readiness"] = clamp01(c["decision_readiness"] + readiness_gain)
        c["perceived_fit"] = clamp01(c["perceived_fit"] + 0.02 * min(supported, 2))

    # --- Claimy fałszywe ---
    has_false = bool(claim_check.get("false_claims"))
    if has_false:
        severity_counts = claim_check.get("severity_counts", {})
        critical_n = int(severity_counts.get("critical", 0))
        major_n = int(severity_counts.get("major", 0))
        minor_n = int(severity_counts.get("minor", 0))
        false_loss_mult = 1.0 + (0.15 if "skeptical" in strategies else 0.0) + (0.10 if "confrontational" in strategies else 0.0)
        if critical_n:
            trust_delta_false = -0.20 * critical_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.15 * critical_n)
        elif major_n:
            trust_delta_false = -0.10 * major_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.08 * major_n)
        elif minor_n:
            trust_delta_false = -0.04 * minor_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.03 * minor_n)
        else:
            trust_delta_false = 0.0
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + trust_delta_false * trust_loss_mult)

    # --- Claimy niepotwierdzone ---
    unsupported = len(claim_check.get("unsupported_claims", []))
    if unsupported:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.03 * min(unsupported, 4))

    # --- Evidence-based ---
    evidence_count = min(len(message_analysis.get("evidence_hits", [])), 2)
    study_count = min(len(message_analysis.get("clinical_study_hits", [])), 2)
    if evidence_count:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.10 * evidence_count * clinical_gain_mult * cc_factor)
        c["interest_level"] = clamp01(c["interest_level"] + 0.05 * evidence_count * interest_gain_mult)
    if study_count:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.06 * study_count * clinical_gain_mult * cc_factor)
        interest_study_gain = 0.03 * study_count * interest_gain_mult
        if "exploratory" in strategies:
            interest_study_gain *= 1.30
        c["interest_level"] = clamp01(c["interest_level"] + interest_study_gain)

    # --- Off-topic ---
    if message_analysis.get("off_topic_hits") and not message_analysis.get("has_drug_focus"):
        offtopic_loss = 0.08
        if "disengaging" in strategies:
            offtopic_loss *= 1.50
        if "transactional" in strategies:
            offtopic_loss *= 1.30
        c["interest_level"] = clamp01(c["interest_level"] - offtopic_loss)

    # --- Marketing buzzwords ---
    marketing_count = len(message_analysis.get("marketing_hits", []))
    if marketing_count:
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + (-0.05 * min(marketing_count, 4)) * trust_loss_mult)

    # --- Anglicyzmy ---
    english_count = len(message_analysis.get("english_hits", []))
    if english_count:
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + (-0.03 * min(english_count, 4)) * trust_loss_mult)

    # --- Disrespect / inappropriate ---
    if message_analysis.get("disrespect_hits") or message_analysis.get("inappropriate_hits"):
        ego = float(traits.get("ego", 0.5))
        disrespect_loss = -0.15 - max(0.0, (ego - 0.5) * 0.10)
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + disrespect_loss * trust_loss_mult)

    # --- Bribery ---
    if message_analysis.get("bribery_hits"):
        c["trust_in_rep"] = 0.0

    # --- Wzrost zaufania z rzetelnego zachowania ---
    topic_adherence = float(turn_metrics.get("topic_adherence", 0.0))
    if not has_false and not marketing_count and not message_analysis.get("bribery_hits"):
        if supported:
            c["trust_in_rep"] = clamp01(c["trust_in_rep"] + 0.010 * min(supported, 2) * trust_gain_mult)
        if evidence_count:
            c["trust_in_rep"] = clamp01(c["trust_in_rep"] + 0.010 * evidence_count * trust_gain_mult)
        if not english_count and not message_analysis.get("inappropriate_hits") and topic_adherence >= 0.7:
            c["trust_in_rep"] = clamp01(c["trust_in_rep"] + 0.012 * trust_gain_mult)

    # --- Pokrycie claimów krytycznych ---
    critical_coverage = float(turn_metrics.get("critical_claim_coverage", 0.0))
    if critical_coverage >= 0.5:
        c["perceived_fit"] = clamp01(c["perceived_fit"] + 0.06)
    if critical_coverage >= 0.8:
        dr_gain = 0.10
        if "skeptical" in strategies:
            dr_gain *= 0.80
        c["decision_readiness"] = clamp01(c["decision_readiness"] + dr_gain)

    # --- Adherencja tematu ---
    if topic_adherence >= 0.7:
        c["interest_level"] = clamp01(c["interest_level"] + 0.03 * interest_gain_mult)
        c["perceived_fit"] = clamp01(c["perceived_fit"] + 0.03)
        c["decision_readiness"] = clamp01(c["decision_readiness"] + 0.02)

    # --- Pasywny carry-over: pewność kliniczna → gotowość decyzyjna ---
    cc_above = max(0.0, c["clinical_confidence"] - 0.40)
    if cc_above > 0:
        c["decision_readiness"] = clamp01(c["decision_readiness"] + round(0.04 * cc_above, 4))

    # --- Wysoka frustracja ---
    if frustration_score >= 6.0:
        c["interest_level"] = clamp01(c["interest_level"] - 0.05)
        c["decision_readiness"] = clamp01(c["decision_readiness"] - 0.10)

    return {k: round(v, 4) for k, v in c.items()}
