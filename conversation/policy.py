"""Reguły rozmowy: pre-check, evidence-first, postprocessing i reakcje traits."""

from typing import Dict, List

from .doctor_traits import clamp_traits
from .message_analysis import limit_sentences


def policy_precheck(message_analysis: Dict, claim_check: Dict) -> Dict:
    """Buduje twarde dyrektywy policy przed wywołaniem LLM."""
    hard_stop = bool(message_analysis["bribery_hits"])
    directives: List[str] = []

    if hard_stop:
        directives.append("Naruszenie etyki: zakończ rozmowę natychmiast.")
    if message_analysis["gender_mismatch_hits"]:
        directives.append(
            f"Skoryguj formę grzecznościową: oczekiwane zwroty to '{message_analysis['expected_address']}'."
        )
    if message_analysis["inappropriate_hits"]:
        directives.append("Wykryto nieelegancką propozycję; stanowczo zaznacz granice rozmowy.")
    if message_analysis["disrespect_hits"]:
        directives.append("Wykryto brak szacunku; odpowiedz stanowczo i skróć rozmowę.")
    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        directives.append("Przerwij off-topic i wróć do rozmowy o leku.")
    if claim_check["false_claims"]:
        directives.append("Wykryto twierdzenia niezgodne z danymi leku.")
    if claim_check["unsupported_claims"]:
        directives.append("Zażądaj potwierdzenia twierdzeń bez pokrycia w danych.")
    if message_analysis["has_drug_focus"] and not claim_check.get("supported_claims"):
        directives.append(
            "Tryb evidence-first: zanim przejdziesz dalej, poproś o konkretne dane potwierdzające tezy."
        )

    missing_critical = claim_check.get("missing_critical_labels", [])
    if missing_critical:
        directives.append("Przedstawiciel nie omówił wszystkich kluczowych claimów; dopytaj o brakujące punkty krytyczne.")

    return {
        "hard_stop": hard_stop,
        "directives": directives,
    }


def evidence_first_requirements(message_analysis: Dict, claim_check: Dict, state: Dict) -> Dict:
    """Wyznacza wymagania evidence-first: dopytanie i weryfikację twierdzeń."""
    supported_count = len(claim_check.get("supported_claims", []))
    false_count = len(claim_check.get("false_claims", []))
    unsupported_count = len(claim_check.get("unsupported_claims", []))
    turn_index = int(state.get("turn_index", 0))
    has_drug_focus = bool(message_analysis.get("has_drug_focus"))

    require_verification = (false_count + unsupported_count) > 0
    require_probe = has_drug_focus and supported_count == 0 and not require_verification
    strict_mode = require_verification or (require_probe and turn_index >= 2)

    directives: List[str] = []
    followup_question = ""
    if require_verification:
        directives.append(
            "Tryb evidence-first: zakwestionuj twierdzenia bez pokrycia i poproś o źródło oraz parametry danych."
        )
        followup_question = (
            "Proszę podać konkretne źródło i dane (np. populacja, punkt końcowy, dawka), "
            "które potwierdzają tę tezę."
        )
    elif require_probe:
        directives.append(
            "Tryb evidence-first: dopytaj o konkret kliniczny potwierdzający skuteczność lub bezpieczeństwo."
        )
        followup_question = (
            "Jakie konkretne dane kliniczne potwierdzają to twierdzenie "
            "(np. liczby, populacja, porównanie, bezpieczeństwo)?"
        )

    return {
        "require_probe": require_probe,
        "require_verification": require_verification,
        "strict_mode": strict_mode,
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "false_count": false_count,
        "directives": directives,
        "followup_question": followup_question,
    }


def policy_postprocess_message(
    raw_message: str,
    style_runtime: Dict,
    message_analysis: Dict,
    claim_check: Dict,
    phase: str,
    register: str = "professional",
) -> str:
    """Nakłada minimalne korekty bezpieczeństwa na odpowiedź LLM."""
    result = limit_sentences(raw_message.strip(), int(style_runtime["max_sentences"]))

    if register != "informal" and message_analysis["gender_mismatch_hits"]:
        correction = f"Proszę zwracać się do mnie poprawnie: '{message_analysis['expected_address']}'."
        if correction not in result:
            result += f" {correction}"

    if message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"]:
        boundary_msg = "Taki ton i takie propozycje są nieakceptowalne w rozmowie zawodowej."
        if boundary_msg not in result:
            result += f" {boundary_msg}"

    if claim_check["false_claims"]:
        if "niezgodne z danymi" not in result:
            result += " To niezgodne z danymi leku."

    if phase == "close":
        if "Kończę" not in result and "kończę" not in result and "zakończ" not in result:
            result += " Na tym etapie kończę spotkanie."

    return result.strip()


def apply_reaction_rules(
    current_traits: Dict[str, float],
    llm_traits: Dict[str, float],
    analysis: Dict,
    detected_errors_count: int,
    frustration_delta: float,
    frustration_total: float,
) -> Dict[str, float]:
    """Nakłada deterministyczne zmiany traits po reakcji LLM i regułach policy."""
    traits = clamp_traits(llm_traits)

    if analysis["bribery_hits"]:
        traits["skepticism"] = 1.0
        traits["patience"] = 0.0
        traits["openness"] = 0.0
        traits["time_pressure"] = 1.0
        traits["ego"] = max(traits["ego"], current_traits.get("ego", 0.5))
        return clamp_traits(traits)

    if detected_errors_count > 0:
        traits["skepticism"] += min(0.18 * detected_errors_count, 0.5)
        traits["openness"] -= min(0.2 * detected_errors_count, 0.5)
        traits["patience"] -= min(0.12 * detected_errors_count, 0.35)
        traits["time_pressure"] += min(0.01 * detected_errors_count, 0.04)

    if analysis["marketing_hits"]:
        penalty = min(0.06 * len(analysis["marketing_hits"]), 0.2)
        traits["skepticism"] += penalty
        traits["openness"] -= penalty

    if analysis.get("empty_praise_hits"):
        traits["skepticism"] += min(0.15 * len(analysis["empty_praise_hits"]), 0.4)

    if analysis.get("evidence_hits"):
        traits["time_pressure"] -= min(0.08 * len(analysis["evidence_hits"]), 0.3)

    if analysis.get("clinical_study_hits"):
        count = len(analysis["clinical_study_hits"])
        traits["time_pressure"] -= min(0.1 * count, 0.4)
        traits["openness"] += min(0.12 * count, 0.35)
        traits["skepticism"] -= min(0.1 * count, 0.3)

    if analysis["english_hits"]:
        penalty = min(0.05 * len(analysis["english_hits"]), 0.2)
        traits["skepticism"] += penalty
        traits["patience"] -= penalty

    if analysis["off_topic_hits"] and not analysis["has_drug_focus"]:
        traits["time_pressure"] += 0.01
        traits["patience"] -= 0.08

    if analysis["gender_mismatch_hits"]:
        penalty = min(0.08 * len(analysis["gender_mismatch_hits"]), 0.24)
        traits["skepticism"] += penalty
        traits["openness"] -= penalty
        traits["patience"] -= min(penalty, 0.15)

    if analysis["inappropriate_hits"] or analysis["disrespect_hits"]:
        traits["skepticism"] += min(0.2 + 0.07 * len(analysis["inappropriate_hits"]), 0.45)
        traits["openness"] -= min(0.25 + 0.05 * len(analysis["disrespect_hits"]), 0.5)
        traits["patience"] -= min(0.2 + 0.05 * len(analysis["disrespect_hits"]), 0.4)
        traits["time_pressure"] += min(0.02 + 0.01 * len(analysis["inappropriate_hits"]), 0.04)

    if frustration_delta > 0:
        traits["skepticism"] += min(0.04 * frustration_delta, 0.2)
        traits["patience"] -= min(0.05 * frustration_delta, 0.25)

    if frustration_total >= 6.0:
        traits["openness"] -= 0.15
        traits["time_pressure"] += 0.01

    return clamp_traits(traits)
