"""Warstwa serwisowa: pipeline rozmowy i ewaluacji."""

import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from sdialog import Context, Dialog

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

from .claims import (
    build_claim_catalog,
    check_medical_claims,
    critical_coverage_summary,
    extract_drug_keywords,
    update_claim_coverage,
)
from .constants import CHAT_COMMANDS, PHASE_OBJECTIVES
from .data import get_doctor_by_id, get_drug_by_id
from .dialog_utils import add_doctor_turn, add_user_turn, append_event, extract_turns
from .policy import (
    advance_phase,
    analyze_message,
    apply_reaction_rules,
    build_style_directives,
    clamp_traits,
    compute_frustration,
    compute_turn_metrics,
    derive_turn_limit,
    difficulty_profile,
    evidence_first_requirements,
    extract_preferred_strategies,
    policy_postprocess_message,
    policy_precheck,
)
from .schemas import ConversationGoal, DoctorResponse, EvaluationResult, MessageRequest
from .types import SessionData, SessionState

logger = logging.getLogger(__name__)

# Frazy zasłyszanej opinii — lekarz odwołuje się do wcześniejszej wiedzy o leku zamiast
# ujawniać dane wprost. Wybór frazy jest deterministyczny (hash sesji i tury).
HEARSAY_PHRASES = (
    "Czytałam gdzieś, że",
    "Słyszałam od koleżanki, że",
    "Wydaje mi się, że widziałam w literaturze, że",
    "O ile dobrze pamiętam z konferencji,",
    "Ktoś mi wspominał, że",
    "Chyba natknęłam się na taki artykuł, że",
    "Mam wrażenie, że czytałam coś na ten temat —",
    "Z tego co obiło mi się o uszy,",
)

# Losowe (deterministycznie próbkowane) zdarzenia środowiskowe, które podnoszą realizm rozmowy.
RANDOM_EVENT_TEMPLATES = (
    {
        "id": "phone_call",
        "event_name": "Telefon",
        "event_details": "Lekarka odebrała krótki telefon z oddziału i ma mniej czasu na rozmowę.",
        "base_probability": 0.11,
        "traits_delta": {"patience": -0.08, "time_pressure": 0.12, "openness": -0.04},
        "frustration_delta": 0.35,
        "turn_limit_adjust": -1,
        "directive": "W tle był pilny telefon; skróć odpowiedź i przejdź do sedna klinicznego.",
        "reasoning_note": "Telefon z oddziału dodatkowo skrócił dostępny czas lekarza.",
    },
    {
        "id": "patient_summons",
        "event_name": "Wezwanie Do Pacjenta",
        "event_details": "Pojawiło się wezwanie do pacjenta, więc rozmowa musi być bardzo krótka i konkretna.",
        "base_probability": 0.08,
        "traits_delta": {"patience": -0.14, "time_pressure": 0.2, "openness": -0.06},
        "frustration_delta": 0.6,
        "turn_limit_adjust": -1,
        "directive": "Masz wezwanie do pacjenta: wymagaj jednego konkretu klinicznego i domykaj rozmowę.",
        "reasoning_note": "Wezwanie do pacjenta drastycznie zwiększyło presję czasu.",
    },
    {
        "id": "waiting_room_surge",
        "event_name": "Wzrost Kolejki Pacjentów",
        "event_details": "Wzrosła liczba pacjentów oczekujących na wizytę, więc lekarz szybciej ucina dygresje.",
        "base_probability": 0.14,
        "traits_delta": {"patience": -0.1, "time_pressure": 0.15},
        "frustration_delta": 0.45,
        "turn_limit_adjust": -1,
        "directive": "Kolejka pacjentów rośnie: nie pozwalaj na ogólniki, egzekwuj konkretne dane o leku.",
        "reasoning_note": "Rosnąca kolejka pacjentów zmniejszyła tolerancję na dygresje.",
    },
)

POSITIVE_DECISIONS = {"trial_use", "will_prescribe", "recommend"}
NEGATIVE_DECISIONS = {"reject"}
INTENT_SIGNAL_KEYWORDS = (
    "lek",
    "leku",
    "preparat",
    "terapi",
    "leczen",
    "wskazan",
    "dawkowan",
    "profil bezpieczenstwa",
    "substanc",
    "refundac",
    "farmakoterapi",
    "skuteczn",
)
GENERIC_CLINICAL_PHRASES = {
    "wskazania",
    "przeciwwskazania",
    "dzialania niepozadane",
    "skutki uboczne",
    "dawkowanie",
    "bezpieczenstwo",
    "skutecznosc",
}


def _log_process_step(session_id: str, turn_index: int, step: str, details: str = "") -> None:
    """Loguje postęp pipeline'u process_message dla diagnostyki i debugowania."""
    suffix = f" | {details}" if details else ""
    logger.info("process_message session=%s turn=%s step=%s%s", session_id, turn_index, step, suffix)


def _normalize_doctor_decision(value: str) -> str:
    """Normalizuje decyzję lekarza do wspieranego enumu."""
    normalized = str(value or "").strip().lower()
    mapping = {
        "trial": "trial_use",
        "trial_use": "trial_use",
        "try": "trial_use",
        "will_prescribe": "will_prescribe",
        "prescribe": "will_prescribe",
        "recommend": "recommend",
        "reject": "reject",
        "decline": "reject",
        "undecided": "undecided",
        "neutral": "undecided",
    }
    return mapping.get(normalized, "undecided")


def _normalize_text_for_match(text: str) -> str:
    """Upraszcza tekst do dopasowań słów-kluczy niezależnie od diakrytyków."""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    without_diacritics = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", without_diacritics)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_drug_intro_keywords(drug_info: Dict) -> List[str]:
    """Wyciąga minimalny zestaw słów pozwalających wykryć przedstawienie leku."""
    raw_values = [
        str(drug_info.get("id", "")),
        str(drug_info.get("nazwa", "")),
    ]
    text = _normalize_text_for_match(" ".join(raw_values))
    tokens = [token for token in re.findall(r"[a-z0-9-]{4,}", text)]
    return list(dict.fromkeys(tokens))


def _detect_drug_introduction(message: str, intro_keywords: List[str]) -> List[str]:
    """Sprawdza, czy przedstawiciel nazwał lek (marka/id/substancja)."""
    text = _normalize_text_for_match(message)
    hits = [keyword for keyword in intro_keywords if keyword and keyword in text]
    return list(dict.fromkeys(hits))


def _detect_conversation_intent(message: str) -> List[str]:
    """Wykrywa, czy przedstawiciel ujawnił już cel spotkania (temat medyczno-lekowy)."""
    text = _normalize_text_for_match(message)
    hits = [keyword for keyword in INTENT_SIGNAL_KEYWORDS if keyword in text]
    return list(dict.fromkeys(hits))


def _extract_representative_history(dialog: Dialog) -> str:
    """Zwraca znormalizowany tekst wszystkich wypowiedzi przedstawiciela."""
    parts: List[str] = []
    for turn in extract_turns(dialog):
        speaker = str(getattr(turn, "speaker", getattr(turn, "role", ""))).lower()
        if "przedstawiciel" in speaker or speaker == "user":
            parts.append(str(getattr(turn, "text", getattr(turn, "content", ""))))
    return _normalize_text_for_match(" ".join(parts))


def _build_sensitive_drug_phrases(drug_info: Dict) -> List[str]:
    """Buduje listę czułych fraz o leku, które nie powinny być dopowiadane przez lekarza."""
    candidates: List[str] = []
    claims = drug_info.get("claims", [])
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        for key in ("statement",):
            value = str(claim.get(key, "")).strip()
            if value:
                candidates.append(value)
        for key in ("keywords", "support_patterns", "contradiction_patterns"):
            values = claim.get(key, [])
            if isinstance(values, list):
                candidates.extend([str(item) for item in values if str(item).strip()])

    for key in ("skład", "wskazania", "przeciwwskazania", "działania_niepożądane", "dawkowanie"):
        value = str(drug_info.get(key, "")).strip()
        if value:
            candidates.append(value)

    phrases: List[str] = []
    for raw in candidates:
        normalized = _normalize_text_for_match(raw)
        if not normalized:
            continue
        for fragment in re.split(r"[,;().]", normalized):
            phrase = fragment.strip()
            if not phrase:
                continue
            has_digits = bool(re.search(r"\d", phrase))
            word_count = len(phrase.split())
            if phrase in GENERIC_CLINICAL_PHRASES:
                continue
            if has_digits or (word_count >= 2 and len(phrase) >= 12):
                phrases.append(phrase)

    return list(dict.fromkeys(phrases))


def _pick_hearsay_phrase(session_id: str, turn_index: int) -> str:
    """Wybiera deterministycznie frazę zasłyszanej opinii na podstawie sesji i tury."""
    seed = f"{session_id}|{turn_index}|hearsay"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(HEARSAY_PHRASES)
    return HEARSAY_PHRASES[idx]


def _apply_knowledge_guard(
    doctor_message: str,
    dialog: Dialog,
    drug_info: Dict,
    session_id: str,
    turn_index: int,
) -> Tuple[str, List[str]]:
    """Przepisuje zdania z danymi leku, których przedstawiciel nie podał, jako zasłyszaną opinię.

    Zamiast usuwać wiedzę lekarza, opakowuje ją frazą typu 'Słyszałam od koleżanki, że...'
    — lekarz zachowuje się realistycznie: coś obiło mu się o uszy, ale chce to potwierdzić.
    """
    representative_history = _extract_representative_history(dialog)
    message_normalized = _normalize_text_for_match(doctor_message)
    if not representative_history or not message_normalized:
        return doctor_message, []

    sensitive_phrases = _build_sensitive_drug_phrases(drug_info)
    leaked_phrases = [
        phrase
        for phrase in sensitive_phrases
        if phrase in message_normalized and phrase not in representative_history
    ]
    leaked_phrases = list(dict.fromkeys(leaked_phrases))
    if not leaked_phrases:
        return doctor_message, []

    hearsay_prefix = _pick_hearsay_phrase(session_id, turn_index)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", doctor_message.strip()) if part.strip()]
    rewritten_sentences: List[str] = []
    for sentence in sentences:
        sentence_normalized = _normalize_text_for_match(sentence)
        if any(phrase in sentence_normalized for phrase in leaked_phrases):
            # Przepisujemy zdanie jako zasłyszaną opinię zamiast je usuwać.
            rewritten = f"{hearsay_prefix} {sentence[0].lower()}{sentence[1:]}"
            # Upewniamy się, że zdanie kończy się znakiem interpunkcyjnym.
            if rewritten and rewritten[-1] not in ".!?":
                rewritten += "."
            rewritten_sentences.append(rewritten)
        else:
            rewritten_sentences.append(sentence)

    return " ".join(rewritten_sentences).strip(), leaked_phrases[:5]


def _infer_decision_from_message(doctor_message: str) -> str:
    """Wnioskuje decyzję lekarza z treści wypowiedzi, gdy LLM nie podał jej jawnie."""
    text = doctor_message.lower()

    reject_patterns = (
        "odrzucam",
        "nie przepis",
        "nie wdroż",
        "nie wdroz",
        "nie zamierzam",
        "nie jestem zainteresowan",
        "kończę rozmowę",
        "koncze rozmowe",
    )
    if any(pattern in text for pattern in reject_patterns):
        return "reject"

    prescribe_patterns = (
        "będę przepisy",
        "bede przepisy",
        "włączę do terapii",
        "wlacze do terapii",
        "zacznę stosować",
        "zaczne stosowac",
        "decyduję się na",
        "decyduje sie na",
    )
    if any(pattern in text for pattern in prescribe_patterns):
        return "will_prescribe"

    recommend_patterns = (
        "zarekomenduj",
        "zalecę",
        "zalece",
        "polecę",
        "polece",
    )
    if any(pattern in text for pattern in recommend_patterns):
        return "recommend"

    trial_patterns = (
        "rozważę",
        "rozwaze",
        "przetestuję",
        "przetestuje",
        "spróbuję",
        "sprobuje",
        "u wybranych pacjent",
        "w pilotażu",
        "w pilotazu",
    )
    if any(pattern in text for pattern in trial_patterns):
        return "trial_use"

    return "undecided"


def _evaluate_conversation_goal(
    state: SessionState,
    message_analysis: Dict,
    claim_check: Dict,
    coverage_summary: Dict,
    turn_metrics: Dict,
    evidence_requirements: Dict,
    doctor_decision: str,
    doctor_attitude: str,
) -> Dict:
    """Wylicza status realizacji celu rozmowy (achieved/partial/not_achieved)."""
    decision = _normalize_doctor_decision(doctor_decision)
    drug_revealed = bool(state.get("drug_revealed", False))

    ethical_ok = not (
        message_analysis["bribery_hits"]
        or message_analysis["inappropriate_hits"]
        or message_analysis["disrespect_hits"]
    )
    clinical_ok = len(claim_check["false_claims"]) == 0 and float(turn_metrics["clinical_precision"]) >= 0.7
    coverage_ratio = float(coverage_summary["coverage_ratio"])
    coverage_ok = coverage_summary["total_critical"] == 0 or coverage_ratio >= 0.8
    evidence_ok = not evidence_requirements.get("require_verification", False)

    # W trybie evidence-first dopytanie (probe) uznajemy za spełnione, jeśli coverage jest wysokie.
    if evidence_requirements.get("require_probe", False) and coverage_ratio < 0.8:
        evidence_ok = False

    doctor_satisfied = (
        doctor_attitude in {"happy", "neutral"}
        and float(state.get("frustration_score", 0.0)) <= 6.0
        and float(turn_metrics["topic_adherence"]) >= 0.6
    )

    decision_positive = decision in POSITIVE_DECISIONS
    decision_negative = decision in NEGATIVE_DECISIONS

    relationship_score = max(0.0, min(1.0, 1.0 - (float(state.get("frustration_score", 0.0)) / 10.0)))
    if doctor_attitude == "happy":
        relationship_score = min(1.0, relationship_score + 0.15)
    elif doctor_attitude == "serious":
        relationship_score = max(0.0, relationship_score - 0.1)
    elif doctor_attitude == "sad":
        relationship_score = max(0.0, relationship_score - 0.2)

    evidence_score = 1.0 if evidence_ok else 0.35
    base_score = (
        0.35 * float(turn_metrics["clinical_precision"])
        + 0.25 * coverage_ratio
        + 0.2 * float(turn_metrics["ethics"])
        + 0.1 * float(turn_metrics["language_quality"])
        + 0.1 * relationship_score
    )
    if decision_positive:
        base_score += 0.12
    elif decision_negative:
        base_score -= 0.1
    if evidence_score < 1.0:
        base_score -= 0.12

    score = int(round(max(0.0, min(1.0, base_score)) * 100))

    achieved = (
        drug_revealed
        and
        decision_positive
        and ethical_ok
        and clinical_ok
        and coverage_ok
        and evidence_ok
        and doctor_satisfied
        and score >= 75
    )
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
        missing.append("Brakuje jasnej pozytywnej decyzji lekarza (trial_use/will_prescribe/recommend).")
    else:
        missing.append("Lekarz odrzucił propozycję leku.")

    if coverage_ok:
        reasons.append(
            f"Pokrycie claimów krytycznych jest wystarczające ({coverage_summary['covered_critical']}/{coverage_summary['total_critical']})."
        )
    else:
        missing.append("Niewystarczające pokrycie claimów krytycznych (<80%).")

    if clinical_ok:
        reasons.append("Brak istotnych sprzeczności merytorycznych i dobra precyzja kliniczna.")
    else:
        missing.append("Wystąpiły błędy merytoryczne lub zbyt niska precyzja kliniczna.")

    if not ethical_ok:
        missing.append("Naruszono standard etyczny/profesjonalny rozmowy.")
    else:
        reasons.append("Zachowano standard etyczny rozmowy.")

    if evidence_ok:
        reasons.append("Wymagania evidence-first zostały spełnione.")
    else:
        missing.append("Nie spełniono wymagań evidence-first (brak wiarygodnej weryfikacji).")

    if doctor_satisfied:
        reasons.append("Lekarz pozostaje na poziomie satysfakcji umożliwiającym decyzję pozytywną.")
    else:
        missing.append("Niski poziom satysfakcji lekarza (frustracja/ton rozmowy).")
    if not drug_revealed:
        missing.append("Lek nie został jeszcze jednoznacznie przedstawiony lekarzowi.")

    goal_payload = ConversationGoal(
        achieved=achieved,
        status=status,  # type: ignore[arg-type]
        score=score,
        doctor_decision=decision,  # type: ignore[arg-type]
        doctor_satisfied=doctor_satisfied,
        reasons=list(dict.fromkeys(reasons))[:5],
        missing=list(dict.fromkeys(missing))[:6],
    )
    return goal_payload.model_dump()


def _forced_goal_payload(state: SessionState, doctor_decision: str, reason: str) -> Dict:
    """Buduje wynik celu dla wymuszonego zakończenia rozmowy."""
    normalized_decision = _normalize_doctor_decision(doctor_decision)
    payload = ConversationGoal(
        achieved=False,
        status="not_achieved",
        score=max(0, int(state.get("goal_score", 0))),
        doctor_decision=normalized_decision,  # type: ignore[arg-type]
        doctor_satisfied=False,
        reasons=["Rozmowa została zakończona wymuszenie przez reguły bezpieczeństwa/czasu."],
        missing=[reason],
    )
    return payload.model_dump()


def _deterministic_probability(seed: str) -> float:
    """Zwraca pseudo-losową wartość 0..1, deterministyczną dla podanego ziarna."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _select_random_event(session_id: str, message: str, state: SessionState) -> Optional[Dict]:
    """Wybiera zdarzenie losowe dla tury (co najwyżej jedno), z uwzględnieniem cooldown."""
    turn_index = int(state.get("turn_index", 0))
    if turn_index < 2:
        return None

    last_event_turn = int(state.get("last_random_event_turn", -999))
    if (turn_index - last_event_turn) < 2:
        return None

    frustration = float(state.get("frustration_score", 0.0))
    candidates: List[Tuple[float, float, float, Dict]] = []

    for template in RANDOM_EVENT_TEMPLATES:
        probability = float(template.get("base_probability", 0.1))
        if template["id"] == "patient_summons":
            probability += 0.08 if frustration >= 5.0 else 0.03
        elif template["id"] == "waiting_room_surge" and turn_index >= 3:
            probability += 0.04
        elif template["id"] == "phone_call" and turn_index >= 4:
            probability += 0.03

        probability = max(0.02, min(0.45, probability))
        roll = _deterministic_probability(f"{session_id}|{turn_index}|{message}|{template['id']}")
        if roll <= probability:
            # Im mniejszy iloraz roll/probability, tym "silniej" zdarzenie zostało trafione.
            candidates.append((roll / max(probability, 1e-6), roll, probability, template))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, roll, probability, template = candidates[0]
    selected = dict(template)
    selected["roll"] = round(roll, 4)
    selected["probability"] = round(probability, 3)
    return selected


def _apply_random_event(
    session_id: str,
    message: str,
    session: SessionData,
    state: SessionState,
    dialog: Dialog,
) -> Optional[Dict]:
    """Nakłada skutki zdarzenia losowego na stan rozmowy i zapisuje je w historii."""
    selected = _select_random_event(session_id=session_id, message=message, state=state)
    if not selected:
        return None

    traits = dict(session["traits"])
    for trait_name, delta in selected.get("traits_delta", {}).items():
        traits[trait_name] = float(traits.get(trait_name, 0.5)) + float(delta)
    session["traits"] = clamp_traits(traits)

    state["frustration_score"] = round(
        max(0.0, min(10.0, float(state.get("frustration_score", 0.0)) + float(selected.get("frustration_delta", 0.0)))),
        2,
    )

    turn_limit_adjust = int(selected.get("turn_limit_adjust", 0))
    if turn_limit_adjust:
        updated_limit = max(3, int(state.get("max_turns", 7)) + turn_limit_adjust)
        state["max_turns"] = max(updated_limit, int(state.get("turn_index", 0)) + 1)

    event_payload = {
        "event_id": selected["id"],
        "event_name": selected["event_name"],
        "event_details": selected["event_details"],
        "directive": selected["directive"],
        "reasoning_note": selected["reasoning_note"],
        "roll": selected["roll"],
        "probability": selected["probability"],
        "turn": int(state.get("turn_index", 0)),
    }

    history = list(state.get("random_events_history", []))
    history.append(event_payload)
    state["random_events_history"] = history[-20:]
    state["last_random_event_turn"] = int(state.get("turn_index", 0))

    append_event(dialog, f"Zdarzenie Losowe: {event_payload['event_name']}", event_payload["event_details"])
    return event_payload


def _build_initial_state(doctor_profile: Dict, drug_info: Dict, initial_traits: Dict[str, float]) -> SessionState:
    """Tworzy początkowy stan state-machine i metryk sesji."""
    preferred_strategies = extract_preferred_strategies(doctor_profile)
    difficulty_cfg = difficulty_profile(doctor_profile.get("difficulty", "medium"))
    claim_catalog = build_claim_catalog(drug_info)
    max_turns = derive_turn_limit(initial_traits, difficulty_cfg)

    return {
        "turn_index": 0,
        "max_turns": max_turns,
        "phase": "opening",
        "frustration_score": 0.0,
        "difficulty": difficulty_cfg["difficulty"],
        "preferred_strategies": preferred_strategies,
        "close_phase_threshold": difficulty_cfg["close_phase_threshold"],
        "termination_frustration_threshold": difficulty_cfg["termination_frustration_threshold"],
        "claim_index": claim_catalog["claim_index"],
        "critical_claim_ids": claim_catalog["critical_claim_ids"],
        "critical_claim_labels": claim_catalog["critical_claim_labels"],
        "intent_revealed": False,
        "intent_revealed_turn": 0,
        "drug_revealed": False,
        "drug_revealed_turn": 0,
        "drug_intro_keywords": _extract_drug_intro_keywords(drug_info),
        "seen_claim_ids": [],
        "covered_critical_claim_ids": [],
        "last_phase_reason": "initial_state",
        "turn_metrics": [],
        "critical_flags": [],
        "random_events_history": [],
        "last_random_event_turn": -999,
        "goal_achieved": False,
        "goal_status": "not_achieved",
        "goal_score": 0,
        "goal_achieved_turn": 0,
        "latest_goal": {},
    }


def _analyze_turn(message: str, session: SessionData, state: SessionState) -> Dict:
    """Analizuje wypowiedź, waliduje claimy i aktualizuje coverage."""
    drug_revealed = bool(state.get("drug_revealed", False))
    if drug_revealed:
        focus_keywords = extract_drug_keywords(session["drug_info"])
    else:
        focus_keywords = set(state.get("drug_intro_keywords", []))

    message_analysis = analyze_message(
        message=message,
        drug_info=session["drug_info"],
        doctor_profile=session["doctor_profile"],
        focus_keywords=focus_keywords,
    )

    if drug_revealed:
        claim_check = check_medical_claims(message, session["drug_info"])
        coverage_update = update_claim_coverage(state, claim_check)
        coverage_summary = critical_coverage_summary(state)
    else:
        claim_check = {
            "false_claims": [],
            "unsupported_claims": [],
            "supported_claims": [],
            "evidence_signals": [],
            "claim_matches": [],
            "severity_counts": {"critical": 0, "major": 0, "minor": 0},
        }
        coverage_update = {"missing_critical_ids": [], "missing_critical_labels": []}
        coverage_summary = {"total_critical": 0, "covered_critical": 0, "missing_critical": 0, "coverage_ratio": 1.0}

    claim_check["coverage"] = coverage_summary
    claim_check["missing_critical_labels"] = coverage_update["missing_critical_labels"]
    claim_check["missing_critical_ids"] = coverage_update["missing_critical_ids"]

    return {
        "message_analysis": message_analysis,
        "claim_check": claim_check,
        "coverage_update": coverage_update,
        "coverage_summary": coverage_summary,
    }


def _update_turn_state(
    state: SessionState,
    message_analysis: Dict,
    claim_check: Dict,
    difficulty_cfg: Dict,
    preferred_strategies: List[str],
) -> Dict:
    """Aktualizuje metryki tury, frustrację i fazę rozmowy."""
    pre_policy = policy_precheck(message_analysis, claim_check)
    turn_metrics = compute_turn_metrics(message_analysis, claim_check, state)
    frustration_update = compute_frustration(
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        metrics=turn_metrics,
        difficulty_cfg=difficulty_cfg,
        preferred_strategies=preferred_strategies,
    )
    state["frustration_score"] = frustration_update["total"]

    phase_update = advance_phase(
        current_phase=state.get("phase", "opening"),
        state=state,
        message_analysis=message_analysis,
        metrics=turn_metrics,
    )
    state["phase"] = phase_update["phase"]
    state["last_phase_reason"] = phase_update["reason"]

    state["turn_metrics"].append(
        {
            "turn": state["turn_index"],
            "phase": state["phase"],
            "metrics": turn_metrics,
            "frustration": frustration_update,
            "critical_claim_coverage": claim_check["coverage"],
            "missing_critical_labels": claim_check["missing_critical_labels"][:3],
        }
    )
    state["turn_metrics"] = state["turn_metrics"][-20:]

    return {
        "pre_policy": pre_policy,
        "turn_metrics": turn_metrics,
        "frustration_update": frustration_update,
        "phase_update": phase_update,
    }


def _record_turn_events(
    dialog: Dialog,
    state: SessionState,
    message_analysis: Dict,
    claim_check: Dict,
    evidence_requirements: Dict,
) -> None:
    """Zapisuje diagnostykę jakościową i etyczną do historii sdialog."""
    if message_analysis["bribery_hits"]:
        state["critical_flags"].append("bribery")
        append_event(
            dialog,
            "Naruszenie Etyki",
            "Wypowiedź przedstawiciela sugeruje korzyść osobistą/łapówkę.",
        )

    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        if state.get("intent_revealed", False):
            append_event(dialog, "Off-topic", "Przedstawiciel odchodzi od tematu leku i rozmowy medycznej.")
        else:
            append_event(dialog, "Off-topic", "Przedstawiciel nie ujawnia celu spotkania i schodzi na temat poboczny.")

    if message_analysis["marketing_hits"]:
        append_event(
            dialog,
            "Marketingowy Żargon",
            "Wykryto chwytliwe slogany marketingowe zamiast konkretów klinicznych.",
        )

    if message_analysis["english_hits"]:
        append_event(
            dialog,
            "Anglicyzmy",
            "Wykryto angielskie słownictwo marketingowe utrudniające precyzję przekazu.",
        )

    if message_analysis["gender_mismatch_hits"]:
        append_event(
            dialog,
            "Forma Grzecznościowa",
            f"Niepoprawna forma zwracania się do lekarza. Oczekiwana: {message_analysis['expected_address']}.",
        )

    if message_analysis["inappropriate_hits"]:
        append_event(
            dialog,
            "Nieelegancka Propozycja",
            "Wykryto nieprofesjonalną i nieelegancką propozycję wobec lekarza.",
        )

    if message_analysis["disrespect_hits"]:
        append_event(
            dialog,
            "Brak Szacunku",
            "Wypowiedź przedstawiciela ma charakter deprecjonujący lub obraźliwy.",
        )

    for claim in claim_check["false_claims"]:
        append_event(dialog, "Twierdzenie Fałszywe", claim)

    for claim in claim_check["unsupported_claims"]:
        append_event(dialog, "Twierdzenie Bez Pokrycia", claim)

    if claim_check["missing_critical_labels"] and state["turn_index"] >= 2:
        append_event(
            dialog,
            "Brak Kluczowych Claimów",
            "Nieporuszone kluczowe kwestie: " + "; ".join(claim_check["missing_critical_labels"][:3]),
        )

    # Eventy audytowe pokazują, który wariant evidence-first uruchomił się w tej turze.
    if evidence_requirements.get("require_verification"):
        append_event(
            dialog,
            "Evidence-First Weryfikacja",
            "Lekarz zażądał weryfikacji twierdzeń oraz wskazania źródła danych.",
        )
    elif evidence_requirements.get("require_probe"):
        append_event(
            dialog,
            "Evidence-First Dopytanie",
            "Lekarz poprosił o twarde dane kliniczne potwierdzające tezę.",
        )


def _handle_time_limit_exceeded(
    session: SessionData,
    state: SessionState,
    dialog: Dialog,
    turn_index: int,
    max_turns: int,
    turn_metrics: Dict,
) -> Optional[Dict]:
    """Wymusza zakończenie, gdy rozmowa przekroczy dozwolony limit tur."""
    if turn_index <= max_turns:
        return None

    state["critical_flags"].append("time_limit_exceeded")
    session["traits"] = clamp_traits(
        {
            **session["traits"],
            "patience": float(session["traits"].get("patience", 0.5)) - 0.2,
            "openness": float(session["traits"].get("openness", 0.5)) - 0.15,
            "time_pressure": max(float(session["traits"].get("time_pressure", 0.5)), 0.9),
        }
    )

    termination_reason = f"Lekarz zakończył rozmowę z powodu presji czasu (limit tur: {max_turns})."
    append_event(dialog, "Limit Czasu", termination_reason)
    append_event(dialog, "Rozmowa Przerwana", termination_reason)

    forced_message = (
        "Kończę spotkanie, nie mam już czasu. Jeśli to możliwe, proszę przesłać najważniejsze informacje o leku pisemnie."
    )
    add_doctor_turn(dialog, forced_message)

    forced_goal = _forced_goal_payload(
        state=state,
        doctor_decision="undecided",
        reason="Nie osiągnięto celu: rozmowa została ucięta przez presję czasu.",
    )
    state["latest_goal"] = forced_goal
    state["goal_status"] = str(forced_goal["status"])
    state["goal_score"] = int(forced_goal["score"])
    state["goal_achieved"] = bool(forced_goal["achieved"])

    session["is_terminated"] = True
    return {
        "doctor_message": forced_message,
        "updated_traits": session["traits"],
        "reasoning": "Przekroczono dopuszczalną liczbę tur wynikającą z presji czasu lekarza.",
        "turn_metrics": _build_turn_metrics_payload(
            turn_metrics=turn_metrics,
            frustration_score=float(state.get("frustration_score", 0.0)),
        ),
        "doctor_attitude": "serious",
        "doctor_decision": forced_goal["doctor_decision"],
        "conversation_goal": forced_goal,
        "is_terminated": True,
        "termination_reason": termination_reason,
    }


def _build_behavior_directives(
    session: SessionData,
    state: SessionState,
    pre_policy: Dict,
    message_analysis: Dict,
    missing_critical_labels: List[str],
    evidence_requirements: Dict,
    random_event: Optional[Dict],
) -> Tuple[List[str], Dict]:
    """Buduje runtime dyrektywy dla odpowiedzi lekarza."""
    behavior_directives: List[str] = []
    style_runtime = build_style_directives(
        doctor_profile=session["doctor_profile"],
        traits=session["traits"],
        state=state,
    )
    behavior_directives.extend(style_runtime["directives"])
    behavior_directives.extend(pre_policy["directives"])
    behavior_directives.extend(evidence_requirements.get("directives", []))
    intent_revealed = bool(state.get("intent_revealed", False))
    if not intent_revealed:
        behavior_directives.append(
            "Nie wiesz po co ta osoba przyszła. Zareaguj naturalnie na to co usłyszałeś i zapytaj o cel wizyty — bez sugerowania tematu."
        )
    elif not state.get("drug_revealed", False):
        behavior_directives.append(
            "Rozmówca ujawnił cel wizyty, ale nie przedstawił jeszcze tematu rozmowy. Poproś o szczegóły."
        )
    if random_event:
        event_directive = str(random_event.get("directive", "")).strip()
        if event_directive:
            behavior_directives.append(event_directive)

    if missing_critical_labels:
        behavior_directives.append(
            "Przedstawiciel nie omówił wszystkich kluczowych claimów klinicznych. "
            "Zadaj pytanie uzupełniające o skuteczność, bezpieczeństwo i dawkowanie, bez dopowiadania niepodanych szczegółów."
        )

    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        if intent_revealed:
            behavior_directives.append(
                "Rozmowa zeszła na temat poboczny; krótko utnij wątek i wróć do konkretów o leku."
            )
        else:
            behavior_directives.append(
                "Rozmowa zeszła na temat poboczny; krótko utnij wątek i poproś o cel spotkania."
            )
    if message_analysis["marketing_hits"]:
        behavior_directives.append(
            "Przedstawiciel używa sloganów marketingowych; żądaj twardych danych i precyzji."
        )
    if message_analysis["english_hits"]:
        behavior_directives.append(
            "Przedstawiciel nadużywa angielskich terminów; poproś o polskie, jednoznaczne sformułowania."
        )

    return behavior_directives, style_runtime


def _build_system_prompt(
    session: SessionData,
    state: SessionState,
    message_analysis: Dict,
    claim_check: Dict,
    turn_metrics: Dict,
    coverage_summary: Dict,
    behavior_directives: List[str],
    style_runtime: Dict,
    preferred_strategies: List[str],
    random_event: Optional[Dict],
) -> str:
    """Składa pełny prompt systemowy z aktualnym kontekstem rozmowy."""
    random_event_line = (
        "brak"
        if not random_event
        else f"{random_event['event_name']}: {random_event['event_details']}"
    )
    intent_revealed = bool(state.get("intent_revealed", False))
    drug_revealed = bool(state.get("drug_revealed", False))
    drug_knowledge = (
        (
            "{'status': 'ujawniony przez przedstawiciela', "
            "'instruction': 'Lekarz zna tylko informacje podane przez przedstawiciela w tej rozmowie i nie dopowiada brakujacych szczegolow.'}"
        )
        if drug_revealed
        else "{'status': 'nieujawniony', 'instruction': 'Lekarz nie zna jeszcze szczegolow leku.'}"
    )
    claim_view = {
        "false_count": len(claim_check.get("false_claims", [])),
        "unsupported_count": len(claim_check.get("unsupported_claims", [])),
        "supported_count": len(claim_check.get("supported_claims", [])),
        "coverage_critical": f"{coverage_summary['covered_critical']}/{coverage_summary['total_critical']}",
        "missing_critical_count": int(coverage_summary.get("missing_critical", 0)),
        "requires_verification": bool(len(claim_check.get("false_claims", [])) + len(claim_check.get("unsupported_claims", []))),
    }
    return f"""
    Jesteś lekarzem. Profil: {session['doctor_profile']['name']} ({session['doctor_profile'].get('description', '')})
    Kontekst sytuacyjny: {session['doctor_profile'].get('context_str', session['doctor_profile'].get('description', ''))}
    Płeć lekarza: {message_analysis['doctor_gender']}. Oczekiwana forma zwracania się: {message_analysis['expected_address']}.
    Poziom trudności rozmowy: {state.get('difficulty', 'medium')}.
    Preferowane strategie lekarza: {json.dumps(preferred_strategies, ensure_ascii=False)}.
    Styl komunikacji: {session['doctor_profile'].get('communication_style', 'profesjonalny')}
    Aktualne cechy psychologiczne: {json.dumps(session['traits'], ensure_ascii=False)}
    Aktualna faza rozmowy: {state['phase']} (powód: {state['last_phase_reason']})
    Cel fazy: {PHASE_OBJECTIVES.get(state['phase'], PHASE_OBJECTIVES['opening'])}
    Numer tury: {state['turn_index']}
    Budżet rozmowy: maksymalnie {state['max_turns']} tur, pozostało {style_runtime['remaining_turns']} tur.
    Frustracja lekarza (0-10): {state['frustration_score']}
    Zdarzenie losowe tej tury: {random_event_line}
    Czy rozmówca ujawnił cel wizyty: {'tak' if intent_revealed else 'nie — lekarz nie wie jeszcze po co ta osoba przyszła'}
    Czy rozmówca przedstawił temat rozmowy: {'tak' if drug_revealed else 'nie — lekarz nie zna jeszcze tematu'}
    Aktualny status celu rozmowy: {state.get('goal_status', 'not_achieved')} (score={state.get('goal_score', 0)})
    Pokrycie kluczowych claimów (critical): {coverage_summary['covered_critical']}/{coverage_summary['total_critical']}

    {f"WAŻNE: Do gabinetu weszła nieznana osoba. Nie wiesz kim jest ani po co przyszła. Nie zakładaj że to przedstawiciel farmaceutyczny, nie zakładaj że chodzi o lek — po prostu zareaguj na to co usłyszałeś i jeśli trzeba zapytaj o cel wizyty." if not intent_revealed else ""}
    Dane leku:
    {drug_knowledge}

    Wykryte sygnały w ostatniej wypowiedzi:
    {json.dumps(message_analysis, ensure_ascii=False)}
    Wynik weryfikacji twierdzeń:
    {json.dumps(claim_view, ensure_ascii=False)}
    Metryki tej tury:
    {json.dumps(turn_metrics, ensure_ascii=False)}

    Dodatkowe dyrektywy:
    {json.dumps(behavior_directives, ensure_ascii=False)}

    Zasady:
    1. Odpowiadaj wyłącznie po polsku.
    2. Jeśli cel wizyty jest ujawniony i dotyczy leku, rozmowę trzymaj przy leku: wskazania, przeciwwskazania, działania niepożądane, dawkowanie.
    3. Jeśli rozmówca poda niezgodne informacje o leku, wpisz to do detected_errors i radykalnie obniż openness.
    4. Na próby korupcji reaguj stanowczo i zakończ rozmowę.
    5. Na slogany i anglicyzmy reaguj krytycznie i proś o medyczny konkret.
    6. Aktywnie stosuj styl komunikacji lekarza oraz cechy psychologiczne przy konstrukcji odpowiedzi i tonu.
    7. Aktywnie stosuj preferowane strategie lekarza oraz poziom trudności rozmowy.
    8. Jeśli to końcowa faza budżetu tur, zamknij rozmowę i nie rozwijaj nowych wątków.
    9. Jeśli rozmówca używa niepoprawnej formy grzecznościowej względem płci lekarza, skoryguj go.
    10. Jeśli rozmówca składa nieeleganckie propozycje lub mówi nieprofesjonalnie, wyraźnie ustaw granice i obniż zaufanie.
    11. Jeśli brakuje pokrycia claimów krytycznych, aktywnie zadawaj pytania o brakujące kwestie.
    12. W doctor_attitude użyj jednej wartości: happy, neutral, serious, sad.
    13. Tryb evidence-first: nie akceptuj tez bez danych; jeśli brak dowodów, dopytaj i zweryfikuj.
    14. W doctor_decision użyj jednej wartości: undecided, trial_use, will_prescribe, recommend, reject.
    15. Jeśli nie wiesz po co ta osoba przyszła, nie sugeruj żadnego tematu — po prostu zapytaj w czym możesz pomóc lub czego dotyczy wizyta.
    16. Jeśli cel wizyty jest znany, ale temat rozmowy nie został jeszcze przedstawiony, poproś o szczegóły.
    17. Jeśli znasz parametry leku z własnej wiedzy, wcześniejszych lektur lub opinii kolegów, możesz się do nich odwołać — ale ZAWSZE jako do zasłyszanej lub przeczytanej opinii (np. "Słyszałam od koleżanki, że...", "Czytałam gdzieś, że...", "Wydaje mi się, że widziałam w literaturze..."). Nigdy nie przedstawiaj takich informacji jako potwierdzonych faktów — to zadanie rozmówcy. Po odwołaniu do zasłyszanej opinii zawsze poproś o potwierdzenie lub doprecyzowanie danych.
    18. Nigdy nie przedstawiaj się słowami "Jestem pani doktor" ani żadnym podobnym zwrotem — lekarz nie przedstawia się z tytułem. Reaguj naturalnie na to, co usłyszałeś.  
    """


def _build_chat_messages(dialog: Dialog, system_prompt: str) -> List[Dict[str, str]]:
    """Składa wiadomości do API LLM z promptu i historii sdialog."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in extract_turns(dialog):
        content = getattr(turn, "text", getattr(turn, "content", str(turn)))
        speaker = str(getattr(turn, "speaker", getattr(turn, "role", ""))).lower()
        role = "user" if "przedstawiciel" in speaker or speaker == "user" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _collect_detected_errors(
    ai_detected_errors: List[str],
    message_analysis: Dict,
    claim_check: Dict,
    coverage_summary: Dict,
    missing_critical_labels: List[str],
    state: SessionState,
    evidence_requirements: Dict,
) -> List[str]:
    """Łączy błędy modelu i błędy wykryte regułowo."""
    final_detected_errors = list(ai_detected_errors)

    if message_analysis["marketing_hits"]:
        final_detected_errors.append("Użycie sloganów marketingowych bez danych klinicznych.")
    if message_analysis["english_hits"]:
        final_detected_errors.append("Nadużycie anglicyzmów zamiast precyzyjnego języka polskiego.")
    if (
        message_analysis["off_topic_hits"]
        and not message_analysis["has_drug_focus"]
        and state.get("intent_revealed", True)
    ):
        final_detected_errors.append("Odejście od tematu leku i wartości klinicznej.")
    if message_analysis["gender_mismatch_hits"]:
        final_detected_errors.append(
            f"Niepoprawna forma zwracania się do lekarza (oczekiwane: {message_analysis['expected_address']})."
        )
    if message_analysis["inappropriate_hits"]:
        final_detected_errors.append("Nieelegancka propozycja naruszająca profesjonalny charakter rozmowy.")
    if message_analysis["disrespect_hits"]:
        final_detected_errors.append("Brak szacunku wobec lekarza i nieprofesjonalny ton wypowiedzi.")

    final_detected_errors.extend(claim_check["false_claims"])
    final_detected_errors.extend(claim_check["unsupported_claims"])

    if coverage_summary["missing_critical"] > 0 and state["turn_index"] >= max(2, state["max_turns"] // 2):
        missing_preview = "; ".join(missing_critical_labels[:2])
        final_detected_errors.append(f"Nie omówiono kluczowych claimów leku: {missing_preview}.")
    # Brak wymaganych danych w evidence-first traktujemy jako dodatkowy błąd jakości rozmowy.
    if evidence_requirements.get("require_verification"):
        final_detected_errors.append("Brak wiarygodnej weryfikacji twierdzeń w trybie evidence-first.")
    elif evidence_requirements.get("require_probe"):
        final_detected_errors.append("Brak konkretu klinicznego w trybie evidence-first.")

    return list(dict.fromkeys(final_detected_errors))


def _derive_context_shift(
    ai_context_shift: Optional[str],
    message_analysis: Dict,
    claim_check: Dict,
    coverage_summary: Dict,
    state: SessionState,
) -> Optional[str]:
    """Wyznacza trwałą zmianę kontekstu, gdy model jej nie zwrócił."""
    final_context_shift = ai_context_shift

    if not final_context_shift and message_analysis["gender_mismatch_hits"]:
        final_context_shift = "Lekarz koryguje formę rozmowy i staje się bardziej zdystansowany."
    if not final_context_shift and (message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"]):
        final_context_shift = "Lekarz utracił zaufanie z powodu nieprofesjonalnego zachowania rozmówcy."
    if not final_context_shift and message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        if state.get("intent_revealed", False):
            final_context_shift = "Lekarz przejął kontrolę nad rozmową i wymaga wyłącznie konkretów o leku."
        else:
            final_context_shift = "Lekarz przerywa dygresję i prosi o jasny cel wizyty."
    if not final_context_shift and claim_check["false_claims"]:
        final_context_shift = "Lekarz utracił zaufanie do rzetelności informacji przedstawiciela."
    if (
        not final_context_shift
        and coverage_summary["missing_critical"] > 0
        and state["turn_index"] >= max(2, state["max_turns"] - 1)
    ):
        final_context_shift = "Lekarz traci zaufanie, bo nie omówiono kluczowych kwestii bezpieczeństwa leku."
    if not final_context_shift and state["frustration_score"] >= 7.0:
        final_context_shift = "Lekarz jest skrajnie zniecierpliwiony i dąży do zakończenia rozmowy."

    return final_context_shift


def _build_final_reasoning(
    base_reasoning: str,
    message_analysis: Dict,
    style_runtime: Dict,
    coverage_summary: Dict,
    evidence_requirements: Dict,
    random_event: Optional[Dict],
) -> str:
    """Składa końcowe uzasadnienie zwracane przez endpoint /message."""
    final_reasoning = base_reasoning

    if message_analysis["marketing_hits"] or message_analysis["english_hits"]:
        final_reasoning += " Lekarz obniża zaufanie przy języku marketingowym i nieprecyzyjnych anglicyzmach."
    if message_analysis["gender_mismatch_hits"]:
        final_reasoning += " Wykryto niewłaściwą formę grzecznościową, co obniża relację i zaufanie."
    if message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"]:
        final_reasoning += " Wykryto nieprofesjonalny ton lub nieelegancką propozycję; lekarz reaguje stanowczo."
    if style_runtime["directives"]:
        final_reasoning += " Zastosowano aktywne reguły stylu i cech lekarza."
    # Uzasadnienie jawnie komunikuje wpływ evidence-first na przebieg tury.
    if evidence_requirements.get("require_verification"):
        final_reasoning += " Aktywny tryb evidence-first wymusił weryfikację i żądanie źródeł danych."
    elif evidence_requirements.get("require_probe"):
        final_reasoning += " Aktywny tryb evidence-first wymusił dopytanie o konkretne dane kliniczne."
    if random_event:
        final_reasoning += f" {random_event.get('reasoning_note', '')}"
    if coverage_summary["missing_critical"] > 0:
        final_reasoning += (
            f" Brakuje pokrycia claimów krytycznych: {coverage_summary['covered_critical']}/"
            f"{coverage_summary['total_critical']}."
        )

    return final_reasoning


def _build_turn_metrics_payload(turn_metrics: Dict, frustration_score: float) -> Dict:
    """Buduje ustrukturyzowany obiekt metryk zwracanych w odpowiedzi /message."""
    return {
        "topic_adherence": float(turn_metrics.get("topic_adherence", 0.0)),
        "clinical_precision": float(turn_metrics.get("clinical_precision", 0.0)),
        "ethics": float(turn_metrics.get("ethics", 0.0)),
        "language_quality": float(turn_metrics.get("language_quality", 0.0)),
        "critical_claim_coverage": float(turn_metrics.get("critical_claim_coverage", 0.0)),
        "frustration": round(float(frustration_score), 2),
    }


def _resolve_doctor_attitude(
    ai_attitude: str,
    detected_errors_count: int,
    remaining_turns: int,
    traits: Dict[str, float],
    frustration_score: float,
) -> str:
    """Wymusza bardziej surowe nastawienie lekarza przy pogorszeniu jakości rozmowy."""
    final_attitude = ai_attitude
    if detected_errors_count >= 2:
        final_attitude = "serious"
    elif remaining_turns <= 1 and traits.get("time_pressure", 0.5) >= 0.7:
        final_attitude = "serious"
    elif frustration_score >= 6.0:
        final_attitude = "serious"
    return final_attitude


def _check_termination(session: SessionData, state: SessionState, message_analysis: Dict) -> Tuple[bool, Optional[str]]:
    """Decyduje o zakończeniu rozmowy po aktualizacji cech lekarza."""
    turn_index = int(state.get("turn_index", 0))

    if (
        state["turn_index"] >= state["max_turns"]
        and (session["traits"]["time_pressure"] >= 0.7 or state["phase"] == "close")
    ):
        termination_reason = (
            f"Lekarz zakończył rozmowę po osiągnięciu limitu tur ({state['max_turns']}) "
            "wynikającego z presji czasu."
        )
        return True, termination_reason

    # Przed turą 3 lekarz nie może zerwać rozmowy przez traits — dajemy przedstawicielowi
    # szansę na powitanie i przedstawienie celu wizyty, zanim oceniamy cierpliwość/presję.
    if turn_index < 3:
        if message_analysis["bribery_hits"]:
            return True, "Lekarz zakończył rozmowę z powodu naruszenia etyki."
        return False, None

    should_terminate = (
        session["traits"]["patience"] <= 0.2
        or session["traits"]["time_pressure"] >= 0.8
        or state["frustration_score"] >= float(state.get("termination_frustration_threshold", 8.0))
        or (
            (message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"])
            and state["frustration_score"] >= 7.0
        )
    )

    if not should_terminate:
        return False, None

    if message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"]:
        return True, "Lekarz zakończył rozmowę z powodu nieprofesjonalnego tonu wypowiedzi."
    return True, "Lekarz stracił cierpliwość i przerwał spotkanie."


def start_session(doctor_id: str, drug_id: str, sessions: Dict[str, SessionData]) -> Dict:
    """Inicjalizuje sesję rozmowy: profil lekarza, lek i stan algorytmu."""
    doctor_profile = get_doctor_by_id(doctor_id)
    if not doctor_profile:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono lekarza o id: {doctor_id}")

    drug_info = get_drug_by_id(drug_id)
    if not drug_info:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono leku o id: {drug_id}")

    session_id = str(uuid.uuid4())
    dialog = Dialog()
    context_str = doctor_profile.get("context_str", doctor_profile.get("description", ""))
    med_context = Context(**{"description": context_str}) if "description" in Context.model_fields else None

    initial_traits = clamp_traits(doctor_profile.get("traits", {}))
    initial_state = _build_initial_state(doctor_profile=doctor_profile, drug_info=drug_info, initial_traits=initial_traits)

    sessions[session_id] = {
        "dialog": dialog,
        "context": med_context,
        "traits": initial_traits,
        "doctor_profile": doctor_profile,
        "drug_info": drug_info,
        "is_terminated": False,
        "state": initial_state,
    }

    return {
        "session_id": session_id,
        "status": "Rozmowa rozpoczęta. Lekarz gotowy i czeka na cel wizyty.",
        "traits": sessions[session_id]["traits"],
    }


def _handle_chat_command(message: str) -> Optional[str]:
    """Zwraca nazwę akcji jeśli wiadomość jest komendą czatu, w przeciwnym razie None."""
    return CHAT_COMMANDS.get(message.strip())


def process_message(req: MessageRequest, sessions: Dict[str, SessionData], client) -> Dict:
    """Obsługuje pojedynczą turę rozmowy i zwraca odpowiedź lekarza."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji")

    session = sessions[req.session_id]
    if session.get("is_terminated"):
        _log_process_step(req.session_id, int(session["state"]["turn_index"]), "already_terminated")
        return {
            "error": "Rozmowa została już zakończona przez lekarza. Wywołaj /finish, aby uzyskać ocenę."
        }

    dialog = session["dialog"]
    state = session["state"]
    state["turn_index"] += 1
    turn_index = state["turn_index"]
    max_turns = state["max_turns"]

    command_action = _handle_chat_command(req.message)
    if command_action == "close_visit":
        forced_message = (
            "Stanowczo odrzucam taką propozycję. To narusza zasady etyki i kończę tę rozmowę natychmiast."
        )
        append_event(dialog, "Rozmowa Przerwana", "Naruszenie etyki zawodowej przez przedstawiciela.")
        add_doctor_turn(dialog, forced_message)
        session["is_terminated"] = True
        state["critical_flags"].append("terminated")
        state["phase"] = "close"
        forced_goal = _forced_goal_payload(
            state=state,
            doctor_decision="reject",
            reason="Nie osiągnięto celu: naruszenie etyki zakończyło rozmowę.",
        )
        state["latest_goal"] = forced_goal
        state["goal_status"] = str(forced_goal["status"])
        state["goal_score"] = int(forced_goal["score"])
        state["goal_achieved"] = bool(forced_goal["achieved"])
        logger.info("session=%s command=close_visit triggered", req.session_id)
        return {
            "doctor_message": forced_message,
            "updated_traits": session["traits"],
            "reasoning": "Policy pre-check wykrył naruszenie etyki i wymusił zakończenie rozmowy.",
            "turn_metrics": _build_turn_metrics_payload(
                turn_metrics={"topic_adherence": 0.0, "clinical_precision": 0.0, "ethics": 0.0, "language_quality": 0.0, "critical_claim_coverage": 0.0},
                frustration_score=float(state.get("frustration_score", 0.0)),
            ),
            "doctor_attitude": "serious",
            "doctor_decision": forced_goal["doctor_decision"],
            "conversation_goal": forced_goal,
            "is_terminated": True,
            "termination_reason": "Lekarz zakończył rozmowę z powodu propozycji korupcyjnej.",
        }

    if command_action == "increase_openness":
        session["traits"]["openness"] = min(1.0, float(session["traits"].get("openness", 0.5)) + 0.3)
        session["traits"] = clamp_traits(session["traits"])
        logger.info("session=%s command=increase_openness openness=%.2f", req.session_id, session["traits"]["openness"])

    _log_process_step(req.session_id, turn_index, "turn_started", f"max_turns={max_turns}")

    difficulty_cfg = difficulty_profile(state.get("difficulty", "medium"))
    preferred_strategies = state.get("preferred_strategies", [])
    _log_process_step(req.session_id, turn_index, "runtime_profiles_loaded", f"difficulty={state.get('difficulty', 'medium')}")

    _log_process_step(req.session_id, turn_index, "analysis_started")
    if not state.get("intent_revealed", False):
        intent_hits = _detect_conversation_intent(req.message)
        if intent_hits:
            state["intent_revealed"] = True
            state["intent_revealed_turn"] = turn_index
            append_event(
                dialog,
                "Ujawniono Cel Spotkania",
                "Przedstawiciel ujawnił cel rozmowy medyczno-lekowej: " + ", ".join(intent_hits[:3]),
            )
            _log_process_step(req.session_id, turn_index, "intent_revealed", f"hits={','.join(intent_hits[:3])}")
    if not state.get("drug_revealed", False):
        intro_hits = _detect_drug_introduction(req.message, state.get("drug_intro_keywords", []))
        if intro_hits:
            if not state.get("intent_revealed", False):
                state["intent_revealed"] = True
                state["intent_revealed_turn"] = turn_index
            state["drug_revealed"] = True
            state["drug_revealed_turn"] = turn_index
            append_event(
                dialog,
                "Przedstawiono Lek",
                "Przedstawiciel ujawnił lek w rozmowie: " + ", ".join(intro_hits[:3]),
            )
            _log_process_step(req.session_id, turn_index, "drug_revealed", f"hits={','.join(intro_hits[:3])}")

    analysis_bundle = _analyze_turn(message=req.message, session=session, state=state)
    message_analysis = analysis_bundle["message_analysis"]
    claim_check = analysis_bundle["claim_check"]
    coverage_update = analysis_bundle["coverage_update"]
    coverage_summary = analysis_bundle["coverage_summary"]
    evidence_requirements = evidence_first_requirements(
        message_analysis=message_analysis,
        claim_check=claim_check,
        state=state,
    )
    # Ten krok wybiera tryb evidence-first: dopytanie o dowód albo twardą weryfikację tez.
    _log_process_step(
        req.session_id,
        turn_index,
        "analysis_completed",
        (
            f"false={len(claim_check['false_claims'])}, "
            f"unsupported={len(claim_check['unsupported_claims'])}, "
            f"coverage={coverage_summary['covered_critical']}/{coverage_summary['total_critical']}, "
            f"evidence_probe={evidence_requirements['require_probe']}, "
            f"evidence_verify={evidence_requirements['require_verification']}"
        ),
    )

    _log_process_step(req.session_id, turn_index, "state_update_started")
    state_bundle = _update_turn_state(
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        difficulty_cfg=difficulty_cfg,
        preferred_strategies=preferred_strategies,
    )
    pre_policy = state_bundle["pre_policy"]
    turn_metrics = state_bundle["turn_metrics"]
    frustration_update = state_bundle["frustration_update"]
    _log_process_step(
        req.session_id,
        turn_index,
        "state_update_completed",
        f"phase={state['phase']}, frustration={state['frustration_score']}",
    )

    # Po aktualizacji metryk wstrzykujemy losowe zdarzenie środowiskowe (telefon/kolejka/wezwanie).
    random_event = _apply_random_event(
        session_id=req.session_id,
        message=req.message,
        session=session,
        state=state,
        dialog=dialog,
    )
    if random_event:
        _log_process_step(
            req.session_id,
            turn_index,
            "random_event_triggered",
            (
                f"id={random_event['event_id']}, roll={random_event['roll']}, "
                f"p={random_event['probability']}, max_turns={state['max_turns']}"
            ),
        )
    else:
        _log_process_step(req.session_id, turn_index, "random_event_none")

    _record_turn_events(
        dialog=dialog,
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        evidence_requirements=evidence_requirements,
    )
    add_user_turn(dialog, req.message)
    _log_process_step(req.session_id, turn_index, "dialog_updated_with_user_turn")

    time_limit_response = _handle_time_limit_exceeded(
        session=session,
        state=state,
        dialog=dialog,
        turn_index=turn_index,
        max_turns=max_turns,
        turn_metrics=turn_metrics,
    )
    if time_limit_response:
        _log_process_step(req.session_id, turn_index, "terminated_time_limit_exceeded")
        return time_limit_response

    # Budowa dyrektyw i opcjonalny hard-stop etyczny.
    _log_process_step(req.session_id, turn_index, "behavior_directives_started")
    behavior_directives, style_runtime = _build_behavior_directives(
        session=session,
        state=state,
        pre_policy=pre_policy,
        message_analysis=message_analysis,
        missing_critical_labels=coverage_update["missing_critical_labels"],
        evidence_requirements=evidence_requirements,
        random_event=random_event,
    )
    _log_process_step(req.session_id, turn_index, "behavior_directives_completed", f"count={len(behavior_directives)}")

    if pre_policy["hard_stop"]:
        _log_process_step(req.session_id, turn_index, "hard_stop_triggered")
        forced_message = (
            "Stanowczo odrzucam taką propozycję. To narusza zasady etyki i kończę tę rozmowę natychmiast."
        )
        append_event(dialog, "Rozmowa Przerwana", "Naruszenie etyki zawodowej przez przedstawiciela.")

        session["traits"] = apply_reaction_rules(
            current_traits=session["traits"],
            llm_traits=session["traits"],
            analysis=message_analysis,
            detected_errors_count=1 + len(claim_check["false_claims"]),
            frustration_delta=frustration_update["delta"],
            frustration_total=frustration_update["total"],
        )

        add_doctor_turn(dialog, forced_message)
        session["is_terminated"] = True
        state["critical_flags"].append("terminated")
        state["phase"] = "close"
        forced_goal = _forced_goal_payload(
            state=state,
            doctor_decision="reject",
            reason="Nie osiągnięto celu: naruszenie etyki zakończyło rozmowę.",
        )
        state["latest_goal"] = forced_goal
        state["goal_status"] = str(forced_goal["status"])
        state["goal_score"] = int(forced_goal["score"])
        state["goal_achieved"] = bool(forced_goal["achieved"])
        _log_process_step(req.session_id, turn_index, "hard_stop_completed")

        return {
            "doctor_message": forced_message,
            "updated_traits": session["traits"],
            "reasoning": "Policy pre-check wykrył naruszenie etyki i wymusił zakończenie rozmowy.",
            "turn_metrics": _build_turn_metrics_payload(
                turn_metrics=turn_metrics,
                frustration_score=float(state.get("frustration_score", 0.0)),
            ),
            "doctor_attitude": "serious",
            "doctor_decision": forced_goal["doctor_decision"],
            "conversation_goal": forced_goal,
            "is_terminated": True,
            "termination_reason": "Lekarz zakończył rozmowę z powodu propozycji korupcyjnej.",
        }

    _log_process_step(req.session_id, turn_index, "prompt_build_started")
    system_prompt = _build_system_prompt(
        session=session,
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        turn_metrics=turn_metrics,
        coverage_summary=coverage_summary,
        behavior_directives=behavior_directives,
        style_runtime=style_runtime,
        preferred_strategies=preferred_strategies,
        random_event=random_event,
    )
    messages = _build_chat_messages(dialog=dialog, system_prompt=system_prompt)
    _log_process_step(req.session_id, turn_index, "prompt_build_completed", f"messages={len(messages)}")

    _log_process_step(req.session_id, turn_index, "llm_call_started")
    completion = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=DoctorResponse,
    )
    ai_response: DoctorResponse = completion.choices[0].message.parsed
    _log_process_step(req.session_id, turn_index, "llm_call_completed")

    _log_process_step(req.session_id, turn_index, "postprocess_started")
    final_detected_errors = _collect_detected_errors(
        ai_detected_errors=list(ai_response.detected_errors),
        message_analysis=message_analysis,
        claim_check=claim_check,
        coverage_summary=coverage_summary,
        missing_critical_labels=coverage_update["missing_critical_labels"],
        state=state,
        evidence_requirements=evidence_requirements,
    )
    for error in final_detected_errors:
        append_event(dialog, "Błąd Merytoryczny", error)

    final_context_shift = _derive_context_shift(
        ai_context_shift=ai_response.context_shift,
        message_analysis=message_analysis,
        claim_check=claim_check,
        coverage_summary=coverage_summary,
        state=state,
    )
    if final_context_shift:
        append_event(dialog, "Zmiana Kontekstu", final_context_shift)
        if session.get("context") and hasattr(session["context"], "description"):
            session["context"].description += f" | NOWA SYTUACJA: {final_context_shift}"

    final_doctor_message = policy_postprocess_message(
        raw_message=ai_response.doctor_message,
        style_runtime=style_runtime,
        message_analysis=message_analysis,
        claim_check=claim_check,
        phase=state["phase"],
        missing_critical_labels=coverage_update["missing_critical_labels"],
        evidence_requirements=evidence_requirements,
        drug_revealed=bool(state.get("drug_revealed", False)),
        intent_revealed=bool(state.get("intent_revealed", False)),
    )
    leaked_details: List[str] = []
    if state.get("drug_revealed", False):
        final_doctor_message, leaked_details = _apply_knowledge_guard(
            doctor_message=final_doctor_message,
            dialog=dialog,
            drug_info=session["drug_info"],
            session_id=req.session_id,
            turn_index=int(state.get("turn_index", 0)),
        )
        if leaked_details:
            append_event(
                dialog,
                "Knowledge Guard",
                "Zablokowano dopowiadanie nieujawnionych szczegółów leku: " + "; ".join(leaked_details[:3]),
            )
            _log_process_step(
                req.session_id,
                turn_index,
                "knowledge_guard_applied",
                f"blocked={len(leaked_details)}",
            )

    final_reasoning = _build_final_reasoning(
        base_reasoning=ai_response.reasoning,
        message_analysis=message_analysis,
        style_runtime=style_runtime,
        coverage_summary=coverage_summary,
        evidence_requirements=evidence_requirements,
        random_event=random_event,
    )
    if leaked_details:
        final_reasoning += " Aktywowano filtr wiedzy: usunięto szczegóły leku, których przedstawiciel nie podał."

    final_attitude = _resolve_doctor_attitude(
        ai_attitude=ai_response.doctor_attitude,
        detected_errors_count=len(final_detected_errors),
        remaining_turns=style_runtime["remaining_turns"],
        traits=session["traits"],
        frustration_score=state["frustration_score"],
    )

    # Decyzja lekarza jest jawnie śledzona; fallback bierze sygnały bezpośrednio z wypowiedzi.
    final_decision = _normalize_doctor_decision(getattr(ai_response, "doctor_decision", "undecided"))
    if final_decision == "undecided":
        final_decision = _infer_decision_from_message(final_doctor_message)

    conversation_goal = _evaluate_conversation_goal(
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        coverage_summary=coverage_summary,
        turn_metrics=turn_metrics,
        evidence_requirements=evidence_requirements,
        doctor_decision=final_decision,
        doctor_attitude=final_attitude,
    )
    state["latest_goal"] = conversation_goal
    state["goal_status"] = str(conversation_goal["status"])
    state["goal_score"] = int(conversation_goal["score"])
    state["goal_achieved"] = bool(state.get("goal_achieved", False) or conversation_goal["achieved"])
    if bool(conversation_goal["achieved"]) and int(state.get("goal_achieved_turn", 0)) == 0:
        state["goal_achieved_turn"] = int(state["turn_index"])

    updated_traits = apply_reaction_rules(
        current_traits=session["traits"],
        llm_traits=ai_response.updated_traits.model_dump(),
        analysis=message_analysis,
        detected_errors_count=len(final_detected_errors),
        frustration_delta=frustration_update["delta"],
        frustration_total=frustration_update["total"],
    )
    session["traits"] = updated_traits

    add_doctor_turn(dialog, final_doctor_message)
    _log_process_step(
        req.session_id,
        turn_index,
        "postprocess_completed",
        (
            f"errors={len(final_detected_errors)}, attitude={final_attitude}, "
            f"goal={conversation_goal['status']}:{conversation_goal['score']}"
        ),
    )

    is_terminated, termination_reason = _check_termination(
        session=session,
        state=state,
        message_analysis=message_analysis,
    )
    if not is_terminated and bool(conversation_goal["achieved"]):
        is_terminated = True
        termination_reason = "Cel rozmowy osiągnięty: lekarz deklaruje pozytywną decyzję wobec leku."

    if is_terminated:
        session["is_terminated"] = True
        state["critical_flags"].append("terminated")
        state["phase"] = "close"
        if termination_reason and "limitu tur" in termination_reason:
            append_event(dialog, "Limit Czasu", termination_reason)
        if bool(conversation_goal["achieved"]):
            append_event(dialog, "Cel Osiągnięty", termination_reason or "Cel rozmowy osiągnięty.")
            append_event(dialog, "Rozmowa Zakończona", termination_reason or "Rozmowa zakończona.")
        else:
            append_event(dialog, "Rozmowa Przerwana", termination_reason or "Rozmowa zakończona.")
        _log_process_step(req.session_id, turn_index, "terminated", termination_reason or "unknown")
    else:
        _log_process_step(req.session_id, turn_index, "turn_completed")

    return {
        "doctor_message": final_doctor_message,
        "updated_traits": session["traits"],
        "reasoning": final_reasoning,
        "turn_metrics": _build_turn_metrics_payload(
            turn_metrics=turn_metrics,
            frustration_score=float(state.get("frustration_score", 0.0)),
        ),
        "doctor_attitude": final_attitude,
        "doctor_decision": conversation_goal["doctor_decision"],
        "conversation_goal": conversation_goal,
        "is_terminated": is_terminated,
        "termination_reason": termination_reason,
    }


def finish_session(session_id: str, sessions: Dict[str, SessionData], client) -> Dict:
    """Kończy sesję i uruchamia końcową ocenę jakości rozmowy."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji")

    session = sessions[session_id]
    dialog = session["dialog"]
    state = session.get("state", {})

    history_text = ""
    history_list = []
    for turn in extract_turns(dialog):
        speaker = getattr(turn, "speaker", getattr(turn, "role", "unknown"))
        text = getattr(turn, "text", getattr(turn, "content", str(turn)))
        history_list.append({"speaker": speaker, "text": text})
        history_text += f"[{speaker}]: {text}\n"

    evaluator_prompt = f"""
    Jesteś doświadczonym trenerem sprzedaży medycznej.

    Profil lekarza:
    {json.dumps(session['doctor_profile'], ensure_ascii=False)}

    Informacje o leku:
    {json.dumps(session['drug_info'], ensure_ascii=False)}

    Zapis rozmowy:
    {history_text}

    Metryki i przebieg algorytmu:
    {json.dumps(state.get('turn_metrics', []), ensure_ascii=False)}
    Ostatnia faza rozmowy: {state.get('phase', 'unknown')}
    Końcowy poziom frustracji lekarza (0-10): {state.get('frustration_score', 'unknown')}

    Oceń przedstawiciela pod kątem:
    - profesjonalizmu,
    - trafności argumentów dot. leku,
    - budowania relacji i radzenia sobie z trudnym lekarzem,
    - utrzymania rozmowy w temacie leku,
    - zgodności etycznej (bez prób niedozwolonych korzyści).
    """

    completion = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Jesteś surowym, ale sprawiedliwym trenerem biznesu."},
            {"role": "user", "content": evaluator_prompt},
        ],
        response_format=EvaluationResult,
    )

    evaluation = completion.choices[0].message.parsed
    latest_goal = state.get("latest_goal") or _forced_goal_payload(
        state=state,
        doctor_decision="undecided",
        reason="Brak tury decyzyjnej zakończonej oceną celu.",
    )
    del sessions[session_id]

    return {
        "status": "Rozmowa zakończona, sesja usunięta.",
        "conversation_history": history_list,
        "conversation_goal": latest_goal,
        "evaluation": evaluation.model_dump(),
    }
