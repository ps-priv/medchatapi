"""Funkcje pomocnicze dla silnika LangGraph — czyste obliczenia bez zależności od sdialog."""

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from conversation.claims import build_claim_catalog
from conversation.policy import (
    clamp_traits,
    derive_turn_limit,
    difficulty_profile,
    extract_preferred_strategies,
)
from conversation.schemas import ConversationGoal, SessionConfig

from .state import ConversationState

# ---------------------------------------------------------------------------
# Konstanty
# ---------------------------------------------------------------------------

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
    "lek", "leku", "preparat", "terapi", "leczen", "wskazan",
    "dawkowan", "profil bezpieczenstwa", "substanc", "refundac",
    "farmakoterapi", "skuteczn",
)

GENERIC_CLINICAL_PHRASES = {
    "wskazania", "przeciwwskazania", "dzialania niepozadane",
    "skutki uboczne", "dawkowanie", "bezpieczenstwo", "skutecznosc",
}


# ---------------------------------------------------------------------------
# Normalizacja tekstu
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalizuje tekst do porównań: małe litery, bez diakrytyków, ASCII."""
    nfkd = unicodedata.normalize("NFKD", str(text).lower())
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Wykrywanie intencji i leku
# ---------------------------------------------------------------------------

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


# Słowa kontekstu farmaceutycznego — sugerują, że po nich następuje nazwa leku.
_DRUG_CONTEXT_WORDS = {"lek", "leku", "leki", "preparatu", "preparat", "produkt", "produktu", "tabletki", "kapsulki"}

def detect_wrong_drug(message: str, drug_intro_keywords: List[str]) -> bool:
    """Zwraca True gdy wiadomość zawiera nazwę leku innego niż lek sesji.

    Algorytm: szuka wielkich liter (nazwy handlowe) w sąsiedztwie słów
    farmaceutycznych. Jeśli znaleziona nazwa nie pasuje do drug_intro_keywords
    sesji — rozmówca mówi o innym leku.
    """
    # Wyciągnij potencjalne nazwy handlowe (wielka litera + ≥3 znaki)
    brand_candidates = re.findall(r"\b[A-ZŁŚĄŹĆÓĘ][a-ząćęłńóśźż]{2,}\b", message)
    if not brand_candidates:
        return False

    # Sprawdź czy w wiadomości jest kontekst farmaceutyczny
    text_lower = normalize_text(message)
    has_drug_context = any(w in text_lower for w in _DRUG_CONTEXT_WORDS)
    if not has_drug_context:
        return False

    # Sprawdź każdą kandydaturę na nazwę leku
    session_keywords = set(drug_intro_keywords)
    for candidate in brand_candidates:
        normalized = normalize_text(candidate)
        # Jeśli znormalizowana nazwa nie pokrywa się z żadnym kluczem sesji → obcy lek
        if normalized not in session_keywords and len(normalized) >= 4:
            return True

    return False


# ---------------------------------------------------------------------------
# Budowanie stanu początkowego
# ---------------------------------------------------------------------------

def build_initial_state(
    doctor_profile: Dict,
    drug_info: Dict,
    session_id: str,
    session_config: Optional[SessionConfig] = None,
) -> ConversationState:
    """Tworzy pełny stan początkowy sesji.

    Parametry:
        doctor_profile: profil lekarza z doctor_archetypes.json
        drug_info: dane leku
        session_id: UUID sesji
        session_config: opcjonalna konfiguracja sesji (familiarity, register, warmth, itp.)
            Brak = wartości domyślne (FIRST_MEETING, PROFESSIONAL, NEUTRAL).
            Zachowuje pełną kompatybilność wsteczną ze starszymi klientami API,
            którzy wołają /start bez body.
    """
    initial_traits = clamp_traits(doctor_profile.get("traits", {}))
    preferred_strategies = extract_preferred_strategies(doctor_profile)
    difficulty_cfg = difficulty_profile(doctor_profile.get("difficulty", "medium"))
    claim_catalog = build_claim_catalog(drug_info)
    max_turns = derive_turn_limit(initial_traits, difficulty_cfg)

    # Domyślna konfiguracja gdy klient nie przekazał body
    if session_config is None:
        session_config = SessionConfig()

    # Inicjalizacja conviction (Etap 2)
    # trust_in_rep zależy od familiarity; interest_level od openness lekarza
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
        # Konfiguracja sesji (Etap 1)
        familiarity=session_config.familiarity.value,
        register=session_config.register.value,
        warmth=session_config.warmth.value,
        rep_name=session_config.rep_name,
        rep_company=session_config.rep_company,
        prior_visits_summary=session_config.prior_visits_summary,
        # Pozostałe pola
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
        critical_flags=[],
        random_events_history=[],
        last_random_event_turn=-999,
        is_terminated=False,
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


# ---------------------------------------------------------------------------
# Normalizacja decyzji lekarza
# ---------------------------------------------------------------------------

DECISION_MAP = {
    "trial": "trial_use", "trial_use": "trial_use", "try": "trial_use",
    "will_prescribe": "will_prescribe", "prescribe": "will_prescribe",
    "recommend": "recommend",
    "reject": "reject", "decline": "reject",
    "undecided": "undecided", "neutral": "undecided",
}


def normalize_doctor_decision(value: str) -> str:
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


# ---------------------------------------------------------------------------
# Ewaluacja celu rozmowy
# ---------------------------------------------------------------------------

def evaluate_conversation_goal(state: ConversationState, turn_metrics: Dict, claim_check: Dict,
                                coverage_summary: Dict, evidence_requirements: Dict,
                                doctor_decision: str, doctor_attitude: str) -> Dict:
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

    # Rekomendacja innemu lekarzowi (wysoki trust + dobra znajomość kliniczna)
    if trust >= 0.75 and clinical >= 0.70 and readiness >= 0.65:
        return "recommend"

    # Pełna decyzja o przepisaniu
    if avg_positive >= 0.65 and readiness >= 0.75:
        return "will_prescribe"

    # Próbne zastosowanie
    if avg_positive >= 0.55 and readiness >= 0.60:
        return "trial_use"

    return "undecided"


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


# ---------------------------------------------------------------------------
# Zdarzenia losowe
# ---------------------------------------------------------------------------

def deterministic_probability(seed: str) -> float:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def select_random_event(session_id: str, message: str, state: ConversationState) -> Optional[Dict]:
    turn_index = int(state.get("turn_index", 0))
    if turn_index < 2:
        return None
    last_event_turn = int(state.get("last_random_event_turn", -999))
    if (turn_index - last_event_turn) < 2:
        return None

    frustration = float(state.get("frustration_score", 0.0))
    candidates = []

    for template in RANDOM_EVENT_TEMPLATES:
        probability = float(template["base_probability"])
        if template["id"] == "patient_summons":
            probability += 0.08 if frustration >= 5.0 else 0.03
        elif template["id"] == "waiting_room_surge" and turn_index >= 3:
            probability += 0.04
        elif template["id"] == "phone_call" and turn_index >= 4:
            probability += 0.03
        probability = max(0.02, min(0.45, probability))
        roll = deterministic_probability(f"{session_id}|{turn_index}|{message}|{template['id']}")
        if roll <= probability:
            candidates.append((roll / max(probability, 1e-6), roll, probability, template))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, roll, probability, template = candidates[0]
    return {**template, "roll": round(roll, 4), "probability": round(probability, 3)}


def apply_random_event(state: ConversationState) -> Dict:
    """Nakłada skutki zdarzenia losowego, zwraca dict zmian stanu."""
    selected = select_random_event(
        session_id=str(state.get("session_id", "")),
        message=str(state.get("current_user_message", "")),
        state=state,
    )
    if not selected:
        return {"current_random_event": None}

    traits = dict(state.get("traits", {}))
    for key, delta in selected.get("traits_delta", {}).items():
        traits[key] = float(traits.get(key, 0.5)) + float(delta)
    from conversation.policy import clamp_traits
    new_traits = clamp_traits(traits)

    new_frustration = round(
        max(0.0, min(10.0, float(state.get("frustration_score", 0.0)) + float(selected.get("frustration_delta", 0.0)))),
        2,
    )

    new_max_turns = state.get("max_turns", 7)
    adjust = int(selected.get("turn_limit_adjust", 0))
    if adjust:
        new_max_turns = max(3, new_max_turns + adjust)
        new_max_turns = max(new_max_turns, int(state.get("turn_index", 0)) + 1)

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

    return {
        "current_random_event": event_payload,
        "traits": new_traits,
        "frustration_score": new_frustration,
        "max_turns": new_max_turns,
        "random_events_history": history[-20:],
        "last_random_event_turn": int(state.get("turn_index", 0)),
    }


# ---------------------------------------------------------------------------
# Knowledge guard — lekarz nie dopowiada danych leku, których nie usłyszał
# ---------------------------------------------------------------------------

def build_sensitive_drug_phrases(drug_info: Dict) -> List[str]:
    candidates: List[str] = []
    for claim in drug_info.get("claims", []) if isinstance(drug_info.get("claims"), list) else []:
        if not isinstance(claim, dict):
            continue
        for key in ("statement", "keywords", "support_patterns", "contradiction_patterns"):
            val = claim.get(key, [])
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list):
                candidates.extend(str(v) for v in val if str(v).strip())
    for key in ("skład", "wskazania", "przeciwwskazania", "działania_niepożądane", "dawkowanie"):
        val = str(drug_info.get(key, "")).strip()
        if val:
            candidates.append(val)

    phrases: List[str] = []
    for raw in candidates:
        norm = normalize_text(raw)
        if not norm:
            continue
        for fragment in re.split(r"[,;().]", norm):
            phrase = fragment.strip()
            if not phrase or phrase in GENERIC_CLINICAL_PHRASES:
                continue
            has_digits = bool(re.search(r"\d", phrase))
            if has_digits or (len(phrase.split()) >= 2 and len(phrase) >= 12):
                phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def pick_hearsay_phrase(session_id: str, turn_index: int) -> str:
    seed = f"{session_id}|{turn_index}|hearsay"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    idx = int(digest[:8], 16) % len(HEARSAY_PHRASES)
    return HEARSAY_PHRASES[idx]


def extract_representative_text_from_messages(messages: List[Dict[str, str]]) -> str:
    """Wyciąga łączony znormalizowany tekst wypowiedzi przedstawiciela z historii LLM."""
    parts = [m["content"] for m in messages if m.get("role") == "user"]
    return normalize_text(" ".join(parts))


def apply_knowledge_guard(
    doctor_message: str,
    messages: List[Dict[str, str]],
    drug_info: Dict,
    session_id: str,
    turn_index: int,
) -> Tuple[str, List[str]]:
    """Przepisuje zdania z danymi leku, których przedstawiciel nie podał, jako zasłyszaną opinię."""
    rep_history = extract_representative_text_from_messages(messages)
    msg_norm = normalize_text(doctor_message)
    if not rep_history or not msg_norm:
        return doctor_message, []

    sensitive = build_sensitive_drug_phrases(drug_info)
    leaked = list(dict.fromkeys(
        p for p in sensitive if p in msg_norm and p not in rep_history
    ))
    if not leaked:
        return doctor_message, []

    hearsay = pick_hearsay_phrase(session_id, turn_index)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doctor_message.strip()) if s.strip()]
    rewritten: List[str] = []
    for sentence in sentences:
        s_norm = normalize_text(sentence)
        if any(p in s_norm for p in leaked):
            rw = f"{hearsay} {sentence[0].lower()}{sentence[1:]}"
            if rw and rw[-1] not in ".!?":
                rw += "."
            rewritten.append(rw)
        else:
            rewritten.append(sentence)

    return " ".join(rewritten).strip(), leaked[:5]


# ---------------------------------------------------------------------------
# Metryki tury — wyjście
# ---------------------------------------------------------------------------

def build_turn_metrics_payload(turn_metrics: Dict, frustration_score: float) -> Dict:
    return {
        "topic_adherence": float(turn_metrics.get("topic_adherence", 0.0)),
        "clinical_precision": float(turn_metrics.get("clinical_precision", 0.0)),
        "ethics": float(turn_metrics.get("ethics", 0.0)),
        "language_quality": float(turn_metrics.get("language_quality", 0.0)),
        "critical_claim_coverage": float(turn_metrics.get("critical_claim_coverage", 0.0)),
        "frustration": round(float(frustration_score), 2),
    }


# ---------------------------------------------------------------------------
# Sprawdzenie zakończenia
# ---------------------------------------------------------------------------

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
