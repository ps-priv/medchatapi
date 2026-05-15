"""Węzły grafu LangGraph — każdy realizuje jeden etap przetwarzania tury."""

import json
import logging
from typing import Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from conversation.claims import (
    check_medical_claims,
    critical_coverage_summary,
    extract_drug_keywords,
    update_claim_coverage,
)
from conversation.constants import PHASE_OBJECTIVES
from conversation.policy import (
    advance_phase,
    apply_reaction_rules,
    build_style_directives,
    clamp_traits,
    compute_frustration,
    compute_turn_metrics,
    evidence_first_requirements,
    policy_postprocess_message,
    policy_precheck,
)
from conversation.schemas import DoctorResponse

from .helpers import (
    apply_knowledge_guard,
    apply_random_event,
    build_turn_metrics_payload,
    check_termination,
    detect_conversation_intent,
    detect_drug_introduction,
    detect_wrong_drug,
    evaluate_conversation_goal,
    forced_goal_payload,
    infer_decision_from_message,
    normalize_doctor_decision,
)
from .state import ConversationState

logger = logging.getLogger(__name__)

EMPTY_TURN_METRICS = {
    "topic_adherence": 0.0, "clinical_precision": 0.0,
    "ethics": 0.0, "language_quality": 0.0, "critical_claim_coverage": 0.0,
}
EMPTY_CLAIM_CHECK = {
    "false_claims": [], "unsupported_claims": [], "supported_claims": [],
    "evidence_signals": [], "claim_matches": [],
    "severity_counts": {"critical": 0, "major": 0, "minor": 0},
    "coverage": {"total_critical": 0, "covered_critical": 0, "missing_critical": 0, "coverage_ratio": 1.0},
    "missing_critical_labels": [], "missing_critical_ids": [],
}


# ---------------------------------------------------------------------------
# Węzeł 1: wykrywanie kontekstu — intencja i przedstawienie leku
# ---------------------------------------------------------------------------

def node_detect_context(state: ConversationState) -> Dict:
    message = str(state.get("current_user_message", ""))
    updates: Dict = {}

    if not state.get("intent_revealed", False):
        hits = detect_conversation_intent(message)
        if hits:
            updates["intent_revealed"] = True
            updates["intent_revealed_turn"] = state.get("turn_index", 0)
            logger.debug("intent_revealed hits=%s", hits[:3])

    if not state.get("drug_revealed", False):
        intro_hits = detect_drug_introduction(message, state.get("drug_intro_keywords", []))
        if intro_hits:
            updates.setdefault("intent_revealed", True)
            updates.setdefault("intent_revealed_turn", state.get("turn_index", 0))
            updates["drug_revealed"] = True
            updates["drug_revealed_turn"] = state.get("turn_index", 0)
            updates["wrong_drug_suspected"] = False  # właściwy lek — resetuj flagę
            logger.debug("drug_revealed hits=%s", intro_hits[:3])
        elif not state.get("wrong_drug_suspected", False):
            if detect_wrong_drug(message, state.get("drug_intro_keywords", [])):
                updates["wrong_drug_suspected"] = True
                logger.info("wrong_drug_suspected session=%s", state.get("session_id", "?"))

    return updates


# ---------------------------------------------------------------------------
# Węzeł 2: analiza wypowiedzi — claimy, pokrycie, sygnały evidence
# ---------------------------------------------------------------------------

def node_analyze(state: ConversationState) -> Dict:
    message = str(state.get("current_user_message", ""))
    drug_info = state.get("drug_info", {})
    doctor_profile = state.get("doctor_profile", {})

    drug_revealed = bool(state.get("drug_revealed", False))
    focus_keywords = extract_drug_keywords(drug_info) if drug_revealed else set(state.get("drug_intro_keywords", []))

    from conversation.policy import analyze_message
    message_analysis = analyze_message(
        message=message,
        drug_info=drug_info,
        doctor_profile=doctor_profile,
        focus_keywords=focus_keywords,
    )

    if drug_revealed:
        claim_check = check_medical_claims(message, drug_info)
        coverage_update = update_claim_coverage(state, claim_check)
        coverage_summary = critical_coverage_summary(state)
    else:
        claim_check = dict(EMPTY_CLAIM_CHECK)
        coverage_update = {"missing_critical_ids": [], "missing_critical_labels": []}
        coverage_summary = {"total_critical": 0, "covered_critical": 0, "missing_critical": 0, "coverage_ratio": 1.0}

    claim_check["coverage"] = coverage_summary
    claim_check["missing_critical_labels"] = coverage_update["missing_critical_labels"]
    claim_check["missing_critical_ids"] = coverage_update["missing_critical_ids"]

    evidence_reqs = evidence_first_requirements(
        message_analysis=message_analysis,
        claim_check=claim_check,
        state=state,
    )

    return {
        "current_analysis": message_analysis,
        "current_claim_check": claim_check,
        "current_coverage_summary": coverage_summary,
        "current_coverage_update": coverage_update,
        "current_evidence_requirements": evidence_reqs,
    }


# ---------------------------------------------------------------------------
# Węzeł 3: policy_check — twarde dyrektywy, detekcja hard-stop
# ---------------------------------------------------------------------------

def node_policy_check(state: ConversationState) -> Dict:
    pre_policy = policy_precheck(
        message_analysis=state.get("current_analysis", {}),
        claim_check=state.get("current_claim_check", {}),
    )
    return {"current_pre_policy": pre_policy}


# ---------------------------------------------------------------------------
# Węzeł 4a: wymuszony stop etyczny
# ---------------------------------------------------------------------------

def node_ethics_stop(state: ConversationState) -> Dict:
    forced_message = "Stanowczo odrzucam taką propozycję. To narusza zasady etyki i kończę tę rozmowę natychmiast."
    forced_goal = forced_goal_payload(
        state=state,
        doctor_decision="reject",
        reason="Nie osiągnięto celu: naruszenie etyki zakończyło rozmowę.",
    )
    turn_index = int(state.get("turn_index", 0))
    new_messages = list(state.get("messages", []))
    new_messages.append({"role": "user", "content": str(state.get("current_user_message", ""))})
    new_messages.append({"role": "assistant", "content": forced_message})

    flags = list(state.get("critical_flags", []))
    flags.append("terminated")

    return {
        "messages": new_messages,
        "current_doctor_response": forced_message,
        "current_doctor_attitude": "serious",
        "current_doctor_decision": "reject",
        "current_reasoning": "Policy pre-check wykrył naruszenie etyki i wymusił zakończenie rozmowy.",
        "current_detected_errors": ["Naruszenie etyki zawodowej."],
        "current_is_terminated": True,
        "current_termination_reason": "Lekarz zakończył rozmowę z powodu propozycji korupcyjnej.",
        "current_conversation_goal": forced_goal,
        "current_turn_metrics_payload": build_turn_metrics_payload(EMPTY_TURN_METRICS, float(state.get("frustration_score", 0.0))),
        "is_terminated": True,
        "phase": "close",
        "critical_flags": flags,
        "latest_goal": forced_goal,
        "goal_status": str(forced_goal["status"]),
        "goal_score": int(forced_goal["score"]),
        "goal_achieved": bool(forced_goal["achieved"]),
    }


# ---------------------------------------------------------------------------
# Węzeł 5: aktualizacja stanu tury — frustracja, faza, metryki, zdarzenia losowe
# ---------------------------------------------------------------------------

def node_update_state(state: ConversationState) -> Dict:
    message_analysis = state.get("current_analysis", {})
    claim_check = state.get("current_claim_check", {})

    from conversation.policy import difficulty_profile
    difficulty_cfg = difficulty_profile(state.get("difficulty", "medium"))
    preferred_strategies = state.get("preferred_strategies", [])

    turn_metrics = compute_turn_metrics(message_analysis, claim_check, state)
    frustration_update = compute_frustration(
        state=state,
        message_analysis=message_analysis,
        claim_check=claim_check,
        metrics=turn_metrics,
        difficulty_cfg=difficulty_cfg,
        preferred_strategies=preferred_strategies,
    )
    new_frustration = frustration_update["total"]

    phase_update = advance_phase(
        current_phase=state.get("phase", "opening"),
        state={**state, "frustration_score": new_frustration},
        message_analysis=message_analysis,
        metrics=turn_metrics,
    )

    turn_log = {
        "turn": state.get("turn_index", 0),
        "phase": phase_update["phase"],
        "metrics": turn_metrics,
        "frustration": frustration_update,
        "critical_claim_coverage": claim_check.get("coverage", {}),
        "missing_critical_labels": claim_check.get("missing_critical_labels", [])[:3],
    }
    history = list(state.get("turn_metrics_history", []))
    history.append(turn_log)

    # Zastosuj zdarzenia losowe
    random_updates = apply_random_event({**state, "frustration_score": new_frustration, "phase": phase_update["phase"]})

    final_frustration = random_updates.get("frustration_score", new_frustration)
    final_traits = random_updates.get("traits", state.get("traits", {}))
    final_max_turns = random_updates.get("max_turns", state.get("max_turns", 10))

    return {
        "current_turn_metrics": turn_metrics,
        "current_frustration_update": frustration_update,
        "frustration_score": final_frustration,
        "phase": phase_update["phase"],
        "last_phase_reason": phase_update["reason"],
        "turn_metrics_history": history[-20:],
        "traits": final_traits,
        "max_turns": final_max_turns,
        "current_random_event": random_updates.get("current_random_event"),
        "random_events_history": random_updates.get("random_events_history", state.get("random_events_history", [])),
        "last_random_event_turn": random_updates.get("last_random_event_turn", state.get("last_random_event_turn", -999)),
    }


# ---------------------------------------------------------------------------
# Węzeł 4b: wymuszony stop — przekroczenie limitu tur
# ---------------------------------------------------------------------------

def node_time_stop(state: ConversationState) -> Dict:
    max_turns = int(state.get("max_turns", 10))
    turn_metrics = state.get("current_turn_metrics", EMPTY_TURN_METRICS)
    forced_message = "Kończę spotkanie, nie mam już czasu. Jeśli to możliwe, proszę przesłać najważniejsze informacje o leku pisemnie."
    termination_reason = f"Lekarz zakończył rozmowę z powodu presji czasu (limit tur: {max_turns})."
    forced_goal = forced_goal_payload(state=state, doctor_decision="undecided", reason="Nie osiągnięto celu: rozmowa ucięta przez presję czasu.")

    new_messages = list(state.get("messages", []))
    new_messages.append({"role": "user", "content": str(state.get("current_user_message", ""))})
    new_messages.append({"role": "assistant", "content": forced_message})

    flags = list(state.get("critical_flags", []))
    flags.append("time_limit_exceeded")

    new_traits = clamp_traits({
        **state.get("traits", {}),
        "patience": float(state.get("traits", {}).get("patience", 0.5)) - 0.2,
        "openness": float(state.get("traits", {}).get("openness", 0.5)) - 0.15,
        "time_pressure": max(float(state.get("traits", {}).get("time_pressure", 0.5)), 0.9),
    })

    return {
        "messages": new_messages,
        "traits": new_traits,
        "current_doctor_response": forced_message,
        "current_doctor_attitude": "serious",
        "current_doctor_decision": forced_goal["doctor_decision"],
        "current_reasoning": "Przekroczono dopuszczalną liczbę tur wynikającą z presji czasu.",
        "current_detected_errors": [],
        "current_is_terminated": True,
        "current_termination_reason": termination_reason,
        "current_conversation_goal": forced_goal,
        "current_turn_metrics_payload": build_turn_metrics_payload(turn_metrics, float(state.get("frustration_score", 0.0))),
        "is_terminated": True,
        "phase": "close",
        "critical_flags": flags,
        "latest_goal": forced_goal,
        "goal_status": str(forced_goal["status"]),
        "goal_score": int(forced_goal["score"]),
        "goal_achieved": bool(forced_goal["achieved"]),
    }


# ---------------------------------------------------------------------------
# Węzeł 6: budowanie dyrektyw zachowania lekarza
# ---------------------------------------------------------------------------

def node_build_directives(state: ConversationState) -> Dict:
    message_analysis = state.get("current_analysis", {})
    claim_check = state.get("current_claim_check", {})
    coverage_update = state.get("current_coverage_update", {})
    evidence_reqs = state.get("current_evidence_requirements", {})
    random_event = state.get("current_random_event")
    pre_policy = state.get("current_pre_policy", {})

    style_runtime = build_style_directives(
        doctor_profile=state.get("doctor_profile", {}),
        traits=state.get("traits", {}),
        state=state,
    )

    intent_revealed = bool(state.get("intent_revealed", False))
    drug_revealed = bool(state.get("drug_revealed", False))
    wrong_drug = bool(state.get("wrong_drug_suspected", False))
    missing_labels = coverage_update.get("missing_critical_labels", [])

    # Dyrektywy stylu i policy — bez modyfikacji
    directives = []
    directives.extend(style_runtime["directives"])
    directives.extend(pre_policy.get("directives", []))

    # Dyrektywa kontekstu wizyty — tylko jedna, priorytetyzowana
    if wrong_drug and not drug_revealed:
        directives.append(
            "WAŻNE: Rozmówca mówi o leku spoza obszaru Twoich zainteresowań. "
            "Wyraź brak zainteresowania tym lekiem i zasugeruj zakończenie wizyty."
        )
    elif not intent_revealed:
        directives.append("Zareaguj naturalnie — nie wiesz po co ta osoba przyszła. Zapytaj o cel wizyty.")
    elif not drug_revealed:
        directives.append("Rozmówca ujawnił cel medyczny, ale nie podał tematu. Poproś o jeden konkret.")

    # Zdarzenie losowe
    if random_event:
        event_directive = str(random_event.get("directive", "")).strip()
        if event_directive:
            directives.append(event_directive)

    # Reakcje na jakość wypowiedzi — łączone w jedną dyrektywę gdy kilka naraz
    quality_issues = []
    if message_analysis.get("off_topic_hits") and not message_analysis.get("has_drug_focus"):
        quality_issues.append("rozmowa zbacza z tematu")
    if message_analysis.get("marketing_hits") and drug_revealed:
        quality_issues.append("slogany marketingowe bez danych")
    if message_analysis.get("english_hits") and drug_revealed:
        quality_issues.append("nadużycie anglicyzmów")
    if quality_issues:
        directives.append(f"Wykryto: {', '.join(quality_issues)}. Zareaguj krótko i wróć do tematu.")

    # Pytanie o brakujące claimy — jako ostatnia dyrektywa, tylko gdy drug revealed
    if missing_labels and drug_revealed:
        first_missing = missing_labels[0]
        directives.append(f"Zadaj jedno pytanie o brakujący temat: {first_missing}.")

    # Dyrektywa evidence-first — tylko gdy nie ma już pytania kontekstowego
    if not wrong_drug and intent_revealed:
        directives.extend(evidence_reqs.get("directives", []))

    return {
        "current_behavior_directives": directives,
        "current_style_runtime": style_runtime,
    }


# ---------------------------------------------------------------------------
# Węzeł 6b: retrieval kontekstu RAG z Chroma
# ---------------------------------------------------------------------------

def node_retrieve_context(state: ConversationState, config: RunnableConfig) -> Dict:
    """Pobiera fragmenty ChPL z Chroma relevantne dla bieżącej wypowiedzi."""
    from .rag import get_rag

    api_key = config.get("configurable", {}).get("api_key", "")
    drug_id = str(state.get("drug_info", {}).get("id", ""))
    message = str(state.get("current_user_message", ""))

    if not drug_id or not message:
        return {"current_rag_context": []}

    rag = get_rag(api_key)
    if not rag.has_drug(drug_id):
        return {"current_rag_context": []}

    query = f"{message} {state.get('drug_info', {}).get('nazwa', '')}"
    chunks = rag.retrieve_context(drug_id=drug_id, query=query, k=3)
    logger.debug("RAG retrieved %d chunks for drug=%s", len(chunks), drug_id)
    return {"current_rag_context": chunks}


# ---------------------------------------------------------------------------
# Węzeł 7: wywołanie LLM — generowanie odpowiedzi lekarza
# ---------------------------------------------------------------------------

def _rag_section(state: ConversationState) -> str:
    """Buduje sekcję RAG dla system promptu jeśli są dostępne fragmenty ChPL."""
    chunks = state.get("current_rag_context", [])
    if not chunks:
        return ""
    joined = "\n---\n".join(chunks)
    return f"""
Poniżej fragmenty oficjalnej dokumentacji leku (ChPL) relevantne do bieżącej wypowiedzi.
Użyj ich jako wiedzy bazowej przy ocenie twierdzeń przedstawiciela — ale TYLKO w trybie zasłyszanej opinii:
{joined}
"""


def _build_system_prompt(state: ConversationState) -> str:
    """Buduje czysty, naturalny prompt systemowy lekarza."""
    doctor = state.get("doctor_profile", {})
    traits = state.get("traits", {})
    message_analysis = state.get("current_analysis", {})
    claim_check = state.get("current_claim_check", {})
    turn_metrics = state.get("current_turn_metrics", {})
    coverage_summary = state.get("current_coverage_summary", {})
    style_runtime = state.get("current_style_runtime", {})
    behavior_directives = state.get("current_behavior_directives", [])
    preferred_strategies = state.get("preferred_strategies", [])
    random_event = state.get("current_random_event")
    intent_revealed = bool(state.get("intent_revealed", False))
    drug_revealed = bool(state.get("drug_revealed", False))

    random_event_line = (
        "brak" if not random_event
        else f"{random_event['event_name']}: {random_event['event_details']}"
    )

    wrong_drug_suspected = bool(state.get("wrong_drug_suspected", False))

    # --- Sekcja wiedzy o leku — tylko gdy cel wizyty jest już ujawniony ---
    if wrong_drug_suspected and not drug_revealed:
        drug_section = ""
        claim_section = ""
        rag_section = ""
        phase_objective = "Rozmówca mówi o leku, który nie jest w obszarze Twoich zainteresowań. Wyraź brak zainteresowania tym lekiem."
        visitor_context = "WAŻNE: Rozmówca przedstawia lek, o którym nie masz wiedzy i którym nie jesteś zainteresowana. Nie masz czasu na dyskusję o leku spoza swojego obszaru zainteresowań. Powiedz wyraźnie, że nie jesteś zainteresowana tym preparatem i zasugeruj zakończenie wizyty."
    elif not intent_revealed:
        drug_section = ""
        claim_section = ""
        rag_section = ""
        phase_objective = "Do gabinetu weszła osoba — nie wiesz jeszcze kim jest ani po co przyszła. Zapytaj uprzejmie w czym możesz pomóc. Nie sugeruj żadnego tematu rozmowy."
        visitor_context = "WAŻNE: Nie wiesz kim jest ta osoba — może to pacjent, student, kolega, przedstawiciel handlowy lub ktoś inny. Nie zakładaj z góry żadnej roli ani tematu wizyty."
    elif not drug_revealed:
        drug_section = "Rozmówca ujawnił, że ma cel medyczny/zawodowy, ale nie podał jeszcze konkretnego tematu.\n"
        claim_section = ""
        rag_section = ""
        phase_objective = "Rozmówca ujawnił cel wizyty o charakterze medycznym, ale nie podał tematu. Zapytaj o szczegóły."
        visitor_context = ""
    else:
        drug_knowledge = "Znasz tylko informacje podane przez rozmówcę w tej rozmowie. Własną wiedzę przytaczaj wyłącznie jako zasłyszaną opinię."
        claim_view = {
            "false_count": len(claim_check.get("false_claims", [])),
            "unsupported_count": len(claim_check.get("unsupported_claims", [])),
            "supported_count": len(claim_check.get("supported_claims", [])),
            "coverage_critical": f"{coverage_summary.get('covered_critical', 0)}/{coverage_summary.get('total_critical', 0)}",
        }
        drug_section = f"Wiedza o leku: {drug_knowledge}\n"
        claim_section = f"Weryfikacja twierdzeń: {json.dumps(claim_view, ensure_ascii=False)}\nPokrycie claimów krytycznych: {coverage_summary.get('covered_critical', 0)}/{coverage_summary.get('total_critical', 0)}\n"
        rag_section = _rag_section(state)
        phase_objective = PHASE_OBJECTIVES.get(state.get("phase", "opening"), "")
        visitor_context = ""

    # --- Zasady kontekstowe ---
    if not intent_revealed:
        rules = """Zasady:
1. Odpowiadaj wyłącznie po polsku.
2. Nie wiesz z kim rozmawiasz ani po co ta osoba przyszła — nie sugeruj żadnego tematu.
3. Zareaguj naturalnie i krótko zapytaj w czym możesz pomóc.
4. Na próby korupcji reaguj stanowczo i zakończ rozmowę.
5. Jeśli rozmówca używa niepoprawnej formy grzecznościowej, skoryguj go.
6. W doctor_attitude użyj: happy, neutral, serious, sad.
7. W doctor_decision użyj: undecided (dopóki nie poznasz celu wizyty).
8. Nigdy nie przedstawiaj się słowami "Jestem pani doktor". Reaguj naturalnie."""
    else:
        rules = """Zasady:
1. Odpowiadaj wyłącznie po polsku.
2. MAKSYMALNIE JEDNO PYTANIE na odpowiedź. Jeśli masz kilka kwestii do wyjaśnienia, wybierz najważniejszą i zapytaj tylko o nią.
3. Trzymaj rozmowę przy temacie wizyty: wskazania, przeciwwskazania, działania niepożądane, dawkowanie.
4. Jeśli rozmówca poda niezgodne informacje, wpisz to do detected_errors i obniż openness.
5. Na próby korupcji reaguj stanowczo i zakończ rozmowę.
6. Na slogany i anglicyzmy reaguj krytycznie i proś o jeden konkretny fakt.
7. Aktywnie stosuj styl komunikacji, cechy psychologiczne i strategie lekarza.
8. Jeśli to końcowa faza budżetu tur, zamknij rozmowę.
9. Jeśli rozmówca używa niepoprawnej formy grzecznościowej, skoryguj go.
10. Jeśli brakuje claimów krytycznych, zadaj jedno celne pytanie uzupełniające — nie wyliczankę.
11. W doctor_attitude użyj: happy, neutral, serious, sad.
12. Tryb evidence-first: nie akceptuj tez bez danych — zapytaj o jedno konkretne potwierdzenie.
13. W doctor_decision użyj: undecided, trial_use, will_prescribe, recommend, reject.
14. Własną wiedzę o leku przytaczaj wyłącznie jako zasłyszaną opinię (np. "Słyszałam od koleżanki, że..."). Nigdy jako potwierdzone fakty.
15. Nigdy nie przedstawiaj się słowami "Jestem pani doktor". Reaguj naturalnie."""

    return f"""Jesteś lekarzem. Profil: {doctor.get('name', '')} ({doctor.get('description', '')})
Kontekst sytuacyjny: {doctor.get('context_str', doctor.get('description', ''))}
Płeć lekarza: {message_analysis.get('doctor_gender', 'female')}. Oczekiwana forma zwracania się: {message_analysis.get('expected_address', 'pani doktor')}.
Poziom trudności rozmowy: {state.get('difficulty', 'medium')}.
Preferowane strategie lekarza: {json.dumps(preferred_strategies, ensure_ascii=False)}.
Styl komunikacji: {doctor.get('communication_style', 'profesjonalny')}
Aktualne cechy psychologiczne: {json.dumps(traits, ensure_ascii=False)}
Aktualna faza rozmowy: {state.get('phase', 'opening')}
Cel bieżącej fazy: {phase_objective}
Numer tury: {state.get('turn_index', 0)}
Budżet rozmowy: maksymalnie {state.get('max_turns', 10)} tur, pozostało {style_runtime.get('remaining_turns', 5)} tur.
Frustracja lekarza (0-10): {state.get('frustration_score', 0.0)}
Zdarzenie losowe tej tury: {random_event_line}
{visitor_context}
{drug_section}{rag_section}{claim_section}
Wykryte sygnały w ostatniej wypowiedzi:
{json.dumps(message_analysis, ensure_ascii=False)}
Metryki tej tury:
{json.dumps(turn_metrics, ensure_ascii=False)}

Dodatkowe dyrektywy:
{json.dumps(behavior_directives, ensure_ascii=False)}

{rules}
"""


def node_generate_response(state: ConversationState, config: RunnableConfig) -> Dict:
    """Wywołuje LLM — generuje ustrukturyzowaną odpowiedź lekarza."""
    api_key = config.get("configurable", {}).get("api_key", "")
    model = config.get("configurable", {}).get("model", "gpt-4o")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.7,
    ).with_structured_output(DoctorResponse)

    system_prompt = _build_system_prompt(state)
    lc_messages = [SystemMessage(content=system_prompt)]

    # Dołącz historię rozmowy z wcześniejszych tur
    for msg in state.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    # Bieżąca wiadomość użytkownika
    lc_messages.append(HumanMessage(content=str(state.get("current_user_message", ""))))

    logger.info(
        "LLM call session=%s turn=%s model=%s",
        state.get("session_id", "?"), state.get("turn_index", 0), model,
    )
    ai_response: DoctorResponse = llm.invoke(lc_messages)

    return {"_raw_ai_response": ai_response}


# ---------------------------------------------------------------------------
# Węzeł 8: finalizacja — postprocessing, ocena celu, aktualizacja traits, zakończenie
# ---------------------------------------------------------------------------

def node_finalize(state: ConversationState) -> Dict:  # noqa: C901
    """Post-processuje odpowiedź LLM, ocenia cel i finalizuje turę."""
    ai_response: DoctorResponse = state.get("_raw_ai_response")  # type: ignore[assignment]
    message_analysis = state.get("current_analysis", {})
    claim_check = state.get("current_claim_check", {})
    coverage_summary = state.get("current_coverage_summary", {})
    coverage_update = state.get("current_coverage_update", {})
    evidence_reqs = state.get("current_evidence_requirements", {})
    turn_metrics = state.get("current_turn_metrics", EMPTY_TURN_METRICS)
    frustration_update = state.get("current_frustration_update", {"delta": 0.0, "total": 0.0})
    style_runtime = state.get("current_style_runtime", {"max_sentences": 3, "remaining_turns": 5, "directives": []})
    random_event = state.get("current_random_event")

    # --- Zbierz błędy wykryte przez model + reguły ---
    final_errors = list(ai_response.detected_errors if ai_response else [])
    if message_analysis.get("marketing_hits"):
        final_errors.append("Użycie sloganów marketingowych bez danych klinicznych.")
    if message_analysis.get("english_hits"):
        final_errors.append("Nadużycie anglicyzmów zamiast precyzyjnego języka polskiego.")
    if message_analysis.get("gender_mismatch_hits"):
        final_errors.append(f"Niepoprawna forma zwracania się (oczekiwane: {message_analysis.get('expected_address', '')}).")
    if message_analysis.get("inappropriate_hits"):
        final_errors.append("Nieelegancka propozycja naruszająca profesjonalny charakter rozmowy.")
    if message_analysis.get("disrespect_hits"):
        final_errors.append("Brak szacunku wobec lekarza i nieprofesjonalny ton.")
    final_errors.extend(claim_check.get("false_claims", []))
    final_errors.extend(claim_check.get("unsupported_claims", []))
    missing_labels = coverage_update.get("missing_critical_labels", [])
    turn_index = int(state.get("turn_index", 0))
    max_turns = int(state.get("max_turns", 10))
    if coverage_summary.get("missing_critical", 0) > 0 and turn_index >= max(2, max_turns // 2):
        final_errors.append(f"Nie omówiono kluczowych claimów leku: {'; '.join(missing_labels[:2])}.")
    if evidence_reqs.get("require_verification"):
        final_errors.append("Brak wiarygodnej weryfikacji twierdzeń w trybie evidence-first.")
    elif evidence_reqs.get("require_probe"):
        final_errors.append("Brak konkretu klinicznego w trybie evidence-first.")
    final_errors = list(dict.fromkeys(final_errors))

    # --- Post-processing odpowiedzi przez policy ---
    raw_message = ai_response.doctor_message if ai_response else ""
    doctor_message = policy_postprocess_message(
        raw_message=raw_message,
        style_runtime=style_runtime,
        message_analysis=message_analysis,
        claim_check=claim_check,
        phase=state.get("phase", "opening"),
        missing_critical_labels=missing_labels,
        evidence_requirements=evidence_reqs,
        drug_revealed=bool(state.get("drug_revealed", False)),
        intent_revealed=bool(state.get("intent_revealed", False)),
    )

    # --- Knowledge guard ---
    leaked: list = []
    if state.get("drug_revealed", False):
        doctor_message, leaked = apply_knowledge_guard(
            doctor_message=doctor_message,
            messages=state.get("messages", []),
            drug_info=state.get("drug_info", {}),
            session_id=str(state.get("session_id", "")),
            turn_index=turn_index,
        )

    # --- Uzasadnienie ---
    reasoning = ai_response.reasoning if ai_response else ""
    if message_analysis.get("marketing_hits") or message_analysis.get("english_hits"):
        reasoning += " Lekarz obniża zaufanie przy języku marketingowym i nieprecyzyjnych anglicyzmach."
    if message_analysis.get("gender_mismatch_hits"):
        reasoning += " Wykryto niewłaściwą formę grzecznościową."
    if style_runtime.get("directives"):
        reasoning += " Zastosowano aktywne reguły stylu i cech lekarza."
    if evidence_reqs.get("require_verification"):
        reasoning += " Aktywny tryb evidence-first wymusił weryfikację i żądanie źródeł danych."
    elif evidence_reqs.get("require_probe"):
        reasoning += " Aktywny tryb evidence-first wymusił dopytanie o dane kliniczne."
    if random_event:
        reasoning += f" {random_event.get('reasoning_note', '')}"
    if coverage_summary.get("missing_critical", 0) > 0:
        reasoning += f" Brakuje pokrycia claimów krytycznych: {coverage_summary.get('covered_critical', 0)}/{coverage_summary.get('total_critical', 0)}."
    if leaked:
        reasoning += " Aktywowano filtr wiedzy: usunięto szczegóły leku, których przedstawiciel nie podał."

    # --- Nastawienie lekarza ---
    ai_attitude = ai_response.doctor_attitude if ai_response else "neutral"
    final_attitude = ai_attitude
    if len(final_errors) >= 2:
        final_attitude = "serious"
    elif style_runtime.get("remaining_turns", 5) <= 1 and float(state.get("traits", {}).get("time_pressure", 0.5)) >= 0.7:
        final_attitude = "serious"
    elif float(state.get("frustration_score", 0.0)) >= 6.0:
        final_attitude = "serious"

    # --- Decyzja lekarza ---
    ai_decision = normalize_doctor_decision(getattr(ai_response, "doctor_decision", "undecided"))
    if ai_decision == "undecided":
        ai_decision = infer_decision_from_message(doctor_message)

    # --- Ocena celu rozmowy ---
    conversation_goal = evaluate_conversation_goal(
        state=state,
        turn_metrics=turn_metrics,
        claim_check=claim_check,
        coverage_summary=coverage_summary,
        evidence_requirements=evidence_reqs,
        doctor_decision=ai_decision,
        doctor_attitude=final_attitude,
    )

    # --- Aktualizacja traits ---
    llm_traits = ai_response.updated_traits.model_dump() if ai_response else dict(state.get("traits", {}))
    updated_traits = apply_reaction_rules(
        current_traits=state.get("traits", {}),
        llm_traits=llm_traits,
        analysis=message_analysis,
        detected_errors_count=len(final_errors),
        frustration_delta=float(frustration_update.get("delta", 0.0)),
        frustration_total=float(frustration_update.get("total", 0.0)),
    )

    # --- Sprawdzenie zakończenia ---
    is_terminated, termination_reason = check_termination(state={**state, "traits": updated_traits}, message_analysis=message_analysis)
    if not is_terminated and bool(conversation_goal.get("achieved")):
        is_terminated = True
        termination_reason = "Cel rozmowy osiągnięty: lekarz deklaruje pozytywną decyzję wobec leku."

    flags = list(state.get("critical_flags", []))
    if is_terminated:
        flags.append("terminated")

    # --- Aktualizacja historii wiadomości ---
    new_messages = list(state.get("messages", []))
    new_messages.append({"role": "user", "content": str(state.get("current_user_message", ""))})
    new_messages.append({"role": "assistant", "content": doctor_message})

    # --- Aktualizacja celu ---
    goal_achieved_turn = int(state.get("goal_achieved_turn", 0))
    if bool(conversation_goal.get("achieved")) and goal_achieved_turn == 0:
        goal_achieved_turn = turn_index

    return {
        "messages": new_messages,
        "traits": updated_traits,
        "is_terminated": is_terminated,
        "phase": "close" if is_terminated else state.get("phase", "opening"),
        "critical_flags": flags,
        "latest_goal": conversation_goal,
        "goal_status": str(conversation_goal["status"]),
        "goal_score": int(conversation_goal["score"]),
        "goal_achieved": bool(state.get("goal_achieved", False) or conversation_goal.get("achieved", False)),
        "goal_achieved_turn": goal_achieved_turn,
        "current_doctor_response": doctor_message,
        "current_doctor_attitude": final_attitude,
        "current_doctor_decision": conversation_goal["doctor_decision"],
        "current_reasoning": reasoning,
        "current_detected_errors": final_errors,
        "current_is_terminated": is_terminated,
        "current_termination_reason": termination_reason,
        "current_conversation_goal": conversation_goal,
        "current_turn_metrics_payload": build_turn_metrics_payload(turn_metrics, float(state.get("frustration_score", 0.0))),
    }


# ---------------------------------------------------------------------------
# Routing — decyduje o kolejnym węźle
# ---------------------------------------------------------------------------

def route_after_policy(state: ConversationState) -> str:
    """Po policy_check: hard_stop → ethics_stop, else → update_state."""
    if state.get("current_pre_policy", {}).get("hard_stop", False):
        return "ethics_stop"
    return "update_state"


def route_after_update(state: ConversationState) -> str:
    """Po update_state: czas minął → time_stop, else → build_directives."""
    if int(state.get("turn_index", 0)) > int(state.get("max_turns", 10)):
        return "time_stop"
    return "build_directives"
