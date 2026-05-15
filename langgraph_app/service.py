"""Warstwa serwisowa — zarządzanie sesjami i orkiestracja grafu LangGraph."""

import json
import logging
import uuid
from typing import Dict, Optional

from fastapi import HTTPException

from conversation.constants import CHAT_COMMANDS
from conversation.data import get_doctor_by_id, get_drug_by_id
from conversation.policy import clamp_traits
from conversation.schemas import EvaluationResult, MessageRequest, SessionConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .agenda_generator import generate_agenda
from .graph import compiled_graph
from .helpers import build_initial_state, forced_goal_payload
from .rag import get_rag
from .state import ConversationState
from .supabase_service import SupabaseService

_supabase = SupabaseService()

logger = logging.getLogger(__name__)

# Globalny magazyn sesji: session_id → ConversationState
sessions: Dict[str, ConversationState] = {}


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------

def _load_drug(drug_id: str, api_key: str) -> Dict:
    """Ładuje dane leku: RAG catalog ma priorytet, fallback to drugs.json."""
    rag = get_rag(api_key)
    if rag.has_drug(drug_id):
        drug_info = rag.get_drug(drug_id)
        logger.info("drug loaded from RAG catalog: %s", drug_id)
        return drug_info  # type: ignore[return-value]
    drug_info = get_drug_by_id(drug_id)
    if drug_info:
        logger.info("drug loaded from drugs.json: %s", drug_id)
    return drug_info  # type: ignore[return-value]


def start_session(
    doctor_id: str,
    drug_id: str,
    api_key: str = "",
    session_config: Optional[SessionConfig] = None,
) -> Dict:
    """Inicjalizuje sesję i zwraca odpowiedź dla endpointu /start.

    Parametry:
        doctor_id: ID archetypu lekarza
        drug_id: ID leku
        api_key: klucz OpenAI (dla RAG)
        session_config: opcjonalna konfiguracja sesji (familiarity, register, warmth, itp.).
            Brak (None) = wartości domyślne, zachowuje 100% kompatybilności wstecznej
            ze starszymi klientami API wywołującymi /start bez body.
    """
    doctor_profile = get_doctor_by_id(doctor_id)
    if not doctor_profile:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono lekarza o id: {doctor_id}")

    drug_info = _load_drug(drug_id, api_key)
    if not drug_info:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono leku o id: {drug_id}")

    session_id = str(uuid.uuid4())
    state = build_initial_state(
        doctor_profile=doctor_profile,
        drug_info=drug_info,
        session_id=session_id,
        session_config=session_config,
    )

    # Generuj agendę lekarza (1 wywołanie LLM ~0.001 USD) — przy błędzie lista pusta
    doctor_agenda = generate_agenda(
        doctor_profile=doctor_profile,
        drug_info=drug_info,
        familiarity=state.get("familiarity", "first_meeting"),
        prior_visits_summary=state.get("prior_visits_summary"),
        api_key=api_key,
        model="gpt-4o-mini",
    )
    state = {**state, "doctor_agenda": doctor_agenda}

    sessions[session_id] = state

    # Logujemy konfigurację - przyda się przy debugowaniu różnych scenariuszy treningowych
    logger.info(
        "start_session id=%s doctor=%s drug=%s familiarity=%s register=%s warmth=%s",
        session_id, doctor_id, drug_id,
        state.get("familiarity"), state.get("register"), state.get("warmth"),
    )

    response: Dict = {
        "session_id": session_id,
        "status": "Rozmowa rozpoczęta. Lekarz gotowy i czeka na cel wizyty.",
        "traits": state["traits"],
    }
    # session_config jest opcjonalne w odpowiedzi - dodajemy tylko gdy klient go podał,
    # żeby starsi klienci dostawali odpowiedź dokładnie taką jak wcześniej.
    if session_config is not None:
        response["session_config"] = session_config.model_dump(mode="json")
    return response


# ---------------------------------------------------------------------------
# process_message
# ---------------------------------------------------------------------------

def _handle_chat_command(session_id: str, message: str, api_key: str, model: str) -> Dict | None:
    """Obsługuje specjalne komendy czatu; zwraca odpowiedź lub None."""
    action = CHAT_COMMANDS.get(message.strip())
    if not action:
        return None

    state = sessions[session_id]

    if action == "close_visit":
        forced_message = "Stanowczo odrzucam taką propozycję. To narusza zasady etyki i kończę tę rozmowę natychmiast."
        forced_goal = forced_goal_payload(
            state=state,
            doctor_decision="reject",
            reason="Naruszenie etyki zakończyło rozmowę.",
        )
        flags = list(state.get("critical_flags", []))
        flags.append("terminated")
        sessions[session_id] = {
            **state,
            "is_terminated": True,
            "phase": "close",
            "critical_flags": flags,
            "latest_goal": forced_goal,
            "goal_status": "not_achieved",
        }
        return {
            "doctor_message": forced_message,
            "updated_traits": state["traits"],
            "reasoning": "Komenda close_visit: natychmiastowe zakończenie rozmowy.",
            "turn_metrics": {"topic_adherence": 0.0, "clinical_precision": 0.0, "ethics": 0.0, "language_quality": 0.0, "critical_claim_coverage": 0.0, "frustration": float(state.get("frustration_score", 0.0))},
            "doctor_attitude": "serious",
            "doctor_decision": "reject",
            "conversation_goal": forced_goal,
            "is_terminated": True,
            "termination_reason": "Lekarz zakończył rozmowę z powodu propozycji korupcyjnej.",
        }

    if action == "increase_openness":
        traits = dict(state.get("traits", {}))
        traits["openness"] = min(1.0, float(traits.get("openness", 0.5)) + 0.3)
        sessions[session_id] = {**state, "traits": clamp_traits(traits)}
        logger.info("session=%s command=increase_openness openness=%.2f", session_id, traits["openness"])

    return None


def process_message(req: MessageRequest, api_key: str, model: str) -> Dict:
    """Obsługuje jedną turę rozmowy przez graf LangGraph."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji")

    state = sessions[req.session_id]

    if state.get("is_terminated"):
        return {"error": "Rozmowa została już zakończona przez lekarza. Wywołaj /finish, aby uzyskać ocenę."}

    # Obsługa komend specjalnych
    command_result = _handle_chat_command(req.session_id, req.message, api_key, model)
    if command_result:
        return command_result

    # Inkrementuj turę
    new_turn_index = int(state.get("turn_index", 0)) + 1
    state = {**state, "turn_index": new_turn_index, "current_user_message": req.message}

    logger.info(
        "process_message session=%s turn=%d/%d",
        req.session_id, new_turn_index, int(state.get("max_turns", 10)),
    )

    # Wywołaj graf
    config = {"configurable": {"api_key": api_key, "model": model}}
    result_state: ConversationState = compiled_graph.invoke(state, config=config)  # type: ignore[assignment]

    # Wyczyść pole robocze raw response przed zapisem
    result_state["_raw_ai_response"] = None  # type: ignore[typeddict-item]

    sessions[req.session_id] = result_state

    return {
        "doctor_message": result_state.get("current_doctor_response", ""),
        "updated_traits": result_state.get("traits", {}),
        "reasoning": result_state.get("current_reasoning", ""),
        "turn_metrics": result_state.get("current_turn_metrics_payload", {}),
        "doctor_attitude": result_state.get("current_doctor_attitude", "neutral"),
        "doctor_decision": result_state.get("current_doctor_decision", "undecided"),
        "conversation_goal": result_state.get("current_conversation_goal", {}),
        "is_terminated": bool(result_state.get("current_is_terminated", False)),
        "termination_reason": result_state.get("current_termination_reason"),
        "conviction": result_state.get("conviction"),
    }


# ---------------------------------------------------------------------------
# finish_session
# ---------------------------------------------------------------------------

def finish_session(session_id: str, api_key: str, model: str) -> Dict:
    """Kończy sesję i generuje ocenę końcową przez LLM."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji")

    state = sessions[session_id]

    # Odtwórz historię rozmowy z messages
    history_list = []
    history_text = ""
    for msg in state.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            speaker = "Przedstawiciel"
        elif role == "assistant":
            speaker = "Lekarz"
        else:
            continue
        history_list.append({"speaker": speaker, "text": content})
        history_text += f"[{speaker}]: {content}\n"

    evaluator_prompt = f"""
Jesteś doświadczonym trenerem sprzedaży medycznej.

Profil lekarza:
{json.dumps(state.get('doctor_profile', {}), ensure_ascii=False)}

Informacje o leku:
{json.dumps(state.get('drug_info', {}), ensure_ascii=False)}

Zapis rozmowy:
{history_text}

Metryki i przebieg algorytmu:
{json.dumps(state.get('turn_metrics_history', []), ensure_ascii=False)}
Ostatnia faza rozmowy: {state.get('phase', 'unknown')}
Końcowy poziom frustracji lekarza (0-10): {state.get('frustration_score', 'unknown')}

Oceń przedstawiciela pod kątem:
- profesjonalizmu,
- trafności argumentów dot. leku,
- budowania relacji i radzenia sobie z trudnym lekarzem,
- utrzymania rozmowy w temacie leku,
- zgodności etycznej (bez prób niedozwolonych korzyści).
"""

    llm = ChatOpenAI(model=model, api_key=api_key).with_structured_output(EvaluationResult)
    messages = [
        SystemMessage(content="Jesteś surowym, ale sprawiedliwym trenerem biznesu."),
        HumanMessage(content=evaluator_prompt),
    ]
    evaluation: EvaluationResult = llm.invoke(messages)

    latest_goal = state.get("latest_goal") or forced_goal_payload(
        state=state,
        doctor_decision="undecided",
        reason="Brak tury decyzyjnej zakończonej oceną celu.",
    )

    evaluation_dict = evaluation.model_dump()

    _supabase.save_conversation(
        session_id=session_id,
        doctor_profile=state.get("doctor_profile", {}),
        drug_info=state.get("drug_info", {}),
        conversation_history=history_list,
        evaluation=evaluation_dict,
        conversation_goal=latest_goal,
        turn_metrics_history=state.get("turn_metrics_history", []),
        final_traits=state.get("traits", {}),
        frustration_score=float(state.get("frustration_score", 0.0)),
        phase=state.get("phase", "unknown"),
        turn_count=int(state.get("turn_index", 0)),
        is_terminated=bool(state.get("is_terminated", False)),
        critical_flags=list(state.get("critical_flags", [])),
    )

    del sessions[session_id]

    return {
        "status": "Rozmowa zakończona, sesja usunięta.",
        "conversation_history": history_list,
        "conversation_goal": latest_goal,
        "evaluation": evaluation_dict,
    }
