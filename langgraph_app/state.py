"""Definicja stanu konwersacji zarządzanego przez LangGraph."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ConversationState(TypedDict, total=False):
    """Pełny stan sesji rozmowy lekarz-przedstawiciel.

    Pola ``current_*`` są nadpisywane w każdej turze i nie akumulują historii.
    Pola bez prefixu persystują przez całą sesję.
    """

    # Historia wiadomości dla LLM — lista dict {role, content}
    messages: List[Dict[str, str]]

    # Stały kontekst sesji
    session_id: str
    doctor_profile: Dict[str, Any]
    drug_info: Dict[str, Any]

    # Cechy psychologiczne lekarza (ewoluują)
    traits: Dict[str, float]

    # Postęp rozmowy
    turn_index: int
    max_turns: int
    phase: str
    last_phase_reason: str
    frustration_score: float
    difficulty: str
    preferred_strategies: List[str]
    close_phase_threshold: float
    termination_frustration_threshold: float

    # Bramy wykrywania intencji i leku
    intent_revealed: bool
    intent_revealed_turn: int
    drug_revealed: bool
    drug_revealed_turn: int
    drug_intro_keywords: List[str]
    wrong_drug_suspected: bool      # True gdy rozmówca mówi o innym leku niż w sesji

    # Śledzenie claimów medycznych
    claim_index: Dict[str, Any]
    critical_claim_ids: List[str]
    critical_claim_labels: List[str]
    seen_claim_ids: List[str]
    covered_critical_claim_ids: List[str]

    # Śledzenie celu rozmowy
    goal_achieved: bool
    goal_status: str
    goal_score: int
    goal_achieved_turn: int
    latest_goal: Dict[str, Any]

    # Logi historyczne
    turn_metrics_history: List[Dict[str, Any]]
    critical_flags: List[str]
    random_events_history: List[Dict[str, Any]]
    last_random_event_turn: int
    is_terminated: bool

    # Dane robocze bieżącej tury (nadpisywane co turę)
    current_user_message: str
    current_analysis: Dict[str, Any]
    current_claim_check: Dict[str, Any]
    current_coverage_summary: Dict[str, Any]
    current_coverage_update: Dict[str, Any]
    current_evidence_requirements: Dict[str, Any]
    current_pre_policy: Dict[str, Any]
    current_turn_metrics: Dict[str, Any]
    current_frustration_update: Dict[str, Any]
    current_random_event: Optional[Dict[str, Any]]
    current_behavior_directives: List[str]
    current_style_runtime: Dict[str, Any]
    current_rag_context: List[str]   # fragmenty ChPL pobrane przez RAG dla bieżącej tury

    # Tymczasowe wyjście LLM (robocze między generate_response a finalize)
    _raw_ai_response: Optional[Any]

    # Wyjście bieżącej tury
    current_doctor_response: str
    current_doctor_attitude: str
    current_doctor_decision: str
    current_reasoning: str
    current_detected_errors: List[str]
    current_termination_reason: Optional[str]
    current_conversation_goal: Dict[str, Any]
    current_turn_metrics_payload: Dict[str, Any]
