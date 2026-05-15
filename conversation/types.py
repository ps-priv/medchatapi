"""Typy stanu sesji i pomocniczych struktur runtime."""

from typing import Any, Dict, List, TypedDict


class SessionState(TypedDict):
    """Stan algorytmu rozmowy przechowywany w sesji."""

    turn_index: int
    max_turns: int
    phase: str
    frustration_score: float
    difficulty: str
    preferred_strategies: List[str]
    close_phase_threshold: float
    termination_frustration_threshold: float
    claim_index: Dict[str, Dict[str, Any]]
    critical_claim_ids: List[str]
    critical_claim_labels: List[str]
    intent_revealed: bool
    intent_revealed_turn: int
    drug_revealed: bool
    drug_revealed_turn: int
    drug_intro_keywords: List[str]
    seen_claim_ids: List[str]
    covered_critical_claim_ids: List[str]
    last_phase_reason: str
    turn_metrics: List[Dict[str, Any]]
    critical_flags: List[str]
    random_events_history: List[Dict[str, Any]]
    last_random_event_turn: int
    goal_achieved: bool
    goal_status: str
    goal_score: int
    goal_achieved_turn: int
    latest_goal: Dict[str, Any]


class SessionData(TypedDict):
    """Pełna sesja rozmowy trzymana w pamięci API."""

    dialog: Any
    context: Any
    traits: Dict[str, float]
    doctor_profile: Dict[str, Any]
    drug_info: Dict[str, Any]
    is_terminated: bool
    state: SessionState
