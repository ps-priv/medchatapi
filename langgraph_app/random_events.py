"""Zdarzenia losowe w trakcie rozmowy (telefon, kolejka pacjentów, itp.)."""

import hashlib
from typing import Dict, List, Optional

from conversation.doctor_traits import clamp_traits

from .state import ConversationState

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
