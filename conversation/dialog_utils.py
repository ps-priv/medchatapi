"""Narzędzia pracy z obiektem Dialog z biblioteki sdialog."""

from typing import Any, List

from sdialog import Dialog, Turn


def append_event(dialog: Dialog, name: str, details: str) -> None:
    """Bezpiecznie dodaje zdarzenie do historii dialogu."""
    try:
        from sdialog import Event

        event = Event(name=name, details=details)
        if hasattr(dialog, "events"):
            dialog.events.append(event)
        elif hasattr(dialog, "turns"):
            dialog.turns.append(event)
    except Exception:
        return


def add_user_turn(dialog: Dialog, message: str) -> None:
    """Dodaje turę przedstawiciela, zgodnie z API sdialog/fallbackiem."""
    try:
        user_turn = Turn(speaker="Przedstawiciel", text=message)
    except Exception:
        user_turn = Turn(role="user", content=message)

    if hasattr(dialog, "add_turn"):
        dialog.add_turn(user_turn)
    elif hasattr(dialog, "turns"):
        dialog.turns.append(user_turn)


def add_doctor_turn(dialog: Dialog, message: str) -> None:
    """Dodaje turę lekarza, zgodnie z API sdialog/fallbackiem."""
    try:
        doctor_turn = Turn(speaker="Lekarz", text=message)
    except Exception:
        doctor_turn = Turn(role="assistant", content=message)

    if hasattr(dialog, "add_turn"):
        dialog.add_turn(doctor_turn)
    elif hasattr(dialog, "turns"):
        dialog.turns.append(doctor_turn)


def extract_turns(dialog: Dialog) -> List[Any]:
    """Zwraca listę tur/zdarzeń z obiektu dialogu."""
    if hasattr(dialog, "turns"):
        return dialog.turns
    return getattr(dialog, "events", [])
