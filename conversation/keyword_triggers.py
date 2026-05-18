"""Detekcja słów-kluczy w wypowiedzi przedstawiciela i korekta cech lekarza."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .doctor_traits import clamp_traits

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "keyword_triggers.json"
_triggers: List[Dict] | None = None


def _load_triggers() -> List[Dict]:
    global _triggers
    if _triggers is None:
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            _triggers = data.get("triggers", [])
            logger.info("keyword_triggers: wczytano %d triggerów z %s", len(_triggers), _CONFIG_PATH)
        except Exception as exc:
            logger.warning("keyword_triggers: błąd wczytywania konfiguracji: %s", exc)
            _triggers = []
    return _triggers


def apply_keyword_triggers(message: str, traits: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Wykrywa frazy-triggery w wiadomości i stosuje delty cech lekarza.

    Zwraca zaktualizowany dict traits oraz listę trafionych fraz (do logowania).
    Każdy trigger odpala się co najwyżej raz na wiadomość, niezależnie od liczby wystąpień.
    """
    triggers = _load_triggers()
    updated = dict(traits)
    triggered_phrases: List[str] = []

    msg_lower = message.lower()

    for trigger in triggers:
        phrase: str = trigger.get("phrase", "")
        match_type: str = trigger.get("match_type", "word")
        deltas: Dict[str, float] = trigger.get("deltas", {})

        if not phrase or not deltas:
            continue

        phrase_lower = phrase.lower()

        if match_type == "word":
            pattern = r"\b" + re.escape(phrase_lower) + r"\b"
            matched = bool(re.search(pattern, msg_lower))
        else:
            matched = phrase_lower in msg_lower

        if matched:
            triggered_phrases.append(phrase)
            for trait, delta in deltas.items():
                if trait in updated:
                    updated[trait] = updated[trait] + float(delta)

    if triggered_phrases:
        updated = clamp_traits(updated)
        logger.debug("keyword_triggers: trafione frazy=%s", triggered_phrases)

    return updated, triggered_phrases
