"""Detekcja słów-kluczy w wypowiedzi przedstawiciela i korekta cech lekarza."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .doctor_traits import clamp_traits

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "keyword_triggers.json"
_triggers: List[Dict] | None = None

# Liczba znaków wokół dopasowanej frazy, w których szukamy wartości liczbowej/% —
# uzasadnia bonus profesjonalizmu (np. "skuteczność 87%" zamiast gołego słowa).
NUMERIC_BONUS_WINDOW = 40
_NUMBER_NEARBY_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")


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


def apply_keyword_triggers(
    message: str,
    traits: Dict[str, float],
    already_bonused_phrases: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], List[str], List[str], Dict[str, float], Optional[str]]:
    """Wykrywa frazy-triggery w wiadomości i stosuje delty cech lekarza.

    Jeśli trigger ma "numeric_bonus" i w pobliżu dopasowanej frazy (± NUMERIC_BONUS_WINDOW
    znaków) pada wartość liczbowa lub %, dolicza bonus do professionalism_score — ale tylko
    raz na daną frazę w całej sesji (already_bonused_phrases pilnuje tego między turami).

    Jeśli trigger ma "flat_bonus" (np. {"professionalism_score": 1, "relevance_score": 1}),
    samo dopasowanie (bez wymogu liczby w pobliżu) dolicza wskazane bonusy do wskazanych pól
    oceny końcowej — też tylko raz na daną frazę w całej sesji. Używane np. dla wymaganych
    claimów marketingowych, które przedstawiciel powinien wypowiedzieć w rozmowie.

    Jeśli trigger ma "forced_reply", dopasowanie wymusza dokładnie ten tekst jako odpowiedź
    lekarza (zamiast tego, co wygenerował LLM) — do twardych, niepodlegających negocjacji
    granic (np. odmowa przyjęcia prezentu). Zwracany jest pierwszy trafiony forced_reply.

    Zwraca (traits, triggered_phrases, newly_bonused_phrases, bonus_deltas, forced_reply),
    gdzie bonus_deltas to {nazwa_pola_oceny: suma_bonusu_w_tej_turze}.
    Każdy trigger odpala się co najwyżej raz na wiadomość, niezależnie od liczby wystąpień.
    """
    triggers = _load_triggers()
    updated = dict(traits)
    triggered_phrases: List[str] = []
    newly_bonused_phrases: List[str] = []
    bonus_deltas: Dict[str, float] = {}
    forced_reply: Optional[str] = None
    already_bonused = already_bonused_phrases or set()

    msg_lower = message.lower()

    for trigger in triggers:
        phrase: str = trigger.get("phrase", "")
        match_type: str = trigger.get("match_type", "word")
        deltas: Dict[str, float] = trigger.get("deltas", {})
        numeric_bonus = float(trigger.get("numeric_bonus", 0.0))
        flat_bonus: Dict[str, float] = trigger.get("flat_bonus", {}) or {}
        trigger_forced_reply: str = str(trigger.get("forced_reply", "") or "")

        if not phrase or (not deltas and not numeric_bonus and not flat_bonus and not trigger_forced_reply):
            continue

        phrase_lower = phrase.lower()
        if match_type == "word":
            pattern = r"\b" + re.escape(phrase_lower) + r"\b"
        elif match_type == "regex":
            # "phrase" to gotowy wzorzec regex (pisany małymi literami — dopasowywany do
            # message.lower()), do konceptów wyrażalnych na wiele sposobów (np. "badanie
            # 4 fazy" / "czterofazowe badanie" / "badanie cztery fazy wykazało...").
            pattern = phrase
        else:
            pattern = re.escape(phrase_lower)
        match = re.search(pattern, msg_lower)

        if not match:
            continue

        display_phrase = str(trigger.get("label") or phrase)
        triggered_phrases.append(display_phrase)
        for trait, delta in deltas.items():
            if trait in updated:
                updated[trait] = updated[trait] + float(delta)

        if (numeric_bonus or flat_bonus) and display_phrase not in already_bonused:
            bonused = False
            if numeric_bonus:
                window_start = max(0, match.start() - NUMERIC_BONUS_WINDOW)
                window_end = min(len(message), match.end() + NUMERIC_BONUS_WINDOW)
                if _NUMBER_NEARBY_RE.search(message[window_start:window_end]):
                    bonus_deltas["professionalism_score"] = bonus_deltas.get("professionalism_score", 0.0) + numeric_bonus
                    bonused = True
            if flat_bonus:
                for score_name, amount in flat_bonus.items():
                    bonus_deltas[score_name] = bonus_deltas.get(score_name, 0.0) + float(amount)
                bonused = True
            if bonused:
                newly_bonused_phrases.append(display_phrase)

        if trigger_forced_reply and forced_reply is None:
            forced_reply = trigger_forced_reply

    if triggered_phrases:
        updated = clamp_traits(updated)
        logger.debug("keyword_triggers: trafione frazy=%s", triggered_phrases)

    return updated, triggered_phrases, newly_bonused_phrases, bonus_deltas, forced_reply
