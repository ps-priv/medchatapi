"""Cechy psychologiczne lekarza i parametry trudności rozmowy."""

from typing import Dict, List

from .constants import SUPPORTED_STRATEGIES, TRAIT_KEYS


def clamp_traits(traits: Dict[str, float]) -> Dict[str, float]:
    """Normalizuje cechy psychologiczne do zakresu 0.0-1.0."""
    clamped: Dict[str, float] = {}
    for key in TRAIT_KEYS:
        value = float(traits.get(key, 0.5))
        clamped[key] = max(0.0, min(1.0, value))
    return clamped


def normalize_difficulty(value: str) -> str:
    """Normalizuje poziom trudności do easy/medium/hard."""
    normalized = str(value or "").strip().lower()
    if normalized in {"easy", "łatwy", "latwy"}:
        return "easy"
    if normalized in {"hard", "trudny"}:
        return "hard"
    return "medium"


def extract_preferred_strategies(doctor_profile: Dict) -> List[str]:
    """Pobiera i filtruje strategie lekarza do wspieranego zbioru."""
    raw = doctor_profile.get("preferred_strategies", [])
    if not isinstance(raw, list):
        return []

    result: List[str] = []
    for item in raw:
        strategy = str(item).strip().lower()
        if strategy in SUPPORTED_STRATEGIES and strategy not in result:
            result.append(strategy)
    return result


def difficulty_profile(difficulty: str) -> Dict:
    """Mapuje trudność na parametry sterujące tempem i surowością rozmowy."""
    normalized = normalize_difficulty(difficulty)
    if normalized == "easy":
        return {
            "difficulty": "easy",
            "frustration_bias": -0.35,
            "close_phase_threshold": 9.5,
            "termination_frustration_threshold": 10.0,
            "turn_limit_adjust": 1,
        }
    if normalized == "hard":
        return {
            "difficulty": "hard",
            "frustration_bias": 0.45,
            "close_phase_threshold": 8.5,
            "termination_frustration_threshold": 9.5,
            "turn_limit_adjust": -1,
        }
    return {
        "difficulty": "medium",
        "frustration_bias": 0.0,
        "close_phase_threshold": 9.0,
        "termination_frustration_threshold": 10.0,
        "turn_limit_adjust": 0,
    }


def derive_turn_limit(traits: Dict[str, float], difficulty_cfg: Dict) -> int:
    """Wylicza limit tur na bazie presji czasu, cierpliwości i trudności."""
    time_pressure = float(traits.get("time_pressure", 0.5))
    patience = float(traits.get("patience", 0.5))

    if time_pressure >= 0.9:
        base = 12
    elif time_pressure >= 0.75:
        base = 14
    elif time_pressure >= 0.55:
        base = 16
    elif patience >= 0.8 and time_pressure <= 0.4:
        base = 20
    else:
        base = 18

    adjusted = base + int(difficulty_cfg.get("turn_limit_adjust", 0))
    return max(10, min(20, adjusted))


def build_style_directives(doctor_profile: Dict, traits: Dict[str, float], state: Dict) -> Dict:
    """Buduje aktywne dyrektywy stylu lekarza do promptu systemowego."""
    style = str(doctor_profile.get("communication_style", "")).lower()
    directives: List[str] = []
    max_sentences = 3
    strategies = state.get("preferred_strategies", [])
    difficulty = state.get("difficulty", "medium")

    concise_markers = ("krótki", "krotki", "bezpośredni", "bezposredni", "suchy", "zdystansowany")
    if (
        any(marker in style for marker in concise_markers)
        or traits.get("time_pressure", 0.5) >= 0.75
        or traits.get("patience", 0.5) <= 0.35
    ):
        directives.append("Odpowiadaj krótko: maksymalnie 1-2 zdania.")
        max_sentences = 2

    if any(marker in style for marker in ("analityczny", "formalny", "precyzyjny", "dociekliwy")):
        directives.append("Żądaj konkretów klinicznych i precyzyjnych danych.")

    if any(marker in style for marker in ("ciepły", "cieply", "empatyczny", "rozmowny", "energiczny")):
        directives.append("Zachowaj życzliwy ton, ale nie odchodź od tematu leku.")

    if any(marker in style for marker in ("dominujący", "dominujacy", "stanowczy")):
        directives.append("Mów stanowczo i wyznaczaj granice rozmowy.")

    if traits.get("skepticism", 0.5) >= 0.75:
        directives.append("Podchodź krytycznie do twierdzeń bez potwierdzenia.")
    if traits.get("openness", 0.5) <= 0.3:
        directives.append("Okazuj ograniczone zaufanie i wymagaj doprecyzowania.")
    if traits.get("ego", 0.5) >= 0.75:
        directives.append("Oczekuj profesjonalnego tonu i szacunku.")
    if float(state.get("frustration_score", 0.0)) >= 6.0:
        directives.append("Masz niski próg tolerancji: stawiaj granice i skracaj rozmowę.")
        max_sentences = 1

    if "transactional" in strategies:
        directives.append("Prowadź rozmowę zadaniowo: krótko, konkretnie, bez dygresji.")
        max_sentences = min(max_sentences, 2)
    if "skeptical" in strategies:
        directives.append("Bądź sceptyczna: każ tezy uzasadniać konkretem klinicznym.")
    if "confrontational" in strategies:
        directives.append("Przyjmij styl konfrontacyjny: zadawaj wymagające pytania i testuj spójność argumentów.")
        max_sentences = min(max_sentences, 2)
    if "exploratory" in strategies:
        directives.append("Przyjmij styl eksploracyjny: zadawaj pytania pogłębiające w obszarze leku.")
    if "disengaging" in strategies:
        directives.append("Przyjmij styl zdystansowany: szybko kończ wątki bez wartości merytorycznej.")
        max_sentences = min(max_sentences, 1)

    if difficulty == "hard":
        directives.append("Wysoka trudność: wymagaj precyzji i nie akceptuj ogólników.")
        max_sentences = min(max_sentences, 2)
    elif difficulty == "easy":
        directives.append("Niższa trudność: zachowaj nieco większą otwartość na doprecyzowanie.")

    max_turns = int(state.get("max_turns", 7))
    turn_index = int(state.get("turn_index", 0))
    remaining_turns = max(0, max_turns - turn_index)
    if remaining_turns <= 1:
        directives.append("To końcowa faza spotkania: domknij temat i przygotuj zakończenie rozmowy.")
        max_sentences = 1

    return {
        "directives": directives,
        "max_sentences": max_sentences,
        "remaining_turns": remaining_turns,
    }
