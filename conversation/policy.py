"""Reguły rozmowy: styl, analiza wypowiedzi, metryki i reakcje lekarza."""

import re
from typing import Dict, List, Optional, Set

from .claims import critical_coverage_summary, extract_drug_keywords
from .constants import (
    BRIBERY_KEYWORDS,
    CLAIM_SEVERITY_WEIGHTS,
    CLINICAL_STUDY_PHRASES,
    DISRESPECTFUL_LANGUAGE_KEYWORDS,
    EMPTY_PRAISE_PHRASES,
    ENGLISH_MARKETING_WORDS,
    EVIDENCE_PHRASES,
    FEMALE_ADDRESS_FORMS,
    INAPPROPRIATE_PROPOSAL_KEYWORDS,
    MALE_ADDRESS_FORMS,
    MARKETING_BUZZWORDS,
    OFF_TOPIC_KEYWORDS,
    PHASES,
    SUPPORTED_STRATEGIES,
    TRAIT_KEYS,
)


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


def limit_sentences(text: str, max_sentences: int) -> str:
    """Przycina wypowiedź do zadanej liczby zdań."""
    cleaned = text.strip()
    if max_sentences <= 0 or not cleaned:
        return cleaned

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(parts) <= max_sentences:
        return cleaned
    return " ".join(parts[:max_sentences])


def normalize_gender(value: str) -> str:
    """Normalizuje zapis płci lekarza do female/male/unknown."""
    normalized = str(value or "").strip().lower()
    if normalized in {"female", "f", "kobieta", "kobieta_lekarz", "żeńska", "zenska"}:
        return "female"
    if normalized in {"male", "m", "mężczyzna", "mezczyzna", "meska", "męska"}:
        return "male"
    return "unknown"


def expected_address_form(gender: str) -> str:
    """Zwraca oczekiwaną formę zwracania się do lekarza."""
    if gender == "female":
        return "pani doktor"
    if gender == "male":
        return "pan doktor"
    return "pani doktor / pan doktor"


def detect_gender_addressing(message: str, doctor_gender: str) -> Dict:
    """Wykrywa niepoprawne formy grzecznościowe względem płci lekarza."""
    text = message.lower()
    male_hits = [form for form in MALE_ADDRESS_FORMS if form in text]
    female_hits = [form for form in FEMALE_ADDRESS_FORMS if form in text]
    generic_pan = bool(re.search(r"\bpan\b|\bpanie\b", text))
    generic_pani = bool(re.search(r"\bpani\b", text))

    mismatch_hits: List[str] = []
    if doctor_gender == "female":
        if male_hits:
            mismatch_hits.extend(male_hits)
        if generic_pan and "pani" not in text:
            mismatch_hits.append("pan")
    elif doctor_gender == "male":
        if female_hits:
            mismatch_hits.extend(female_hits)
        if generic_pani and "pan" not in text:
            mismatch_hits.append("pani")

    return {
        "gender_mismatch_hits": list(dict.fromkeys(mismatch_hits)),
        "expected_address": expected_address_form(doctor_gender),
    }


def analyze_message(
    message: str,
    drug_info: Dict,
    doctor_profile: Dict,
    focus_keywords: Optional[Set[str]] = None,
) -> Dict:
    """Wykrywa sygnały jakościowe i etyczne w wypowiedzi przedstawiciela."""
    text = message.lower()
    drug_keywords = focus_keywords if focus_keywords is not None else extract_drug_keywords(drug_info)
    doctor_gender = normalize_gender(doctor_profile.get("gender", "unknown"))
    gender_info = detect_gender_addressing(message, doctor_gender)

    bribery_hits = [keyword for keyword in BRIBERY_KEYWORDS if keyword in text]
    off_topic_hits = [keyword for keyword in OFF_TOPIC_KEYWORDS if keyword in text]
    marketing_hits = [keyword for keyword in MARKETING_BUZZWORDS if keyword in text]
    english_hits = [keyword for keyword in ENGLISH_MARKETING_WORDS if keyword in text]
    inappropriate_hits = [keyword for keyword in INAPPROPRIATE_PROPOSAL_KEYWORDS if keyword in text]
    disrespect_hits = [keyword for keyword in DISRESPECTFUL_LANGUAGE_KEYWORDS if keyword in text]
    empty_praise_hits = [phrase for phrase in EMPTY_PRAISE_PHRASES if phrase in text]
    evidence_hits = [phrase for phrase in EVIDENCE_PHRASES if phrase in text]
    clinical_study_hits = [phrase for phrase in CLINICAL_STUDY_PHRASES if phrase in text]
    drug_mentions = [keyword for keyword in drug_keywords if keyword in text]

    return {
        "bribery_hits": bribery_hits,
        "off_topic_hits": off_topic_hits,
        "marketing_hits": marketing_hits,
        "english_hits": english_hits,
        "inappropriate_hits": inappropriate_hits,
        "disrespect_hits": disrespect_hits,
        "empty_praise_hits": empty_praise_hits,
        "evidence_hits": evidence_hits,
        "clinical_study_hits": clinical_study_hits,
        "gender_mismatch_hits": gender_info["gender_mismatch_hits"],
        "expected_address": gender_info["expected_address"],
        "doctor_gender": doctor_gender,
        "has_drug_focus": len(drug_mentions) > 0,
        "drug_mentions_sample": sorted(drug_mentions)[:8],
    }


def policy_precheck(message_analysis: Dict, claim_check: Dict) -> Dict:
    """Buduje twarde dyrektywy policy przed wywołaniem LLM."""
    hard_stop = bool(message_analysis["bribery_hits"])
    directives: List[str] = []

    if hard_stop:
        directives.append("Naruszenie etyki: zakończ rozmowę natychmiast.")
    if message_analysis["gender_mismatch_hits"]:
        directives.append(
            f"Skoryguj formę grzecznościową: oczekiwane zwroty to '{message_analysis['expected_address']}'."
        )
    if message_analysis["inappropriate_hits"]:
        directives.append("Wykryto nieelegancką propozycję; stanowczo zaznacz granice rozmowy.")
    if message_analysis["disrespect_hits"]:
        directives.append("Wykryto brak szacunku; odpowiedz stanowczo i skróć rozmowę.")
    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        directives.append("Przerwij off-topic i wróć do rozmowy o leku.")
    if claim_check["false_claims"]:
        directives.append("Wykryto twierdzenia niezgodne z danymi leku.")
    if claim_check["unsupported_claims"]:
        directives.append("Zażądaj potwierdzenia twierdzeń bez pokrycia w danych.")
    # Evidence-first aktywuje się też przy "miękkich" tezach o leku bez żadnego potwierdzonego faktu.
    if message_analysis["has_drug_focus"] and not claim_check.get("supported_claims"):
        directives.append(
            "Tryb evidence-first: zanim przejdziesz dalej, poproś o konkretne dane potwierdzające tezy."
        )

    missing_critical = claim_check.get("missing_critical_labels", [])
    if missing_critical:
        directives.append("Przedstawiciel nie omówił wszystkich kluczowych claimów; dopytaj o brakujące punkty krytyczne.")

    return {
        "hard_stop": hard_stop,
        "directives": directives,
    }


def evidence_first_requirements(message_analysis: Dict, claim_check: Dict, state: Dict) -> Dict:
    """Wyznacza wymagania evidence-first: dopytanie i weryfikację twierdzeń."""
    supported_count = len(claim_check.get("supported_claims", []))
    false_count = len(claim_check.get("false_claims", []))
    unsupported_count = len(claim_check.get("unsupported_claims", []))
    turn_index = int(state.get("turn_index", 0))
    has_drug_focus = bool(message_analysis.get("has_drug_focus"))

    # "verification" to twarde prostowanie błędów; "probe" to dopytanie o dowód przy braku wsparcia.
    require_verification = (false_count + unsupported_count) > 0
    require_probe = has_drug_focus and supported_count == 0 and not require_verification
    strict_mode = require_verification or (require_probe and turn_index >= 2)

    directives: List[str] = []
    followup_question = ""
    if require_verification:
        # Najpierw walidacja i żądanie źródła, dopiero potem kontynuacja rozmowy.
        directives.append(
            "Tryb evidence-first: zakwestionuj twierdzenia bez pokrycia i poproś o źródło oraz parametry danych."
        )
        followup_question = (
            "Proszę podać konkretne źródło i dane (np. populacja, punkt końcowy, dawka), "
            "które potwierdzają tę tezę."
        )
    elif require_probe:
        # Przy braku potwierdzonych claimów lekarz wymusza doprecyzowanie na poziomie danych klinicznych.
        directives.append(
            "Tryb evidence-first: dopytaj o konkret kliniczny potwierdzający skuteczność lub bezpieczeństwo."
        )
        followup_question = (
            "Jakie konkretne dane kliniczne potwierdzają to twierdzenie "
            "(np. liczby, populacja, porównanie, bezpieczeństwo)?"
        )

    return {
        "require_probe": require_probe,
        "require_verification": require_verification,
        "strict_mode": strict_mode,
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "false_count": false_count,
        "directives": directives,
        "followup_question": followup_question,
    }


def compute_turn_metrics(message_analysis: Dict, claim_check: Dict, state: Dict) -> Dict:
    """Liczy metryki jakości tury, w tym coverage claimów krytycznych."""
    coverage = critical_coverage_summary(state)
    turn_index = int(state.get("turn_index", 0))
    max_turns = max(1, int(state.get("max_turns", 7)))
    progress_ratio = min(1.0, turn_index / max_turns)

    topic_adherence = 0.3
    if message_analysis["has_drug_focus"]:
        topic_adherence += 0.5
    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        topic_adherence -= 0.35

    clinical_precision = 0.8
    clinical_precision -= min(0.25 * len(claim_check["false_claims"]), 0.75)
    clinical_precision -= min(0.12 * len(claim_check["unsupported_claims"]), 0.35)
    clinical_precision -= min(0.07 * len(message_analysis["marketing_hits"]), 0.2)

    drug_revealed = bool(state.get("drug_revealed", True))
    if drug_revealed and coverage["total_critical"] > 0 and turn_index >= 2:
        missing_ratio = coverage["missing_critical"] / max(1, coverage["total_critical"])
        coverage_penalty = min((0.25 * missing_ratio) + (0.2 * progress_ratio), 0.45)
        clinical_precision -= coverage_penalty

    ethics = 1.0 if not message_analysis["bribery_hits"] else 0.0
    ethics -= min(0.3 * len(message_analysis["inappropriate_hits"]), 0.7)
    ethics -= min(0.2 * len(message_analysis["disrespect_hits"]), 0.6)
    ethics -= min(0.15 * len(message_analysis["gender_mismatch_hits"]), 0.4)

    language_quality = 1.0
    language_quality -= min(0.12 * len(message_analysis["english_hits"]), 0.5)
    language_quality -= min(0.08 * len(message_analysis["marketing_hits"]), 0.3)
    language_quality -= min(0.2 * len(message_analysis["disrespect_hits"]), 0.6)
    language_quality -= min(0.1 * len(message_analysis["gender_mismatch_hits"]), 0.3)

    return {
        "topic_adherence": round(max(0.0, min(1.0, topic_adherence)), 2),
        "clinical_precision": round(max(0.0, min(1.0, clinical_precision)), 2),
        "ethics": round(max(0.0, min(1.0, ethics)), 2),
        "language_quality": round(max(0.0, min(1.0, language_quality)), 2),
        "critical_claim_coverage": coverage["coverage_ratio"],
    }


def compute_frustration(
    state: Dict,
    message_analysis: Dict,
    claim_check: Dict,
    metrics: Dict,
    difficulty_cfg: Dict,
    preferred_strategies: List[str],
) -> Dict:
    """Aktualizuje frustrację lekarza na podstawie jakości tury."""
    delta = 0.0
    severity_counts = claim_check.get("severity_counts", {})
    critical_count = int(severity_counts.get("critical", 0))
    major_count = int(severity_counts.get("major", 0))
    minor_count = int(severity_counts.get("minor", 0))

    coverage = critical_coverage_summary(state)
    turn_index = int(state.get("turn_index", 0))
    max_turns = max(1, int(state.get("max_turns", 7)))
    progress_ratio = min(1.0, turn_index / max_turns)

    if message_analysis["bribery_hits"]:
        delta += 8.0
    delta += 2.2 * len(message_analysis["inappropriate_hits"])
    delta += 1.5 * len(message_analysis["disrespect_hits"])
    delta += 0.9 * len(message_analysis["gender_mismatch_hits"])
    delta += CLAIM_SEVERITY_WEIGHTS["critical"] * critical_count
    delta += CLAIM_SEVERITY_WEIGHTS["major"] * major_count
    delta += CLAIM_SEVERITY_WEIGHTS["minor"] * minor_count
    delta += 1.4 if (message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]) else 0.0
    delta += 0.45 * len(message_analysis["marketing_hits"])
    delta += 0.35 * len(message_analysis["english_hits"])

    drug_revealed = bool(state.get("drug_revealed", True))
    if drug_revealed and coverage["total_critical"] > 0 and turn_index >= 2:
        missing_ratio = coverage["missing_critical"] / max(1, coverage["total_critical"])
        delta += missing_ratio * (0.6 + 1.1 * progress_ratio)
        if progress_ratio >= 0.75 and coverage["missing_critical"] > 0:
            delta += 0.5

    if "skeptical" in preferred_strategies:
        delta += 0.35 * len(claim_check["unsupported_claims"])
        delta += 0.45 * len(claim_check["false_claims"])

    if "confrontational" in preferred_strategies and (claim_check["false_claims"] or claim_check["unsupported_claims"]):
        delta += 0.5

    if "transactional" in preferred_strategies and (
        message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]
    ):
        delta += 0.7

    if "disengaging" in preferred_strategies:
        # Kara za brak drug focus dopiero gdy cel wizyty jest już znany —
        # nie karamy za powitanie, zanim przedstawiciel miał szansę powiedzieć po co przyszedł.
        if not message_analysis["has_drug_focus"] and state.get("intent_revealed", False):
            delta += 0.6
        if int(state.get("turn_index", 0)) >= int(state.get("max_turns", 7)) - 1:
            delta += 0.5

    if "exploratory" in preferred_strategies:
        if (
            message_analysis["has_drug_focus"]
            and not claim_check["false_claims"]
            and not claim_check["unsupported_claims"]
        ):
            delta -= 0.35

    if metrics["topic_adherence"] >= 0.7:
        delta -= 0.4
    if metrics["clinical_precision"] >= 0.8:
        delta -= 0.2
    if claim_check.get("supported_claims"):
        delta -= min(0.15 * len(claim_check["supported_claims"]), 0.5)
    if turn_index > max_turns:
        delta += 2.0

    # Bias trudności stosujemy dopiero od tury 2 — nie ma sensu karać lekarza
    # za wstępne powitanie, zanim rozmowa w ogóle się zaczęła.
    if turn_index >= 2:
        delta += float(difficulty_cfg.get("frustration_bias", 0.0))
    delta = max(-1.0, min(8.0, delta))
    total = max(0.0, min(10.0, float(state.get("frustration_score", 0.0)) + delta))
    return {"delta": round(delta, 2), "total": round(total, 2)}


def advance_phase(current_phase: str, state: Dict, message_analysis: Dict, metrics: Dict) -> Dict:
    """Przełącza fazę rozmowy zgodnie z regułami state-machine."""
    phase = current_phase if current_phase in PHASES else "opening"
    turn_index = int(state.get("turn_index", 0))
    max_turns = int(state.get("max_turns", 7))
    frustration = float(state.get("frustration_score", 0.0))
    close_phase_threshold = float(state.get("close_phase_threshold", 7.0))

    if message_analysis["bribery_hits"]:
        return {"phase": "close", "reason": "ethical_breach"}
    # Frustracja i limit tur mogą zamknąć rozmowę dopiero od tury 2 —
    # przed tym czasem dajemy przedstawicielowi szansę na przedstawienie się i celu wizyty.
    if turn_index >= 2 and (frustration >= close_phase_threshold or turn_index >= max_turns):
        return {"phase": "close", "reason": "time_or_frustration_limit"}

    if phase == "opening":
        if message_analysis["has_drug_focus"]:
            return {"phase": "needs", "reason": "drug_focus_established"}
        return {"phase": "opening", "reason": "awaiting_relevance"}

    if phase == "needs":
        if metrics["clinical_precision"] >= 0.6:
            return {"phase": "objection", "reason": "objection_phase_started"}
        return {"phase": "needs", "reason": "needs_clarification"}

    if phase == "objection":
        if metrics["clinical_precision"] >= 0.7 and not message_analysis["marketing_hits"]:
            return {"phase": "evidence", "reason": "evidence_requested"}
        return {"phase": "objection", "reason": "objection_not_resolved"}

    if phase == "evidence":
        if turn_index >= max_turns - 1 or frustration >= 8.5:
            return {"phase": "close", "reason": "closing_window"}
        return {"phase": "evidence", "reason": "continue_evidence"}

    return {"phase": "close", "reason": "already_closing"}


def policy_postprocess_message(
    raw_message: str,
    style_runtime: Dict,
    message_analysis: Dict,
    claim_check: Dict,
    phase: str,
    missing_critical_labels: List[str],
    evidence_requirements: Dict,
    drug_revealed: bool,
    intent_revealed: bool,
) -> str:
    """Dopisuje obowiązkowe korekty policy do odpowiedzi LLM.

    Korekty bezpieczeństwa (błędna forma, niestosowność, false claims) są zawsze
    dopisywane. Pytania dodatkowe — co najwyżej jedno, i tylko gdy LLM sam
    nie zadał żadnego pytania.
    """
    result = limit_sentences(raw_message.strip(), int(style_runtime["max_sentences"]))

    # --- Korekty bezpieczeństwa — zawsze dopisywane ---
    if message_analysis["gender_mismatch_hits"]:
        correction = f"Proszę zwracać się do mnie poprawnie: '{message_analysis['expected_address']}'."
        if correction not in result:
            result += f" {correction}"

    if message_analysis["inappropriate_hits"] or message_analysis["disrespect_hits"]:
        boundary_msg = "Taki ton i takie propozycje są nieakceptowalne w rozmowie zawodowej."
        if boundary_msg not in result:
            result += f" {boundary_msg}"

    if message_analysis["off_topic_hits"] and not message_analysis["has_drug_focus"]:
        redirect = "Proszę wrócić do tematu wizyty." if intent_revealed else "Proszę przejść do celu spotkania."
        if redirect not in result:
            result += f" {redirect}"

    if claim_check["false_claims"]:
        if "niezgodne z danymi" not in result:
            result += " To niezgodne z danymi leku."

    if phase == "close":
        if "Kończę" not in result and "kończę" not in result and "zakończ" not in result:
            result += " Na tym etapie kończę spotkanie."

    # --- Pytanie uzupełniające — co najwyżej jedno, tylko gdy LLM nie zadał żadnego ---
    if phase != "close" and "?" not in result:
        followup = str(evidence_requirements.get("followup_question", "")).strip()
        if followup:
            result += f" {followup}"
        elif missing_critical_labels:
            result += " Proszę odnieść się do kluczowych danych klinicznych leku."
        elif not intent_revealed:
            result += " W czym mogę pomóc?"
        elif not drug_revealed:
            result += " Proszę powiedzieć, czego dokładnie dotyczy wizyta."

    return result.strip()


def apply_reaction_rules(
    current_traits: Dict[str, float],
    llm_traits: Dict[str, float],
    analysis: Dict,
    detected_errors_count: int,
    frustration_delta: float,
    frustration_total: float,
) -> Dict[str, float]:
    """Nakłada deterministyczne zmiany traits po reakcji LLM i regułach policy."""
    traits = clamp_traits(llm_traits)

    if analysis["bribery_hits"]:
        traits["skepticism"] = 1.0
        traits["patience"] = 0.0
        traits["openness"] = 0.0
        traits["time_pressure"] = 1.0
        traits["ego"] = max(traits["ego"], current_traits.get("ego", 0.5))
        return clamp_traits(traits)

    if detected_errors_count > 0:
        traits["skepticism"] += min(0.18 * detected_errors_count, 0.5)
        traits["openness"] -= min(0.2 * detected_errors_count, 0.5)
        traits["patience"] -= min(0.12 * detected_errors_count, 0.35)
        traits["time_pressure"] += min(0.01 * detected_errors_count, 0.04)

    if analysis["marketing_hits"]:
        penalty = min(0.06 * len(analysis["marketing_hits"]), 0.2)
        traits["skepticism"] += penalty
        traits["openness"] -= penalty

    if analysis.get("empty_praise_hits"):
        traits["skepticism"] += min(0.15 * len(analysis["empty_praise_hits"]), 0.4)

    if analysis.get("evidence_hits"):
        traits["time_pressure"] -= min(0.08 * len(analysis["evidence_hits"]), 0.3)

    if analysis.get("clinical_study_hits"):
        count = len(analysis["clinical_study_hits"])
        traits["time_pressure"] -= min(0.1 * count, 0.4)
        traits["openness"] += min(0.12 * count, 0.35)
        traits["skepticism"] -= min(0.1 * count, 0.3)

    if analysis["english_hits"]:
        penalty = min(0.05 * len(analysis["english_hits"]), 0.2)
        traits["skepticism"] += penalty
        traits["patience"] -= penalty

    if analysis["off_topic_hits"] and not analysis["has_drug_focus"]:
        traits["time_pressure"] += 0.01
        traits["patience"] -= 0.08

    if analysis["gender_mismatch_hits"]:
        penalty = min(0.08 * len(analysis["gender_mismatch_hits"]), 0.24)
        traits["skepticism"] += penalty
        traits["openness"] -= penalty
        traits["patience"] -= min(penalty, 0.15)

    if analysis["inappropriate_hits"] or analysis["disrespect_hits"]:
        traits["skepticism"] += min(0.2 + 0.07 * len(analysis["inappropriate_hits"]), 0.45)
        traits["openness"] -= min(0.25 + 0.05 * len(analysis["disrespect_hits"]), 0.5)
        traits["patience"] -= min(0.2 + 0.05 * len(analysis["disrespect_hits"]), 0.4)
        traits["time_pressure"] += min(0.02 + 0.01 * len(analysis["inappropriate_hits"]), 0.04)

    if frustration_delta > 0:
        traits["skepticism"] += min(0.04 * frustration_delta, 0.2)
        traits["patience"] -= min(0.05 * frustration_delta, 0.25)

    if frustration_total >= 6.0:
        traits["openness"] -= 0.15
        traits["time_pressure"] += 0.01

    return clamp_traits(traits)


def update_conviction(
    conviction: Dict[str, float],
    claim_check: Dict,
    turn_metrics: Dict,
    message_analysis: Dict,
    familiarity: str,
    frustration_score: float = 0.0,
    doctor_traits: Optional[Dict[str, float]] = None,
    preferred_strategies: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Aktualizuje 5-wymiarowy stan przekonań lekarza na podstawie zachowania przedstawiciela.

    Delty są modyfikowane przez cechy i strategie konkretnego lekarza — sceptyk trudniej
    uwierzy w dane kliniczne, lekarz eksploracyjny chętniej podniesie zainteresowanie.
    """
    c = {
        "interest_level": float(conviction.get("interest_level", 0.3)),
        "trust_in_rep": float(conviction.get("trust_in_rep", 0.3)),
        "clinical_confidence": float(conviction.get("clinical_confidence", 0.2)),
        "perceived_fit": float(conviction.get("perceived_fit", 0.2)),
        "decision_readiness": float(conviction.get("decision_readiness", 0.0)),
    }
    traits = doctor_traits or {}
    strategies = preferred_strategies or []

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    # --- Mnożniki z cech i strategii lekarza ---
    skepticism = float(traits.get("skepticism", 0.5))
    openness = float(traits.get("openness", 0.5))

    # Sceptyk trudniej uwierzy w dane kliniczne (wymaga więcej dowodów)
    clinical_gain_mult = 1.0 - max(0.0, (skepticism - 0.5) * 0.6)  # sceptycyzm 0.9 → mult ≈ 0.76
    # Otwarty lekarz chętniej podnosi zainteresowanie
    interest_gain_mult = 0.85 + openness * 0.3  # openness 0.1 → 0.88; openness 0.9 → 1.12

    # Modyfikatory strategii dla trust_in_rep
    trust_gain_mult = {"acquainted": 1.15, "familiar": 1.30}.get(familiarity, 1.0)
    trust_loss_mult = {"familiar": 1.20}.get(familiarity, 1.0)
    if "confrontational" in strategies:
        trust_gain_mult *= 0.80   # trudniejszy w budowaniu relacji

    # --- Claimy potwierdzone → wzrost pewności klinicznej i gotowości ---
    supported = len(claim_check.get("supported_claims", []))
    if supported:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.08 * min(supported, 3) * clinical_gain_mult)
        readiness_gain = 0.05 * min(supported, 3)
        if "transactional" in strategies:
            readiness_gain *= 1.25  # pragmatyk szybciej decyduje gdy dane się zgadzają
        c["decision_readiness"] = clamp01(c["decision_readiness"] + readiness_gain)

    # --- Claimy fałszywe → strata zaufania i pewności (zależna od severity) ---
    has_false = bool(claim_check.get("false_claims"))
    if has_false:
        severity_counts = claim_check.get("severity_counts", {})
        critical_n = int(severity_counts.get("critical", 0))
        major_n = int(severity_counts.get("major", 0))
        minor_n = int(severity_counts.get("minor", 0))
        # Sceptyk i konfrontacyjny bardziej karze za błędy
        false_loss_mult = 1.0 + (0.15 if "skeptical" in strategies else 0.0) + (0.10 if "confrontational" in strategies else 0.0)
        if critical_n:
            trust_delta_false = -0.20 * critical_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.15 * critical_n)
        elif major_n:
            trust_delta_false = -0.10 * major_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.08 * major_n)
        elif minor_n:
            trust_delta_false = -0.04 * minor_n * false_loss_mult
            c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.03 * minor_n)
        else:
            trust_delta_false = 0.0
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + trust_delta_false * trust_loss_mult)

    # --- Claimy niepotwierdzone → drobna strata pewności klinicznej ---
    unsupported = len(claim_check.get("unsupported_claims", []))
    if unsupported:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] - 0.03 * min(unsupported, 4))

    # --- Evidence-based (badania, dane) → wzrost pewności i zainteresowania ---
    evidence_count = min(len(message_analysis.get("evidence_hits", [])), 2)
    study_count = min(len(message_analysis.get("clinical_study_hits", [])), 2)
    if evidence_count:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.10 * evidence_count * clinical_gain_mult)
        c["interest_level"] = clamp01(c["interest_level"] + 0.05 * evidence_count * interest_gain_mult)
    if study_count:
        c["clinical_confidence"] = clamp01(c["clinical_confidence"] + 0.06 * study_count * clinical_gain_mult)
        interest_study_gain = 0.03 * study_count * interest_gain_mult
        if "exploratory" in strategies:
            interest_study_gain *= 1.30  # ciekawy lekarz bardziej reaguje na badania
        c["interest_level"] = clamp01(c["interest_level"] + interest_study_gain)

    # --- Off-topic → spadek zainteresowania ---
    if message_analysis.get("off_topic_hits") and not message_analysis.get("has_drug_focus"):
        offtopic_loss = 0.08
        if "disengaging" in strategies:
            offtopic_loss *= 1.50  # zdystansowany szybko traci zainteresowanie
        if "transactional" in strategies:
            offtopic_loss *= 1.30  # pragmatyk nie toleruje dygresji
        c["interest_level"] = clamp01(c["interest_level"] - offtopic_loss)

    # --- Marketing buzzwords → erozja zaufania ---
    marketing_count = len(message_analysis.get("marketing_hits", []))
    if marketing_count:
        trust_delta_mkt = -0.05 * min(marketing_count, 4)
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + trust_delta_mkt * trust_loss_mult)

    # --- Anglicyzmy → drobna erozja zaufania ---
    english_count = len(message_analysis.get("english_hits", []))
    if english_count:
        trust_delta_eng = -0.03 * min(english_count, 4)
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + trust_delta_eng * trust_loss_mult)

    # --- Disrespect / inappropriate → silna strata zaufania ---
    if message_analysis.get("disrespect_hits") or message_analysis.get("inappropriate_hits"):
        ego = float(traits.get("ego", 0.5))
        disrespect_loss = -0.15 - max(0.0, (ego - 0.5) * 0.10)  # wysokie ego → większa kara
        c["trust_in_rep"] = clamp01(c["trust_in_rep"] + disrespect_loss * trust_loss_mult)

    # --- Bribery → reset zaufania do zera ---
    if message_analysis.get("bribery_hits"):
        c["trust_in_rep"] = 0.0

    # --- Pokrycie claimów krytycznych >= 80% → wzrost dopasowania i gotowości ---
    if float(turn_metrics.get("critical_claim_coverage", 0.0)) >= 0.8:
        c["perceived_fit"] = clamp01(c["perceived_fit"] + 0.10)
        dr_gain = 0.10
        if "skeptical" in strategies:
            dr_gain *= 0.80  # sceptyk wolniej przechodzi do decyzji nawet przy dobrym pokryciu
        c["decision_readiness"] = clamp01(c["decision_readiness"] + dr_gain)

    # --- Dobra adherencja tematu → lekki wzrost zainteresowania ---
    if float(turn_metrics.get("topic_adherence", 0.0)) >= 0.7:
        c["interest_level"] = clamp01(c["interest_level"] + 0.03 * interest_gain_mult)

    # --- Wysoka frustracja → spadek zainteresowania i gotowości ---
    if frustration_score >= 6.0:
        c["interest_level"] = clamp01(c["interest_level"] - 0.05)
        c["decision_readiness"] = clamp01(c["decision_readiness"] - 0.10)

    return {k: round(v, 4) for k, v in c.items()}
