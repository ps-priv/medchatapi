"""Knowledge guard — lekarz nie dopowiada danych leku, których nie usłyszał od przedstawiciela."""

import hashlib
import re
from typing import Dict, List, Tuple

from .helpers import normalize_text

HEARSAY_PHRASES = (
    "Czytałam gdzieś, że",
    "Słyszałam od koleżanki, że",
    "Wydaje mi się, że widziałam w literaturze, że",
    "O ile dobrze pamiętam z konferencji,",
    "Ktoś mi wspominał, że",
    "Chyba natknęłam się na taki artykuł, że",
    "Mam wrażenie, że czytałam coś na ten temat —",
    "Z tego co obiło mi się o uszy,",
)

GENERIC_CLINICAL_PHRASES = {
    "wskazania", "przeciwwskazania", "dzialania niepozadane",
    "skutki uboczne", "dawkowanie", "bezpieczenstwo", "skutecznosc",
}


def build_sensitive_drug_phrases(drug_info: Dict) -> List[str]:
    candidates: List[str] = []
    for claim in drug_info.get("claims", []) if isinstance(drug_info.get("claims"), list) else []:
        if not isinstance(claim, dict):
            continue
        for key in ("statement", "keywords", "support_patterns", "contradiction_patterns"):
            val = claim.get(key, [])
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list):
                candidates.extend(str(v) for v in val if str(v).strip())
    for key in ("skład", "wskazania", "przeciwwskazania", "działania_niepożądane", "dawkowanie"):
        val = str(drug_info.get(key, "")).strip()
        if val:
            candidates.append(val)

    phrases: List[str] = []
    for raw in candidates:
        norm = normalize_text(raw)
        if not norm:
            continue
        for fragment in re.split(r"[,;().]", norm):
            phrase = fragment.strip()
            if not phrase or phrase in GENERIC_CLINICAL_PHRASES:
                continue
            has_digits = bool(re.search(r"\d", phrase))
            if has_digits or (len(phrase.split()) >= 2 and len(phrase) >= 12):
                phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def pick_hearsay_phrase(session_id: str, turn_index: int) -> str:
    seed = f"{session_id}|{turn_index}|hearsay"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    idx = int(digest[:8], 16) % len(HEARSAY_PHRASES)
    return HEARSAY_PHRASES[idx]


def extract_representative_text_from_messages(messages: List[Dict[str, str]]) -> str:
    """Wyciąga łączony znormalizowany tekst wypowiedzi przedstawiciela z historii LLM."""
    parts = [m["content"] for m in messages if m.get("role") == "user"]
    return normalize_text(" ".join(parts))


def apply_knowledge_guard(
    doctor_message: str,
    messages: List[Dict[str, str]],
    drug_info: Dict,
    session_id: str,
    turn_index: int,
) -> Tuple[str, List[str]]:
    """Przepisuje zdania z danymi leku, których przedstawiciel nie podał, jako zasłyszaną opinię."""
    rep_history = extract_representative_text_from_messages(messages)
    msg_norm = normalize_text(doctor_message)
    if not rep_history or not msg_norm:
        return doctor_message, []

    sensitive = build_sensitive_drug_phrases(drug_info)
    leaked = list(dict.fromkeys(
        p for p in sensitive if p in msg_norm and p not in rep_history
    ))
    if not leaked:
        return doctor_message, []

    hearsay = pick_hearsay_phrase(session_id, turn_index)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doctor_message.strip()) if s.strip()]
    rewritten: List[str] = []
    for sentence in sentences:
        s_norm = normalize_text(sentence)
        if any(p in s_norm for p in leaked):
            rw = f"{hearsay} {sentence[0].lower()}{sentence[1:]}"
            if rw and rw[-1] not in ".!?":
                rw += "."
            rewritten.append(rw)
        else:
            rewritten.append(sentence)

    return " ".join(rewritten).strip(), leaked[:5]
