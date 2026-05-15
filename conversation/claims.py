"""Silnik claimow: dopasowanie semantyczne, walidacja i coverage."""

import re
import unicodedata
from typing import Dict, List, Set

from .constants import POLISH_STOPWORDS


def normalize_for_semantic(text: str) -> str:
    """Upraszcza tekst do porównań semantycznych."""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    without_diacritics = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", without_diacritics)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize_semantic(text: str) -> List[str]:
    """Rozbija tekst na tokeny do dopasowania claimów."""
    return [token for token in re.findall(r"[a-z0-9-]{3,}", text) if token not in POLISH_STOPWORDS]


def unique_texts(values: List[str]) -> List[str]:
    """Usuwa duplikaty przy zachowaniu kolejności."""
    return list(dict.fromkeys([value.strip() for value in values if value and value.strip()]))


def normalize_claim_severity(value: str) -> str:
    """Normalizuje ważność claimu do critical/major/minor."""
    normalized = str(value or "").strip().lower()
    if normalized in {"critical", "krytyczny", "krytyczna"}:
        return "critical"
    if normalized in {"minor", "mniejszy", "niewielki"}:
        return "minor"
    return "major"


def extract_claim_numbers(claim: Dict) -> Set[str]:
    """Wyciąga liczby z claimu dla walidacji dawkowania."""
    explicit_numbers = claim.get("numbers")
    if isinstance(explicit_numbers, list):
        return {str(number) for number in explicit_numbers if str(number).isdigit()}

    text_blob = " ".join(
        [
            str(claim.get("statement", "")),
            " ".join(str(keyword) for keyword in claim.get("keywords", [])),
            " ".join(str(pattern) for pattern in claim.get("support_patterns", [])),
        ]
    )
    return set(re.findall(r"\b\d{2,4}\b", normalize_for_semantic(text_blob)))


def fallback_claims_for_drug(drug_info: Dict) -> List[Dict]:
    """Buduje minimalny zestaw claimów, gdy lek nie ma sekcji `claims`."""
    return [
        {
            "id": f"{drug_info.get('id', 'drug')}_contraindications",
            "type": "contraindication",
            "severity": "critical",
            "statement": f"Uwzględnij przeciwwskazania: {drug_info.get('przeciwwskazania', '')}",
            "keywords": ["przeciwwskazania", "niewydolnosc nerek", "ciaza", "kwasica", "zakazenia"],
            "support_patterns": ["przeciwwskaz", "ostroznie", "niewydolnosc nerek", "ciaza"],
            "contradiction_patterns": [
                "brak przeciwwskazan",
                "nie ma przeciwwskazan",
                "bez przeciwwskazan",
                "bezpieczny przy niewydolnosci nerek",
                "mozna przy niewydolnosci nerek",
                "bezpieczny w ciazy",
            ],
        },
        {
            "id": f"{drug_info.get('id', 'drug')}_side_effects",
            "type": "side_effect",
            "severity": "major",
            "statement": f"Dzialania niepozadane: {drug_info.get('działania_niepożądane', '')}",
            "keywords": ["dzialania niepozadane", "skutki uboczne", "nudnosci", "biegunka", "swiad"],
            "support_patterns": ["dzialan niepozad", "skutk uboczn", "nudnos", "biegun"],
            "contradiction_patterns": [
                "brak dzialan niepozadanych",
                "nie ma dzialan niepozadanych",
                "zero skutkow ubocznych",
                "bez skutkow ubocznych",
            ],
        },
        {
            "id": f"{drug_info.get('id', 'drug')}_dosage",
            "type": "dosage",
            "severity": "major",
            "statement": f"Dawkowanie: {drug_info.get('dawkowanie', '')}",
            "keywords": ["dawkowanie", "mg", "ampulka", "dziennie", "doba"],
            "support_patterns": ["dawk", "mg", "ampulka", "dziennie"],
            "contradiction_patterns": ["dowolna dawka", "bez ograniczen dawkowania"],
        },
    ]


def load_claims_for_drug(drug_info: Dict) -> List[Dict]:
    """Zwraca listę claimów leku, lub fallback gdy brak danych."""
    claims = drug_info.get("claims", [])
    if isinstance(claims, list) and claims:
        return [claim for claim in claims if isinstance(claim, dict)]
    return fallback_claims_for_drug(drug_info)


def build_claim_catalog(drug_info: Dict) -> Dict:
    """Tworzy indeks claimów oraz listy claimów krytycznych."""
    claims = load_claims_for_drug(drug_info)
    claim_index: Dict[str, Dict] = {}
    critical_claim_ids: List[str] = []
    critical_claim_labels: List[str] = []

    for claim in claims:
        claim_id = str(claim.get("id", "unknown_claim"))
        severity = normalize_claim_severity(claim.get("severity", "major"))
        statement = str(claim.get("statement", "")).strip() or claim_id
        claim_index[claim_id] = {"severity": severity, "statement": statement}
        if severity == "critical":
            critical_claim_ids.append(claim_id)
            critical_claim_labels.append(statement)

    return {
        "claim_index": claim_index,
        "critical_claim_ids": critical_claim_ids,
        "critical_claim_labels": critical_claim_labels,
    }


def update_claim_coverage(state: Dict, claim_check: Dict) -> Dict:
    """Aktualizuje pokrycie claimów na bazie bieżącej wypowiedzi."""
    seen_claim_ids = set(state.get("seen_claim_ids", []))
    covered_critical_ids = set(state.get("covered_critical_claim_ids", []))
    critical_claim_ids = set(state.get("critical_claim_ids", []))

    for match in claim_check.get("claim_matches", []):
        claim_id = str(match.get("id", ""))
        if not claim_id:
            continue
        seen_claim_ids.add(claim_id)
        if claim_id in critical_claim_ids:
            covered_critical_ids.add(claim_id)

    state["seen_claim_ids"] = sorted(seen_claim_ids)
    state["covered_critical_claim_ids"] = sorted(covered_critical_ids)

    missing_critical_ids = sorted(critical_claim_ids.difference(covered_critical_ids))
    claim_index = state.get("claim_index", {})
    missing_critical_labels = [
        str(claim_index.get(claim_id, {}).get("statement", claim_id)) for claim_id in missing_critical_ids
    ]

    return {
        "missing_critical_ids": missing_critical_ids,
        "missing_critical_labels": missing_critical_labels,
    }


def critical_coverage_summary(state: Dict) -> Dict:
    """Zwraca podsumowanie coverage claimów krytycznych."""
    total_critical = len(state.get("critical_claim_ids", []))
    covered_critical = len(state.get("covered_critical_claim_ids", []))
    missing_critical = max(0, total_critical - covered_critical)
    coverage_ratio = 1.0 if total_critical == 0 else covered_critical / max(1, total_critical)

    return {
        "total_critical": total_critical,
        "covered_critical": covered_critical,
        "missing_critical": missing_critical,
        "coverage_ratio": round(max(0.0, min(1.0, coverage_ratio)), 2),
    }


def semantic_claim_score(message_norm: str, message_tokens: Set[str], claim: Dict) -> Dict:
    """Liczy podobieństwo wypowiedzi do claimu."""
    statement = normalize_for_semantic(claim.get("statement", ""))
    keywords = [normalize_for_semantic(keyword) for keyword in claim.get("keywords", []) if str(keyword).strip()]

    claim_tokens = set(tokenize_semantic(statement))
    for keyword in keywords:
        claim_tokens.update(tokenize_semantic(keyword))

    keyword_hits = [keyword for keyword in keywords if keyword and keyword in message_norm]

    overlap = 0.0
    if claim_tokens:
        overlap = len(message_tokens.intersection(claim_tokens)) / max(1, len(claim_tokens))

    phrase_bonus = 0.8 if keyword_hits else 0.0
    return {
        "score": round(max(overlap, phrase_bonus), 3),
        "keyword_hits": keyword_hits,
        "claim_tokens": sorted(claim_tokens),
    }


def check_medical_claims(message: str, drug_info: Dict) -> Dict:
    """Porównuje wypowiedź z faktami o leku i zwraca zgodności/sprzeczności."""
    message_norm = normalize_for_semantic(message)
    message_tokens = set(tokenize_semantic(message_norm))
    has_clinical_intent = bool(re.search(r"\b(wskazan|przeciwwskaz|dzialan|skutk|dawk|mg|bezpiecz)\w*", message_norm))

    false_entries: List[Dict] = []
    unsupported_entries: List[Dict] = []
    supported_entries: List[Dict] = []
    evidence_signals: List[str] = []
    claim_matches: List[Dict] = []
    severity_counts = {"critical": 0, "major": 0, "minor": 0}

    for claim in load_claims_for_drug(drug_info):
        claim_id = str(claim.get("id", "unknown_claim"))
        claim_type = str(claim.get("type", "general")).strip().lower()
        severity = normalize_claim_severity(claim.get("severity", "major"))
        statement = str(claim.get("statement", "")).strip()

        semantic = semantic_claim_score(message_norm, message_tokens, claim)
        semantic_score = semantic["score"]
        keyword_hits = semantic["keyword_hits"]

        contradiction_patterns = [
            normalize_for_semantic(pattern) for pattern in claim.get("contradiction_patterns", []) if str(pattern).strip()
        ]
        support_patterns = [
            normalize_for_semantic(pattern) for pattern in claim.get("support_patterns", []) if str(pattern).strip()
        ]
        contradiction_hits = [pattern for pattern in contradiction_patterns if pattern in message_norm]
        support_hits = [pattern for pattern in support_patterns if pattern in message_norm]

        dosage_intent = bool(re.search(r"\b(dawk|mg|ampulk|tablet)\w*", message_norm))
        message_numbers = set(re.findall(r"\b\d{2,4}\b", message_norm))
        claim_numbers = extract_claim_numbers(claim)
        if claim_type == "dosage" and dosage_intent and message_numbers:
            if claim_numbers and message_numbers.intersection(claim_numbers):
                support_hits.append("numeric_match")
                evidence_signals.append("Padly liczby zgodne z dawkowaniem z bazy leku.")
            elif claim_numbers and not message_numbers.intersection(claim_numbers):
                contradiction_hits.append("numeric_mismatch")

        is_relevant = bool(keyword_hits) or bool(contradiction_hits) or bool(support_hits) or semantic_score >= 0.34
        if not is_relevant:
            continue

        claim_matches.append(
            {
                "id": claim_id,
                "type": claim_type,
                "severity": severity,
                "semantic_score": semantic_score,
                "keyword_hits": keyword_hits,
                "contradiction_hits": contradiction_hits,
                "support_hits": support_hits,
            }
        )

        claim_label = statement or f"Fakt kliniczny ({claim_id})"
        if contradiction_hits:
            text = f"[{severity}] Twierdzenie sprzeczne z faktem: {claim_label}"
            false_entries.append({"text": text, "severity": severity, "claim_id": claim_id})
            severity_counts[severity] += 1
            continue

        if support_hits or semantic_score >= 0.6:
            supported_entries.append({"text": f"[{severity}] Potwierdzono fakt: {claim_label}", "claim_id": claim_id})
            evidence_signals.append(f"Rozmowa odwoluje sie do faktu: {claim_label}")
            continue

        if has_clinical_intent:
            text = f"[{severity}] Twierdzenie bez pokrycia: {claim_label}"
            unsupported_entries.append({"text": text, "severity": severity, "claim_id": claim_id})
            severity_counts[severity] += 1

    return {
        "false_claims": unique_texts([entry["text"] for entry in false_entries]),
        "unsupported_claims": unique_texts([entry["text"] for entry in unsupported_entries]),
        "supported_claims": unique_texts([entry["text"] for entry in supported_entries]),
        "evidence_signals": unique_texts(evidence_signals),
        "claim_matches": claim_matches,
        "severity_counts": severity_counts,
    }


def extract_drug_keywords(drug_info: Dict) -> Set[str]:
    """Buduje zestaw słów-kluczy leku do wykrywania on-topic."""
    raw_text = " ".join(
        str(drug_info.get(field, ""))
        for field in (
            "id",
            "nazwa",
            "skład",
            "wskazania",
            "przeciwwskazania",
            "działania_niepożądane",
            "dawkowanie",
        )
    ).lower()
    tokens = set(re.findall(r"[a-ząćęłńóśźż0-9-]{4,}", raw_text))
    return {token for token in tokens if token not in POLISH_STOPWORDS}
