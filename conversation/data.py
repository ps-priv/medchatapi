"""Ladowanie danych domenowych: profile lekarzy i leki."""

import json
import os
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCTOR_ARCHETYPES_PATH = os.path.join(BASE_DIR, "doctor_archetypes.json")
DRUGS_PATH = os.path.join(BASE_DIR, "drugs.json")


def load_doctor_archetypes() -> List[Dict]:
    """Zwraca liste archetypow lekarzy z pliku JSON."""
    with open(DOCTOR_ARCHETYPES_PATH, encoding="utf-8") as file:
        return json.load(file)


def load_drugs() -> List[Dict]:
    """Zwraca liste lekow z pliku JSON."""
    with open(DRUGS_PATH, encoding="utf-8") as file:
        return json.load(file)


def get_doctor_by_id(doctor_id: str) -> Optional[Dict]:
    """Wyszukuje archetyp lekarza po identyfikatorze."""
    for doctor in load_doctor_archetypes():
        if doctor.get("id") == doctor_id:
            return doctor
    return None


def get_drug_by_id(drug_id: str) -> Optional[Dict]:
    """Wyszukuje lek po identyfikatorze."""
    for drug in load_drugs():
        if drug.get("id") == drug_id:
            return drug
    return None
