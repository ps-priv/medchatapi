"""Test spójności rejestru komunikacji — lekarz nie powinien dryfować w trakcie rozmowy.

Sprawdza regresję dla poprawki w prompt_builder.py (reguła #3 BEZPIECZNIKÓW),
która ma pilnować formy zwracania się LEKARZ -> PRZEDSTAWICIEL przez całą
rozmowę, niezależnie od frustracji czy zmiany tonu.

Scenariusz A (familiar + informal, "ty"):
  - lekarz nie powinien w żadnej turze przejść na 'Panie {imię}' / 'Pani {imię}'
  - wplatam tury z marketingiem/fałszywym claimem, żeby podnieść frustrację
    i sprawdzić, czy to nie wywołuje dryfu w stronę formalności

Scenariusz B (acquainted + professional, "Pan/Pani"):
  - lekarz nie powinien w żadnej turze użyć bezpośrednio zaimka 'Ty'
    (heurystyka — nie łapie wszystkich form czasownikowych, ale łapie
    najbardziej rażący przypadek)

Użycie: python3 tests/test_register_consistency.py
Wymaga: uvicorn api4:app --port 8000
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DRUG = "noacid"

REP_NAME_INFORMAL = "Paweł"
REP_STEM_INFORMAL = "Paw"  # łapie odmianę wołacza: Pawle, Pawła...

TURNS_INFORMAL = [
    "Cześć Marek, wpadłem pogadać o Noacid — znasz mnie, więc będę konkretny.",
    "Noacid to absolutnie rewolucyjny, przełomowy lek — game-changer w leczeniu refluksu!",
    "Noacid można bezpiecznie stosować u wszystkich pacjentów — nie ma żadnych przeciwwskazań.",
    "Wiem, że masz sporo pacjentów na NLPZ — Noacid skutecznie chroni ich przed owrzodzeniami.",
    "Dawkowanie: 20 mg raz na dobę. Jedno przeciwwskazanie: alergia na soję.",
    "Ostatnie badanie z 2024 pokazało 15% wyższy compliance niż przy starszych IPP.",
    "Masz pacjentów, którzy wciąż refluksują mimo starszych IPP? Noacid mógłby być zmianą.",
    "Zostawię ci próbki — spróbujesz u kilku pacjentów?",
]

TURNS_PROFESSIONAL = [
    "Dzień dobry, jestem przedstawicielem PharmaX, chciałbym omówić Noacid.",
    "Noacid to pantoprazol 20 mg, inhibitor pompy protonowej na refluks.",
    "To najlepszy lek na rynku, absolutny przełom — proszę zaufać naszym danym.",
    "W badaniu na 800 pacjentach skuteczność po 4 tygodniach wynosiła 87%.",
    "Dawkowanie: 20 mg raz na dobę przed posiłkiem.",
    "Czy ma Pan/Pani pacjentów z przewlekłym refluksem, u których Noacid mógłby pomóc?",
]

FORMAL_ADDRESS_RE = re.compile(rf"\b[Pp]an(ie)?\s+{REP_STEM_INFORMAL}\w*|\b[Pp]ani\s+{REP_STEM_INFORMAL}\w*")
BARE_TY_RE = re.compile(r"\bTy\b|\btobie\b|\bciebie\b", re.IGNORECASE)


def run_scenario(label, doctor, session_config, turns):
    print_header(label)
    start_resp = post(f"/start?id={doctor}&drug_id={DRUG}", session_config)
    if "error" in start_resp:
        print("BŁĄD /start:", start_resp)
        return None, [], True

    session_id = start_resp["session_id"]
    print(f"Sesja: {session_id}\n")

    messages = []
    terminated_early = False
    last_resp = None

    for i, msg in enumerate(turns, 1):
        print(f"[Tura {i}] Przedstawiciel: {msg}")
        resp = post("/message", {"session_id": session_id, "message": msg})
        if "error" in resp:
            print(f"  BŁĄD: {resp['error']}")
            terminated_early = True
            break

        doctor_message = resp.get("doctor_message", "")
        print(f"[Tura {i}] Lekarz ({resp.get('doctor_attitude', '?')} | {resp.get('doctor_decision', '?')}):")
        print(f"  {doctor_message}")
        print(fmt_conviction(resp.get("conviction")))
        print(fmt_metrics(resp.get("turn_metrics", {})))
        print()

        messages.append((i, doctor_message))
        last_resp = resp
        if resp.get("is_terminated"):
            print(f"  *** Zakończono wcześnie: {resp.get('termination_reason', '')} ***")
            terminated_early = True
            break

    print_separator()
    print("Kończę sesję (/finish)...")
    post(f"/finish?session_id={session_id}&api_key=", {})

    return last_resp, messages, terminated_early


passes = []

# --- Scenariusz A: familiar + informal ---
last_a, messages_a, terminated_a = run_scenario(
    "TEST SPÓJNOŚCI REJESTRU | familiar + informal (ty)",
    doctor="friendly_generalist",
    session_config={
        "familiarity": "familiar",
        "register": "informal",
        "warmth": "warm",
        "rep_name": REP_NAME_INFORMAL,
    },
    turns=TURNS_INFORMAL,
)

passes.append(assert_pass(
    len(messages_a) >= 5,
    f"Scenariusz A: zebrano wystarczającą próbkę tur do sprawdzenia dryfu (jest: {len(messages_a)}, "
    f"terminated_early={terminated_a})"
))

drift_hits_a = [(i, msg) for i, msg in messages_a if FORMAL_ADDRESS_RE.search(msg)]
passes.append(assert_pass(
    len(drift_hits_a) == 0,
    f"Scenariusz A: lekarz nie zwrócił się formalnie ('Panie/Pani {REP_NAME_INFORMAL}') "
    f"w żadnej turze (trafienia: {[i for i, _ in drift_hits_a]})"
))

# --- Scenariusz B: acquainted + professional ---
last_b, messages_b, terminated_b = run_scenario(
    "TEST SPÓJNOŚCI REJESTRU | acquainted + professional (Pan/Pani)",
    doctor="skeptical_expert",
    session_config={
        "familiarity": "acquainted",
        "register": "professional",
        "warmth": "neutral",
    },
    turns=TURNS_PROFESSIONAL,
)

passes.append(assert_pass(
    len(messages_b) >= 4,
    f"Scenariusz B: zebrano wystarczającą próbkę tur do sprawdzenia dryfu (jest: {len(messages_b)}, "
    f"terminated_early={terminated_b})"
))

drift_hits_b = [(i, msg) for i, msg in messages_b if BARE_TY_RE.search(msg)]
passes.append(assert_pass(
    len(drift_hits_b) == 0,
    f"Scenariusz B: lekarz nie zwrócił się na 'ty' w żadnej turze "
    f"(trafienia: {[i for i, _ in drift_hits_b]})"
))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
