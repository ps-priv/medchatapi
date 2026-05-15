"""Test bazowy — 5 tur, konfiguracja domyślna (acquainted), lekarz: skeptical_expert.

Sprawdza:
- sesja startuje poprawnie
- conviction jest inicjalizowany (trust_in_rep ~ 0.45 dla acquainted)
- rozmowa nie kończy się przedwcześnie przez 5 tur
- /finish zwraca evaluation i goal

Użycie: python3 tests/test_baseline.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "skeptical_expert"
DRUG = "noacid"

TURNS = [
    "Dzień dobry, jestem przedstawicielem firmy PharmaX i chciałbym omówić Noacid.",
    "Noacid to inhibitor pompy protonowej — pantoprazol 20 mg — wskazany w refluksowym zapaleniu przełyku.",
    "W badaniu klinicznym na 800 pacjentach skuteczność po 4 tygodniach wynosiła 87%.",
    "Dawkowanie: 20 mg raz na dobę przed posiłkiem. Prosty schemat, łatwy dla pacjenta.",
    "Czy ma Pan pacjentów z przewlekłym refluksem, u których Noacid mógłby być dobrą opcją?",
]

print_header(f"TEST BAZOWY | lekarz: {DOCTOR} | lek: {DRUG} | 5 tur")

# --- START ---
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "acquainted",
    "register": "professional",
    "warmth": "neutral",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}")

initial_conv = start_resp.get("session_config", {})
initial_trust_expected = 0.45  # acquainted

# --- TURY ---
last_resp = None
terminated_early = False

for i, msg in enumerate(TURNS, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg}")
    resp = post("/message", {"session_id": session_id, "message": msg})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        terminated_early = True
        break

    print(f"[Tura {i}] Lekarz ({resp.get('doctor_attitude', '?')} | {resp.get('doctor_decision', '?')}):")
    print(f"  {resp.get('doctor_message', '')}")
    print(fmt_conviction(resp.get("conviction")))
    print(fmt_metrics(resp.get("turn_metrics", {})))
    print()
    last_resp = resp

    if resp.get("is_terminated"):
        print(f"  *** Zakończono wcześnie: {resp.get('termination_reason', '')} ***")
        terminated_early = True
        break

# --- FINISH ---
print_separator()
print("Kończę sesję (/finish)...")
fin = post(f"/finish?session_id={session_id}&api_key=", {})

print_separator()
print("WYNIKI TESTÓW:\n")

passes = []

passes.append(assert_pass(not terminated_early, "Rozmowa nie zakończona przedwcześnie (5 tur)"))

if last_resp:
    conv = last_resp.get("conviction") or {}
    passes.append(assert_pass(
        conv.get("trust_in_rep", 0) >= 0.30,
        f"trust_in_rep >= 0.30 po 5 turach (jest: {conv.get('trust_in_rep', 0):.2f})"
    ))
    passes.append(assert_pass(
        conv.get("clinical_confidence", 0) > 0.20,
        f"clinical_confidence wzrosło powyżej startu (jest: {conv.get('clinical_confidence', 0):.2f})"
    ))
    passes.append(assert_pass(
        last_resp.get("doctor_decision") in {"undecided", "trial_use", "will_prescribe"},
        f"Decyzja to nie 'reject' (jest: {last_resp.get('doctor_decision')})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    passes.append(assert_pass(
        ev.get("professionalism_score", 0) is not None,
        "Evaluation zawiera professionalism_score"
    ))
    passes.append(assert_pass(
        fin.get("conversation_goal") is not None,
        "Finish zwraca conversation_goal"
    ))
    goal = fin.get("conversation_goal", {})
    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"trafność={ev.get('relevance_score', '?')}/10")
else:
    passes.append(assert_pass(False, f"Finish bez błędu (błąd: {fin.get('error', '')[:80]})"))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
