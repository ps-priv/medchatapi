"""Test pierwszego spotkania — familiarity=first_meeting, zimny start.

Sprawdza:
- trust_in_rep startuje od ~0.20 (niżej niż acquainted=0.45)
- lekarz jest bardziej powściągliwy (decyzja raczej undecided po 6 turach)
- nawet dobre claimy nie prowadzą do will_prescribe tak szybko jak przy acquainted
- INFORMAL + FIRST_MEETING jest odrzucone przez API (HTTP 422)

Użycie: python3 tests/test_first_meeting.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "skeptical_expert"
DRUG = "noacid"

TURNS = [
    "Dzień dobry, nazywam się Jan Kowalski i reprezentuję firmę PharmaX. Chciałbym omówić nasz lek Noacid.",
    "Noacid to pantoprazol 20 mg — inhibitor pompy protonowej wskazany w refluksowym zapaleniu przełyku.",
    "W badaniu klinicznym na 800 pacjentach Noacid wykazał 87% skuteczność po 4 tygodniach leczenia.",
    "Dawkowanie: 20 mg raz na dobę przed posiłkiem. Lek jest przeciwwskazany przy nadwrażliwości na pantoprazol lub soję.",
    "Przy długotrwałym stosowaniu zalecamy monitorowanie poziomu magnezu — ryzyko hipomagnezemii jest znane i zarządzalne.",
    "Czy jest coś, co chciałby Pan dowiedzieć się więcej o profilu bezpieczeństwa Noacid?",
]

print_header(f"TEST PIERWSZEGO SPOTKANIA | lekarz: {DOCTOR} | lek: {DRUG}")

# --- Sprawdź że INFORMAL + FIRST_MEETING jest odrzucone ---
print("Sprawdzam walidację: informal + first_meeting powinno dać HTTP 422...")
invalid_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "first_meeting",
    "register": "informal",
    "warmth": "warm",
})
validation_rejected = (
    "error" in invalid_resp
    and ("422" in str(invalid_resp.get("status_code", "")) or "validate" in str(invalid_resp.get("error", "")).lower()
         or "informal" in str(invalid_resp.get("error", "")).lower()
         or "first_meeting" in str(invalid_resp.get("error", "")).lower())
)
print(f"  Odpowiedź walidacji: {str(invalid_resp)[:120]}")
print()

# --- START (poprawna konfiguracja) ---
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "first_meeting",
    "register": "professional",
    "warmth": "neutral",
    "rep_name": "Jan Kowalski",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}")
print("(oczekiwany trust_in_rep startowy: ~0.20 dla first_meeting)\n")

# --- TURY ---
first_conviction = None
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

    if i == 1 and resp.get("conviction"):
        first_conviction = dict(resp["conviction"])
    last_resp = resp

    if resp.get("is_terminated"):
        print(f"  *** Zakończono wcześnie: {resp.get('termination_reason', '')} ***")
        terminated_early = True
        break

# --- FINISH ---
print_separator()
print("Kończę sesję (/finish)...")
fin = post(f"/finish?session_id={session_id}&api_key=", {})

# --- WYNIKI ---
print_separator()
print("WYNIKI TESTÓW:\n")

passes = []

passes.append(assert_pass(
    validation_rejected,
    "API odrzuca informal + first_meeting (HTTP 422 lub błąd walidacji)"
))

passes.append(assert_pass(
    not terminated_early,
    "Rozmowa nie zakończyła się przedwcześnie (6 tur)"
))

if first_conviction:
    initial_trust = first_conviction.get("trust_in_rep", 0)
    passes.append(assert_pass(
        initial_trust <= 0.30,
        f"trust_in_rep startuje nisko dla first_meeting (jest: {initial_trust:.2f}, oczekiwane: <=0.30)"
    ))

if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    # Po 6 turach przy pierwszym spotkaniu — raczej undecided lub trial_use, nie will_prescribe
    passes.append(assert_pass(
        decision in {"undecided", "trial_use"},
        f"Po 6 turach pierwszego spotkania decyzja to undecided/trial_use (jest: {decision})"
    ))

    # trust powinno być niższe niż przy acquainted w tym samym momencie
    trust_after_6 = conv.get("trust_in_rep", 0)
    passes.append(assert_pass(
        trust_after_6 < 0.60,
        f"trust_in_rep < 0.60 po 6 turach pierwszego spotkania (jest: {trust_after_6:.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    goal = fin.get("conversation_goal", {})
    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"trafność={ev.get('relevance_score', '?')}/10")

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
