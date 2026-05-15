"""Test marketingowego języka — buzzwords zamiast danych klinicznych.

Scenariusz: przedstawiciel używa tylko ogólnikowych marketingowych zwrotów
bez konkretnych danych (brak liczb, brak evidence, samo "najlepszy", "rewolucyjny").
Sprawdza:
- ethics score w turn_metrics spada (flagowanie marketingowego języka)
- trust_in_rep nie rośnie (brak evidence)
- clinical_confidence nie rośnie (brak claimów do walidacji)
- lekarz jest sfrustrowany lub kończący rozmowę
- finalna ocena trafności argumentów <= 5/10

Użycie: python3 tests/test_marketing_language.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "evidence_seeker"
DRUG = "noacid"

# Czysto marketingowy język — bez liczb, bez wskazań, bez danych
TURNS = [
    "Dzień dobry! Mam dziś dla Pana coś absolutnie rewolucyjnego — Noacid, przełom w leczeniu żołądka!",
    "Noacid to innowacyjny, przełomowy lek nowej generacji, który odmieni życie Pana pacjentów na zawsze.",
    "Lekarze na całym świecie już pokochali Noacid — to must-have w każdej praktyce medycznej!",
    "Pacjenci są zachwyceni Noacid — mamy fantastyczne opinie, wszyscy mówią, że to najlepszy lek!",
    "Noacid jest zdecydowanie lepszy od konkurencji pod każdym względem — to po prostu najlepsza opcja.",
    "Polecam gorąco — Noacid to game-changer, który zrewolucjonizuje leczenie Pana pacjentów!",
]

print_header(f"TEST MARKETINGOWEGO JĘZYKA | lekarz: {DOCTOR} | lek: {DRUG}")

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
print(f"Sesja: {session_id}\n")

# --- TURY ---
first_conviction = None
last_resp = None
terminated_early = False
ethics_scores = []

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

    tm = resp.get("turn_metrics", {})
    print(fmt_metrics(tm))
    print()

    ethics = tm.get("ethics", None)
    if ethics is not None:
        ethics_scores.append(ethics)

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

# Marketingowy język nie powinien dać will_prescribe
if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    passes.append(assert_pass(
        decision != "will_prescribe",
        f"Decyzja NIE jest will_prescribe przy czystym marketingu (jest: {decision})"
    ))

    # clinical_confidence nie powinna urosnąć — brak claimów medycznych
    cc = conv.get("clinical_confidence", 0)
    cc_start = first_conviction.get("clinical_confidence", 0) if first_conviction else 0.20
    passes.append(assert_pass(
        cc < cc_start + 0.15,
        f"clinical_confidence nie rośnie znacząco bez danych klinicznych "
        f"(start: {cc_start:.2f}, koniec: {cc:.2f})"
    ))

# Ocena ethics powinna być niska przy marketingowym języku
if ethics_scores:
    avg_ethics = sum(ethics_scores) / len(ethics_scores)
    passes.append(assert_pass(
        avg_ethics < 0.80,
        f"Średni ethics score < 0.80 przy marketingowym języku (jest: {avg_ethics:.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    relevance = ev.get("relevance_score", 10)
    if relevance is not None:
        passes.append(assert_pass(
            relevance <= 5,
            f"Ocena trafności argumentów <= 5/10 przy czystym marketingu (jest: {relevance})"
        ))
    goal = fin.get("conversation_goal", {})
    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"trafność={ev.get('relevance_score', '?')}/10")
    print(f"  Feedback: {ev.get('final_feedback', '')[:200]}")
else:
    passes.append(assert_pass(False, f"Finish bez błędu (błąd: {fin.get('error', '')[:80]})"))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
