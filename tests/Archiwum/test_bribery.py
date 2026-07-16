"""Test naruszenia etyki (łapówka) — natychmiastowe zakończenie rozmowy.

Scenariusz: w 3. turze przedstawiciel proponuje korzyść majątkową za przepisywanie leku.
Oczekiwane zachowanie:
- lekarz natychmiast kończy rozmowę (is_terminated = True)
- termination_reason zawiera informację o naruszeniu etyki
- /finish zwraca historię i ocenę z niskim wynikiem etycznym
- conviction trust_in_rep spada do bardzo niskiego poziomu

Użycie: python3 tests/test_bribery.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "dominant_authority"
DRUG = "noacid"

TURNS_BEFORE_BRIBERY = [
    "Dzień dobry, jestem przedstawicielem PharmaX, chciałbym omówić Noacid.",
    "Noacid to pantoprazol 20 mg, skuteczny w leczeniu refluksowego zapalenia przełyku.",
]

BRIBERY_MESSAGE = (
    "Panie Doktorze, mamy dla lekarzy specjalny program lojalnościowy — "
    "za każde wypisanie 10 recept na Noacid dostaje Pan voucher do spa o wartości 500 zł."
)

TURNS_AFTER = [
    "Oczywiście to tylko symboliczne podziękowanie za Pana czas i zaangażowanie...",
]

print_header(f"TEST ŁAPÓWKI (ETYKA) | lekarz: {DOCTOR} | lek: {DRUG}")

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

# --- TURY PRZED ŁAPÓWKĄ ---
terminated_at_bribery = False
terminated_before = False
bribery_resp = None

for i, msg in enumerate(TURNS_BEFORE_BRIBERY, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg}")
    resp = post("/message", {"session_id": session_id, "message": msg})
    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        terminated_before = True
        break
    print(f"[Tura {i}] Lekarz: {resp.get('doctor_message', '')}")
    print(fmt_conviction(resp.get("conviction")))
    print()
    if resp.get("is_terminated"):
        print("  *** Zakończono PRZED łapówką — nieoczekiwane ***")
        terminated_before = True
        break

# --- ŁAPÓWKA ---
if not terminated_before:
    bribery_turn = len(TURNS_BEFORE_BRIBERY) + 1
    print(f"[Tura {bribery_turn}] Przedstawiciel (ŁAPÓWKA): {BRIBERY_MESSAGE}")
    resp = post("/message", {"session_id": session_id, "message": BRIBERY_MESSAGE})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
    else:
        print(f"[Tura {bribery_turn}] Lekarz ({resp.get('doctor_attitude', '?')} | {resp.get('doctor_decision', '?')}):")
        print(f"  {resp.get('doctor_message', '')}")
        print(fmt_conviction(resp.get("conviction")))
        print(f"  is_terminated: {resp.get('is_terminated')}")
        print(f"  termination_reason: {resp.get('termination_reason', '')}")
        print()
        bribery_resp = resp
        if resp.get("is_terminated"):
            terminated_at_bribery = True
            print("  *** Rozmowa zakończona po łapówce — poprawnie ***")

    # Spróbuj wysłać kolejną wiadomość — powinna być zablokowana
    if terminated_at_bribery:
        print(f"\n[Sprawdzam czy sesja jest zablokowana po zakończeniu...]")
        after_resp = post("/message", {"session_id": session_id, "message": TURNS_AFTER[0]})
        print(f"  Odpowiedź: {after_resp}")
        blocked = "error" in after_resp or "zakończona" in str(after_resp.get("error", "")).lower()

# --- FINISH ---
print_separator()
print("Kończę sesję (/finish)...")
fin = post(f"/finish?session_id={session_id}&api_key=", {})

# --- WYNIKI ---
print_separator()
print("WYNIKI TESTÓW:\n")

passes = []

passes.append(assert_pass(
    not terminated_before,
    "Rozmowa nie zakończyła się PRZED łapówką"
))

passes.append(assert_pass(
    terminated_at_bribery,
    "Rozmowa zakończyła się bezpośrednio po propozycji łapówki (is_terminated=True)"
))

if bribery_resp:
    reason = bribery_resp.get("termination_reason", "") or ""
    passes.append(assert_pass(
        bool(reason),
        f"termination_reason jest niepuste (jest: '{reason[:80]}')"
    ))
    passes.append(assert_pass(
        bribery_resp.get("doctor_decision") == "reject",
        f"Decyzja to 'reject' (jest: {bribery_resp.get('doctor_decision')})"
    ))
    conv = bribery_resp.get("conviction") or {}
    passes.append(assert_pass(
        conv.get("trust_in_rep", 1.0) < 0.30,
        f"trust_in_rep < 0.30 po łapówce (jest: {conv.get('trust_in_rep', 0):.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    ethics = ev.get("ethics_score", 10)
    if ethics is not None:
        passes.append(assert_pass(
            ethics <= 5,
            f"Ocena etyczna <= 5/10 po łapówce (jest: {ethics})"
        ))
    print(f"\n  Feedback: {ev.get('final_feedback', '')[:200]}")
else:
    passes.append(assert_pass(False, f"Finish bez błędu (błąd: {fin.get('error', '')[:80]})"))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
