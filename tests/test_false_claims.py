"""Test fałszywych claimów medycznych — lekarz: evidence_seeker.

Scenariusz: przedstawiciel podaje błędne informacje kliniczne o Noacid
(sprzeczne z katalogiem claims) — sprawdzamy czy:
- trust_in_rep spada
- clinical_confidence nie rośnie (bo claims odrzucone)
- finalna decyzja to reject lub undecided (nie will_prescribe)
- evaluation nisko ocenia trafność argumentów

Fałszywe claimy użyte w teście (sprzeczne z claims.json):
- "Noacid można stosować u wszystkich pacjentów, brak przeciwwskazań" [critical — naruszenie noacid_contraindications]
- "Nie odnotowano żadnych działań niepożądanych" [major — naruszenie noacid_side_effects]
- "Dawka jest elastyczna — od 5 do 80 mg" [major — naruszenie noacid_dosage]
- "Skuteczność 99% w ciągu 1 tygodnia" [major — przesada, naruszenie noacid_efficacy]

Użycie: python3 tests/test_false_claims.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "evidence_seeker"
DRUG = "noacid"

TURNS = [
    "Dzień dobry, chciałbym omówić Noacid — nasz nowoczesny lek na refluks.",
    # Fałszywy claim 1: brak przeciwwskazań [critical]
    "Noacid można bezpiecznie stosować u wszystkich pacjentów — produkt nie ma żadnych przeciwwskazań.",
    # Fałszywy claim 2: brak działań niepożądanych [major]
    "W badaniach klinicznych nie odnotowano żadnych działań niepożądanych — lek jest absolutnie bezpieczny.",
    # Fałszywy claim 3: błędne dawkowanie [major]
    "Dawkowanie jest elastyczne — od 5 do 80 mg dziennie, lekarz może dobrać dowolną dawkę.",
    # Fałszywy claim 4: przesadzona skuteczność [major]
    "Noacid wykazuje 99% skuteczność już po 7 dniach — to najszybciej działający IPP na rynku.",
    "Polecam włączenie Noacid dla wszystkich pacjentów z problemami żołądkowymi.",
]

print_header(f"TEST FAŁSZYWYCH CLAIMÓW | lekarz: {DOCTOR} | lek: {DRUG}")

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
conviction_start = None
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
        conviction_start = dict(resp["conviction"])
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

# Fałszywe claimy powinny obniżyć zaufanie — reject lub undecided
if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    passes.append(assert_pass(
        decision in {"reject", "undecided"},
        f"Decyzja to reject lub undecided po fałszywych claimach (jest: {decision})"
    ))
    passes.append(assert_pass(
        decision != "will_prescribe",
        "Decyzja NIE jest will_prescribe po fałszywych claimach"
    ))

    # trust_in_rep powinno spaść lub nie rosnąć po fałszywych claimach
    if conviction_start:
        trust_start = conviction_start.get("trust_in_rep", 0)
        trust_end = conv.get("trust_in_rep", 0)
        passes.append(assert_pass(
            trust_end <= trust_start + 0.05,
            f"trust_in_rep nie rośnie przy fałszywych claimach "
            f"({trust_start:.2f} → {trust_end:.2f})"
        ))

    # clinical_confidence nie powinna mocno rosnąć — fałszywe claimy są odrzucane
    cc = conv.get("clinical_confidence", 0)
    passes.append(assert_pass(
        cc < 0.60,
        f"clinical_confidence nie skoczyło wysoko przy odrzuconych claimach (jest: {cc:.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    relevance = ev.get("relevance_score", 10) or 10
    passes.append(assert_pass(
        relevance <= 7,
        f"Ocena trafności argumentów <= 7/10 po fałszywych claimach (jest: {relevance})"
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
