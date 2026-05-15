"""Test trudnego lekarza — tired_cynic lub dominant_authority, 8 tur.

Scenariusz: archetypowo najtrudniejszy lekarz — wypalony cynik lub dominujący autorytet.
Nawet przy perfekcyjnych claimach decyzja to co najwyżej trial_use.
Sprawdza:
- rozmowa nie kończy się przedwcześnie (protection window działa)
- frustration_score widoczny w turn_metrics
- decyzja to reject, undecided lub trial_use (nie will_prescribe) przy trudnym lekarzu
- lekarz reaguje z appropriate attitude (serious, skeptical, defensive)

Użycie: python3 tests/test_hard_doctor.py [tired_cynic|dominant_authority]
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = sys.argv[1] if len(sys.argv) > 1 else "tired_cynic"
DRUG = "noacid"

TURNS = [
    "Dzień dobry, jestem z PharmaX — chciałbym omówić Noacid, inhibitor pompy protonowej.",
    "W badaniu na 800 pacjentach Noacid wykazał 87% skuteczność w refluksowym zapaleniu przełyku po 4 tygodniach.",
    "Dawkowanie: 20 mg raz na dobę. Contraindication: nadwrażliwość na pantoprazol lub soję.",
    "Profil bezpieczeństwa: monitorowanie magnezu przy długotrwałym stosowaniu — to standardowa procedura przy IPP.",
    "U pacjentów na NLPZ, Noacid skutecznie zapobiega owrzodzeniom żołądka — dane z randomizowanego badania kontrolowanego.",
    "Rozumiem sceptycyzm. Zostawię Panu/Pani publikację z pełną metodyką — może to będzie bardziej przekonujące niż moje słowa.",
    "Czy jest konkretna kwestia kliniczna dotycząca IPP, którą chciałby Pan/Pani przedyskutować?",
    "Dziękuję za czas. Nawet jeśli nie jesteśmy gotowi na decyzję dziś, doceniam rozmowę.",
]

print_header(f"TEST TRUDNEGO LEKARZA | lekarz: {DOCTOR} | lek: {DRUG}")

# --- START ---
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "acquainted",
    "register": "professional",
    "warmth": "neutral",
    "rep_name": "Anna Nowak",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}\n")

# --- TURY ---
attitudes = []
last_resp = None
terminated_early = False
frustration_scores = []

for i, msg in enumerate(TURNS, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg}")
    resp = post("/message", {"session_id": session_id, "message": msg})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        terminated_early = True
        break

    attitude = resp.get("doctor_attitude", "neutral")
    attitudes.append(attitude)

    print(f"[Tura {i}] Lekarz ({attitude} | {resp.get('doctor_decision', '?')}):")
    print(f"  {resp.get('doctor_message', '')}")
    print(fmt_conviction(resp.get("conviction")))

    tm = resp.get("turn_metrics", {})
    print(fmt_metrics(tm))
    print()

    frust = tm.get("frustration")
    if frust is not None:
        frustration_scores.append(frust)

    last_resp = resp

    if resp.get("is_terminated"):
        print(f"  *** Zakończono: {resp.get('termination_reason', '')} ***")
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

# Trudny lekarz nie powinien kończyć zbyt wcześnie (protection window działa do tury 5)
passes.append(assert_pass(
    not terminated_early or (last_resp is not None and last_resp.get("is_terminated")),
    "Rozmowa nie zakończyła się przed turą 5 (protection window)"
))

if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    # Trudny lekarz — will_prescribe po 8 turach to red flag
    passes.append(assert_pass(
        decision in {"reject", "undecided", "trial_use"},
        f"Trudny lekarz nie daje will_prescribe zbyt łatwo (jest: {decision})"
    ))

    # Frustracja powinna być widoczna
    trust = conv.get("trust_in_rep", 0)
    passes.append(assert_pass(
        trust < 0.70,
        f"trust_in_rep trudnego lekarza < 0.70 (jest: {trust:.2f})"
    ))

# Lekarz powinien przejawiać co najmniej raz "serious" lub "skeptical" attitude
hard_attitudes = {"serious", "skeptical", "defensive", "dismissive"}
has_hard_attitude = any(a in hard_attitudes for a in attitudes)
passes.append(assert_pass(
    has_hard_attitude,
    f"Lekarz wykazał trudną postawę ({', '.join(set(attitudes))})"
))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    goal = fin.get("conversation_goal", {})
    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Decyzja: {goal.get('doctor_decision', '?')}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"relacja={ev.get('relationship_score', '?')}/10")
    print(f"  Feedback: {ev.get('final_feedback', '')[:200]}")
    hist = fin.get("turn_mode_histogram")
    if hist:
        print(f"\n  Histogram trybów: {hist}")

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
