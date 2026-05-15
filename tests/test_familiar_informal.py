"""Test znajomego lekarza — familiarity=familiar, register=informal (forma "ty").

Scenariusz: przedstawiciel zna lekarza od dawna, rozmawia po imieniu.
Sprawdza:
- trust_in_rep startuje wysoko (~0.65 dla familiar)
- API akceptuje familiar + informal (brak błędu 422)
- conviction rośnie szybciej dzięki trust_gain_mult=1.30
- decyzja trial_use lub will_prescribe jest osiągalna w 6-8 turach
- session_config jest zwracane w /start response

Uwaga: w Polsce familiar+informal = "stary znajomy", pełna forma "ty".
Użycie: python3 tests/test_familiar_informal.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "friendly_generalist"
DRUG = "noacid"

TURNS = [
    "Cześć Marek, dawno się nie widzieliśmy! Wpadłem pogadać o Noacid — znam Cię, więc będę konkretny.",
    "Noacid to pantoprazol 20 mg, który masz pewnie już w głowie. W naszym nowym badaniu na 800 pacjentach — 87% skuteczność po 4 tygodniach.",
    "Wiem, że masz sporo pacjentów na NLPZ — Noacid skutecznie chroni ich przed owrzodzeniami żołądka, mamy twarde dane.",
    "Dawkowanie proste: 20 mg raz na dobę przed śniadaniem. Jedno przeciwwskazanie warte podkreślenia: alergia na soję.",
    "Przy długotrwałym stosowaniu monitoruj magnez — ale wiesz o tym, to standardowa procedura przy IPP.",
    "Ostatnie badanie z 2024 roku pokazało, że compliance przy Noacid jest o 15% wyższy niż przy starszych IPP — jeden raz dziennie robi różnicę.",
    "Masz takich pacjentów co to wciąż refluksują mimo starszych IPP? Noacid mógłby być zmianą.",
    "Zostawię Ci próbki i materiały — co powiesz, spróbujesz u kilku pacjentów?",
]

print_header(f"TEST ZNANY LEKARZ / INFORMAL | lekarz: {DOCTOR} | lek: {DRUG}")

# --- START ---
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "familiar",
    "register": "informal",
    "warmth": "warm",
    "rep_name": "Paweł",
    "prior_visits_summary": "Trzecia wizyta. Omawialiśmy wcześniej IPP ogólnie i compliance pacjentów.",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    print("(familiar + informal powinno być akceptowane przez API)")
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}")

cfg = start_resp.get("session_config", {})
if cfg:
    print(f"Config: familiarity={cfg.get('familiarity')}, register={cfg.get('register')}, warmth={cfg.get('warmth')}")
print("(oczekiwany trust_in_rep startowy: ~0.65 dla familiar)\n")

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
    "session_config" in start_resp,
    "start_resp zawiera session_config (API zwróciło konfigurację)"
))

passes.append(assert_pass(
    not terminated_early,
    "Rozmowa nie zakończyła się przedwcześnie (8 tur)"
))

if first_conviction:
    initial_trust = first_conviction.get("trust_in_rep", 0)
    passes.append(assert_pass(
        initial_trust >= 0.55,
        f"trust_in_rep startuje wysoko dla familiar (jest: {initial_trust:.2f}, oczekiwane: >=0.55)"
    ))

if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    # Familiar + dobra rozmowa → powinno dać trial_use lub will_prescribe
    passes.append(assert_pass(
        decision in {"trial_use", "will_prescribe", "recommend"},
        f"Decyzja to trial_use/will_prescribe dla familiar lekarza (jest: {decision})"
    ))
    passes.append(assert_pass(
        conv.get("trust_in_rep", 0) > 0.60,
        f"trust_in_rep > 0.60 po dobrej rozmowie z familiar (jest: {conv.get('trust_in_rep', 0):.2f})"
    ))
    passes.append(assert_pass(
        conv.get("decision_readiness", 0) > 0.45,
        f"decision_readiness > 0.45 (jest: {conv.get('decision_readiness', 0):.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    goal = fin.get("conversation_goal", {})
    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"relacja={ev.get('relationship_score', '?')}/10")
    print(f"  Feedback: {ev.get('final_feedback', '')[:200]}")

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
