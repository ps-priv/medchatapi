"""Test wzorcowej rozmowy — 10 tur, lekarz: friendly_generalist, familiarity: acquainted.

Sprawdza:
- conviction rośnie przez całą rozmowę
- trust_in_rep rośnie (nie stoi w miejscu)
- clinical_confidence rośnie z diminishing returns
- finalna decyzja: trial_use lub will_prescribe
- score w evaluation >= 6/10

Użycie: python3 tests/test_good_conversation.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "friendly_generalist"
DRUG = "noacid"

TURNS = [
    "Dzień dobry, jestem Jan Kowalski z PharmaX. Dziś chciałbym omówić Noacid.",
    "Noacid to pantoprazol 20 mg — inhibitor pompy protonowej wskazany w refluksowym zapaleniu przełyku i chorobie wrzodowej.",
    "W badaniu klinicznym na 800 pacjentach skuteczność po 4 tygodniach wynosiła 87% w grupie z refluksem.",
    "Dawkowanie jest proste: 20 mg raz na dobę przed śniadaniem. Pacjenci dobrze to zapamiętują.",
    "Noacid jest wskazany w długoterminowym leczeniu i zapobieganiu nawrotom refluksowego zapalenia przełyku.",
    "W zakresie bezpieczeństwa: przeciwwskazany u pacjentów z nadwrażliwością na pantoprazol lub soję — to ważna informacja kliniczna.",
    "W badaniu na pacjentach stosujących NLPZ, Noacid skutecznie zapobiegał owrzodzeniom żołądka i dwunastnicy.",
    "Przy długotrwałym stosowaniu warto monitorować poziom magnezu — ryzyko hipomagnezemii jest udokumentowane i zarządzalne.",
    "Podsumowując: Noacid łączy skuteczność 87% po 4 tygodniach z prostym dawkowaniem 20 mg/dobę i dobrym profilem bezpieczeństwa.",
    "Czy widzi Pani pacjentów z przewlekłym refluksem lub na NLPZ, którym Noacid mógłby pomóc? Chętnie zostawię próbki.",
]

print_header(f"TEST WZORCOWY | lekarz: {DOCTOR} | lek: {DRUG} | 10 tur")

# --- START ---
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "acquainted",
    "register": "professional",
    "warmth": "warm",
    "rep_name": "Jan Kowalski",
    "prior_visits_summary": "Druga wizyta. Poprzednio rozmawialiśmy ogólnie o IPP.",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}")
if start_resp.get("session_config"):
    cfg = start_resp["session_config"]
    print(f"Config: familiarity={cfg['familiarity']}, register={cfg['register']}, warmth={cfg['warmth']}\n")

# --- TURY ---
conviction_history = []
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

    if resp.get("conviction"):
        conviction_history.append(resp["conviction"])
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

passes.append(assert_pass(not terminated_early, "Rozmowa nie zakończona przedwcześnie (10 tur)"))

if last_resp:
    conv = last_resp.get("conviction") or {}
    decision = last_resp.get("doctor_decision", "")

    passes.append(assert_pass(
        decision in {"trial_use", "will_prescribe", "recommend"},
        f"Finalna decyzja to trial_use/will_prescribe (jest: {decision})"
    ))
    passes.append(assert_pass(
        conv.get("trust_in_rep", 0) > 0.50,
        f"trust_in_rep > 0.50 po 10 dobrych turach (jest: {conv.get('trust_in_rep', 0):.2f})"
    ))
    passes.append(assert_pass(
        conv.get("clinical_confidence", 0) > 0.50,
        f"clinical_confidence > 0.50 (jest: {conv.get('clinical_confidence', 0):.2f})"
    ))
    passes.append(assert_pass(
        conv.get("decision_readiness", 0) > 0.40,
        f"decision_readiness > 0.40 (jest: {conv.get('decision_readiness', 0):.2f})"
    ))

# Sprawdź czy conviction rosło
if len(conviction_history) >= 4:
    first_ready = conviction_history[0].get("decision_readiness", 0)
    last_ready = conviction_history[-1].get("decision_readiness", 0)
    passes.append(assert_pass(
        last_ready > first_ready,
        f"decision_readiness rosło przez rozmowę ({first_ready:.2f} → {last_ready:.2f})"
    ))
    first_trust = conviction_history[0].get("trust_in_rep", 0)
    last_trust = conviction_history[-1].get("trust_in_rep", 0)
    passes.append(assert_pass(
        last_trust >= first_trust,
        f"trust_in_rep nie spadło ({first_trust:.2f} → {last_trust:.2f})"
    ))

if "error" not in fin:
    ev = fin.get("evaluation", {})
    prof = ev.get("professionalism_score", 0) or 0
    passes.append(assert_pass(
        prof >= 6,
        f"Ocena profesjonalizmu >= 6/10 (jest: {prof})"
    ))
    goal = fin.get("conversation_goal", {})
    passes.append(assert_pass(
        goal.get("score", 0) >= 50,
        f"Score celu >= 50 (jest: {goal.get('score', 0)})"
    ))
    hist = fin.get("turn_mode_histogram")
    if hist:
        print(f"\n  Histogram trybów tur: {hist}")
    flag = fin.get("turn_mode_flag")
    if flag:
        print(f"  UWAGA: {flag}")
    print(f"\n  Feedback: {ev.get('final_feedback', '')[:200]}")
else:
    passes.append(assert_pass(False, f"Finish bez błędu (błąd: {fin.get('error', '')[:80]})"))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
