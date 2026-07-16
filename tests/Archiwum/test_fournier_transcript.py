"""Test odtwarzający konkretny transkrypt rozmowy: Tomasz Nowak / Fournier / Noacid.

Scenariusz:
  Lekarz: merytoryczny, sztywny, konkretny, nie zainteresowany dłuższą rozmową.
  Archetyp: busy_pragmatist (transakcyjny, presja czasu).
  Znajomość: acquainted — spotkanie było umówione.

Tury odwzorowane 1:1 z transkryptu:
  1. Przedstawienie się (samo przywitanie, bez leku)
  2. Intro leku + claim o braku nawrotów i interakcji (niepoparty danymi)
  3. Statystyki epidemiologiczne + claim interakcyjny (niepoparty badaniem)
  4. Dowód: tabela + mechanizm sulfotransferazy (evidence_hits)
  5. Zamknięcie z frazą marketingową "dla wszystkich pacjentów z GERD"

Oczekiwane zachowanie:
  - Po turze 2: lek ujawniony, claim niepoparty → unsupported_claims, mały wzrost frustracji
  - Po turze 4: clinical_confidence rośnie, trust_in_rep rośnie dzięki evidencji
  - Po turze 5: marketing_hits triggerowane, trust_in_rep nieznacznie spada
  - Rozmowa NIE jest przerywana przez system (brak bribery, frustration poniżej progu)
  - Finalna decyzja: trial_use lub undecided (nie reject — lekarz obiecał przetestować)

Użycie: python3 tests/test_fournier_transcript.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "busy_pragmatist"   # transakcyjny, presja czasu — najbliższy opisowi
DRUG = "noacid"
REP_NAME = "Tomasz Nowak"
REP_COMPANY = "Fournier"

# --- Wiadomości z transkryptu (dosłownie lub zbliżone) ---
TURNS = [
    # Tura 1: samo przedstawienie, brak wprowadzenia leku
    "Dzień dobry, panie doktorze. Tomasz Nowak, firma Fournier. Umawialiśmy się na krótkie spotkanie.",

    # Tura 2: intro leku + claim o nawrotach i braku interakcji (bez dowodów)
    (
        "Dziękuję, że ma Pan dla mnie chwilę. Rozumiem sytuację, więc przejdę od razu do konkretów. "
        "Chciałbym dziś skupić się na terapii Pantoprazol u Pana pacjentów z nowo rozpoznaną chorobą refluksową. "
        "Chciałbym przekonać Pana do stosowania Noacid jako leku pierwszego rzutu, ze względu na to, "
        "że w porównaniu z innymi prazolami znacząco zmniejszy on liczbę nawrotów i interakcji z innymi lekami."
    ),

    # Tura 3: statystyki epidemiologiczne + claim o braku interakcji (bez cytowania badania)
    (
        "Panie doktorze, statystyki badania mówią, że objawy refluksu u około 7% populacji dorosłej w Polsce "
        "występują codziennie. Poza tym pacjenci często przyjmują inne leki, np. przeciwpadaczkowe, "
        "a Pantoprazol, czyli Noacid, nie wchodzi w interakcje w odróżnieniu od omeprazolu czy lansoprazolu."
    ),

    # Tura 4: dowód tabelaryczny + mechanizm sulfotransferazy (evidence_hits)
    (
        "Proszę spojrzeć na to zestawienie tabelaryczne. Pantoprazol nie wchodzi w reakcję z kofeiną, "
        "warfaryną czy diazepamem, ponieważ oprócz standardowego szlaku metabolicznego jako jedyny "
        "posiada dodatkową sulfotransferazę — to udowodnione badaniem klinicznym."
    ),

    # Tura 5: zamknięcie z frazą marketingową i buzzwordem "dla wszystkich pacjentów"
    (
        "Mam nadzieję, że moje argumenty i Pan doświadczenie pozwolą nam zgodzić się na następnej wizycie, "
        "że NoAcid to skuteczna terapia dla wszystkich pacjentów z GERD, "
        "dzięki optymalnemu profilowi bezpieczeństwa i wysokiemu stopniowi uzyskania remisji. "
        "Dziękuję za konkretną rozmowę. Do widzenia i pozwoli Pan, że wrócę za około 2 tygodnie. "
        "Liczę że przekona się Pan do NoAcid 20 mg."
    ),
]

print_header(f"TEST TRANSKRYPT FOURNIER | {DOCTOR} | {DRUG} | {len(TURNS)} tur")
print(f"Przedstawiciel: {REP_NAME}, firma: {REP_COMPANY}")

# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "acquainted",
    "register": "professional",
    "warmth": "neutral",
    "rep_name": REP_NAME,
    "rep_company": REP_COMPANY,
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}\n")

# Weryfikacja stanu startowego
initial_conv = start_resp.get("conviction") or {}
initial_trust = float(initial_conv.get("trust_in_rep", 0))

# ---------------------------------------------------------------------------
# TURY
# ---------------------------------------------------------------------------
responses = []
terminated_early = False
drug_revealed_turn = None

for i, msg in enumerate(TURNS, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg[:120]}{'...' if len(msg) > 120 else ''}")
    resp = post("/message", {"session_id": session_id, "message": msg})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        terminated_early = True
        break

    attitude = resp.get("doctor_attitude", "?")
    decision = resp.get("doctor_decision", "?")
    print(f"[Tura {i}] Lekarz ({attitude} | {decision}):")
    print(f"  {resp.get('doctor_message', '')}")
    print(fmt_conviction(resp.get("conviction")))
    print(fmt_metrics(resp.get("turn_metrics", {})))

    # Śledź kiedy lek został ujawniony
    if resp.get("drug_revealed") and drug_revealed_turn is None:
        drug_revealed_turn = i
        print(f"  >>> Lek ujawniony w turze {i} <<<")

    print()
    responses.append(resp)

    if resp.get("is_terminated"):
        print(f"  *** System zakończył rozmowę: {resp.get('termination_reason', '')} ***")
        terminated_early = True
        break

# ---------------------------------------------------------------------------
# FINISH
# ---------------------------------------------------------------------------
print_separator()
print("Kończę sesję (/finish)...\n")
fin = post(f"/finish?session_id={session_id}&api_key=", {})

# ---------------------------------------------------------------------------
# ASERCJE
# ---------------------------------------------------------------------------
print_separator()
print("WYNIKI TESTÓW:\n")
passes = []

# --- 1. Stan startowy ---
passes.append(assert_pass(
    0.40 <= initial_trust <= 0.55,
    f"[start] trust_in_rep w zakresie acquainted [0.40–0.55] (jest: {initial_trust:.2f})"
))

# --- 2. Lek ujawniony do tury 2 ---
passes.append(assert_pass(
    drug_revealed_turn is not None and drug_revealed_turn <= 2,
    f"[tura ≤ 2] Noacid ujawniony nie później niż w turze 2 (tura: {drug_revealed_turn})"
))

# --- 3. Rozmowa nie zakończona wcześnie przez system ---
passes.append(assert_pass(
    not terminated_early,
    "[system] Rozmowa nie przerwana przez policy/presję czasu w trakcie 5 tur"
))

# --- 4. Po turze 4 (evidence) — clinical_confidence wzrosła ---
if len(responses) >= 4:
    conv4 = responses[3].get("conviction") or {}
    cc4 = float(conv4.get("clinical_confidence", 0))
    passes.append(assert_pass(
        cc4 > 0.22,
        f"[tura 4] clinical_confidence wzrosła ponad start po dowodzie sulfotransferazy (jest: {cc4:.2f})"
    ))
    trust4 = float(conv4.get("trust_in_rep", 0))
    passes.append(assert_pass(
        trust4 >= 0.40,
        f"[tura 4] trust_in_rep utrzymany po evidencji (jest: {trust4:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[tura 4] Brak odpowiedzi — za mało tur"))
    passes.append(assert_pass(False, "[tura 4] Brak odpowiedzi — za mało tur"))

# --- 5. Po turze 5 (marketing) — marketing_hits wykryte w metrics ---
if len(responses) >= 5:
    tm5 = responses[4].get("turn_metrics") or {}
    # Fraza "skuteczna terapia dla wszystkich" powinna obniżyć clinical_precision
    cp5 = float(tm5.get("clinical_precision", 1.0))
    passes.append(assert_pass(
        cp5 < 0.90,
        f"[tura 5] clinical_precision obniżona przez marketing/false claim (jest: {cp5:.2f})"
    ))
    # Frustracja nie powinna eksplodować — lekarz był merytoryczny
    frustration5 = float(tm5.get("frustration", 0))
    passes.append(assert_pass(
        frustration5 < 7.0,
        f"[tura 5] Frustracja poniżej progu terminacji 7.0 (jest: {frustration5:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[tura 5] Brak odpowiedzi na ostatnią turę"))
    passes.append(assert_pass(False, "[tura 5] Brak odpowiedzi na ostatnią turę"))

# --- 6. Finalna decyzja: trial_use lub undecided (nie reject) ---
if responses:
    last = responses[-1]
    final_decision = last.get("doctor_decision", "undecided")
    passes.append(assert_pass(
        final_decision in {"trial_use", "undecided"},
        f"[finał] Decyzja to trial_use lub undecided — lekarz obiecał przetestować (jest: {final_decision})"
    ))
    # Finalna conviction — decision_readiness powinna być niezerowa
    final_conv = last.get("conviction") or {}
    dr_final = float(final_conv.get("decision_readiness", 0))
    passes.append(assert_pass(
        dr_final > 0.0,
        f"[finał] decision_readiness > 0 — jakiś postęp po 5 turach (jest: {dr_final:.2f})"
    ))

# --- 7. /finish — bez błędu, evaluation present ---
if "error" not in fin:
    ev = fin.get("evaluation") or {}
    goal = fin.get("conversation_goal") or {}

    passes.append(assert_pass(
        goal.get("status") in {"partial", "not_achieved"},
        f"[finish] goal.status to partial lub not_achieved (lekarz powiedział 'sprawdzimy') (jest: {goal.get('status')})"
    ))
    prof = ev.get("professionalism_score")
    passes.append(assert_pass(
        prof is not None and prof >= 4,
        f"[finish] professionalism_score ≥ 4 (konkretna, merytoryczna rozmowa) (jest: {prof})"
    ))
    # Brak naruszeń etyki
    passes.append(assert_pass(
        goal.get("doctor_decision") != "reject" or goal.get("score", 0) >= 0,
        "[finish] Brak bribery — rozmowa zakończyła się bez etycznego odrzucenia"
    ))

    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Decyzja końcowa: {goal.get('doctor_decision', '?')}")
    print(f"  Powody: {goal.get('reasons', [])}")
    print(f"  Braki: {goal.get('missing', [])}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"trafność={ev.get('relevance_score', '?')}/10")
else:
    passes.append(assert_pass(False, f"[finish] Błąd /finish: {fin.get('error', '')[:80]}"))
    passes.append(assert_pass(False, "[finish] Brak evaluation"))
    passes.append(assert_pass(False, "[finish] Brak goal"))

# ---------------------------------------------------------------------------
# PODSUMOWANIE
# ---------------------------------------------------------------------------
print_separator()
failed = sum(1 for p in passes if not p)
total = len(passes)
print(f"\nWynik: {total - failed}/{total} testów przeszło")
if failed:
    print("\nOczekiwany przebieg rozmowy:")
    print("  Tura 2: lek ujawniony, claimy o interakcjach bez dowodów → unsupported")
    print("  Tura 4: tabela + sulfotransferaza → evidence_hits, wzrost clinical_confidence")
    print("  Tura 5: 'dla wszystkich pacjentów z GERD' → false claim/marketing → kara")
    print("  Finał: trial_use (lekarz obiecał 'sprawdzimy u kilku pacjentów')")
sys.exit(0 if failed == 0 else 1)
