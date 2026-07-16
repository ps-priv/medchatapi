"""Test odtwarzający transkrypt: przyjazny tradycjonalista / Noacid.

Scenariusz:
  Lekarz: przyjazny, mało otwarty na nowinki, przywiązany do sprawdzonych schematów.
  Archetyp: defensive_traditionalist (skeptical, hard, open=0.10, skeptic=0.80).
  Znajomość: familiar — lekarz mówi do reps. "panie Tomaszu", znają się z konferencji.
  Rejestr: professional (formalny "Panie doktorze").

Tury z transkryptu (TYLKO wypowiedzi przedstawiciela):
  1. Small talk o konferencji — off-topic, brak wzmianki o leku
  2. Pytanie-pułapka o skuteczność krótko/długoterminową — intencja kliniczna, brak nazwy leku
  3. Intro Noacid + claim o skuteczności krótko- i długoterminowej vs. inne leki (porównanie nieudokumentowane)
  4. Zamknięcie marketingowe: "dla wszystkich pacjentów z GERD" — FALSE CLAIM + buzzwordy

Oczekiwane zachowanie:
  - Tura 1: off_topic_hits — lekarz reaguje niechętnie na small talk, brak drug_revealed
  - Tura 2: intent_revealed (refluks), drug_revealed = False
  - Tura 3: drug_revealed = True; claim noacid_efficacy częściowo wsparta; porównanie → unsupported
  - Tura 4: marketing_hits + false claim ("dla wszystkich") → kara w metrykach
  - Frustracja umiarkowana — lekarz jest "przyjazny" nawet jeśli oporny
  - Decyzja: trial_use lub undecided (nie reject — lekarz obiecał "dla świętego spokoju spróbuję")
  - Brak wczesnej terminacji (etyka OK, brak korupcji)

Użycie: python3 tests/test_traditionalist_transcript.py
Wymaga: uvicorn api4:app --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, fmt_conviction, fmt_metrics, assert_pass, print_header, print_separator

DOCTOR = "defensive_traditionalist"   # low openness=0.10, skeptic=0.80 — "przywiązany do schematów"
DRUG = "noacid"
REP_NAME = "Tomasz"
REP_COMPANY = "Fournier"

TURNS = [
    # Tura 1: small talk — konferencja, nowinki; zero treści klinicznej (off-topic)
    (
        "Dzień dobry, panie doktorze. Miło pana znów widzieć. "
        "Jak tam po konferencji w zeszłym tygodniu? Jakieś nowinki, ciekawe wykłady?"
    ),

    # Tura 2: pytanie-pułapka o skuteczność — rejestruje intencję kliniczną (refluks),
    # ale leku jeszcze nie wprowadza (sprytne bridge)
    (
        "Rozumiem. Stabilizacja to podstawa sukcesu. "
        "Z ciekawości zapytam, czy dla Pana ważniejsza jest w leczeniu refluksu "
        "skuteczność krótko- czy długoterminowa?"
    ),

    # Tura 3: wprowadzenie Noacid + claim o skuteczności krótko i długoterminowej;
    # porównanie "większości leków" jest nieudokumentowane → unsupported
    (
        "No właśnie — w przypadku gdy zastosuje Pan Noacid, nie musi Pan wybierać. "
        "Zarówno krótko-, jak i długoterminowo będzie Pan pewien skuteczności, "
        "w odróżnieniu od większości leków stosowanych w refluksie, "
        "które zapewniają skuteczność jedynie krótkoterminowo."
    ),

    # Tura 4: zamknięcie marketingowe — "dla wszystkich pacjentów z GERD" to false claim
    # (contradicts noacid_contraindications: "mozna stosowac u wszystkich");
    # buzzwordy: "unikalny mechanizm działania", "optymalny profil bezpieczeństwa"
    (
        "Cieszę się, że moje argumenty pozwolą nam się zgodzić, że NoAcid to skuteczna terapia "
        "dla wszystkich pacjentów z GERD, dzięki unikalnemu mechanizmowi działania, "
        "optymalnemu profilowi bezpieczeństwa i wysokiemu stopniowi uzyskania remisji. "
        "Bardzo dziękuję. Do zobaczenia!"
    ),
]

print_header(f"TEST TRANSKRYPT TRADYCJONALISTA | {DOCTOR} | {DRUG} | {len(TURNS)} tury")
print(f"Przedstawiciel: {REP_NAME}, firma: {REP_COMPANY} | familiar + professional\n")

# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "familiar",
    "register": "professional",
    "warmth": "neutral",
    "rep_name": REP_NAME,
    "rep_company": REP_COMPANY,
    "prior_visits_summary": "Wieloletnia znajomość zawodowa, lekarz dobrze zna przedstawiciela z regularnych wizyt.",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"Sesja: {session_id}\n")

initial_conv = start_resp.get("conviction") or {}
initial_trust = float(initial_conv.get("trust_in_rep", 0))

# ---------------------------------------------------------------------------
# TURY
# ---------------------------------------------------------------------------
responses = []
terminated_early = False
drug_revealed_turn = None
intent_revealed_turn = None

TURN_LABELS = [
    "small talk / off-topic",
    "pytanie o skuteczność / intent signal",
    "intro Noacid + claim porównawczy",
    "zamknięcie marketingowe / false claim",
]

for i, msg in enumerate(TURNS, 1):
    label = TURN_LABELS[i - 1]
    print(f"[Tura {i} — {label}]")
    print(f"  Przedstawiciel: {msg[:120]}{'...' if len(msg) > 120 else ''}")
    resp = post("/message", {"session_id": session_id, "message": msg})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        terminated_early = True
        break

    attitude = resp.get("doctor_attitude", "?")
    decision = resp.get("doctor_decision", "?")
    print(f"  Lekarz ({attitude} | {decision}): {resp.get('doctor_message', '')[:160]}")
    print(fmt_conviction(resp.get("conviction")))
    print(fmt_metrics(resp.get("turn_metrics", {})))

    if resp.get("drug_revealed") and drug_revealed_turn is None:
        drug_revealed_turn = i
        print(f"  >>> Lek ujawniony w turze {i} <<<")
    if resp.get("intent_revealed") and intent_revealed_turn is None:
        intent_revealed_turn = i
        print(f"  >>> Intencja rozpoznana w turze {i} <<<")

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

# --- 1. Stan startowy: familiar → trust ~0.65 ---
passes.append(assert_pass(
    0.55 <= initial_trust <= 0.75,
    f"[start] trust_in_rep w zakresie familiar [0.55–0.75] (jest: {initial_trust:.2f})"
))

# --- 2. Tura 1 (off-topic): lek NIE ujawniony, intent NIE ujawniony ---
if responses:
    r1 = responses[0]
    passes.append(assert_pass(
        not r1.get("drug_revealed", False),
        "[tura 1] drug_revealed=False po samym small talk (brak nazwy leku)"
    ))
    # Small talk powinien obniżyć topic_adherence
    tm1 = r1.get("turn_metrics") or {}
    ta1 = float(tm1.get("topic_adherence", 1.0))
    passes.append(assert_pass(
        ta1 < 0.85,
        f"[tura 1] topic_adherence obniżona przez off-topic small talk (jest: {ta1:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[tura 1] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[tura 1] Brak odpowiedzi"))

# --- 3. Tura 2 (pytanie o refluks): intent wykryty, lek jeszcze nie ---
if len(responses) >= 2:
    r2 = responses[1]
    passes.append(assert_pass(
        r2.get("intent_revealed", False),
        "[tura 2] intent_revealed=True po pytaniu o leczenie refluksu"
    ))
    passes.append(assert_pass(
        not r2.get("drug_revealed", False),
        "[tura 2] drug_revealed=False — Noacid jeszcze nie wspomniany"
    ))
else:
    passes.append(assert_pass(False, "[tura 2] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[tura 2] Brak odpowiedzi"))

# --- 4. Tura 3 (intro Noacid): lek ujawniony ---
if len(responses) >= 3:
    r3 = responses[2]
    passes.append(assert_pass(
        r3.get("drug_revealed", False),
        "[tura 3] drug_revealed=True po wprowadzeniu nazwy Noacid"
    ))
    conv3 = r3.get("conviction") or {}
    cc3 = float(conv3.get("clinical_confidence", 0.2))
    passes.append(assert_pass(
        cc3 >= 0.20,
        f"[tura 3] clinical_confidence nie spada poniżej startu mimo low openness (jest: {cc3:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[tura 3] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[tura 3] Brak odpowiedzi"))

# --- 5. Tura 4 (marketing/false claim): kary w metrykach ---
if len(responses) >= 4:
    r4 = responses[3]
    tm4 = r4.get("turn_metrics") or {}
    # "dla wszystkich pacjentów z GERD" contradicts noacid_contraindications → false claim → clinical_precision ↓
    cp4 = float(tm4.get("clinical_precision", 1.0))
    passes.append(assert_pass(
        cp4 < 0.85,
        f"[tura 4] clinical_precision obniżona przez false claim + marketing (jest: {cp4:.2f})"
    ))
    # Etyka nie naruszona (brak korupcji)
    ethics4 = float(tm4.get("ethics", 1.0))
    passes.append(assert_pass(
        ethics4 >= 0.90,
        f"[tura 4] ethics pozostaje wysoka — brak korupcji ani niestosownych propozycji (jest: {ethics4:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[tura 4] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[tura 4] Brak odpowiedzi"))

# --- 6. Brak wczesnej terminacji ---
passes.append(assert_pass(
    not terminated_early,
    "[system] Rozmowa nie przerwana — lekarz przyjazny, brak naruszeń etyki"
))

# --- 7. Finalna frustracja — umiarkowana (lekarz "przyjazny") ---
if responses:
    last = responses[-1]
    tm_last = last.get("turn_metrics") or {}
    fr_last = float(tm_last.get("frustration", 0))
    passes.append(assert_pass(
        fr_last < 8.0,
        f"[finał] Frustracja poniżej 8.0 — lekarz jest przyjazny mimo oporności (jest: {fr_last:.2f})"
    ))
    # Finalna decyzja: nie reject (lekarz obiecał "dla świętego spokoju spróbuję")
    final_decision = last.get("doctor_decision", "undecided")
    passes.append(assert_pass(
        final_decision in {"trial_use", "undecided"},
        f"[finał] Decyzja ≠ reject — lekarz obiecał przetestować (jest: {final_decision})"
    ))
    # Mimo low openness, trust nie powinno spaść zbyt nisko (familiar + brak naruszeń)
    final_conv = last.get("conviction") or {}
    tr_final = float(final_conv.get("trust_in_rep", 0))
    passes.append(assert_pass(
        tr_final >= 0.40,
        f"[finał] trust_in_rep utrzymany mimo oporności i marketingu (jest: {tr_final:.2f})"
    ))
else:
    passes.append(assert_pass(False, "[finał] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[finał] Brak odpowiedzi"))
    passes.append(assert_pass(False, "[finał] Brak odpowiedzi"))

# --- 8. /finish ---
if "error" not in fin:
    ev = fin.get("evaluation") or {}
    goal = fin.get("conversation_goal") or {}

    passes.append(assert_pass(
        goal.get("status") in {"partial", "not_achieved"},
        f"[finish] goal.status partial lub not_achieved (tradycjonalista daje tylko trial) (jest: {goal.get('status')})"
    ))
    prof = ev.get("professionalism_score")
    passes.append(assert_pass(
        prof is not None and prof >= 3,
        f"[finish] professionalism_score ≥ 3 (off-topic start obniżył, ale etyka OK) (jest: {prof})"
    ))
    # Sprawdzamy że bribery nie zostało wykryte (brak odrzucenia etycznego)
    passes.append(assert_pass(
        goal.get("doctor_decision") != "reject" or "etyk" not in str(goal.get("missing", [])),
        "[finish] Brak odrzucenia z przyczyn etycznych"
    ))

    print(f"\n  Cel: {goal.get('status', '?')} | score={goal.get('score', '?')}")
    print(f"  Decyzja końcowa: {goal.get('doctor_decision', '?')}")
    print(f"  Powody: {goal.get('reasons', [])}")
    print(f"  Braki: {goal.get('missing', [])}")
    print(f"  Ocena: prof={ev.get('professionalism_score', '?')}/10 | "
          f"trafność={ev.get('relevance_score', '?')}/10")
else:
    passes.append(assert_pass(False, f"[finish] Błąd: {fin.get('error', '')[:80]}"))
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
    print("\nOczekiwany przebieg:")
    print("  Tura 1: small talk → off_topic, brak drug_revealed, topic_adherence < 0.85")
    print("  Tura 2: pytanie o refluks → intent_revealed, drug_revealed=False")
    print("  Tura 3: Noacid intro → drug_revealed, unsupported claim porównawczy")
    print("  Tura 4: 'dla wszystkich' = false claim + buzzwordy → clinical_precision < 0.85")
    print("  Finał: trial_use lub undecided, frustration < 8.0, trust >= 0.40")
sys.exit(0 if failed == 0 else 1)
