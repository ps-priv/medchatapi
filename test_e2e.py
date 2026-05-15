"""Test end-to-end sesji — uruchom po `uvicorn api4:app --port 8000`.

Przeprowadza pełną rozmowę z wybranym lekarzem i drukuje conviction po każdej turze.
Użycie:
    python3 test_e2e.py [doctor_id] [drug_id]
Domyślnie: skeptical_expert + noacid
"""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
DOCTOR = sys.argv[1] if len(sys.argv) > 1 else "skeptical_expert"
DRUG   = sys.argv[2] if len(sys.argv) > 2 else "noacid"

TURNS = [
    "Dzień dobry, jestem przedstawicielem firmy PharmaX.",
    "Chciałbym omówić lek Noacid — nowoczesny inhibitor pompy protonowej.",
    "W badaniu klinicznym na 1200 pacjentach Noacid wykazał skuteczność 89% w gojeniu nadżerek.",
    "Lek jest wskazany przy refluksie i chorobie wrzodowej żołądka. Profil bezpieczeństwa jest bardzo dobry.",
    "Czy rozważyłaby Pani przepisanie go pacjentom z przewlekłym refluksem?",
]


def post(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}


def get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}


def fmt_conviction(c: dict | None) -> str:
    if not c:
        return "  (brak conviction)"
    return (
        f"  interest={c.get('interest_level',0):.2f} | "
        f"trust={c.get('trust_in_rep',0):.2f} | "
        f"clinical={c.get('clinical_confidence',0):.2f} | "
        f"fit={c.get('perceived_fit',0):.2f} | "
        f"readiness={c.get('decision_readiness',0):.2f}"
    )


# ── START ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TEST E2E  |  lekarz: {DOCTOR}  |  lek: {DRUG}")
print(f"{'='*60}\n")

start_resp = post(f"/start?id={DOCTOR}&drug_id={DRUG}", {
    "familiarity": "acquainted",
    "register": "professional",
    "warmth": "warm",
    "rep_name": "Jan Testowy",
    "prior_visits_summary": "Druga wizyta. Poprzednio rozmawialiśmy ogólnie o IPP.",
})

if "error" in start_resp:
    print("BŁĄD /start:", start_resp)
    sys.exit(1)

session_id = start_resp["session_id"]
print(f"✓ Sesja: {session_id}")
if start_resp.get("session_config"):
    cfg = start_resp["session_config"]
    print(f"  Config: familiarity={cfg['familiarity']}, register={cfg['register']}, warmth={cfg['warmth']}")

agenda_info = "(agenda w stanie serwera, niewidoczna w /start response)"
print(f"  Agenda: {agenda_info}\n")

# ── TURY ───────────────────────────────────────────────────────────────────
for i, msg in enumerate(TURNS, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg}")

    resp = post("/message", {"session_id": session_id, "message": msg})

    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        break

    print(f"[Tura {i}] Lekarz ({resp.get('doctor_attitude','?')} | {resp.get('doctor_decision','?')}):")
    print(f"  {resp.get('doctor_message','')}")
    print(fmt_conviction(resp.get("conviction")))

    tm = resp.get("turn_metrics", {})
    print(
        f"  metrics: topic={tm.get('topic_adherence',0):.2f} | "
        f"clinical={tm.get('clinical_precision',0):.2f} | "
        f"ethics={tm.get('ethics',0):.2f} | "
        f"frustration={tm.get('frustration',0):.2f}"
    )
    print()

    if resp.get("is_terminated"):
        print(f"  ⚠ Rozmowa zakończona: {resp.get('termination_reason','')}")
        break

# ── FINISH ─────────────────────────────────────────────────────────────────
print("─" * 60)
print("Kończę sesję (/finish)...")
fin = post(f"/finish?session_id={session_id}&api_key=", {})

if "error" not in fin:
    goal = fin.get("conversation_goal", {})
    ev = fin.get("evaluation", {})
    print(f"\nWynik celu: {goal.get('status','?')} | score={goal.get('score','?')} | decyzja={goal.get('doctor_decision','?')}")
    print(f"Ocena końcowa:")
    print(f"  profesjonalizm={ev.get('professionalism_score','?')}/10")
    print(f"  trafność={ev.get('relevance_score','?')}/10")
    print(f"  relacja={ev.get('relationship_score','?')}/10")
    print(f"  feedback: {ev.get('final_feedback','')}")
    hist = fin.get("turn_mode_histogram", {})
    if hist:
        print(f"\nHistogram trybów tur: {hist}")
    flag = fin.get("turn_mode_flag")
    if flag:
        print(f"  ⚠ {flag}")
else:
    print("BŁĄD /finish:", fin)
