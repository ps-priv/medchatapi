"""Test przeciw powtórkom lekarza.

Sprawdza dwie rzeczy po serii tur, w których przedstawiciel NIGDY nie używa
złej formy grzecznościowej (nie ma więc powodu do korekty):
- fraza korygująca formę grzecznościową ("proszę mówić pani/pan doktor")
  nie pojawia się bez powodu w kolejnych turach (patrz prompt_builder.py,
  reguła #3 warunkowana na gender_mismatch_hits z bieżącej tury),
- kolejne wypowiedzi lekarza nie są niemal identyczne (agenda filtrowana do
  nieużytych wątków + przypomnienie ostatniej wypowiedzi w prompt_builder.py).

Użycie: python3 tests/test_no_repetition.py
Wymaga: uvicorn api7:app --port 8000
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from _common import post, assert_pass, print_header, print_separator

DOCTOR = "busy_pragmatist"
DRUG = "noacid"

TURNS = [
    "Dzien dobry, chcialbym omowic Noacid na refluks.",
    "Jakie dane kliniczne moge przedstawic?",
    "Lek redukuje objawy refluksu w 2-4 tygodnie wedlug badan.",
    "Czy moge zapytac o Pani doswiadczenia z podobnymi pacjentami?",
    "Dawkowanie to 20 mg raz na dobe zgodnie z ChPL.",
]

CORRECTION_PATTERN = re.compile(
    r"prosz[eę][^.!?]{0,20}m[oó]wi[cć][^.!?]{0,20}(pani|pan)\s+doktor",
    re.IGNORECASE,
)


def word_overlap_ratio(a: str, b: str) -> float:
    """Jaccard similarity tokenów (>=4 znaki) między dwiema wypowiedziami."""
    ta = {w for w in re.findall(r"[a-ząćęłńóśźż]{4,}", a.lower())}
    tb = {w for w in re.findall(r"[a-ząćęłńóśźż]{4,}", b.lower())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


print_header(f"TEST BRAKU POWTÓREK | lekarz: {DOCTOR} | lek: {DRUG}")

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

doctor_messages = []
for i, msg in enumerate(TURNS, 1):
    print(f"[Tura {i}] Przedstawiciel: {msg}")
    resp = post("/message", {"session_id": session_id, "message": msg})
    if "error" in resp:
        print(f"  BŁĄD: {resp['error']}")
        break
    doctor_message = resp.get("doctor_message", "")
    print(f"[Tura {i}] Lekarz: {doctor_message}\n")
    doctor_messages.append(doctor_message)
    if resp.get("is_terminated"):
        print(f"  *** Zakończono wcześnie: {resp.get('termination_reason', '')} ***")
        break

print_separator()
print("Kończę sesję (/finish)...")
post(f"/finish?session_id={session_id}", {})

print_separator()
print("WYNIKI TESTÓW:\n")
passes = []

# Rep nigdy nie użył złej formy — korekta nie powinna pojawiać się w więcej niż 1 turze
correction_hits = [bool(CORRECTION_PATTERN.search(m)) for m in doctor_messages]
passes.append(assert_pass(
    sum(correction_hits) <= 1,
    f"Fraza korygująca formę grzecznościową nie powtarza się bez powodu "
    f"(wystąpienia: {sum(correction_hits)}/{len(doctor_messages)})"
))

# Kolejne wypowiedzi lekarza nie powinny być niemal identyczne (agenda / pytania)
max_overlap = 0.0
for i in range(1, len(doctor_messages)):
    overlap = word_overlap_ratio(doctor_messages[i - 1], doctor_messages[i])
    max_overlap = max(max_overlap, overlap)
passes.append(assert_pass(
    max_overlap < 0.6,
    f"Kolejne wypowiedzi lekarza nie są niemal identyczne (max nakładanie słów: {max_overlap:.2f})"
))

print_separator()
failed = sum(1 for p in passes if not p)
print(f"\nWynik: {len(passes) - failed}/{len(passes)} testów przeszło")
sys.exit(0 if failed == 0 else 1)
