"""Uruchamia wszystkie testy sekwencyjnie i drukuje podsumowanie.

Użycie: python3 tests/run_all.py
Wymaga: uvicorn api4:app --port 8000
"""

import subprocess
import sys
import os
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)

TESTS = [
    ("test_baseline.py", "Bazowy (5 tur, acquainted)"),
    ("test_good_conversation.py", "Wzorcowa rozmowa (10 tur, friendly_generalist)"),
    ("test_false_claims.py", "Fałszywe claimy (evidence_seeker)"),
    ("test_bribery.py", "Łapówka / naruszenie etyki"),
    ("test_first_meeting.py", "Pierwsze spotkanie (first_meeting + walidacja)"),
    ("test_familiar_informal.py", "Znany lekarz / informal"),
    ("test_marketing_language.py", "Marketingowy język (buzzwords)"),
    ("test_hard_doctor.py", "Trudny lekarz (tired_cynic)"),
]

results = []

print(f"\n{'=' * 60}")
print("  URUCHAMIANIE WSZYSTKICH TESTÓW")
print(f"{'=' * 60}\n")

for filename, label in TESTS:
    path = os.path.join(TESTS_DIR, filename)
    print(f"[ {label} ]")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, path],
        cwd=ROOT_DIR,
        capture_output=False,
    )
    elapsed = time.time() - start
    status = "PASS" if proc.returncode == 0 else "FAIL"
    results.append((label, status, elapsed))
    print(f"\n  → {status} ({elapsed:.1f}s)\n{'─' * 60}\n")

print(f"\n{'=' * 60}")
print("  PODSUMOWANIE")
print(f"{'=' * 60}")
passed = sum(1 for _, s, _ in results if s == "PASS")
for label, status, elapsed in results:
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} [{status}] {label} ({elapsed:.1f}s)")

print(f"\n  Wynik: {passed}/{len(results)} testów przeszło")
sys.exit(0 if passed == len(results) else 1)
