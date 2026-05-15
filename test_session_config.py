"""Szybki test walidatora SessionConfig (Etap 1)."""

import sys
sys.path.insert(0, "/home/claude/work")

from conversation.schemas import (
    SessionConfig,
    Familiarity,
    CommunicationRegister,
    CommunicationWarmth,
)

# Test 1: domyślne wartości (kompatybilność wsteczna)
cfg = SessionConfig()
assert cfg.familiarity == Familiarity.FIRST_MEETING
assert cfg.register == CommunicationRegister.PROFESSIONAL
assert cfg.warmth == CommunicationWarmth.NEUTRAL
assert cfg.rep_name is None
print("✓ Test 1: domyślne wartości")

# Test 2: pełna konfiguracja - poprawna
cfg = SessionConfig(
    familiarity="acquainted",
    register="professional",
    warmth="warm",
    rep_name="Anna Kowalska",
    rep_company="PharmaCo",
    prior_visits_summary="Trzecia wizyta. Wcześniej omawiali bezpieczeństwo u pacjentów >65 r.ż.",
)
assert cfg.familiarity == Familiarity.ACQUAINTED
assert cfg.warmth == CommunicationWarmth.WARM
print("✓ Test 2: poprawna konfiguracja acquainted+professional+warm")

# Test 3: familiar + informal - dozwolone
cfg = SessionConfig(
    familiarity="familiar",
    register="informal",
    rep_name="Paweł",
)
assert cfg.register == CommunicationRegister.INFORMAL
print("✓ Test 3: familiar + informal - dozwolone")

# Test 4: first_meeting + informal - musi rzucić ValueError
try:
    cfg = SessionConfig(
        familiarity="first_meeting",
        register="informal",
    )
    assert False, "Powinno rzucić ValueError"
except ValueError as e:
    assert "Komunikacja nieformalna" in str(e)
    print("✓ Test 4: first_meeting + informal -> ValueError (poprawne odrzucenie)")

# Test 5: acquainted + informal - też powinno być odrzucone (INFORMAL wymaga FAMILIAR)
try:
    cfg = SessionConfig(
        familiarity="acquainted",
        register="informal",
    )
    assert False, "Powinno rzucić ValueError"
except ValueError as e:
    print("✓ Test 5: acquainted + informal -> ValueError (poprawne odrzucenie)")

# Test 6: first_meeting + prior_visits_summary - logiczny konflikt
try:
    cfg = SessionConfig(
        familiarity="first_meeting",
        prior_visits_summary="Spotykaliśmy się już",
    )
    assert False, "Powinno rzucić ValueError"
except ValueError as e:
    assert "prior_visits_summary" in str(e)
    print("✓ Test 6: first_meeting + prior_visits_summary -> ValueError")

# Test 7: serializacja do dict (do response)
cfg = SessionConfig(familiarity="familiar", register="informal", warmth="warm")
d = cfg.model_dump(mode="json")
assert d["familiarity"] == "familiar"
assert d["register"] == "informal"
print("✓ Test 7: serializacja do JSON-compatible dict")

# Test 8: zbyt długie prior_visits_summary -> walidacja max_length
try:
    cfg = SessionConfig(
        familiarity="familiar",
        prior_visits_summary="x" * 600,
    )
    assert False, "Powinno rzucić ValidationError za max_length"
except Exception as e:
    print("✓ Test 8: prior_visits_summary > 500 znaków -> błąd walidacji")

print("\n🎉 Wszystkie 8 testów SessionConfig przeszło.")
