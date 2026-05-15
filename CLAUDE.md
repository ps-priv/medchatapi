# CLAUDE.md — kontekst projektu dla Claude Code

> Ten plik jest automatycznie czytany przez Claude Code przy starcie każdej sesji w tym repo.
> Aktualny stan: **refaktoring zakończony** — wszystkie 4 etapy zaimplementowane i commitowane.

---

## O czym jest ten projekt

`medchatapi` to silnik symulacji rozmowy między **lekarzem (AI)** a **przedstawicielem farmaceutycznym (człowiek)**. Służy do treningu handlowców — bezpieczne środowisko do uczenia się prowadzenia rozmów sprzedażowych z różnymi typami lekarzy.

Stack: **Python 3.10+, FastAPI, LangGraph, OpenAI GPT (gpt-4o / gpt-4o-mini), ChromaDB (RAG), Supabase (persystencja).**

Główne pliki:
- `api4.py` — FastAPI app (6 endpointów: /start, /message, /finish, /transcribe, /tts, /rate)
- `langgraph_app/` — graf konwersacji, węzły, state, service
- `conversation/` — claims, policy, schemas, metrics, conviction
- `doctor_archetypes.json` — 10 archetyp lekarzy
- `drug_catalog.json` — leki z claims do walidacji

Szczegółowy opis algorytmu: `docs/ALGORITHM.md` ← przeczytaj przed zmianami w logice rozmowy.

---

## Stan refaktoringu — wszystko zakończone ✅

Oryginalny problem: rozmowa brzmiała jak przesłuchanie, lekarz tylko reagował, brak własnej agendy i jasnego kryterium decyzji.

### Etap 1 — Konfiguracja sesji ✅
- `SessionConfig` z `familiarity` / `register` / `warmth` / `rep_name` / `rep_company`
- `POST /start` przyjmuje opcjonalne body — 100% kompatybilność wsteczna
- `INFORMAL + FIRST_MEETING` → HTTP 422 (naruszenie etykiety)

### Etap 2 — Doctor Conviction ✅
- 5-wymiarowy stan decyzyjny: `interest_level`, `trust_in_rep`, `clinical_confidence`, `perceived_fit`, `decision_readiness`
- Startowe `trust_in_rep` zależy od `familiarity` (first_meeting=0.20, acquainted=0.45, familiar=0.65)
- `derive_decision_from_conviction()` zastępuje `evaluate_conversation_goal` jako źródło decyzji
- `evaluate_conversation_goal` pozostaje jako logger metryk dla raportu końcowego

### Etap 3 — DoctorAgenda + przepisanie promptu ✅
- `agenda_generator.py` — jednorazowe wywołanie gpt-4o-mini na starcie (~0.001 USD), generuje 3-5 wątków lekarza (patient_case, clinical_curiosity, concern, personal, time_pressure)
- `prompt_builder.py` — `_build_system_prompt` z 3 sekcjami: KIM JESTEŚ / CO ROBISZ / BEZPIECZNIKI
- Familiarity, register, warmth i conviction wbudowane w prompt

### Etap 4 — TurnMode ✅
- `turn_planner.py` — `plan_turn_mode()`: heurystyka bez LLM, wybiera REACT/PROBE/SHARE/CHALLENGE/DRIFT/CLOSE
- Węzeł `node_plan_turn` w grafie między `build_directives` a `retrieve_context`
- DRIFT: ~10% szans od tury 3, deterministyczne z SHA-256 seed

### Dodatkowe prace ✅
- **SRP refaktoring**: rozbicie `helpers.py` (704 linii), `nodes.py` (883 linii), `policy.py` (813 linii) na 9 nowych modułów
- **Docstringi**: każda funkcja we wszystkich modułach ma docstring po polsku
- **`docs/ALGORITHM.md`**: pełny opis algorytmu z polskimi nazwami parametrów, tabelami, progami i schematem grafu
- **10 testów integracyjnych** w `tests/` (w tym 2 oparte na transkryptach rzeczywistych rozmów)

---

## Aktualna struktura modułów

### `langgraph_app/`

| Plik | Zawartość |
|---|---|
| `graph.py` | Topologia grafu LangGraph — kolejność węzłów, conditional routing |
| `nodes.py` | Węzły grafu (node_detect_context → node_finalize) |
| `state.py` | `ConversationState` — pełna struktura TypedDict sesji |
| `service.py` | `start_session`, `process_message`, `finish_session` |
| `session_builder.py` | `build_initial_state`, `evaluate_conversation_goal`, `normalize_doctor_decision` |
| `prompt_builder.py` | `_build_system_prompt` — 3-sekcyjny prompt lekarza |
| `agenda_generator.py` | `generate_agenda` — LLM generuje wątki lekarza na starcie |
| `turn_planner.py` | `plan_turn_mode`, `derive_decision_from_conviction` |
| `knowledge_guard.py` | `apply_knowledge_guard` — lekarz nie cytuje danych których nie słyszał |
| `random_events.py` | `apply_random_event` — deterministyczne zdarzenia zakłócające |
| `helpers.py` | Utilitki tekstowe: `normalize_text`, `detect_drug_introduction`, `check_termination` |
| `rag.py` | ChromaDB retrieval (ChPL) |

### `conversation/`

| Plik | Zawartość |
|---|---|
| `schemas.py` | Pydantic models: `SessionConfig`, `DoctorResponse`, `MessageSuccessResponse`, itp. |
| `claims.py` | Walidacja claimów medycznych — **stabilny, nie ruszać bez rozmowy** |
| `conviction.py` | `update_conviction` — aktualizacja 5-wymiarowego stanu lekarza |
| `metrics.py` | `compute_turn_metrics`, `compute_frustration`, `advance_phase` |
| `message_analysis.py` | `analyze_message` — bribery, off-topic, marketing, gender, evidence |
| `doctor_traits.py` | `clamp_traits`, `build_style_directives`, `difficulty_profile` |
| `policy.py` | `policy_precheck`, `evidence_first_requirements`, `policy_postprocess_message` |
| `constants.py` | Słowniki słów kluczowych (bribery, marketing, off-topic, itd.) |

---

## Co teraz jest do zrobienia

### 1. Uruchomić testy integracyjne
Żaden test nie był jeszcze faktycznie uruchomiony. Potrzebny żywy serwer:

```bash
uvicorn api4:app --port 8000 --reload   # terminal 1
python3 tests/run_all.py                # terminal 2
```

Albo pojedynczo (przykład):
```bash
python3 tests/test_fournier_transcript.py
python3 tests/test_traditionalist_transcript.py
```

### 2. Kalibracja na podstawie wyników testów
Po uruchomieniu testów prawie na pewno wyjdą rozbieżności — progi conviction, wartości frustration, goal.score. Miejsca do kalibracji:
- Progi decyzji: `turn_planner.py` → `derive_decision_from_conviction()`
- Delty conviction: `conversation/conviction.py` → `update_conviction()`
- Bias frustracji per trudność: `conversation/doctor_traits.py` → `difficulty_profile()`

---

## Kluczowe decyzje architektury — nie zmieniaj bez rozmowy z Pawłem

1. **100% kompatybilność wsteczna.** `POST /start?id=X&drug_id=Y` bez body musi działać identycznie jak przed refaktoringiem.
2. **Agenda przez LLM**, nie hardcode w JSON. Błąd generacji → pusta lista (graceful fallback).
3. **Conviction jest źródłem decyzji.** `evaluate_conversation_goal` to tylko logger dla raportu — nie steruje decyzją.
4. **INFORMAL + FIRST_MEETING → HTTP 422.** Etykieta polska: na ty przy pierwszej wizycie to błąd.
5. **Nie ruszaj `conversation/claims.py`** — silnik walidacji jest stabilny i przetestowany ręcznie.
6. **Nie dodawaj nowych zależności** bez rozmowy. Obecne: pydantic, langgraph, langchain-openai, openai, fastapi, langchain-chroma, supabase.

---

## Konwencje kodu

- **Komentarze i docstringi po polsku** — tak jest w całym repo
- **Pydantic v2**: `model_validator`, `model_dump`, `ConfigDict`
- **Type hints wszędzie**
- **TypedDict dla state** (`ConversationState`) — nie BaseModel
- **Brak persystencji w trakcie sesji** — sesje w pamięci (`sessions: Dict` w `service.py`), zapis do Supabase tylko po `/finish`
- **Loggery per moduł**: `logger = logging.getLogger(__name__)`
- **Docstringi jednolinijkowe** — tylko nieoczywiste WHY, bez opisywania WHAT

---

## Co robić, gdy wracasz do pracy

1. `git log --oneline -5` — sprawdź co ostatnio zmieniano
2. `python3 -c "from langgraph_app.nodes import node_finalize; print('OK')"` — szybki import check
3. Jeśli serwer działa → `python3 tests/run_all.py` — sprawdź które testy przechodzą
4. Na podstawie wyników ustaj co kalibrować lub co budować dalej

---

## Antywzorce — czego NIE robić

❌ **Nie pisz `/start-v2`** — wszystko musi działać na `/start` z opcjonalnym body.

❌ **Nie wyrzucaj `evaluate_conversation_goal`** — zostaje jako logger metryk dla `/finish`.

❌ **Nie dodawaj nowych zasad do promptu lekarza** — sekcja BEZPIECZNIKI ma 8 reguł i to już dużo. Zmniejszaj, nie zwiększaj.

❌ **Nie ruszaj `conversation/claims.py`** bez rozmowy — silnik claimów jest stabilny.

❌ **Nie hardcoduj agendy** — musi być generowana przez LLM (agenda_generator.py). Fallback = pusta lista.

❌ **Nie dodawaj modeli Pydantic do `ConversationState`** — state to TypedDict, nie BaseModel.
