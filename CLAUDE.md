# CLAUDE.md — kontekst projektu dla Claude Code

> Ten plik jest automatycznie czytany przez Claude Code przy starcie każdej sesji w tym repo.
> Zawiera kontekst niezbędny do dalszej pracy nad refaktoringiem silnika symulacji rozmowy.

---

## O czym jest ten projekt

`medchatapi` to silnik symulacji rozmowy między **lekarzem (AI)** a **przedstawicielem farmaceutycznym (człowiek)**. Służy do treningu handlowców — bezpieczne środowisko do uczenia się prowadzenia rozmów sprzedażowych z różnymi typami lekarzy.

Stack: **Python 3.10+, FastAPI, LangGraph, OpenAI GPT (gpt-5.4-mini/gpt-4o), ChromaDB (RAG), Supabase (persystencja).**

Główne pliki:
- `api4.py` — FastAPI app (6 endpointów: /start, /message, /finish, /transcribe, /tts, /rate)
- `langgraph_app/` — graf konwersacji, węzły, state, service
- `conversation/` — claims, policy, schemas, data, RAG
- `doctor_archetypes.json` — 10 archetyp lekarzy
- `drug_catalog.json` — leki z claims do walidacji

Szczegóły architektury: `SPEC_API4_LANGGRAPH.md`.

---

## Aktualnie trwa refaktoring

**Problem:** rozmowa brzmi jak przesłuchanie, nie konwersacja. Lekarz tylko reaguje i zadaje pytania — nie wnosi własnej agendy. Brak jasnego kryterium sukcesu/porażki wewnątrz rozmowy.

**Pełny plan refaktoringu:** `docs/REFACTORING_PLAN.md` ← **PRZECZYTAJ TO PRZED ZMIANAMI.**

**Krótkie streszczenie planu (4 etapy):**

1. **Etap 1 — Konfiguracja sesji** ✅ ZAKOŃCZONY (w chacie z Claude'em web)
   - Nowe modele `SessionConfig`, `Familiarity`, `CommunicationRegister`, `CommunicationWarmth`
   - Endpoint `/start` przyjmuje opcjonalne body (100% kompatybilność wsteczna)
   - Pola konfiguracji w `ConversationState`
   - **Pliki Etapu 1 zostały już wgrane do repo, ale wymagają testów u użytkownika.**

2. **Etap 2 — Doctor Conviction (NASTĘPNY)**
   - 5-wymiarowy stan decyzyjny lekarza: `interest_level`, `trust_in_rep`, `clinical_confidence`, `perceived_fit`, `decision_readiness`
   - Zastępuje `evaluate_conversation_goal` jako *źródło* decyzji
   - Wartości startowe zależą od `familiarity` (z Etapu 1)
   - Decyzja lekarza wynika z osiągnięcia progów, nie z metryk postfactum

3. **Etap 3 — Agenda + przepisanie promptu**
   - `DoctorAgenda`: 3-5 wątków własnych lekarza (clinical_curiosity, patient_case, concern, personal, time_pressure)
   - Generowane LLM-em raz na starcie sesji
   - Przepisanie `_build_system_prompt` w `langgraph_app/nodes.py`: krócej zasad, więcej character, używać familiarity/register/warmth z Etapu 1
   - Zmniejszenie ingerencji `policy_postprocess_message` (zostają tylko bezpieczniki etyczne i Knowledge Guard na liczbach)

4. **Etap 4 — TurnMode**
   - Tryb tury wybierany heurystycznie: REACT/PROBE/SHARE/CHALLENGE/DRIFT/CLOSE
   - Nowy węzeł `node_plan_turn` w grafie (między `build_directives` a `retrieve_context`)
   - Łamie schemat "lekarz zawsze odpowiada"

---

## Kluczowe decyzje dotyczące architektury

Te ustalenia zostały podjęte z użytkownikiem (Pawłem) — **nie podważaj ich bez rozmowy.**

1. **100% kompatybilność wsteczna z istniejącym frontendem.** Aplikacja webowa istnieje i nie może być zmieniana. Wszelkie nowe parametry MUSZĄ być opcjonalne. `POST /start?id=X&drug_id=Y` bez body MUSI działać identycznie jak wcześniej.

2. **Agenda generowana przez LLM** na starcie sesji (1 dodatkowe wywołanie, ~0.001 USD). Nie hardcodowana w `doctor_archetypes.json`.

3. **Conviction zastępuje stary system decyzyjny od razu** — bez równoległego logowania starego. Stary `evaluate_conversation_goal` zostaje tylko jako logger metryk dla raportu końcowego.

4. **INFORMAL + FIRST_MEETING jest błędem walidacji (HTTP 422).** W polskiej kulturze biznesowej "na ty" przy pierwszej wizycie to poważne naruszenie etykiety — nie wolno na to pozwolić w konfiguracji.

5. **Nie wyrzucamy istniejących mechanizmów.** Claims engine, RAG, policy etyki, Knowledge Guard, frustracja, fazy — to wszystko zostaje. Zmieniamy *jak prompt z tego korzysta*, nie *czy istnieje*.

---

## Stan kodu — co już jest, co trzeba zrobić

### ✅ Zaimplementowane w Etapie 1

Zmodyfikowane pliki:
- `conversation/schemas.py` — modele `Familiarity`, `CommunicationRegister`, `CommunicationWarmth`, `SessionConfig`, walidator `validate_consistency`
- `langgraph_app/state.py` — 6 nowych pól (`familiarity`, `register`, `warmth`, `rep_name`, `rep_company`, `prior_visits_summary`)
- `langgraph_app/helpers.py` — `build_initial_state(...)` przyjmuje `session_config: Optional[SessionConfig]`
- `langgraph_app/service.py` — `start_session(...)` przyjmuje `session_config`, loguje konfigurację, opcjonalnie zwraca w response
- `api4.py` — endpoint `/start` ręcznie parsuje body z `Request` (obsługa: brak body / pusty / null / JSON), walidacja Pydantic → HTTP 422

Test ręczny do uruchomienia: `test_session_config.py` (8 testów walidacji `SessionConfig`).

### ⏳ Do zrobienia w Etapie 2 (kolejny krok)

**Cel:** zastąpić `evaluate_conversation_goal` (z `langgraph_app/helpers.py`) jako *źródło decyzji lekarza* przez 5-wymiarowy `DoctorConviction`.

**Pliki do zmiany:**

1. `conversation/schemas.py` — dodać model `DoctorConviction`:
   ```python
   class DoctorConviction(BaseModel):
       interest_level: float = 0.3       # zainteresowanie tematem (0-1)
       trust_in_rep: float = 0.3         # zaufanie do osoby (zależne od familiarity!)
       clinical_confidence: float = 0.2  # zrozumienie leku
       perceived_fit: float = 0.2        # dopasowanie do swoich pacjentów
       decision_readiness: float = 0.0   # gotowość do decyzji
   ```

2. `langgraph_app/state.py` — dodać pole `conviction: Dict[str, float]` (analogicznie do `traits`).

3. `langgraph_app/helpers.py` — w `build_initial_state`:
   - Inicjalizacja `conviction` z wartościami zależnymi od `familiarity`:
     - `first_meeting` → trust_in_rep = 0.2
     - `acquainted` → trust_in_rep = 0.45
     - `familiar` → trust_in_rep = 0.65
   - Pozostałe wymiary startują niezależnie od familiarity.

4. `conversation/policy.py` — nowa funkcja `update_conviction(state, claim_check, turn_metrics, message_analysis)`:
   - Wpływ na każdy wymiar zdefiniowany jako delta z mapowania zachowań przedstawiciela.
   - Przykład: claim potwierdzony → `clinical_confidence += 0.08`; fałszywy claim → `trust_in_rep -= 0.15`; evidence-based → `clinical_confidence += 0.1`.
   - Wartości klampowane do [0.0, 1.0].

5. `langgraph_app/nodes.py` — w `node_finalize`:
   - Po `compute_turn_metrics` wywołać `update_conviction`.
   - **Decyzja lekarza wynika z conviction**, nie z metryk:
     - `will_prescribe` jeśli wszystkie 5 wymiarów >= 0.65 + decision_readiness >= 0.75
     - `trial_use` jeśli 4/5 wymiarów >= 0.55 + decision_readiness >= 0.6
     - `reject` jeśli trust_in_rep < 0.25 OR (false_claims > 0 AND severity == critical)
     - `undecided` w pozostałych przypadkach
   - W prompcie (na razie nie ruszamy `_build_system_prompt`, to Etap 3 — ALE dodaj sekcję `Twój wewnętrzny stan: ...` z wartościami conviction tak, żeby LLM wiedział na czym stoi).

6. `langgraph_app/helpers.py` — funkcję `evaluate_conversation_goal` ZACHOWAĆ, ale przestać używać jako źródło decyzji:
   - Wywołanie zostaje (zwraca metryki dla raportu końcowego)
   - Ale `current_doctor_decision` w `node_finalize` MA wynikać z `derive_decision_from_conviction(conviction)`, nie z `evaluate_conversation_goal`.

7. `conversation/schemas.py` — `MessageSuccessResponse` (response z `/message`): dodać OPCJONALNE pole `conviction: Optional[DoctorConviction] = None`, żeby nowy frontend mógł zobaczyć trajektorię.

**Test, który trzeba napisać po Etapie 2:**
- Conviction startowy zależy od familiarity (3 scenariusze).
- Po fałszywym claimie krytycznym trust_in_rep spada.
- Po 3 udanych claimach potwierdzonych decision_readiness rośnie.
- Decyzja `reject` wymuszona przy trust_in_rep < 0.25.

---

## Konwencje kodu w tym repo

- **Język komentarzy i docstringów: polski** (tak jest w istniejącym kodzie).
- **Pydantic v2** (`model_validator`, `model_dump`, `ConfigDict`).
- **Type hints wszędzie**, szczególnie w `helpers.py` i `policy.py`.
- **TypedDict dla state**, nie BaseModel — `ConversationState` jest TypedDict z `total=False`.
- **Brak persystencji w trakcie sesji** — sesje w pamięci (`sessions: Dict` w `service.py`). Persystencja tylko na końcu (Supabase).
- **Loggery per moduł**: `logger = logging.getLogger(__name__)`.
- **Nie wprowadzaj nowych zależności** bez rozmowy z użytkownikiem. Obecne: pydantic, langgraph, langchain-openai, openai, fastapi, langchain-chroma, supabase.

---

## Pliki kluczowe — kogo czytać o czym

| Plik | Po co tam zaglądać |
|------|-------------------|
| `langgraph_app/graph.py` | Topologia grafu — kolejność węzłów, conditional routing |
| `langgraph_app/nodes.py` | Logika węzłów — szczególnie `_build_system_prompt` (linia ~416) i `node_finalize` (~566) |
| `langgraph_app/helpers.py` | `build_initial_state`, `evaluate_conversation_goal`, knowledge guard, random events |
| `langgraph_app/state.py` | Pełna struktura ConversationState |
| `conversation/policy.py` | Frustracja, fazy, traits, dyrektywy stylu, postprocess, evidence-first |
| `conversation/claims.py` | Walidacja claimów medycznych |
| `conversation/schemas.py` | Pydantic models — kontrakt API |
| `conversation/constants.py` | Słowniki słów kluczowych (bribery, off-topic, marketing, etc.) |

---

## Co robić, gdy użytkownik wraca do pracy

1. **Najpierw zapytaj na czym skończyliśmy.** Stan zapisany tu może być nieaktualny.
2. **Sprawdź czy Etap 1 jest fizycznie wgrany do repo** (`git status`, `git log --oneline -5`). Jeśli nie — pomóż wgrać.
3. **Sprawdź czy testy z Etapu 1 przechodzą** (`python3 test_session_config.py`).
4. **Wtedy przechodź do Etapu 2** zgodnie z planem powyżej.
5. **Każdy etap jest osobnym PR-em / commitem.** Nie mieszaj etapów.

---

## Antywzorce — czego NIE robić

❌ **Nie przepisuj `_build_system_prompt` w Etapie 2.** To Etap 3. Skup się na conviction.

❌ **Nie wyrzucaj `evaluate_conversation_goal`** — zostaje jako logger, tylko przestaje sterować decyzją.

❌ **Nie dodawaj nowych zasad do promptu lekarza** — obecnie jest ich 15, to za dużo. Plan Etapu 3 to *zmniejszyć* do 4-5.

❌ **Nie robisz `/start-v2`** — wszystko musi działać na `/start` z opcjonalnym body.

❌ **Nie wymuszaj `rep_name` w SessionConfig** — to opcjonalne, walidator tylko ostrzega softowo.

❌ **Nie ruszaj `conversation/claims.py`** bez rozmowy — silnik walidacji claimów jest sprawdzony i stabilny.
