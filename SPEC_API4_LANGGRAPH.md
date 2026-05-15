# Specyfikacja Aplikacji — API v4 (LangGraph)

> Dokument opisuje aktualny stan aplikacji `api4.py` oraz modułu `langgraph_app/`.  
> Przeznaczony jako punkt odniesienia przy dalszych modyfikacjach.

---

## 1. Cel aplikacji

Symulator rozmowy między **lekarzem** (AI) a **przedstawicielem farmaceutycznym** (człowiek).  
Przedstawiciel ma za zadanie przekonać lekarza do stosowania konkretnego leku — prezentując dane kliniczne, budując relację i odpowiadając na obiekcje. Lekarz jest agentem AI z profilem psychologicznym, który ewoluuje w trakcie rozmowy.

---

## 2. Architektura ogólna

```
api4.py                          ← FastAPI: 6 endpointów (identyczne z api3)
└── langgraph_app/
    ├── service.py               ← Zarządzanie sesjami + wywołanie grafu
    ├── graph.py                 ← Kompilacja StateGraph
    ├── nodes.py                 ← Węzły grafu (logika przetwarzania)
    ├── helpers.py               ← Czyste funkcje pomocnicze
    └── state.py                 ← ConversationState (TypedDict)
conversation/                    ← Moduły wspólne (api3 + api4)
    ├── claims.py                ← Silnik walidacji claimów medycznych
    ├── policy.py                ← Reguły policy, metryki, traits
    ├── schemas.py               ← Modele Pydantic (kontrakty API)
    ├── data.py                  ← Ładowanie doctor_archetypes.json, drugs.json
    ├── tts.py                   ← Synteza mowy (OpenAI TTS)
    └── transcribe.py            ← Transkrypcja audio (OpenAI STT)
```

**Sesje przechowywane w pamięci** — `sessions: Dict[str, ConversationState]` w `service.py`.  
Brak persystencji do bazy danych. Brak skalowania horyzontalnego.

---

## 3. Endpointy FastAPI

Wszystkie endpointy są **identyczne z api3.py** — aplikacja webowa nie wymaga zmian.

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| `POST` | `/start` | Inicjalizacja nowej sesji. Query params: `?id=<doctor_id>&drug_id=<drug_id>` |
| `POST` | `/message` | Przetworzenie jednej tury rozmowy. Body: `{session_id, message}` |
| `POST` | `/finish` | Zakończenie sesji + ocena końcowa. Query param: `?session_id=<id>` |
| `POST` | `/transcribe` | Transkrypcja audio base64 → tekst PL |
| `POST` | `/transcribe-file` | Transkrypcja pliku audio (multipart/form-data) |
| `POST` | `/tts` | Synteza mowy lekarza: tekst → MP3 base64 |

### Uruchomienie

```bash
uvicorn api4:app --reload --log-level debug
# Swagger: http://127.0.0.1:8000/docs
```

### Zmienne środowiskowe

| Zmienna | Domyślna | Opis |
|---------|----------|------|
| `OPENAI_API_KEY` | *(hardcoded w api4.py)* | Klucz OpenAI |
| `OPENAI_MODEL` | `gpt-4o` | Model LLM do generowania odpowiedzi |
| `CORS_ORIGINS` | `""` | Dozwolone originy CORS, `"*"` = wszystkie |
| `OPENAI_TRANSCRIBE_MODEL` | `gpt-4o-mini-transcribe` | Model STT |
| `OPENAI_TRANSCRIBE_FALLBACKS` | `whisper-1` | Fallback STT |

---

## 4. Graf LangGraph

### 4.1 Przegląd

Każde wywołanie `/message` uruchamia **jeden przebieg grafu** przez `compiled_graph.invoke(state, config)`.  
Graf jest bezstanowy (brak checkpointera) — sesja persystuje w słowniku `sessions` po stronie `service.py`.  
Model i klucz API przekazywane są przez `RunnableConfig["configurable"]`.

### 4.2 Diagram przepływu

```
START
  │
  ▼
detect_context          ← Wykrycie intencji i leku
  │
  ▼
analyze                 ← Analiza claimów, pokrycia, evidence
  │
  ▼
policy_check            ← Hard-stop? Dyrektywy wstępne?
  │
  ├─[hard_stop = True]──→  ethics_stop ──→ END
  │
  └─[hard_stop = False]──▼
                      update_state     ← Frustracja, faza, zdarzenia losowe
                          │
                          ├─[turn_index > max_turns]──→  time_stop ──→ END
                          │
                          └─[czas OK]──▼
                                  build_directives  ← Dyrektywy dla LLM
                                      │
                                      ▼
                                  generate_response ← Wywołanie LLM (ChatOpenAI)
                                      │
                                      ▼
                                  finalize          ← Post-processing, ocena, traits
                                      │
                                      ▼
                                     END
```

### 4.3 Opis węzłów

#### `detect_context`
**Plik:** `nodes.py` → `node_detect_context`

Wykrywa, czy przedstawiciel ujawnił:
- **cel wizyty** — obecność słów kluczowych z grupy `INTENT_SIGNAL_KEYWORDS` (lek, terapia, dawkowanie, refundacja, itd.)
- **temat rozmowy** — konkretny lek, wykrywany po `drug_intro_keywords` (id leku, nazwa, substancja czynna)

Ustawia w stanie: `intent_revealed`, `intent_revealed_turn`, `drug_revealed`, `drug_revealed_turn`.

**Zasada:** Lekarz nie wie po co przyszła osoba, dopóki `intent_revealed = False`. Dopóki `drug_revealed = False`, lekarz nie zna tematu rozmowy i nie ocenia twierdzeń o leku.

---

#### `analyze`
**Plik:** `nodes.py` → `node_analyze`

Wykonuje pełną analizę bieżącej wypowiedzi przedstawiciela:

1. **`analyze_message`** (`conversation/policy.py`) — wykrywa sygnały jakościowe:
   - `bribery_hits` — próby przekupstwa (`BRIBERY_KEYWORDS`)
   - `off_topic_hits` — odbieganie od tematu (`OFF_TOPIC_KEYWORDS`)
   - `marketing_hits` — slogany marketingowe (`MARKETING_BUZZWORDS`)
   - `english_hits` — anglicyzmy marketingowe (`ENGLISH_MARKETING_WORDS`)
   - `inappropriate_hits` — nieeleganckie propozycje (`INAPPROPRIATE_PROPOSAL_KEYWORDS`)
   - `disrespect_hits` — brak szacunku (`DISRESPECTFUL_LANGUAGE_KEYWORDS`)
   - `gender_mismatch_hits` — niepoprawna forma grzecznościowa
   - `evidence_hits` — frazy evidence-based (`EVIDENCE_PHRASES`)
   - `clinical_study_hits` — odwołania do badań klinicznych (`CLINICAL_STUDY_PHRASES`)
   - `has_drug_focus` — czy wypowiedź dotyczy leku

2. **`check_medical_claims`** (`conversation/claims.py`) — walidacja twierdzeń medycznych:
   - `false_claims` — sprzeczne z faktami leku
   - `unsupported_claims` — bez pokrycia w danych leku
   - `supported_claims` — zgodne z faktami leku
   - `severity_counts` — liczba błędów krytycznych / major / minor

3. **`update_claim_coverage`** — aktualizuje, które kluczowe claimy zostały poruszone
4. **`critical_coverage_summary`** — procent pokrycia claimów krytycznych
5. **`evidence_first_requirements`** — czy tryb weryfikacji jest aktywny:
   - `require_verification = True` → fałszywe / niepotwierdzone twierdzenia (żądaj źródła)
   - `require_probe = True` → brak potwierdzonych claimów, choć jest mowa o leku (dopytaj o dane)

Analiza claimów działa **tylko gdy `drug_revealed = True`** — zanim lek zostanie przedstawiony, pola claimów są puste.

---

#### `policy_check`
**Plik:** `nodes.py` → `node_policy_check`

Wywołuje `policy_precheck()` z `conversation/policy.py`. Buduje dyrektywy wymuszone przez reguły:
- `hard_stop = True` jeśli wykryto `bribery_hits` → natychmiastowe zakończenie
- Dodatkowe dyrektywy: korekta formy grzecznościowej, żądanie konkretu klinicznego, zatrzymanie off-topic, itp.

**Routing po tym węźle:**
- `hard_stop = True` → `ethics_stop`
- `hard_stop = False` → `update_state`

---

#### `ethics_stop`
**Plik:** `nodes.py` → `node_ethics_stop`

Wymuszone zakończenie z powodu naruszenia etyki. Emituje stały komunikat lekarza, ustawia `is_terminated = True`, decyzja lekarza `= reject`, status celu `= not_achieved`.

---

#### `update_state`
**Plik:** `nodes.py` → `node_update_state`

Aktualizuje stan algorytmiczny po analizie wypowiedzi:

1. **Metryki tury** (`compute_turn_metrics`):
   - `topic_adherence` — zgodność z tematem leku
   - `clinical_precision` — precyzja merytoryczna
   - `ethics` — zachowanie etyczne
   - `language_quality` — jakość języka

2. **Frustracja** (`compute_frustration`) — delta na podstawie wykrytych problemów, dodawana do `frustration_score` (skala 0–10)

3. **Faza rozmowy** (`advance_phase`) — state-machine przejść faz

4. **Zdarzenia losowe** — losowe zdarzenia środowiskowe (telefon, kolejka, wezwanie do pacjenta), determinizowane przez hash sesji+tury

**Routing po tym węźle:**
- `turn_index > max_turns` → `time_stop`
- w przeciwnym razie → `build_directives`

---

#### `time_stop`
**Plik:** `nodes.py` → `node_time_stop`

Wymuszone zakończenie z powodu przekroczenia limitu tur. Lekarz informuje, że nie ma już czasu, prosi o przesłanie materiałów pisemnie.

---

#### `build_directives`
**Plik:** `nodes.py` → `node_build_directives`

Składa listę dyrektyw behawioralnych dla LLM z kilku źródeł:
- Dyrektywy stylu z `build_style_directives()` (na podstawie `communication_style` i `traits`)
- Dyrektywy z `policy_precheck` (reguły policy)
- Dyrektywy `evidence_first_requirements` (żądanie danych / weryfikacji)
- Dyrektywy kontekstowe: `intent_revealed`, `drug_revealed`, brakujące claime, off-topic, marketing, anglicyzmy
- Dyrektywa zdarzenia losowego (jeśli wystąpiło)

Wynik: lista ciągów tekstowych wstrzykiwana do systemu promptu lekarza.

---

#### `generate_response`
**Plik:** `nodes.py` → `node_generate_response`

Wywołuje LLM (`ChatOpenAI.with_structured_output(DoctorResponse)`):

1. Buduje **system prompt** z:
   - Profilem lekarza (nazwa, opis, kontekst sytuacyjny)
   - Aktualnymi cechami psychologicznymi
   - Fazą rozmowy i jej celem
   - Numerem tury i budżetem tur
   - Poziomem frustracji
   - Danymi leku (status: ujawniony / nieujawniony)
   - Wynikami analizy, metryki tury
   - Listą dyrektyw behawioralnych
   - 18 zasadami zachowania lekarza

2. Dołącza **historię rozmowy** z `state["messages"]` jako `HumanMessage` / `AIMessage`

3. Dołącza **bieżącą wiadomość** jako `HumanMessage`

Zwraca obiekt `DoctorResponse`:
```python
class DoctorResponse(BaseModel):
    doctor_message: str          # Treść odpowiedzi lekarza
    updated_traits: TraitsUpdate # Nowe cechy psychologiczne (proponowane przez LLM)
    reasoning: str               # Uzasadnienie reakcji
    detected_errors: List[str]   # Błędy merytoryczne wg LLM
    context_shift: Optional[str] # Trwała zmiana kontekstu/atmosfery
    doctor_attitude: DoctorAttitude  # happy | neutral | serious | sad
    doctor_decision: DoctorDecision  # undecided | trial_use | will_prescribe | recommend | reject
```

---

#### `finalize`
**Plik:** `nodes.py` → `node_finalize`

Post-processuje surową odpowiedź LLM przez kolejne kroki:

1. **Zbieranie błędów** — łączy `detected_errors` z LLM z błędami wykrytymi regułowo (marketing, anglicyzmy, forma grzecznościowa, brak szacunku, fałszywe claime, niezweryfikowane claime)

2. **`policy_postprocess_message`** — wymuszony dopisek korekt:
   - Korekta formy grzecznościowej jeśli `gender_mismatch_hits`
   - Granica profesjonalna jeśli `inappropriate_hits` lub `disrespect_hits`
   - Powrót do tematu jeśli `off_topic_hits`
   - Sygnalizacja błędu merytorycznego jeśli `false_claims`
   - Pytanie o brakujące claime krytyczne
   - Pytanie evidence-first (weryfikacja / sondaż)
   - W fazie close: zamknięcie rozmowy

3. **Knowledge Guard** — przepisuje zdania z danymi leku, których przedstawiciel nie podał, jako *zasłyszaną opinię* (np. `"Słyszałam od koleżanki, że..."`). Chroni przed dopowiadaniem lekarzowi nieujawnionych faktów.

4. **Uzasadnienie końcowe** — uzupełnia `reasoning` o kontekst: marketing, evidence-first, zdarzenia losowe, braki coverage

5. **Nastawienie lekarza** — wymusza `"serious"` gdy:
   - ≥ 2 wykryte błędy, LUB
   - ostatnia tura z dużą presją czasu, LUB
   - frustracja ≥ 6.0

6. **Decyzja lekarza** — pobiera od LLM, z fallbackiem `infer_decision_from_message()` gdy `undecided`

7. **Ocena celu rozmowy** — `evaluate_conversation_goal()` (szczegóły w sekcji 6)

8. **Aktualizacja traits** — `apply_reaction_rules()` nakłada deterministyczne korekty na traits zaproponowane przez LLM

9. **Sprawdzenie zakończenia** — `check_termination()`:
   - Limit tur osiągnięty + presja czasu / faza close
   - Cierpliwość ≤ 0.2, LUB presja czasu ≥ 0.8, LUB frustracja ≥ próg, LUB po turze 3 nieprofesjonalny ton
   - Cel osiągnięty → wymuszone zakończenie (pozytywne)

10. Aktualizacja `state["messages"]` — dołączenie `user` + `assistant`

---

## 5. Stan konwersacji (`ConversationState`)

Definicja w `langgraph_app/state.py`. TypedDict z `total=False`.  
Pełna lista pól:

### Statyczny kontekst sesji
| Pole | Typ | Opis |
|------|-----|------|
| `session_id` | `str` | UUID sesji |
| `doctor_profile` | `Dict` | Profil lekarza z `doctor_archetypes.json` |
| `drug_info` | `Dict` | Dane leku z `drugs.json` |
| `messages` | `List[Dict]` | Historia wiadomości LLM `[{role, content}]` |

### Cechy psychologiczne lekarza
| Pole | Zakres | Opis |
|------|--------|------|
| `traits.skepticism` | 0.0–1.0 | Sceptycyzm wobec twierdzeń |
| `traits.patience` | 0.0–1.0 | Cierpliwość lekarza |
| `traits.openness` | 0.0–1.0 | Otwartość na nowe informacje |
| `traits.ego` | 0.0–1.0 | Poczucie własnej ważności |
| `traits.time_pressure` | 0.0–1.0 | Presja czasu / niecierpliwość |

### Postęp rozmowy
| Pole | Typ | Opis |
|------|-----|------|
| `turn_index` | `int` | Numer aktualnej tury (od 1) |
| `max_turns` | `int` | Limit tur (wyliczany z traits i difficulty) |
| `phase` | `str` | Aktualna faza: `opening`, `needs`, `objection`, `evidence`, `close` |
| `frustration_score` | `float` | Poziom frustracji 0–10 |
| `difficulty` | `str` | `easy`, `medium`, `hard` |
| `close_phase_threshold` | `float` | Frustracja przy której wchodzi faza close |
| `termination_frustration_threshold` | `float` | Próg frustracji kończący rozmowę |

### Bramy wykrywania
| Pole | Opis |
|------|------|
| `intent_revealed` | Czy przedstawiciel ujawnił cel wizyty |
| `drug_revealed` | Czy lek został przedstawiony |
| `drug_intro_keywords` | Tokeny do wykrywania nazwy leku |

### Śledzenie claimów
| Pole | Opis |
|------|------|
| `claim_index` | Mapa `claim_id → {severity, statement}` |
| `critical_claim_ids` | ID claimów o severity = critical |
| `seen_claim_ids` | Wszystkie poruszone claime |
| `covered_critical_claim_ids` | Potwierdzone claime krytyczne |

### Śledzenie celu
| Pole | Opis |
|------|------|
| `goal_achieved` | Czy cel kiedykolwiek osiągnięty |
| `goal_status` | `achieved`, `partial`, `not_achieved` |
| `goal_score` | 0–100 |
| `latest_goal` | Pełny obiekt `ConversationGoal` z ostatniej tury |

### Pola robocze bieżącej tury (`current_*`)
Nadpisywane przy każdym wywołaniu grafu. Przechowują wyniki poszczególnych węzłów.

---

## 6. Zasady oceny rozmowy

### 6.1 Metryki tury

Obliczane przez `compute_turn_metrics()` w `conversation/policy.py` na podstawie każdej wypowiedzi:

| Metryka | Zakres | Obliczenie |
|---------|--------|-----------|
| `topic_adherence` | 0.0–1.0 | Bazowo 0.3; +0.5 gdy `has_drug_focus`; -0.35 gdy `off_topic` bez focus |
| `clinical_precision` | 0.0–1.0 | Bazowo 0.8; -0.25/claim za fałszywe, -0.12/claim za niepotwierdzone, -0.07/hit za marketing; kara za brakujące claime krytyczne |
| `ethics` | 0.0–1.0 | 1.0 bazowo; 0.0 gdy przekupstwo; -0.3/hit za nieeleganckie; -0.2/hit za brak szacunku; -0.15/hit za złą formę grzecznościową |
| `language_quality` | 0.0–1.0 | 1.0 bazowo; -0.12/hit za anglicyzmy; -0.08/hit za marketing; -0.2/hit za brak szacunku |
| `critical_claim_coverage` | 0.0–1.0 | Ratio pokrytych claimów krytycznych |

### 6.2 Frustracja lekarza

Kumulatywna skala 0–10. Delta tury obliczana przez `compute_frustration()`:

| Zdarzenie | Delta |
|-----------|-------|
| Próba przekupstwa | +8.0 |
| Nieelegancka propozycja | +2.2 / hit |
| Brak szacunku | +1.5 / hit |
| Zła forma grzecznościowa | +0.9 / hit |
| Claim krytyczny fałszywy | +1.6 |
| Claim major fałszywy | +1.0 |
| Claim minor fałszywy | +0.5 |
| Off-topic (cel znany) | +1.4 |
| Slogan marketingowy | +0.45 / hit |
| Anglicyzm | +0.35 / hit |
| Brakujące claime krytyczne | +0.6–1.7 (rośnie z postępem tury) |
| Dobry topic adherence ≥ 0.7 | -0.4 |
| Dobra precyzja kliniczna ≥ 0.8 | -0.2 |
| Potwierdzony claim | -0.15 / claim |
| Bias trudności (od tury 2) | -0.35 (easy) / 0.0 (medium) / +0.45 (hard) |

**Próg zakończenia rozmowy** ustawiony przez `difficulty`:
- easy: 10.0 (praktycznie nie kończy)
- medium: 10.0
- hard: 9.5

### 6.3 Fazy rozmowy (state-machine)

| Faza | Cel lekarza | Warunek przejścia |
|------|-------------|-------------------|
| `opening` | Ustalenie kontekstu, pytanie o wartość kliniczną | `has_drug_focus = True` → `needs` |
| `needs` | Doprecyzowanie profilu pacjenta i zastosowań | `clinical_precision ≥ 0.6` → `objection` |
| `objection` | Wyrażenie obiekcji, wymaganie odpowiedzi | `clinical_precision ≥ 0.7` i brak marketingu → `evidence` |
| `evidence` | Żądanie danych: bezpieczeństwo, dawkowanie | Tura `≥ max_turns - 1` lub frustracja `≥ 8.5` → `close` |
| `close` | Domknięcie, zakończenie rozmowy | Stan końcowy |

**Wymuszenia fazy close:**
- `bribery_hits` → natychmiast `close`
- `frustration ≥ close_phase_threshold` (domyślnie 9.0)
- `turn_index ≥ max_turns`

### 6.4 Limit tur

Obliczany przy starcie sesji przez `derive_turn_limit()`:

| Warunki | Limit bazowy |
|---------|--------------|
| `time_pressure ≥ 0.9` | 12 tur |
| `time_pressure ≥ 0.75` | 14 tur |
| `time_pressure ≥ 0.55` | 16 tur |
| `patience ≥ 0.8` i `time_pressure ≤ 0.4` | 20 tur |
| Pozostałe | 18 tur |

Modyfikatory: difficulty easy +1, difficulty hard -1. Zakres ostateczny: 10–20.  
Zdarzenia losowe mogą skrócić limit o 1.

### 6.5 Ocena celu rozmowy (`evaluate_conversation_goal`)

Wywoływana po każdej turze. Zwraca `ConversationGoal`:

#### Warunki `status = "achieved"` (wszystkie muszą być spełnione):
1. `drug_revealed = True`
2. Decyzja lekarza ∈ `{trial_use, will_prescribe, recommend}`
3. Brak `bribery_hits`, `inappropriate_hits`, `disrespect_hits`
4. `clinical_precision ≥ 0.7` i brak `false_claims`
5. Pokrycie claimów krytycznych ≥ 80% (lub brak claimów krytycznych)
6. Wymagania evidence-first spełnione (`require_verification = False`)
7. `doctor_attitude ∈ {happy, neutral}` i `frustration ≤ 6.0` i `topic_adherence ≥ 0.6`
8. `score ≥ 75`

#### Formuła score (0–100):
```
base_score =  0.35 × clinical_precision
            + 0.25 × coverage_ratio
            + 0.20 × ethics
            + 0.10 × language_quality
            + 0.10 × relationship_score   ← 1 - frustration/10, korekta za attitude

bonus: +0.12 jeśli decyzja pozytywna
kara:  -0.10 jeśli reject
kara:  -0.12 jeśli evidence_first nie spełnione
```

#### Status `"partial"`: `score ≥ 55` i brak reject i brak naruszenia etyki  
#### Status `"not_achieved"`: pozostałe przypadki

### 6.6 Decyzja lekarza

LLM zwraca jedną z wartości `DoctorDecision`:

| Wartość | Znaczenie |
|---------|-----------|
| `undecided` | Bez decyzji (domyślna) |
| `trial_use` | Rozważy zastosowanie u wybranych pacjentów |
| `will_prescribe` | Włączy lek do terapii |
| `recommend` | Zarekomenduje lek |
| `reject` | Odrzuca propozycję |

Gdy LLM zwróci `undecided`, system próbuje wywnioskować decyzję z treści wypowiedzi (`infer_decision_from_message`).

### 6.7 Ocena końcowa (`/finish`)

Po zakończeniu sesji LLM generuje `EvaluationResult`:
- `professionalism_score` (1–10)
- `relevance_score` (1–10) — trafność argumentów
- `relationship_score` (1–10) — budowanie relacji
- `strengths` — 2–3 mocne strony
- `areas_for_improvement` — 2–3 obszary do poprawy
- `final_feedback` — krótkie podsumowanie i główna porada

---

## 7. Profile lekarzy (`doctor_archetypes.json`)

10 dostępnych archetyp:

| ID | Nazwa | Trudność | Styl | Strategia |
|----|-------|----------|------|-----------|
| `busy_pragmatist` | Zabiegany pragmatyk | easy | krótki, bezpośredni | transactional |
| `skeptical_expert` | Sceptyczny ekspert | medium | analityczny, krytyczny | skeptical, confrontational |
| `friendly_generalist` | Przyjazny ogólny | easy | ciepły, rozmowny | exploratory |
| `tired_cynic` | Zmęczony cynik | hard | suchy, zdystansowany | disengaging, confrontational |
| `academic_researcher` | Naukowiec akademicki | hard | formalny, precyzyjny | skeptical, exploratory |
| `young_enthusiast` | Młody entuzjasta | medium | energiczny, ciekawy | exploratory |
| `dominant_authority` | Dominujący autorytet | hard | dominujący, stanowczy | confrontational |
| `curious_specialist` | Ciekawy specjalista | medium | dociekliwy | exploratory |
| `defensive_traditionalist` | Defensywny tradycjonalista | hard | ostrożny, zachowawczy | skeptical |
| `empathetic_clinician` | Empatyczny klinicysta | medium | empatyczny, spokojny | exploratory |

### Cechy psychologiczne — interpretacja w dyrektywach stylu

| Trait | Wartość | Dyrektywa |
|-------|---------|-----------|
| `time_pressure ≥ 0.75` | wysoka | Max 1–2 zdania odpowiedzi |
| `patience ≤ 0.35` | niska | Max 1–2 zdania |
| `skepticism ≥ 0.75` | wysoki | Krytyczne podejście do twierdzeń |
| `openness ≤ 0.30` | niska | Ograniczone zaufanie, wymaga doprecyzowania |
| `ego ≥ 0.75` | wysokie | Oczekuje profesjonalnego tonu i szacunku |
| `frustration ≥ 6.0` | wysoka | Max 1 zdanie, ucinanie rozmowy |

### Strategie lekarza — wpływ na zachowanie

| Strategia | Efekt |
|-----------|-------|
| `transactional` | Krótko, konkretnie, bez dygresji; +0.7 frustracja za off-topic |
| `skeptical` | Wymaga uzasadnienia każdej tezy; +0.35 frustracja za niepotwierdzone claime |
| `confrontational` | Styl konfrontacyjny, testuje spójność argumentów; +0.5 frustracja za błędy |
| `exploratory` | Zadaje pytania pogłębiające; -0.35 frustracja gdy dobre dane bez błędów |
| `disengaging` | Szybko kończy wątki bez wartości; +0.6 za brak focus (gdy cel znany) |

---

## 8. Silnik claimów medycznych

### Walidacja (`check_medical_claims`, `conversation/claims.py`)

Dla każdego claimu z `drugs.json` sprawdzane jest:
1. **Similarity semantyczna** — overlap tokenów między wypowiedzią a claimem
2. **Keyword hits** — bezpośrednie wystąpienie słów kluczowych claimu
3. **Wzorce sprzeczności** (`contradiction_patterns`) → `false_claims`
4. **Wzorce wsparcia** (`support_patterns`) lub wysoka similarity → `supported_claims`
5. **Intent kliniczny + brak potwierdzenia** → `unsupported_claims`

Specjalna obsługa dawkowania: porównanie liczb z wypowiedzi z liczbami z definicji claimu.

### Coverage claimów krytycznych

Claime z `severity = "critical"` są śledzone osobno.  
`coverage_ratio = covered_critical / total_critical`  
Używany w metrykach, ocenie celu i dyrektywach dopytujących.

---

## 9. Evidence-First

Mechanizm wymuszający podawanie danych klinicznych. Aktywuje się gdy:

| Tryb | Warunek | Akcja lekarza |
|------|---------|--------------|
| `require_verification` | `false_claims > 0` LUB `unsupported_claims > 0` | Zakwestionuj twierdzenia, żądaj źródła i parametrów |
| `require_probe` | `has_drug_focus = True` i `supported_claims = 0` i brak verification | Dopytaj o konkretne dane kliniczne |

Wpływ na `score`: jeśli `require_verification = True` → kara -0.12 w ocenie celu.

---

## 10. Zdarzenia losowe

Deterministyczne (hash sesji + tury) "zdarzenia środowiskowe" zwiększające realizm:

| Zdarzenie | Baza prob. | Efekt |
|-----------|-----------|-------|
| Telefon z oddziału | 11% | patience -0.08, time_pressure +0.12, frustration +0.35, limit tur -1 |
| Wezwanie do pacjenta | 8% (+8% gdy frustration ≥ 5) | patience -0.14, time_pressure +0.20, frustration +0.60, limit tur -1 |
| Wzrost kolejki | 14% | patience -0.10, time_pressure +0.15, frustration +0.45, limit tur -1 |

Cooldown: minimum 2 tury między zdarzeniami. Aktywacja dopiero od tury 2.

---

## 11. Knowledge Guard

Chroni przed sytuacją, gdy lekarz "dopowiada" szczegóły leku, których przedstawiciel nie podał.

**Mechanizm:**
1. Buduje listę wrażliwych fraz z claimów i danych leku (`build_sensitive_drug_phrases`)
2. Sprawdza, czy frazy te pojawiają się w odpowiedzi lekarza, ale **nie** w historii wypowiedzi przedstawiciela
3. Przepisuje takie zdania jako *zasłyszaną opinię*: `"Słyszałam od koleżanki, że..."`, `"Czytałam gdzieś, że..."` itd.

Wybór frazy jest deterministyczny: hash(`session_id|turn_index|hearsay`).

---

## 12. Komendy specjalne

Definiowane w `conversation/constants.py`, `CHAT_COMMANDS`:

| Komenda (dokładna treść) | Akcja |
|--------------------------|-------|
| `"Zamienić"` | `close_visit` — wymuszony hard-stop etyczny |
| `"Skuteczny"` | `increase_openness` — zwiększa openness lekarza o +0.3 |

Komendy obsługiwane przez `service.py` przed wywołaniem grafu.

---

## 13. TTS i Transkrypcja

### TTS (`/tts`, `conversation/tts.py`)
- Model: `tts-1` (OpenAI)
- Głos pobierany z `doctor_profile["tts_voice"]` (alloy / shimmer / nova)
- Tempo (`speed`) zależne od `doctor_attitude`:
  - `happy` → 1.05
  - `neutral` → 1.0
  - `serious` → 0.95
  - `sad` → 0.88

### Transkrypcja (`/transcribe`, `conversation/transcribe.py`)
- Model główny: `gpt-4o-mini-transcribe` (env: `OPENAI_TRANSCRIBE_MODEL`)
- Fallback: `whisper-1` (env: `OPENAI_TRANSCRIBE_FALLBACKS`)
- Zawsze język: `pl`

---

## 14. Jak modyfikować aplikację

### Dodanie nowego lekarza
Edytuj `doctor_archetypes.json` — dodaj obiekt zgodny ze schematem:
```json
{
  "id": "unique_id",
  "name": "Wyświetlana nazwa",
  "gender": "female",
  "tts_voice": "alloy",
  "description": "Krótki opis",
  "communication_style": "np. analityczny, krótki",
  "context_str": "Opis sytuacji lekarza",
  "traits": { "skepticism": 0.5, "patience": 0.5, "openness": 0.5, "ego": 0.5, "time_pressure": 0.5 },
  "preferred_strategies": ["skeptical"],
  "difficulty": "medium"
}
```

### Dodanie nowego leku
Edytuj `drugs.json` — każdy lek może mieć sekcję `claims[]` z walidowanymi twierdzeniami.

### Zmiana modelu LLM
Ustaw zmienną środowiskową `OPENAI_MODEL=gpt-4o-mini` lub inny model.

### Dodanie nowego węzła do grafu
1. Napisz funkcję `def node_moj_wezel(state: ConversationState) -> Dict:` w `nodes.py`
2. Zarejestruj węzeł w `graph.py`: `graph.add_node("moj_wezel", node_moj_wezel)`
3. Dodaj krawędź: `graph.add_edge("poprzedni_wezel", "moj_wezel")`

### Zmiana progów frustacji / zakończenia
Edytuj `difficulty_profile()` w `conversation/policy.py`:
```python
return {
    "frustration_bias": 0.0,
    "close_phase_threshold": 9.0,
    "termination_frustration_threshold": 10.0,
    "turn_limit_adjust": 0,
}
```

### Dodanie nowej komendy specjalnej
Dodaj wpis do `CHAT_COMMANDS` w `conversation/constants.py`:
```python
CHAT_COMMANDS = {
    "Zamienić": "close_visit",
    "Skuteczny": "increase_openness",
    "MojaKomenda": "moja_akcja",   # ← nowa
}
```
Obsłuż akcję w `_handle_chat_command()` w `langgraph_app/service.py`.

---

## 15. Znane ograniczenia i uwagi

- **In-memory sessions** — restart serwera usuwa wszystkie aktywne sesje.
- **Hardcoded API key** — `OPENAI_API_KEY` jest wpisany na stałe w `api4.py` (tak samo jak w api3). Należy przenieść do env przed wdrożeniem produkcyjnym.
- **Brak testów regresji dla api4** — testy w `test_api3_regression.py` i `test_api3_e2e.py` testują api3; api4 wymaga osobnych testów.
- **Model `gpt-4o`** — domyślny w api4, w odróżnieniu od `gpt-5.4` używanego w api3 (może być aliasem wewnętrznym).
- **Brak wsparcia dla `context_shift`** w api4 — pole jest zwracane przez LLM, ale nie jest wykorzystywane do aktualizacji contextu sesji (w odróżnieniu od api3 gdzie dopisywano do `sdialog.Context`).
