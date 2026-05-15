# Algorytm symulacji rozmowy — medchatapi

Dokument opisuje jak działa silnik rozmowy między lekarzem (AI) a przedstawicielem farmaceutycznym (człowiek). Parametry podane są w formacie: `nazwa_anglielska` (polska nazwa).

---

## 1. Konfiguracja sesji (SessionConfig)

Przekazywana przy starcie przez `POST /start`. Wszystkie pola opcjonalne — brak body działa identycznie jak wcześniej.

| Parametr | Polska nazwa | Wartości | Znaczenie |
|---|---|---|---|
| `familiarity` | znajomość | `first_meeting`, `acquainted`, `familiar` | Jak dobrze lekarz zna przedstawiciela — wpływa na startowe zaufanie |
| `register` | rejestr | `professional`, `formal`, `informal` | Stopień formalności języka |
| `warmth` | ciepłość | `neutral`, `warm`, `cool` | Ton emocjonalny (tylko przy `professional`) |
| `rep_name` | imię przedstawiciela | string | Jeśli podane, pojawi się w promptcie lekarza |
| `rep_company` | firma przedstawiciela | string | Opcjonalny kontekst |
| `prior_visits_summary` | podsumowanie poprzednich wizyt | string | Krótki opis historii relacji |

**Walidacja:** `informal + first_meeting` → HTTP 422 (naruszenie etykiety polskiej kultury biznesowej).

---

## 2. Profil lekarza (doctor_profile)

Pobierany z `doctor_archetypes.json` na podstawie `id` przekazanego do `/start`.

| Parametr | Polska nazwa | Znaczenie |
|---|---|---|
| `name` | imię i nazwisko | Nazwa lekarza wyświetlana w prompcie |
| `communication_style` | styl komunikacji | Np. „analityczny", „empatyczny", „dominujący" — wpływa na dyrektywy stylu |
| `gender` | płeć | `female`/`male` — kształtuje oczekiwaną formę grzecznościową |
| `difficulty` | trudność | `easy`, `medium`, `hard` — ogólny bias frustracji i progi terminacji |
| `preferred_strategies` | preferowane strategie | Lista z: `skeptical`, `transactional`, `confrontational`, `exploratory`, `disengaging` |
| `traits` | cechy psychologiczne | Słownik 5 wartości liczbowych (patrz sekcja 3) |

---

## 3. Cechy psychologiczne lekarza (traits)

Wartości zmiennoprzecinkowe w zakresie [0.0, 1.0]. Zmieniają się w trakcie rozmowy pod wpływem zachowania przedstawiciela.

| Cecha | Polska nazwa | Niska wartość | Wysoka wartość |
|---|---|---|---|
| `skepticism` | sceptycyzm | Otwartość na twierdzenia | Żądanie dowodów przy każdym twie. |
| `patience` | cierpliwość | Szybko kończy rozmowę | Toleruje dłuższą wymianę |
| `openness` | otwartość | Zamknięty, wymaga doprecyzowania | Chętnie słucha nowych argumentów |
| `ego` | ego | Luźne oczekiwania | Wymaga profesjonalnego tonu i szacunku |
| `time_pressure` | presja czasu | Ma czas na rozmowę | Pilnie skraca każdą wymianę |

**Jak zmieniają się traits:**
- Marketing buzzwords → `skepticism` ↑, `openness` ↓
- Evidence/badania kliniczne → `time_pressure` ↓, `openness` ↑, `skepticism` ↓
- Bribery → `skepticism = 1.0`, `patience = 0.0`, `openness = 0.0`, `time_pressure = 1.0`
- Fałszywe claimy → `skepticism` ↑, `openness` ↓, `patience` ↓
- Frustracja ≥ 6.0 → `openness` ↓

---

## 4. Stan przekonań lekarza — DoctorConviction (conviction)

5-wymiarowy model wewnętrzny lekarza. To on decyduje o decyzji końcowej, nie stare metryki.

| Wymiar | Polska nazwa | Start (first_meeting) | Start (acquainted) | Start (familiar) |
|---|---|---|---|---|
| `interest_level` | zainteresowanie tematem | ~0.20–0.28 | ~0.20–0.28 | ~0.20–0.28 |
| `trust_in_rep` | zaufanie do osoby | 0.20 | 0.45 | 0.65 |
| `clinical_confidence` | pewność kliniczna | 0.20 | 0.20 | 0.20 |
| `perceived_fit` | dopasowanie do pacjentów | 0.20 | 0.20 | 0.20 |
| `decision_readiness` | gotowość do decyzji | 0.00 | 0.00 | 0.00 |

**Jak zmieniają się wartości conviction w trakcie rozmowy:**

| Zdarzenie | Wpływ |
|---|---|
| Claim potwierdzony | `clinical_confidence` ↑ (malejące zwroty), `decision_readiness` ↑, `perceived_fit` ↑ |
| Claim fałszywy (krytyczny) | `trust_in_rep` −0.20×n, `clinical_confidence` −0.15×n |
| Claim fałszywy (major) | `trust_in_rep` −0.10×n, `clinical_confidence` −0.08×n |
| Evidence (badania) | `clinical_confidence` ↑, `interest_level` ↑ |
| Off-topic | `interest_level` −0.08 (więcej przy `disengaging`/`transactional`) |
| Marketing buzzwords | `trust_in_rep` −0.05×n |
| Anglicyzmy | `trust_in_rep` −0.03×n |
| Bribery | `trust_in_rep = 0.0` |
| Rzetelne zachowanie (bez błędów) | `trust_in_rep` +0.01–0.012 × mnożnik znajomości |
| Pokrycie claimów krytycznych ≥ 50% | `perceived_fit` ↑ |
| Pokrycie claimów krytycznych ≥ 80% | `decision_readiness` ↑ |
| Frustracja ≥ 6.0 | `interest_level` −0.05, `decision_readiness` −0.10 |

**Mnożniki zależne od znajomości (familiarity):**
- `acquainted` → zyski zaufania × 1.15
- `familiar` → zyski zaufania × 1.30, straty × 1.20

**Malejące zwroty dla `clinical_confidence`:**
`cc_factor = max(0.40, 1.0 − clinical_confidence × 0.70)` — trudniej podnieść pewność kliniczną gdy jest już wysoka.

---

## 5. Decyzja lekarza (derive_decision_from_conviction)

Wynika wyłącznie z wartości conviction, nie z subiektywnej oceny:

```
avg_positive = (interest_level + trust_in_rep + clinical_confidence + perceived_fit) / 4

trust_in_rep < 0.25                          → reject
avg_positive ≥ 0.60 AND decision_readiness ≥ 0.65  → will_prescribe
trust_in_rep ≥ 0.70 AND clinical_confidence ≥ 0.65 AND decision_readiness ≥ 0.60 → recommend
avg_positive ≥ 0.50 AND decision_readiness ≥ 0.50  → trial_use
w pozostałych przypadkach                    → undecided
```

---

## 6. Metryki tury (turn_metrics)

Obliczane deterministycznie po każdej wypowiedzi przedstawiciela. Służą jako sygnały dla systemu, nie sterują decyzją lekarza.

| Metryka | Polska nazwa | Co mierzy |
|---|---|---|
| `topic_adherence` | adherencja tematu | Czy wypowiedź dotyczy leku (0.3 base + 0.5 za drug_focus) |
| `clinical_precision` | precyzja kliniczna | Czy brak fałszywych/niepotwierdzonych claimów, penalizuje marketing |
| `ethics` | etyka | 1.0 jeśli brak bribery; kary za niestosowność, brak szacunku, zły zwrot grzecznościowy |
| `language_quality` | jakość języka | Kary za anglicyzmy, buzzwordy, brak szacunku |
| `critical_claim_coverage` | pokrycie claimów krytycznych | Ile z obowiązkowych twierdzeń o leku zostało omówionych |

---

## 7. Frustracja lekarza (frustration_score)

Liczba zmiennoprzecinkowa [0.0, 10.0]. Im wyższa, tym krótsze i bardziej zdystansowane odpowiedzi lekarza.

**Główne składowe delty frustracji:**
- Bribery: +8.0 (natychmiastowe)
- Niestosowna propozycja: +2.2 za każde trafienie
- Brak szacunku: +1.5 za każde trafienie
- Fałszywy claim krytyczny: duże kary (wagowane przez `CLAIM_SEVERITY_WEIGHTS`)
- Off-topic bez fokusa na lek: +1.4
- Marketing buzzwords: +0.45 za każde
- Niepokryte claimy krytyczne: rośnie proporcjonalnie do postępu rozmowy
- Dobra prezentacja: redukcja δ o 0.2–0.5

**Progi:**
- ≥ 6.0 → lekarz staje się bardziej sarkastyczny, dyrektywy stylu skracają wypowiedź do 1 zdania
- ≥ `close_phase_threshold` (domyślnie 9.0) → faza przechodzi do `close`
- ≥ `termination_frustration_threshold` → rozmowa może zostać zakończona

---

## 8. Fazy rozmowy (phase)

State machine z 5 stanami:

```
opening → needs → objection → evidence → close
```

| Faza | Polska nazwa | Warunek przejścia |
|---|---|---|
| `opening` | otwarcie | Wejście domyślne; przejście do needs gdy drug_focus wykryty |
| `needs` | potrzeby | clinical_precision ≥ 0.6 → objection |
| `objection` | obiekcje | clinical_precision ≥ 0.7 AND brak marketingu → evidence |
| `evidence` | dowody | turn_index ≥ max_turns−1 OR frustration ≥ 8.5 → close |
| `close` | zamknięcie | Stan końcowy |

**Natychmiastowe przejście do close:** bribery LUB (frustration ≥ close_phase_threshold LUB turn_index ≥ max_turns).

---

## 9. Tryb tury (TurnMode / current_turn_mode)

Heurystyka bez LLM, wybierana przed wywołaniem modelu. Mówi lekarzowi jak się zachować w tej turze.

| Tryb | Opis |
|---|---|
| `REACT` | Odpowiedz na to co powiedział rozmówca — tryb domyślny |
| `PROBE` | Zadaj pytanie z własnej agendy (gdy cel nie ujawniony) |
| `SHARE` | Opowiedz własne doświadczenie kliniczne (co 4 tury, gdy jest patient_case w agendzie) |
| `CHALLENGE` | Zakwestionuj konkretne twierdzenie (gdy fałszywe claimy lub marketing) |
| `DRIFT` | Wpleć naturalnie wątek z agendy (~10% szans od tury 3, deterministyczne) |
| `CLOSE` | Zmierzaj do zakończenia (frustracja > 7.0 lub limit tur) |

**Hierarchia reguł (hard rules pierwsze):**
1. `frustration > 7.0` lub `turn >= max_turns` → CLOSE
2. Cel nieujawniony → PROBE
3. Fałszywe claimy lub naruszenia → CHALLENGE
4. `turn % 4 == 3` i niezużyty patient_case i `interest_level > 0.4` → SHARE
5. `turn >= 3` i SHA-256 seed < 0.10 → DRIFT
6. Else → REACT

---

## 10. Graf przepływu (LangGraph)

Każda tura to przebieg przez graf węzłów. Kolejność:

```
START
  ↓
node_detect_context       — flagi intent_revealed / drug_revealed
  ↓
node_analyze              — claimy, pokrycie, sygnały etyczne
  ↓
node_policy_check         — hard_stop? dyrektywy policy
  ↓
  ├─ [hard_stop=True] → node_ethics_stop → END
  └─ [hard_stop=False]
       ↓
       node_update_state       — metryki, frustracja, faza, zdarzenia losowe
       ↓
       [turn_index > max_turns] → node_time_stop → END
       [else]
         ↓
         node_build_directives  — kompilacja dyrektyw dla LLM
           ↓
           node_plan_turn        — TurnMode (heurystyka)
             ↓
             node_retrieve_context  — RAG z ChromaDB (opcjonalnie)
               ↓
               node_generate_response  — wywołanie LLM (GPT-4o)
                 ↓
                 node_finalize         — postprocessing, Knowledge Guard, conviction, decyzja
                   ↓
                   END (odpowiedź do klienta)
```

---

## 11. Knowledge Guard

Zabezpieczenie: lekarz nie może cytować danych leku, których przedstawiciel nie podał wcześniej.

**Mechanizm:**
1. Z historii wiadomości (role=`user`) budowana jest znormalizowana baza tekstu wypowiedzi przedstawiciela.
2. Z `drug_info` wyciągane są „wrażliwe frazy" — konkretne liczby i wielowyrazowe sformułowania specyficzne dla leku.
3. Jeśli lekarz użył wrażliwej frazy której nie słyszał od przedstawiciela — zdanie jest przepisywane jako zasłyszana opinia: *„Czytałam gdzieś, że..."*, *„Słyszałam od koleżanki, że..."* itp.
4. Fraza zasłyszana dobierana jest deterministycznie z SHA-256 seed (session_id + turn_index).

---

## 12. Zdarzenia losowe (Random Events)

Co kilka tur może wystąpić zdarzenie zakłócające rozmowę. Deterministyczne — te same dane sesji zawsze dają ten sam wynik.

| Zdarzenie | Bazowe p | Efekt |
|---|---|---|
| `phone_call` (telefon) | 0.11 | patience −0.08, time_pressure +0.12, frustracja +0.35, max_turns −1 |
| `patient_summons` (wezwanie do pacjenta) | 0.08 | patience −0.14, time_pressure +0.20, frustracja +0.60, max_turns −1 |
| `waiting_room_surge` (wzrost kolejki) | 0.14 | patience −0.10, time_pressure +0.15, frustracja +0.45, max_turns −1 |

**Ograniczenia:** nie przed turą 2, nie częściej niż co 2 tury.

---

## 13. Limit tur (max_turns)

Ustalany przy starcie na podstawie cech lekarza:

```
time_pressure ≥ 0.90 → base = 12
time_pressure ≥ 0.75 → base = 14
time_pressure ≥ 0.55 → base = 16
patience ≥ 0.80 AND time_pressure ≤ 0.40 → base = 20
else → base = 18

max_turns = clamp(base + difficulty_adjust, 10, 20)
```

Zdarzenia losowe mogą skracać `max_turns` o 1 (do minimum max(3, turn_index+1)).

---

## 14. Ocena celu (evaluate_conversation_goal)

Logger metryk — nie steruje decyzją lekarza (to robi conviction). Oblicza score [0–100] i status:

- `achieved` — wszystkie warunki spełnione i score ≥ 75
- `partial` — brak odrzucenia, etyka OK, score ≥ 55
- `not_achieved` — w pozostałych przypadkach

**Warunki `achieved`:** lek przedstawiony + decyzja pozytywna + etyka OK + brak fałszywych claimów + pokrycie ≥ 80% + evidence-first OK + lekarz usatysfakcjonowany.

---

## 15. Terminacja rozmowy

Rozmowa kończy się gdy:

- **Natychmiast (bez okna ochronnego):** bribery wykryte → `node_ethics_stop`
- **Po turze 5 i wyżej:**
  - `patience ≤ 0.10`
  - `time_pressure ≥ 0.80`
  - `frustration_score ≥ termination_frustration_threshold`
  - Niestosowność/brak szacunku AND `frustration ≥ 7.0`
- **Przekroczenie `max_turns`:** `node_time_stop`
- **Cel osiągnięty:** conviction decyduje `will_prescribe`/`recommend` + `goal.achieved = True`

**Okno ochronne:** brak terminacji przed turą 5 (z wyjątkiem bribery) — daje szansę na naprawę błędów.
