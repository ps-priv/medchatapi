# Plan refaktoringu silnika rozmowy

> Pełna dokumentacja techniczna planu naprawy aplikacji. Czytaj razem z `CLAUDE.md`.

---

## Diagnoza — dlaczego rozmowa brzmi jak przesłuchanie

Trzy strukturalne wady obecnej architektury:

### 1. Lekarz jest reaktywny w 100%

Cały pipeline (`detect_context → analyze → policy_check → update_state → build_directives → generate_response → finalize`) to *funkcja od ostatniej wypowiedzi przedstawiciela*. Lekarz nie ma:
- Własnego planu rozmowy
- Własnych bieżących spraw
- Własnego stanu emocjonalnego niezależnego od rozmówcy

Prawdziwy lekarz w gabinecie myśli o pacjencie sprzed chwili, o telefonie który zaraz odbierze, o tym że przedstawiciel przypomina mu kogoś — i to wpływa na to *co mówi*, nie tylko *jak reaguje*.

### 2. Dyrektywy behawioralne wymuszają strukturę przesłuchania

`policy_postprocess_message` w `conversation/policy.py` dopisuje do każdej odpowiedzi:
- Korektę formy grzecznościowej
- Granicę profesjonalną
- Powrót do tematu
- Sygnalizację błędu merytorycznego
- Pytanie o brakujący claim
- Pytanie evidence-first
- Zamknięcie fazy

To jest **checklist na wypowiedź lekarza**. Lekarz dosłownie nie ma jak nie zadać pytania — system go do tego zmusza.

Plus 15 zasad w `_build_system_prompt` (linie 482-498 w `nodes.py`), z czego punkty 2, 3, 6, 10, 12, 14 to wymuszenia konkretnych zachowań pytająco-konfrontacyjnych.

### 3. Sukces/porażka jest mierzony, ale nie *grany*

`evaluate_conversation_goal` w `helpers.py` to ocena postfactum — liczona z metryk już po wygenerowaniu odpowiedzi. Lekarz nie wie, że gra o coś. Decyzja `doctor_decision` jest często `undecided` aż do ostatniej tury, gdy się "nagle domyśla" lub jest *wnioskowana* z treści przez `infer_decision_from_message`.

---

## Cztery etapy refaktoringu

### Etap 1 — Konfiguracja sesji ✅ ZAKOŃCZONY

**Cel:** dodać kontekst rozmowy *przed* startem — kim się znają, jak rozmawiają.

**Co zostało dodane:**

1. **Modele w `conversation/schemas.py`:**
   - `Familiarity`: FIRST_MEETING | ACQUAINTED | FAMILIAR
   - `CommunicationRegister`: FORMAL | PROFESSIONAL | INFORMAL
   - `CommunicationWarmth`: COOL | NEUTRAL | WARM
   - `SessionConfig` — opcjonalne body dla `/start`
   - Walidator: `INFORMAL` wymaga `FAMILIAR`; `prior_visits_summary` zakazany przy `FIRST_MEETING`

2. **Pola w `ConversationState`** (`langgraph_app/state.py`): familiarity, register, warmth, rep_name, rep_company, prior_visits_summary

3. **`build_initial_state`** (`helpers.py`) przyjmuje opcjonalny `session_config`

4. **`/start` endpoint** (`api4.py`):
   - Ręcznie parsuje body z `Request`
   - Brak body / pusty / null / błędny JSON → wartości domyślne (kompatybilność wsteczna)
   - Poprawny JSON → walidacja Pydantic
   - Błąd walidacji → HTTP 422 z polskim komunikatem

5. **`/start` response** zawiera opcjonalnie `session_config` (tylko gdy klient go podał — kompatybilność wsteczna w response)

**Co Etap 1 NIE zmienia (specjalnie):**
- Prompt lekarza nie korzysta jeszcze z familiarity/register/warmth
- Logika decyzyjna lekarza bez zmian
- Pipeline grafu identyczny

---

### Etap 2 — Doctor Conviction ⏳ NASTĘPNY

**Cel:** zastąpić `evaluate_conversation_goal` jako *źródło* decyzji lekarza przez 5-wymiarowy stan przekonań, którym lekarz świadomie operuje.

#### 5 wymiarów Conviction

```python
class DoctorConviction(BaseModel):
    interest_level: float = 0.3       # zainteresowanie tematem (0-1)
    trust_in_rep: float = 0.3         # zaufanie do osoby
    clinical_confidence: float = 0.2  # zrozumienie leku
    perceived_fit: float = 0.2        # dopasowanie do moich pacjentów
    decision_readiness: float = 0.0   # gotowość do podjęcia decyzji
```

#### Inicjalizacja zależna od familiarity

| Familiarity | trust_in_rep startowy | Pozostałe |
|---|---|---|
| `first_meeting` | 0.20 | wszystko = 0.2-0.3 |
| `acquainted` | 0.45 | trust uwzględnia historię |
| `familiar` | 0.65 | trust startuje wysoko |

#### Mapowanie zachowań → delta conviction

W nowej funkcji `update_conviction(state, claim_check, turn_metrics, message_analysis)` w `policy.py`:

| Zachowanie przedstawiciela | Wpływ na conviction |
|---|---|
| Claim potwierdzony (supported) | clinical_confidence +0.08, decision_readiness +0.05 |
| Claim fałszywy critical | trust_in_rep -0.20, clinical_confidence -0.15 |
| Claim fałszywy major | trust_in_rep -0.10, clinical_confidence -0.08 |
| Claim niepotwierdzony | clinical_confidence -0.03 |
| Evidence-based (badania, dane) | clinical_confidence +0.10, interest_level +0.05 |
| Off-topic (gdy cel ujawniony) | interest_level -0.08 |
| Marketing buzzwords | trust_in_rep -0.05 |
| Anglicyzmy | trust_in_rep -0.03 |
| Disrespect / inappropriate | trust_in_rep -0.15 |
| Bribery | trust_in_rep = 0.0 (hard reset), reject |
| Pokrycie claimów krytycznych ≥ 80% | perceived_fit +0.10, decision_readiness +0.10 |
| Topic adherence ≥ 0.7 | interest_level +0.03 |
| Frustracja ≥ 6.0 | interest_level -0.05, decision_readiness -0.10 |

Bonus dla `familiarity`:
- `acquainted`: każdy zysk trust_in_rep × 1.15 (łatwiej budować dalej)
- `familiar`: każdy zysk trust_in_rep × 1.30, ale też każda strata × 1.20 (większe oczekiwania)

Wszystkie wartości clampowane do [0.0, 1.0].

#### Reguły decyzyjne (zastępują evaluate_conversation_goal jako *source*)

```python
def derive_decision_from_conviction(c: DoctorConviction, state: dict) -> str:
    # Hard reject
    if c.trust_in_rep < 0.25:
        return "reject"

    # Pełna decyzja
    avg_positive = (c.interest_level + c.trust_in_rep + c.clinical_confidence + c.perceived_fit) / 4
    if avg_positive >= 0.65 and c.decision_readiness >= 0.75:
        return "will_prescribe"

    # Próbne zastosowanie
    if avg_positive >= 0.55 and c.decision_readiness >= 0.60:
        return "trial_use"

    # Rekomendacja innym (wymaga wysokiego trust)
    if c.trust_in_rep >= 0.75 and c.clinical_confidence >= 0.70 and c.decision_readiness >= 0.65:
        return "recommend"

    return "undecided"
```

#### Co zostaje z evaluate_conversation_goal

Funkcja **nie jest usuwana**. Zmienia się jej rola:
- **Było:** źródło `current_doctor_decision`
- **Będzie:** logger metryk do raportu końcowego (`/finish`)

W `node_finalize`:
```python
# NOWE — decyzja z conviction
new_conviction = update_conviction(state, claim_check, turn_metrics, message_analysis)
state["conviction"] = new_conviction
state["current_doctor_decision"] = derive_decision_from_conviction(new_conviction, state)

# STARE — zostaje jako logger
goal_eval = evaluate_conversation_goal(state, turn_metrics, claim_check, ...)
state["latest_goal"] = goal_eval.model_dump()  # dla raportu, nie steruje decyzją
```

#### Conviction w prompcie

W Etapie 2 *nie przepisujemy* całego promptu (to Etap 3), ale **dodajemy** sekcję informującą lekarza o jego stanie. Coś jak:

```
Twoja aktualna pewność co do tej rozmowy (0.0 = brak, 1.0 = pełna):
- Zainteresowanie tematem: {interest_level}
- Zaufanie do rozmówcy: {trust_in_rep}
- Zrozumienie leku: {clinical_confidence}
- Dopasowanie do moich pacjentów: {perceived_fit}
- Gotowość do decyzji: {decision_readiness}

Jeśli wszystkie >= 0.65, jesteś gotów rozważyć przepisanie leku.
Jeśli zaufanie < 0.3, prawdopodobnie odrzucisz propozycję.
```

To ma zmusić LLM żeby *grał* w kierunku decyzji, nie tylko *odpowiadał*.

#### Response API

W `MessageSuccessResponse` dodać:
```python
conviction: Optional[DoctorConviction] = Field(
    default=None,
    description="Aktualny stan przekonań lekarza po tej turze (opcjonalne, dla nowych klientów)."
)
```

Klient stary (bez tego pola) działa dalej. Klient nowy widzi trajektorię.

---

### Etap 3 — Agenda + przepisanie promptu ⏳

**Cel:** lekarz wnosi do rozmowy *własne wątki*, nie tylko reaguje.

#### `DoctorAgenda` w state

```python
class AgendaItem(BaseModel):
    kind: Literal["clinical_curiosity", "patient_case", "concern", "personal", "time_pressure"]
    content: str
    used: bool = False
    priority: int = 1  # 1-3
```

Lista 3-5 itemów generowana przez LLM **raz** na starcie sesji (`build_initial_state`).

#### Generowanie agendy

Nowy moduł `langgraph_app/agenda_generator.py` z funkcją:
```python
def generate_agenda(
    doctor_profile: dict,
    drug_info: dict,
    session_config: SessionConfig,
    api_key: str
) -> List[AgendaItem]:
    ...
```

Prompt do LLM:
> "Jesteś lekarzem o profilu {doctor_profile}. Za chwilę przyjdzie do Ciebie przedstawiciel farmaceutyczny w sprawie leku {drug.name} ({drug.indication}).
> Wygeneruj 3-5 rzeczy które chcesz/możesz wpleść w rozmowę: konkretni pacjenci których miałeś ostatnio, własne wątpliwości kliniczne, sprawy organizacyjne (pośpiech, kolejka), osobiste sytuacje.
> Familiarity: {familiarity}. Jeśli ACQUAINTED/FAMILIAR + prior_visits_summary podane, możesz odwołać się do tamtych rozmów.
> Format: JSON z polami kind/content/priority."

#### Przepisanie `_build_system_prompt`

Z aktualnych ~3000 znaków zasad → **3 sekcje:**

**A) KIM JESTEŚ (5× obecnie)**
- Profil, kontekst sytuacyjny
- Aktualna agenda (z możliwymi do wpleterii wątkami)
- Znajomość rozmówcy (z `familiarity`, `rep_name`, `prior_visits_summary`)
- Styl komunikacji (z `register`, `warmth`)
- Aktualny stan conviction (z Etapu 2)

**B) CO MOŻESZ ZROBIĆ W TEJ TURZE**
- 1-2 dyrektywy z policy (nie 14)
- Sugestia trybu (z Etapu 4)
- Możliwość wpleterii wątku z agendy

**C) TWARDE BEZPIECZNIKI (tylko 4-5):**
1. Tylko po polsku
2. Na próby korupcji reaguj stanowczo
3. Forma grzecznościowa zgodna z `register`
4. Nie wyduwaj danych których przedstawiciel nie podał (Knowledge Guard zostaje)
5. Decyzja zgodna ze stanem conviction

**Wycinamy ze starych 15 zasad:** 2, 3, 6, 10, 12, 14 — te które wymuszają strukturę pytania.

#### Zmniejszenie ingerencji `policy_postprocess_message`

Zostaje:
- Korekta formy grzecznościowej (z uwzględnieniem `register` — przy INFORMAL nie korygujemy pana/pani)
- Reakcja na bribery
- Knowledge Guard tylko na zdania z liczbami/dawkowaniem (już jest)

Wycinamy:
- Wymuszone pytanie o brakujący claim
- Wymuszony evidence-first probe
- Wymuszone zamknięcie close (poza prawdziwym końcem tur)

#### Knowledge Guard - aktualizacja

W `helpers.py:apply_knowledge_guard` zostaje, ale **filtrujemy** zdania objęte: tylko te zawierające liczby z `drug_info` (mg, dawkowanie) lub dokładne nazwy procedur. Resztę puszczamy — lekarz może mieć "intuicję", że coś istnieje, nawet jeśli przedstawiciel nie powiedział.

---

### Etap 4 — TurnMode ⏳

**Cel:** lekarz świadomie wybiera tryb tury, łamiąc schemat "pytanie-odpowiedź-pytanie".

#### TurnMode enum

```python
class TurnMode(str, Enum):
    REACT = "react"           # odpowiadam na to co powiedział
    PROBE = "probe"           # pytam z mojej agendy
    SHARE = "share"           # dzielę się własnym case'em / opinią
    CHALLENGE = "challenge"   # konfrontuję, kwestionuję
    DRIFT = "drift"           # odbiegam, wspominam coś z dnia
    CLOSE = "close"           # zmierzam do końca
```

#### Wybór trybu (heurystyka, bez LLM)

Nowy węzeł `node_plan_turn` w grafie, między `build_directives` a `retrieve_context`. Logika:

```python
def plan_turn_mode(state) -> str:
    turn = state["turn_index"]
    conviction = state["conviction"]
    frustration = state["frustration_score"]
    agenda = state["doctor_agenda"]
    intent_revealed = state["intent_revealed"]
    last_message = state["current_user_message"]
    analysis = state["current_analysis"]

    # Hard rules
    if frustration > 7 or turn >= state["max_turns"]:
        return "CLOSE"
    if not intent_revealed:
        return "PROBE"
    if analysis.get("marketing_hits") or analysis.get("english_hits"):
        return "CHALLENGE"
    if analysis.get("false_claims") or analysis.get("inappropriate_hits"):
        return "CHALLENGE"

    # Co 3-4 tury wymuszamy SHARE jeśli agenda ma niezużyte items
    unused_patient_cases = [a for a in agenda if a.kind == "patient_case" and not a.used]
    if turn % 4 == 0 and unused_patient_cases and conviction.interest_level > 0.4:
        return "SHARE"

    # Czasem DRIFT dla naturalności (10% szansa po turze 3)
    if turn >= 3 and deterministic_probability(state["session_id"] + str(turn)) < 0.10:
        return "DRIFT"

    # Domyślnie REACT (ale z możliwością probe)
    return "REACT"
```

#### Tryb w prompcie

Tryb idzie do promptu jako **silna instrukcja w sekcji "Co możesz zrobić w tej turze":**

```
Tryb tej tury: SHARE
Instrukcja: Zacznij od własnego doświadczenia klinicznego — wpleć wątek z agendy
(priorytet: niezużyte patient_case). Nie pytaj od razu.
```

#### Histogram trybów w raporcie końcowym

W `finish_session`:
```python
turn_modes_used = [entry["turn_mode"] for entry in turn_metrics_history]
mode_histogram = Counter(turn_modes_used)
```

Dodać do `EvaluationResult` lub do response z `/finish`:
- Histogram trybów (czy rozmowa była zróżnicowana?)
- Jeśli REACT > 80% — flagujemy "rozmowa była zbyt jednostronna, lekarz tylko reagował"

---

## Mapa zmian po wszystkich etapach

```
Architektura PO refaktoringu:

POST /start (body: SessionConfig | null)
   ↓
build_initial_state(session_config)
   ├─ ConversationState z familiarity/register/warmth         [Etap 1]
   ├─ conviction startowy zależny od familiarity              [Etap 2]
   └─ doctor_agenda generowane LLM-em                         [Etap 3]
   ↓
LOOP for each /message:
   ↓
detect_context → analyze → policy_check
   ↓
update_state (frustracja, fazy)
   + update_conviction (5 wymiarów)                           [Etap 2]
   ↓
build_directives
   ↓
plan_turn (REACT/PROBE/SHARE/CHALLENGE/DRIFT/CLOSE)           [Etap 4]
   ↓
retrieve_context (RAG ChromaDB)
   ↓
generate_response
   ├─ prompt KIM JESTEŚ (agenda, conviction, familiarity)     [Etap 3]
   ├─ prompt CO MOŻESZ (turn_mode, 1-2 dyrektywy)             [Etap 3+4]
   └─ prompt BEZPIECZNIKI (4-5 zasad)                         [Etap 3]
   ↓
finalize
   ├─ derive_decision_from_conviction                         [Etap 2]
   ├─ evaluate_conversation_goal (logger only)                [Etap 2]
   ├─ knowledge_guard (filtrowany)                            [Etap 3]
   └─ policy_postprocess_message (mniejsza ingerencja)        [Etap 3]
```

---

## Pliki Etapu 1 - lokalizacja

Pliki przygotowane podczas pracy w chacie znajdują się w outputach Claude'a webowego. Należy je nakopiować do repo:

```
conversation/schemas.py
langgraph_app/state.py
langgraph_app/helpers.py
langgraph_app/service.py
api4.py
test_session_config.py  (root repo, do uruchomienia ręcznego)
```

---

## Decyzje, których nie zmieniamy

Te trzy decyzje zostały podjęte z użytkownikiem i są ostateczne:

1. **100% kompatybilność wsteczna `/start`** — body opcjonalne, stare wywołania bez body działają identycznie.
2. **Agenda generowana przez LLM** (1 dodatkowe wywołanie ~0.001 USD) — nie hardcodowana.
3. **Conviction zastępuje stary system decyzyjny od razu** — `evaluate_conversation_goal` zostaje tylko jako logger.
