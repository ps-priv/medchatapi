from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Konfiguracja sesji rozmowy (opcjonalna, dodana w Etapie 1)
# ---------------------------------------------------------------------------


class Familiarity(str, Enum):
    """Poziom znajomości między lekarzem a przedstawicielem."""

    FIRST_MEETING = "first_meeting"   # pierwsza wizyta, lekarz nie zna rozmówcy
    ACQUAINTED = "acquainted"         # znają się z poprzednich wizyt (relacja zawodowa)
    FAMILIAR = "familiar"             # dobrze się znają, swobodna relacja


class CommunicationRegister(str, Enum):
    """Rejestr (formalność) komunikacji."""

    FORMAL = "formal"                 # bardzo oficjalny, sztywny
    PROFESSIONAL = "professional"     # Pan/Pani z ciepłem (domyślne polskie biznesowe)
    INFORMAL = "informal"             # na "ty" - dopuszczalne tylko przy FAMILIAR


class CommunicationWarmth(str, Enum):
    """Wymiar ciepła komunikacji (niezależny od rejestru)."""

    COOL = "cool"                     # rzeczowy dystans
    NEUTRAL = "neutral"               # zawodowo, bez ekstra ciepła
    WARM = "warm"                     # serdeczny, ciepły ton


class SessionConfig(BaseModel):
    """Opcjonalna konfiguracja sesji przekazywana w body POST /start.

    Brak konfiguracji = wartości domyślne (pełna kompatybilność wsteczna ze starszymi
    klientami, którzy wołają /start bez body).
    """

    familiarity: Familiarity = Field(
        default=Familiarity.FIRST_MEETING,
        description="Poziom znajomości lekarza z przedstawicielem przed rozmową.",
    )
    register: CommunicationRegister = Field(
        default=CommunicationRegister.PROFESSIONAL,
        description="Formalność komunikacji. INFORMAL dopuszczalny tylko z FAMILIAR.",
    )
    warmth: CommunicationWarmth = Field(
        default=CommunicationWarmth.NEUTRAL,
        description="Ciepło komunikacji - niezależne od rejestru.",
    )
    rep_name: Optional[str] = Field(
        default=None,
        description="Imię/imię i nazwisko przedstawiciela. Wymagane przy familiarity != first_meeting.",
    )
    rep_company: Optional[str] = Field(
        default=None,
        description="Firma reprezentowana przez przedstawiciela.",
    )
    prior_visits_summary: Optional[str] = Field(
        default=None,
        description=(
            "Krótki opis poprzednich wizyt/relacji. Używane tylko przy "
            "familiarity in {acquainted, familiar}. Max 500 znaków."
        ),
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "SessionConfig":
        # Reguła 1: INFORMAL wymaga FAMILIAR
        if (
            self.register == CommunicationRegister.INFORMAL
            and self.familiarity != Familiarity.FAMILIAR
        ):
            raise ValueError(
                "Komunikacja nieformalna (na 'ty') jest możliwa tylko gdy "
                "familiarity = 'familiar'. Przy pierwszym lub okazjonalnym kontakcie "
                "lekarz oczekuje formy 'Pan/Pani'."
            )
        # Reguła 2: prior_visits_summary tylko gdy się znają
        if (
            self.prior_visits_summary
            and self.familiarity == Familiarity.FIRST_MEETING
        ):
            raise ValueError(
                "prior_visits_summary nie ma sensu przy first_meeting "
                "(lekarz nie pamięta wcześniejszych wizyt z tym przedstawicielem)."
            )
        # Reguła 3: rep_name dla relacji ciągłej (ostrzeżenie miękkie - dopuszczamy brak, ale logujemy)
        # Nie raise - tylko walidator może to zalogować, jeśli zostanie pusty
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "familiarity": "acquainted",
                "register": "professional",
                "warmth": "warm",
                "rep_name": "Anna Kowalska",
                "rep_company": "PharmaCo",
                "prior_visits_summary": "Trzecia wizyta. Poprzednio omawiali profil bezpieczeństwa leku u pacjentów >65 r.ż.",
            }
        }
    )


class AgendaItem(BaseModel):
    """Jeden wątek własny lekarza do naturalnego wplecenia w rozmowę (Etap 3)."""

    kind: Literal["clinical_curiosity", "patient_case", "concern", "personal", "time_pressure"]
    content: str = Field(description="Treść wątku w pierwszej osobie, 1-2 zdania po polsku.")
    used: bool = Field(default=False, description="Czy wątek został już użyty w rozmowie.")
    priority: int = Field(default=2, ge=1, le=3, description="Priorytet: 1=niski, 2=średni, 3=ważny.")


class DoctorConviction(BaseModel):
    """5-wymiarowy stan przekonań lekarza wobec leku i przedstawiciela (Etap 2)."""

    interest_level: float = Field(default=0.3, ge=0.0, le=1.0, description="Zainteresowanie tematem (0-1).")
    trust_in_rep: float = Field(default=0.3, ge=0.0, le=1.0, description="Zaufanie do przedstawiciela (0-1).")
    clinical_confidence: float = Field(default=0.2, ge=0.0, le=1.0, description="Zrozumienie i pewność co do leku (0-1).")
    perceived_fit: float = Field(default=0.2, ge=0.0, le=1.0, description="Dopasowanie leku do swoich pacjentów (0-1).")
    decision_readiness: float = Field(default=0.0, ge=0.0, le=1.0, description="Gotowość do podjęcia decyzji (0-1).")


class TraitsUpdate(BaseModel):
    """Aktualny stan cech psychologicznych lekarza."""

    skepticism: float = Field(description="Poziom sceptycyzmu (0.0 - 1.0)")
    patience: float = Field(description="Poziom cierpliwosci (0.0 - 1.0)")
    openness: float = Field(description="Otwartosc (0.0 - 1.0)")
    ego: float = Field(description="Ego (0.0 - 1.0)")
    time_pressure: float = Field(description="Presja czasu/Zniecierpliwienie (0.0 - 1.0)")


DoctorAttitude = Literal["happy", "neutral", "serious", "sad"]
DoctorDecision = Literal["undecided", "trial_use", "will_prescribe", "recommend", "reject"]
GoalStatus = Literal["achieved", "partial", "not_achieved"]


class DoctorResponse(BaseModel):
    """Ustrukturyzowana odpowiedz lekarza zwracana przez LLM."""

    doctor_message: str = Field(description="Odpowiedz lekarza.")
    updated_traits: TraitsUpdate = Field(description="Nowe parametry psychologiczne.")
    reasoning: str = Field(description="Uzasadnienie reakcji.")
    detected_errors: List[str] = Field(
        description="Lista bledow merytorycznych popelnionych przez przedstawiciela."
    )
    context_shift: Optional[str] = Field(
        description="Krotki opis trwalej zmiany atmosfery/kontekstu."
    )
    doctor_attitude: DoctorAttitude = Field(description="Szacowane nastawienie lekarza.")
    doctor_decision: DoctorDecision = Field(
        default="undecided",
        description="Decyzja lekarza po tej turze: undecided/trial_use/will_prescribe/recommend/reject.",
    )


class ConversationGoal(BaseModel):
    """Wynik silnika celu rozmowy (czy przedstawiciel osiagnal cel biznesowy)."""

    achieved: bool = Field(description="Czy cel rozmowy zostal osiagniety.")
    status: GoalStatus = Field(description="Status celu: achieved/partial/not_achieved.")
    score: int = Field(description="Wynik realizacji celu 0-100.")
    doctor_decision: DoctorDecision = Field(description="Decyzja lekarza na koniec tury/sesji.")
    doctor_satisfied: bool = Field(description="Czy lekarz jest ogolnie zadowolony z rozmowy.")
    reasons: List[str] = Field(description="Powody, dla ktorych cel uznano za osiagniety/czesciowy.")
    missing: List[str] = Field(description="Brakujace warunki do pelnego osiagniecia celu.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "achieved": False,
                "status": "partial",
                "score": 67,
                "doctor_decision": "trial_use",
                "doctor_satisfied": True,
                "reasons": ["Pokryto kluczowe claimy bez bledow krytycznych."],
                "missing": ["Brakuje jasnej decyzji will_prescribe/recommend."],
            }
        }
    )


class MessageRequest(BaseModel):
    """Payload wejscia dla endpointu /message."""

    session_id: str
    message: str


class TranscribeRequest(BaseModel):
    """Payload wejscia endpointu /transcribe (audio base64 z aplikacji webowej)."""

    audio_base64: str = Field(description="Zawartosc audio zakodowana jako base64 (bez data URL header).")
    filename: str = Field(default="recording.webm", description="Nazwa pliku audio, np. recording.webm.")
    mime_type: str = Field(default="audio/webm", description="Typ MIME audio, np. audio/webm lub audio/wav.")
    prompt: Optional[str] = Field(
        default=None,
        description="Opcjonalny prompt wspierajacy STT (np. slownictwo medyczne).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
                "filename": "representative.webm",
                "mime_type": "audio/webm",
                "prompt": "Rozmowa medyczna po polsku, nazwy lekow i dawkowania.",
            }
        }
    )


class TranscribeResponse(BaseModel):
    """Kontrakt odpowiedzi endpointu /transcribe."""

    text: str = Field(description="Rozpoznany tekst wypowiedzi przedstawiciela.")
    language: str = Field(description="Jezyk transkrypcji (dla tego endpointu: pl).")
    model: str = Field(description="Model STT wykorzystany do transkrypcji.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Dzien dobry Pani Doktor, chcialbym omowic profil bezpieczenstwa leku.",
                "language": "pl",
                "model": "gpt-4o-mini-transcribe",
            }
        }
    )


class EvaluationResult(BaseModel):
    """Ocena koncowa rozmowy przygotowana przez ewaluatora LLM."""

    professionalism_score: int = Field(description="Ocena profesjonalizmu 1-10.")
    relevance_score: int = Field(description="Ocena trafnosci argumentow 1-10.")
    relationship_score: int = Field(description="Ocena budowania relacji 1-10.")
    strengths: List[str] = Field(description="2-3 mocne strony przedstawiciela.")
    areas_for_improvement: List[str] = Field(
        description="2-3 obszary do poprawy przedstawiciela."
    )
    final_feedback: str = Field(description="Krotkie podsumowanie i glowna porada.")


class StartConversationResponse(BaseModel):
    """Kontrakt odpowiedzi endpointu /start."""

    session_id: str = Field(description="Identyfikator sesji rozmowy.")
    status: str = Field(description="Status uruchomienia sesji.")
    traits: TraitsUpdate = Field(description="Poczatkowe cechy psychologiczne lekarza.")
    session_config: Optional[SessionConfig] = Field(
        default=None,
        description=(
            "Konfiguracja sesji zastosowana do rozmowy. Pole opcjonalne dla "
            "zachowania pełnej kompatybilności wstecznej ze starszymi klientami."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "b1f1d07b-6a5f-4f64-bac5-ef3f0c9e7cbf",
                "status": "Rozmowa rozpoczęta. Lekarz gotowy i czeka na cel wizyty.",
                "traits": {
                    "skepticism": 0.6,
                    "patience": 0.5,
                    "openness": 0.5,
                    "ego": 0.4,
                    "time_pressure": 0.5,
                },
            }
        }
    )


class TurnMetricsResponse(BaseModel):
    """Metryki jakości rozmowy dla bieżącej tury."""

    topic_adherence: float = Field(description="Zgodność wypowiedzi przedstawiciela z tematem leku (0.0-1.0).")
    clinical_precision: float = Field(description="Precyzja merytoryczno-kliniczna wypowiedzi (0.0-1.0).")
    ethics: float = Field(description="Ocena etyczna i profesjonalna wypowiedzi (0.0-1.0).")
    language_quality: float = Field(description="Jakość języka (precyzja, brak żargonu) (0.0-1.0).")
    critical_claim_coverage: float = Field(description="Pokrycie claimów krytycznych (0.0-1.0).")
    frustration: float = Field(description="Aktualny poziom frustracji lekarza po tej turze (0.0-10.0).")


class MessageSuccessResponse(BaseModel):
    """Kontrakt sukcesu endpointu /message."""

    doctor_message: str = Field(description="Wypowiedz lekarza po analizie tury.")
    updated_traits: TraitsUpdate = Field(description="Zaktualizowane cechy psychologiczne lekarza.")
    reasoning: str = Field(description="Wyjasnienie reakcji lekarza i algorytmu.")
    turn_metrics: TurnMetricsResponse = Field(
        description="Ustrukturyzowane metryki jakości aktualnej tury rozmowy."
    )
    doctor_attitude: DoctorAttitude = Field(description="Nastawienie lekarza po tej turze.")
    doctor_decision: DoctorDecision = Field(description="Decyzja lekarza po tej turze.")
    conversation_goal: ConversationGoal = Field(description="Biezacy status realizacji celu rozmowy.")
    is_terminated: bool = Field(description="Czy rozmowa zostala zakonczona po tej turze.")
    termination_reason: Optional[str] = Field(
        default=None,
        description="Powod zakonczenia rozmowy, jesli is_terminated=true.",
    )
    conviction: Optional[DoctorConviction] = Field(
        default=None,
        description="Aktualny stan przekonań lekarza po tej turze (opcjonalne, dla nowych klientów API).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doctor_message": "Dziekuje za konkrety. Wroce do tego leku u wybranych pacjentow.",
                "updated_traits": {
                    "skepticism": 0.48,
                    "patience": 0.52,
                    "openness": 0.62,
                    "ego": 0.4,
                    "time_pressure": 0.58,
                },
                "reasoning": "Pokryto kluczowe kwestie i utrzymano standard evidence-first.",
                "turn_metrics": {
                    "topic_adherence": 0.8,
                    "clinical_precision": 0.76,
                    "ethics": 1.0,
                    "language_quality": 0.9,
                    "critical_claim_coverage": 0.67,
                    "frustration": 2.15,
                },
                "doctor_attitude": "neutral",
                "doctor_decision": "trial_use",
                "conversation_goal": {
                    "achieved": False,
                    "status": "partial",
                    "score": 71,
                    "doctor_decision": "trial_use",
                    "doctor_satisfied": True,
                    "reasons": ["Poprawne omowienie claimow krytycznych."],
                    "missing": ["Brakuje finalnej deklaracji prescribe/recommend."],
                },
                "is_terminated": False,
                "termination_reason": None,
            }
        }
    )


class MessageErrorResponse(BaseModel):
    """Kontrakt bledu biznesowego endpointu /message (sesja juz zakonczona)."""

    error: str = Field(description="Opis bledu dla klienta API.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Rozmowa została już zakończona przez lekarza. Wywołaj /finish, aby uzyskać ocenę."
            }
        }
    )


class ConversationTurn(BaseModel):
    """Pojedyncza pozycja historii rozmowy."""

    speaker: str = Field(description="Rola/nadawca wypowiedzi.")
    text: str = Field(description="Tresc wypowiedzi.")


class FinishConversationResponse(BaseModel):
    """Kontrakt odpowiedzi endpointu /finish."""

    status: str = Field(description="Status finalizacji i usuniecia sesji.")
    conversation_history: List[ConversationTurn] = Field(description="Pelna historia rozmowy.")
    conversation_goal: ConversationGoal = Field(description="Koncowy status realizacji celu rozmowy.")
    evaluation: EvaluationResult = Field(description="Ocena koncowa przygotowana przez ewaluator LLM.")
    turn_mode_histogram: Optional[Dict[str, int]] = Field(
        default=None,
        description="Histogram trybów tur (REACT/PROBE/SHARE/CHALLENGE/DRIFT/CLOSE) — Etap 4.",
    )
    turn_mode_flag: Optional[str] = Field(
        default=None,
        description="Flaga jeśli rozmowa była zbyt jednostronna (>80% REACT przy ≥4 turach).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "Rozmowa zakończona, sesja usunięta.",
                "conversation_history": [
                    {"speaker": "Przedstawiciel", "text": "Dzien dobry, mam konkretne dane o leku."},
                    {"speaker": "Lekarz", "text": "Prosze podac populacje i wyniki kliniczne."},
                ],
                "conversation_goal": {
                    "achieved": True,
                    "status": "achieved",
                    "score": 84,
                    "doctor_decision": "will_prescribe",
                    "doctor_satisfied": True,
                    "reasons": ["Lekarz zadeklarowal pozytywna decyzje o leku."],
                    "missing": [],
                },
                "evaluation": {
                    "professionalism_score": 8,
                    "relevance_score": 8,
                    "relationship_score": 7,
                    "strengths": ["Merytoryczne dane", "Dobra reakcja na obiekcje"],
                    "areas_for_improvement": ["Krotsze otwarcie"],
                    "final_feedback": "Utrzymuj evidence-first od pierwszej tury.",
                },
            }
        }
    )


class RateConversationRequest(BaseModel):
    """Payload wejściowy endpointu /rate."""

    session_id: str = Field(description="Identyfikator zakończonej sesji rozmowy.")
    rating: int = Field(ge=0, le=5, description="Ocena rozmowy w skali 0-5.")
    description: Optional[str] = Field(default=None, description="Opcjonalny komentarz przedstawiciela.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "b1f1d07b-6a5f-4f64-bac5-ef3f0c9e7cbf",
                "rating": 4,
                "description": "Dobra symulacja, lekarz był wymagający ale realistyczny.",
            }
        }
    )


class RateConversationResponse(BaseModel):
    """Kontrakt odpowiedzi endpointu /rate."""

    status: str = Field(description="Potwierdzenie zapisu oceny.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "Ocena zapisana."}}
    )


class TTSRequest(BaseModel):
    """Payload wejściowy endpointu /tts."""

    text: str = Field(description="Tekst do syntezy mowy (wypowiedź lekarza).")
    doctor_attitude: DoctorAttitude = Field(
        default="neutral",
        description="Nastawienie lekarza wpływające na tempo mowy.",
    )
    doctor_id: Optional[str] = Field(
        default=None,
        description="ID archetypu lekarza. Jeśli podane, głos dobierany jest z profilu lekarza.",
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Tekst do syntezy nie może być pusty.")
        return stripped

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Proszę podać konkretne dane kliniczne potwierdzające skuteczność leku.",
                "doctor_attitude": "serious",
                "doctor_id": "skeptical_expert",
            }
        }
    )


class TTSResponse(BaseModel):
    """Kontrakt odpowiedzi endpointu /tts."""

    audio_base64: str = Field(description="Audio MP3 zakodowane jako base64.")
    format: str = Field(default="mp3", description="Format audio.")
    voice: str = Field(description="Głos OpenAI użyty do syntezy.")
    speed: float = Field(description="Prędkość mowy zastosowana do syntezy.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio_base64": "//NExAAA...",
                "format": "mp3",
                "voice": "alloy",
                "speed": 0.92,
            }
        }
    )
