"""API v4 — symulacja rozmowy lekarz-przedstawiciel oparta na LangGraph."""

import os
from typing import Union

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from conversation.schemas import (
    ConversationGoal,
    DoctorAttitude,
    DoctorResponse,
    EvaluationResult,
    FinishConversationResponse,
    MessageErrorResponse,
    MessageRequest,
    MessageSuccessResponse,
    RateConversationRequest,
    RateConversationResponse,
    StartConversationResponse,
    TraitsUpdate,
    TranscribeRequest,
    TranscribeResponse,
    TTSRequest,
    TTSResponse,
)
from conversation.transcribe import transcribe_audio_file_request, transcribe_audio_payload
from conversation.tts import synthesize_speech
from langgraph_app.service import finish_session, process_message, start_session
from langgraph_app.supabase_service import SupabaseService

_supabase = SupabaseService()

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych.")

# Klient OpenAI dla TTS i transkrypcji (spójny z api3)
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Symulator Medyczny AI - API v4 (LangGraph)")

CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "")
allow_origins = (
    ["*"]
    if CORS_ORIGINS_RAW == "*"
    else [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins if allow_origins else [],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpointy — identyczne z api3.py
# ---------------------------------------------------------------------------

@app.post(
    "/start",
    response_model=StartConversationResponse,
    summary="Start nowej sesji rozmowy",
)
async def start_conversation(
    doctor_id: str = Query(..., alias="id", description="ID archetypu lekarza"),
    drug_id: str = Query(..., description="ID leku"),
):
    """Tworzy nową sesję rozmowy."""
    return start_session(doctor_id=doctor_id, drug_id=drug_id, api_key=OPENAI_API_KEY)


@app.post(
    "/message",
    response_model=Union[MessageSuccessResponse, MessageErrorResponse],
    summary="Obsługa pojedynczej tury rozmowy",
)
async def send_message(req: MessageRequest):
    """Obsługuje jedną turę rozmowy przez graf LangGraph."""
    return process_message(req=req, api_key=OPENAI_API_KEY, model=OPENAI_MODEL)


@app.post(
    "/finish",
    response_model=FinishConversationResponse,
    summary="Finalizacja rozmowy i raport końcowy",
)
async def finish_conversation(session_id: str):
    """Kończy sesję i zwraca ewaluację rozmowy."""
    return finish_session(session_id=session_id, api_key=OPENAI_API_KEY, model=OPENAI_MODEL)


@app.post(
    "/rate",
    response_model=RateConversationResponse,
    summary="Ocena rozmowy przez przedstawiciela",
)
async def rate_conversation(req: RateConversationRequest):
    """Zapisuje ocenę (0-5) i komentarz przedstawiciela po zakończonej rozmowie."""
    _supabase.save_rating(
        session_id=req.session_id,
        rating=req.rating,
        description=req.description,
    )
    return {"status": "Ocena zapisana."}


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transkrypcja audio przedstawiciela (PL -> tekst)",
)
async def transcribe_audio(req: TranscribeRequest):
    """Transkrybuje wypowiedź przedstawiciela z audio base64 na tekst po polsku."""
    return transcribe_audio_payload(req=req, client=client)


@app.post(
    "/transcribe-file",
    response_model=TranscribeResponse,
    summary="Transkrypcja audio z pliku (multipart/form-data, PL -> tekst)",
)
async def transcribe_audio_file(request: Request):
    """Transkrybuje wypowiedź przedstawiciela z pliku audio."""
    return await transcribe_audio_file_request(request=request, client=client)


@app.post(
    "/tts",
    response_model=TTSResponse,
    summary="Synteza mowy lekarza (tekst -> audio MP3 base64)",
)
async def text_to_speech(req: TTSRequest):
    """Syntezuje wypowiedź lekarza na audio MP3."""
    return synthesize_speech(req=req, client=client)


# ---------------------------------------------------------------------------
# Backward compatibility exports
# ---------------------------------------------------------------------------

__all__ = [
    "app",
    "client",
    "TraitsUpdate",
    "DoctorAttitude",
    "DoctorResponse",
    "ConversationGoal",
    "MessageRequest",
    "StartConversationResponse",
    "MessageSuccessResponse",
    "MessageErrorResponse",
    "FinishConversationResponse",
    "TranscribeRequest",
    "TranscribeResponse",
    "EvaluationResult",
]
