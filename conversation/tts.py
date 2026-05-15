"""Synteza mowy (TTS) dla odpowiedzi lekarza z modulacją głosu wg archetypu i nastawienia."""

import base64
import logging
from typing import Dict, Optional

from fastapi import HTTPException

from .data import get_doctor_by_id
from .schemas import TTSRequest, TTSResponse

logger = logging.getLogger(__name__)

# Tempo mowy zależne od nastawienia lekarza.
# Głos (voice) jest stały per lekarz — tempo jest jedynym parametrem modulowanym przez nastrój.
#
# Prędkość: zakres 0.25–4.0, wartość 1.0 = normalna.
#   happy   → lekkie przyspieszenie (energiczny ton)
#   neutral → normalna prędkość
#   serious → lekkie zwolnienie (powaga, namysł)
#   sad     → wyraźne zwolnienie (zmęczenie, dystans)
ATTITUDE_SPEED: Dict[str, float] = {
    "happy":   1.08,
    "neutral": 1.00,
    "serious": 0.90,
    "sad":     0.82,
}

# Domyślny głos gdy lekarz nie ma przypisanego tts_voice lub doctor_id nie podano.
# Mapowanie nastawienie → głos zachowane jako fallback (stare zachowanie).
ATTITUDE_FALLBACK_VOICE: Dict[str, str] = {
    "happy":   "nova",
    "neutral": "shimmer",
    "serious": "alloy",
    "sad":     "shimmer",
}

OPENAI_TTS_MODEL = "tts-1"


def _resolve_voice(doctor_id: Optional[str], attitude: str) -> str:
    """Zwraca głos lekarza: z profilu jeśli podano doctor_id, fallback z nastawienia."""
    if doctor_id:
        doctor_profile = get_doctor_by_id(doctor_id)
        if doctor_profile:
            voice = str(doctor_profile.get("tts_voice", "")).strip()
            if voice:
                return voice
    return ATTITUDE_FALLBACK_VOICE.get(attitude, "shimmer")


def synthesize_speech(req: TTSRequest, client) -> TTSResponse:
    """Syntetyzuje wypowiedź lekarza na audio MP3.

    Głos dobierany jest z profilu lekarza (tts_voice w doctor_archetypes.json).
    Tempo mowy modulowane jest przez nastawienie lekarza (doctor_attitude).
    """
    attitude = str(req.doctor_attitude) if req.doctor_attitude else "neutral"
    voice = _resolve_voice(req.doctor_id, attitude)
    speed = ATTITUDE_SPEED.get(attitude, 1.00)

    logger.info(
        "tts doctor_id=%s attitude=%s voice=%s speed=%s chars=%d",
        req.doctor_id, attitude, voice, speed, len(req.text),
    )

    try:
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=voice,
            input=req.text,
            speed=speed,
            response_format="mp3",
        )
        audio_bytes = response.content
    except Exception as exc:
        logger.error("tts error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Błąd syntezy mowy: {exc}") from exc

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return TTSResponse(
        audio_base64=audio_base64,
        format="mp3",
        voice=voice,
        speed=speed,
    )
