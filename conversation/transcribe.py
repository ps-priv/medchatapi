"""Logika transkrypcji audio (STT) dla endpointu /transcribe."""

import base64
import io
import importlib.util
import os
from typing import Dict, List

from fastapi import HTTPException, Request

from .schemas import TranscribeRequest

PRIMARY_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
FALLBACK_TRANSCRIBE_MODELS = [
    model.strip()
    for model in os.getenv("OPENAI_TRANSCRIBE_FALLBACKS", "whisper-1").split(",")
    if model.strip()
]


def _decode_audio_base64(audio_base64: str) -> bytes:
    """Dekoduje base64 audio; akceptuje tez format data URL."""
    raw = str(audio_base64 or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Pole 'audio_base64' jest puste.")

    # Pozwala frontendowi wyslac takze data URL: data:audio/webm;base64,....
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]

    try:
        return base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Niepoprawny format base64 w 'audio_base64'.") from exc


def _extract_text(response) -> str:
    """Wyciaga tekst transkrypcji z odpowiedzi SDK niezaleznie od jej ksztaltu."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    if isinstance(response, dict):
        candidate = str(response.get("text", "")).strip()
        if candidate:
            return candidate

    candidate = str(response).strip()
    if candidate:
        return candidate

    raise HTTPException(status_code=502, detail="Brak tekstu w odpowiedzi transkrypcji.")


def _call_transcribe(client, audio_bytes: bytes, filename: str, prompt: str | None, model: str) -> str:
    """Wykonuje pojedyncza probe STT wybranym modelem."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    response = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        language="pl",
        prompt=prompt,
    )
    return _extract_text(response)


def _transcribe_bytes(audio_bytes: bytes, filename: str, prompt: str | None, client) -> Dict:
    """Transkrybuje bajty audio, próbując kolejno modele główny i fallback."""
    models_to_try: List[str] = [PRIMARY_TRANSCRIBE_MODEL, *FALLBACK_TRANSCRIBE_MODELS]

    last_error: Exception | None = None
    for model in models_to_try:
        try:
            text = _call_transcribe(
                client=client,
                audio_bytes=audio_bytes,
                filename=filename,
                prompt=prompt,
                model=model,
            )
            return {
                "text": text,
                "language": "pl",
                "model": model,
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise HTTPException(
        status_code=502,
        detail=(
            "Transkrypcja nie powiodla sie dla dostepnych modeli STT. "
            f"Ostatni blad: {type(last_error).__name__ if last_error else 'unknown'}"
        ),
    )


def _multipart_is_available() -> bool:
    """Sprawdza, czy parser multipart jest dostępny w środowisku."""
    return importlib.util.find_spec("multipart") is not None


def transcribe_audio_payload(req: TranscribeRequest, client) -> Dict:
    """Transkrybuje polskie audio przedstawiciela z payloadu base64."""
    audio_bytes = _decode_audio_base64(req.audio_base64)
    return _transcribe_bytes(
        audio_bytes=audio_bytes,
        filename=req.filename,
        prompt=req.prompt,
        client=client,
    )


async def transcribe_audio_file_request(request: Request, client) -> Dict:
    """Transkrybuje audio przesłane jako multipart/form-data (`file` + opcjonalnie `prompt`)."""
    if not _multipart_is_available():
        raise HTTPException(
            status_code=503,
            detail="Brak pakietu python-multipart. Zainstaluj go, aby uzyc endpointu /transcribe-file.",
        )

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Endpoint /transcribe-file wymaga Content-Type: multipart/form-data.",
        )

    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Nie udalo sie odczytac danych multipart/form-data.") from exc

    upload = form.get("file")
    if upload is None:
        raise HTTPException(status_code=400, detail="Brak pola 'file' w formularzu.")

    filename = str(getattr(upload, "filename", "") or "recording.webm")
    prompt_raw = form.get("prompt")
    prompt = str(prompt_raw).strip() if prompt_raw is not None else None
    if prompt == "":
        prompt = None

    try:
        audio_bytes = await upload.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Nie udalo sie odczytac przeslanego pliku audio.") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Przeslany plik audio jest pusty.")

    return _transcribe_bytes(
        audio_bytes=audio_bytes,
        filename=filename,
        prompt=prompt,
        client=client,
    )
