"""Serwis zapisu podsumowań rozmów do bazy MongoDB (testowa)."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)

# TODO(Paweł): uzupełnić namiarami do testowej bazy MongoDB.
MONGO_URI = ""
MONGO_DB_NAME = ""
MONGO_COLLECTION = "conversation_summaries"

_client: Optional[MongoClient] = None


def _get_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[MONGO_DB_NAME][MONGO_COLLECTION]


class MongoService:
    """Zapisuje podsumowania zakończonych rozmów do kolekcji MongoDB, na żądanie klienta (POST /save-summary)."""

    def save_summary(self, summary: Dict[str, Any]) -> None:
        document = {**summary, "saved_at": datetime.now(timezone.utc)}
        try:
            _get_collection().insert_one(document)
            logger.info("mongo: saved summary session_id=%s", summary.get("session_id"))
        except Exception as exc:
            logger.error("mongo: failed to save summary session_id=%s error=%s", summary.get("session_id"), exc)
            raise
