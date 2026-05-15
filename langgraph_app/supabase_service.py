"""Serwis zapisu rozmów do bazy Supabase."""

import logging
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)

SUPABASE_URL = "https://tsmhuzesyigczoqewred.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRzbWh1emVzeWlnY3pvcWV3cmVkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3Nzg3OTc1MywiZXhwIjoyMDkzNDU1NzUzfQ.VJSv8TTyPZYjMlsStgUazDtVCDSn0pbGVFLN_Nm0KFU"

_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


class SupabaseService:
    """Zapisuje zakończone sesje rozmów do tabeli `conversations` w Supabase."""

    TABLE = "conversations"

    def save_conversation(
        self,
        session_id: str,
        doctor_profile: Dict[str, Any],
        drug_info: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        evaluation: Dict[str, Any],
        conversation_goal: Dict[str, Any],
        turn_metrics_history: List[Dict[str, Any]],
        final_traits: Dict[str, float],
        frustration_score: float,
        phase: str,
        turn_count: int,
        is_terminated: bool,
        critical_flags: List[str],
    ) -> None:
        row = {
            "session_id": session_id,
            "doctor_id": doctor_profile.get("id"),
            "drug_id": drug_info.get("id"),
            "conversation_history": conversation_history,
            "evaluation": evaluation,
            "conversation_goal": conversation_goal,
            "turn_metrics_history": turn_metrics_history,
            "final_traits": final_traits,
            "frustration_score": frustration_score,
            "phase": phase,
            "turn_count": turn_count,
            "is_terminated": is_terminated,
            "critical_flags": critical_flags,
        }
        try:
            _get_client().table(self.TABLE).insert(row).execute()
            logger.info("supabase: saved conversation session_id=%s", session_id)
        except Exception as exc:
            logger.error("supabase: failed to save conversation session_id=%s error=%s", session_id, exc)

    def save_rating(self, session_id: str, rating: int, description: Optional[str]) -> None:
        row = {
            "session_id": session_id,
            "rating": rating,
            "description": description,
        }
        try:
            _get_client().table("conversation_ratings").insert(row).execute()
            logger.info("supabase: saved rating session_id=%s rating=%d", session_id, rating)
        except Exception as exc:
            logger.error("supabase: failed to save rating session_id=%s error=%s", session_id, exc)
            raise
