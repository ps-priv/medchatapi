"""Generator agendy lekarza — jednorazowe wywołanie LLM na starcie sesji."""

import logging
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_VALID_KINDS = {"clinical_curiosity", "patient_case", "concern", "personal", "time_pressure"}


class _AgendaItem(BaseModel):
    kind: str
    content: str
    priority: int = Field(default=2, ge=1, le=3)


class _AgendaList(BaseModel):
    items: List[_AgendaItem] = Field(description="Lista 3-5 wątków lekarza.")


def generate_agenda(
    doctor_profile: Dict,
    drug_info: Dict,
    familiarity: str,
    prior_visits_summary: Optional[str],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Generuje 3-5 własnych wątków lekarza na starcie sesji.

    Zwraca listę dictów zgodnych ze schematem AgendaItem lub [] przy błędzie LLM.
    Koszt: ~1 wywołanie gpt-4o-mini ≈ 0.001 USD.
    """
    try:
        rep_ref = ""
        if familiarity == "first_meeting":
            familiarity_context = "To pierwsza wizyta tego przedstawiciela — nie znasz go."
        elif familiarity == "acquainted":
            familiarity_context = f"Znasz tego przedstawiciela z poprzednich wizyt zawodowych.{' ' + prior_visits_summary if prior_visits_summary else ''}"
        else:
            familiarity_context = f"Dobrze znasz tego przedstawiciela, macie swobodną relację.{' ' + prior_visits_summary if prior_visits_summary else ''}"

        drug_name = drug_info.get("nazwa", drug_info.get("id", "nieznany lek"))
        drug_indication = drug_info.get("wskazania", "")

        prompt = f"""Wcielasz się w lekarza: {doctor_profile.get('name', '')} ({doctor_profile.get('description', '')}).
Styl komunikacji: {doctor_profile.get('communication_style', 'profesjonalny')}.
Kontekst relacji: {familiarity_context}

Za chwilę przyjdzie do Ciebie przedstawiciel farmaceutyczny z lekiem "{drug_name}" (wskazania: {drug_indication}).

Wygeneruj 3-5 własnych wątków które możesz NATURALNIE wpleć w rozmowę. Mają być konkretne i osobiste — żadnych ogólników. Rodzaje:
- patient_case: konkretny pacjent którego ostatnio widziałeś i który nasuwa pytanie o ten lek (wiek, sytuacja kliniczna)
- clinical_curiosity: własna wątpliwość lub ciekawostka kliniczna którą chcesz poruszyć
- concern: obawa lub zastrzeżenie wobec tego leku lub tej klasy leków
- personal: coś z dnia — coś co odebrałeś przed chwilą, nastrój, okoliczność
- time_pressure: konkretny powód braku czasu (kolejna wizyta, dyżur, szkolenie)

Zasady:
- Pisz w pierwszej osobie, konkretnie, po polsku
- Każdy wątek max 1-2 zdania
- Priorytet: 1=mała waga, 2=średnia, 3=ważne dla mnie
- Min 3, max 5 pozycji
"""

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.85,
        ).with_structured_output(_AgendaList)

        result: _AgendaList = llm.invoke([
            SystemMessage(content="Generujesz wewnętrzną agendę lekarza do realistycznej symulacji rozmowy medycznej."),
            HumanMessage(content=prompt),
        ])

        items = []
        for item in result.items[:5]:
            kind = item.kind if item.kind in _VALID_KINDS else "clinical_curiosity"
            items.append({
                "kind": kind,
                "content": str(item.content).strip(),
                "used": False,
                "priority": max(1, min(3, int(item.priority))),
            })

        logger.info(
            "agenda generated: %d items for doctor=%s drug=%s",
            len(items), doctor_profile.get("id", "?"), drug_info.get("id", "?"),
        )
        return items

    except Exception as exc:
        logger.warning("agenda generation failed (%s) — using empty agenda", exc)
        return []
