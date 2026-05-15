"""RAG engine — Chroma vector store dla dokumentów leków."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
DRUG_CATALOG_PATH = Path(__file__).parent.parent / "drug_catalog.json"
COLLECTION_PREFIX = "drug_"


def _collection_name(drug_id: str) -> str:
    return f"{COLLECTION_PREFIX}{drug_id}".replace("-", "_").replace(" ", "_")


class DrugRAG:
    """Zarządza vector store leków i katalogiem strukturalnym.

    Catalog (drug_catalog.json) — ustrukturyzowane dane leku w formacie identycznym
    z drugs.json, generowane automatycznie przez LLM z dokumentów PDF.

    Chroma (chroma_db/) — fragmenty tekstu źródłowego do retrieval w trakcie rozmowy.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._embeddings = None
        self._catalog: Optional[Dict[str, Dict]] = None

    # ------------------------------------------------------------------
    # Embeddings (lazy init)
    # ------------------------------------------------------------------

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                model="text-embedding-3-small",
            )
        return self._embeddings

    # ------------------------------------------------------------------
    # Katalog strukturalny
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> Dict[str, Dict]:
        if self._catalog is None:
            self._reload_catalog()
        return self._catalog  # type: ignore[return-value]

    def _reload_catalog(self) -> None:
        if DRUG_CATALOG_PATH.exists():
            with open(DRUG_CATALOG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._catalog = {entry["id"]: entry for entry in data}
            logger.info("drug_catalog loaded: %d entries", len(self._catalog))
        else:
            self._catalog = {}

    def reload(self) -> None:
        """Przeładowuje katalog (po dodaniu nowego leku)."""
        self._catalog = None

    def has_drug(self, drug_id: str) -> bool:
        return drug_id in self.catalog

    def get_drug(self, drug_id: str) -> Optional[Dict]:
        """Zwraca strukturalne dane leku z katalogu."""
        return self.catalog.get(drug_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_context(self, drug_id: str, query: str, k: int = 4) -> List[str]:
        """Pobiera k najbardziej relevantnych fragmentów ChPL dla zapytania."""
        from langchain_chroma import Chroma

        col = _collection_name(drug_id)
        try:
            db = Chroma(
                collection_name=col,
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            if db._collection.count() == 0:
                return []
            docs = db.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as exc:
            logger.warning("RAG retrieve failed for %s: %s", drug_id, exc)
            return []


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_rag_instance: Optional[DrugRAG] = None


def get_rag(api_key: str) -> DrugRAG:
    """Zwraca singleton DrugRAG (reinicjalizuje jeśli zmienił się klucz)."""
    global _rag_instance
    if _rag_instance is None or _rag_instance.api_key != api_key:
        _rag_instance = DrugRAG(api_key=api_key)
    return _rag_instance
