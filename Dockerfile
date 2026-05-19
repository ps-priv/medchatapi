FROM python:3.12-slim

WORKDIR /app

# Zależności systemowe potrzebne przez chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalacja zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji
COPY api4.py ./
COPY conversation/ ./conversation/
COPY langgraph_app/ ./langgraph_app/

# Dane statyczne
COPY doctor_archetypes.json drug_catalog.json keyword_triggers.json ./

# Baza RAG (Chroma) — domyślnie baked-in; można nadpisać przez volume
COPY chroma_db/ ./chroma_db/

# Pozwala podmienić chroma_db przez: docker run -v /host/chroma_db:/app/chroma_db ...
VOLUME ["/app/chroma_db"]

EXPOSE 8000

CMD ["uvicorn", "api4:app", "--host", "0.0.0.0", "--port", "8000"]
