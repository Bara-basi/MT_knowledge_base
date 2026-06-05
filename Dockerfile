FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && python -c "import nltk.chunk.util as u; assert hasattr(u, 'ChunkScore'); from pymilvus.model.sparse import BM25EmbeddingFunction; from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer; BM25EmbeddingFunction(build_default_analyzer(language='zh'))"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
