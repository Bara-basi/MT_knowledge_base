FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

WORKDIR /app

COPY pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu" \
    && python -m pip install . \
    && python -c "from minio import Minio; import nltk.chunk.util as u; assert hasattr(u, 'ChunkScore'); from pymilvus.model.sparse import BM25EmbeddingFunction; from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer; BM25EmbeddingFunction(build_default_analyzer(language='zh'))"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
