ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

WORKDIR /app

COPY pyproject.toml ./

ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ARG INSTALL_TORCH=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi \
    && if [ -n "$PIP_TRUSTED_HOST" ]; then python -m pip config set global.trusted-host "$PIP_TRUSTED_HOST"; fi \
    && python -m pip install --upgrade pip setuptools wheel \
    && if [ "$INSTALL_TORCH" = "true" ]; then python -m pip install --index-url "$TORCH_INDEX_URL" "torch==2.9.1+cpu"; fi \
    && python -m pip install . \
    && python -c "from minio import Minio; import nltk.chunk.util as u; assert hasattr(u, 'ChunkScore'); from pymilvus.model.sparse import BM25EmbeddingFunction; from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer; BM25EmbeddingFunction(build_default_analyzer(language='zh'))"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
