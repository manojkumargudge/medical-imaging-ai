# Inference worker image: code and configs only. Checkpoints are mounted at run time
# (docker-compose.worker.yml), never copied in; .dockerignore is an allow-list. CPU only.
FROM python:3.11.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml params.yaml ./
COPY src ./src
COPY configs ./configs
RUN pip install . && useradd --create-home --uid 10001 worker && mkdir /results \
    && chown worker /results
USER worker

CMD ["python", "-m", "src.service.worker"]
