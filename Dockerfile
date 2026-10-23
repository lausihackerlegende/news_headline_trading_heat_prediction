FROM python:3.12-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

WORKDIR /app

COPY src src
COPY api api
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY models models

ENV UV_NO_DEV=1 \
    UV_FROZEN=1 \
    KERAS_BACKEND=torch \
    PORT=8080

RUN uv sync
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"
RUN uv run python -c "import os, zipfile; z='models/finbert_tone_heat_best.zip'; d='models/finbert_tone_heat_best_extracted'; zipfile.ZipFile(z).extractall(d) if os.path.exists(z) else None"

CMD ["sh", "-c", "uv run uvicorn api.fast:app --host 0.0.0.0 --port ${PORT:-8080}"]
