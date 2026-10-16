FROM python:3.12-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY src src
COPY api api
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY models models

ENV UV_NO_DEV=1 \
    UV_FROZEN=1

RUN uv sync
RUN uv run python -c "import nltk; nltk.download('punkt_tab')"

CMD ["uv", "run", "uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
