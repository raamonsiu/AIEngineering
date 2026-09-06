# =============================================================================
# Stage 1, Builder
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --no-install-project --no-dev


# =============================================================================
# Stage 2, Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
# scripts/ holds compare.py, which the exercise expects to be runnable
# with `docker compose exec`. Without this it only exists on the host.
COPY scripts/ /app/scripts/

RUN chown -R appuser:appgroup /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
