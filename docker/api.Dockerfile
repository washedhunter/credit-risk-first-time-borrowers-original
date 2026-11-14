# Scoring API: core + `api` extra only (no sklearn / shap / optuna in this image).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra api --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra api
COPY models ./models
COPY data/sample ./data/sample
ENV PATH="/app/.venv/bin:$PATH" \
    MODELS_DIR=/app/models \
    SAMPLE_PATH=/app/data/sample/synthetic_applicants.csv
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "credit_risk.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
