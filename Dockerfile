FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv 
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
RUN python -m runway_radar.api.load_db
EXPOSE 8000

CMD ["fastapi", "run", "src/runway_radar/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
