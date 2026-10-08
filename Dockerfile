# syntax=docker/dockerfile:1
# cspell:words PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED venv
ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS gateway
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 HOME=/tmp
WORKDIR /app
COPY pyproject.toml uv.lock README.md chat.py evaluate.py ./
COPY schematic_model ./schematic_model
RUN uv sync --locked --no-dev --no-editable
USER 10001:10001
ENTRYPOINT ["/app/.venv/bin/schematic-model"]
CMD ["deploy", "gateway", "--config", "/config/agent.json", "--bind", "0.0.0.0", "--port", "8893"]
