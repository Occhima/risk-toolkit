FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/workspace/risk-toolkit/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        graphviz \
        jq \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /uvx /usr/local/bin/

WORKDIR /workspace/risk-toolkit

# Keep the image focused on the Python workspace for now. The future CLI module can
# add its own system packages or language runtime here without changing the
# devcontainer contract.
