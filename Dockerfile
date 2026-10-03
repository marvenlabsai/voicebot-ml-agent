# Voice agent worker
#   docker build -t voicebot-agent .
#   docker run --env-file .env -p 8081:8081 -v voicebot-recordings:/app/.cache/recordings voicebot-agent
# Give it time to drain on stop: docker stop -t 300 (compose: stop_grace_period: 5m)

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEALTH_PORT=8081

WORKDIR /app

# curl: health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 agent
COPY --chown=agent:agent . .
USER agent

# Model files the plugins need at runtime. PRELOAD_SCRIPT_MODEL=1 also bakes in the scripted
# replies embedding model (~220 MB) so the first scripted call doesn't download it.
ARG PRELOAD_SCRIPT_MODEL=0
RUN python agent.py download-files \
    && if [ "$PRELOAD_SCRIPT_MODEL" = "1" ]; then python -c "from scripted.router import load_model; load_model()"; fi \
    && mkdir -p .cache/recordings

# Recordings waiting for upload survive restarts when this is a volume
VOLUME ["/app/.cache/recordings"]

EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${HEALTH_PORT}/" || exit 1

STOPSIGNAL SIGTERM
CMD ["python", "agent.py", "start"]
