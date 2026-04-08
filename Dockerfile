# ─────────────────────────────────────────────────────────────
#  FlexTime — Production Dockerfile
#  Compatible with HuggingFace Spaces (Docker SDK, port 7860)
#  Run: docker build -t flextime . && docker run -p 7860:7860 flextime
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces metadata labels
LABEL org.opencontainers.image.title="FlexTime"
LABEL org.opencontainers.image.description="Real-world AI workforce scheduling OpenEnv environment"
LABEL org.opencontainers.image.version="1.0.0"
LABEL space.tag="openenv"

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with uid=1000
RUN useradd -m -u 1000 -s /bin/bash flextime

WORKDIR /app

# ── Install Python dependencies (cached layer) ────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────
COPY --chown=flextime:flextime . .

# Switch to non-root user
USER flextime

# HF Spaces always uses port 7860
EXPOSE 7860

# Health check — HF Spaces pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start server ──────────────────────────────────────────────
# Single worker for stateful in-memory environment
# PYTHONPATH ensures scripts.baseline can be imported by app/main.py
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
