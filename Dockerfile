FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --production=false
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY data /app/data
COPY scripts /app/scripts
COPY openenv.yaml /app/openenv.yaml
COPY .env.example /app/.env.example
COPY inference.py /app/inference.py
COPY train_rl.py /app/train_rl.py
COPY train_bandit.py /app/train_bandit.py

COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

RUN python3 /app/scripts/preprocess_data.py

# Ensure the user owns the app directory and has a writable home (HF Spaces)
RUN chown -R user:user /app && \
    mkdir -p /home/user/.cache && chown -R user:user /home/user

ENV PORT=7860
ENV PYTHONPATH="/app/backend/src:/app"
ENV PYTHONUNBUFFERED=1

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
