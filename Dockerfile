FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY openenv-polypharmacy/frontend/package*.json ./
RUN npm ci
COPY openenv-polypharmacy/frontend/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY openenv-polypharmacy/backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY openenv-polypharmacy/backend /app/backend
COPY openenv-polypharmacy/data /app/data
COPY openenv-polypharmacy/scripts /app/scripts
COPY openenv-polypharmacy/openenv.yaml /app/openenv.yaml
COPY openenv-polypharmacy/.env.example /app/.env.example
COPY openenv-polypharmacy/inference.py /app/inference.py

COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

RUN python3 /app/scripts/preprocess_data.py

ENV PORT=7860
ENV PYTHONPATH="/app/backend/src:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
