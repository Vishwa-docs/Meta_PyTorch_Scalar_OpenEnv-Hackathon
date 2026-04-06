---
title: Polypharmacy
emoji: 📉
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# PolypharmacyEnv

Monorepo for an OpenEnv-compatible medication safety environment with:

- a FastAPI backend (`backend/`)
- a React frontend (`frontend/`)
- data assets (`data/`)
- utility scripts (`scripts/`)

---

## Repository Structure

```text
backend/
    main.py                      # ASGI entrypoint (uvicorn target)
    requirements.txt             # Backend dependencies
    Dockerfile                   # Backend container
    src/polypharmacy_env/        # Python package source
      api/
        app.py                   # FastAPI/OpenEnv app assembly
        server.py                # Compatibility import wrapper
        routes/agent.py          # /agent/suggest route
      services/
        groq_agent.py            # Groq-based action suggestion logic
      env_core.py                # OpenEnv environment core
      models.py                  # Action/observation/state models
      data_loader.py             # CSV loading
      ddi_simulator.py           # DDI and Beers lookups
      rewards.py                 # Reward shaping
      graders.py                 # Task graders
      tasks.py                   # Task/episode selection
      tests/                     # Backend tests
frontend/
    src/                         # React UI code
    package.json
    Dockerfile                   # Frontend container
data/
    lookups/                     # drug_metadata.csv, ddi_rules.csv, beers_criteria.csv
    processed/                   # patients_polypharmacy.csv
scripts/
    preprocess_data.py           # Synthetic data generation
    dev_backend.sh               # Local backend run helper
    dev_frontend.sh              # Local frontend run helper
    run_validation.sh            # Tests + baseline validation
docker-compose.yml             # Full stack orchestration
openenv.yaml                   # OpenEnv manifest
inference.py                   # Baseline inference script (required at root)
.env.example                   # Environment template
```

---

## What It Does

The environment simulates elderly polypharmacy review. Agent actions:

- `query_ddi`
- `propose_intervention`
- `finish_review`

Supported tasks:

- `easy_screening`
- `budgeted_screening`
- `complex_tradeoff`

---

## Prerequisites

- Python 3.10+
- Node.js 18+ (or 20+ recommended)
- npm
- Docker + Docker Compose (optional, for containerized run)

---

## Environment Setup

Create `.env`:

```bash
cp .env.example .env
```

Set values for local backend integrations as needed.

---

## Local Run (Recommended During Development)

### 1) Install dependencies

Backend:

```bash
pip install -r backend/requirements.txt
```

Frontend:

```bash
cd frontend
npm install
cd ..
```

### 2) Generate/update synthetic data (if needed)

```bash
python scripts/preprocess_data.py
```

### 3) Start services in two terminals

Terminal A:

```bash
./scripts/dev_backend.sh
```

Terminal B:

```bash
./scripts/dev_frontend.sh
```

### 4) Open app

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend health: [http://localhost:7860/health](http://localhost:7860/health)

---

## Docker Run

Run both services:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

Ports:

- backend: `7860`
- frontend: `5173`

---

## Hugging Face Spaces Deployment (Docker)

This repo now includes a **root `Dockerfile`** that builds frontend + backend into one container, so Spaces can host both API and UI together.

### 1) Create a new Space

- Go to [Hugging Face Spaces](https://huggingface.co/new-space)
- Choose **Docker** SDK
- Create the Space

### 2) Add Space secrets/variables

In Space Settings -> Variables and Secrets:

- Secret: `HF_TOKEN`
- Variable: `API_BASE_URL=https://router.huggingface.co/v1`
- Variable: `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`

### 3) Push this repository to the Space

Commit and push all files, including root `Dockerfile`.

### 4) Verify after build

- Space root URL loads the React UI
- `/health` returns healthy status
- OpenEnv endpoints are available (`/reset`, `/step`, `/state`, `/schema`)

Notes:

- Container reads `PORT` (defaults to `7860`) which is Space-friendly.
- Frontend static assets are served by FastAPI from `frontend/dist`.

---

## API Endpoints

OpenEnv/health:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /schema`
- `WS /ws` (stateful session)

AI helper:

- `POST /agent/suggest`

---

## Testing

Run backend tests:

```bash
python -m pytest backend/src/polypharmacy_env/tests -v
```

Or run validation script:

```bash
./scripts/run_validation.sh
```

### Submission validation

```bash
openenv validate
python inference.py
```

---

## Notes

- OpenEnv HTTP reset/step is stateless; multi-step episode continuity should use websocket (`/ws`).
- The frontend uses websocket for episode continuity and HTTP for AI suggestion.
- AI behavior includes rule-based guardrails to avoid repetitive low-value loops.

---

## Troubleshooting

- `ModuleNotFoundError: polypharmacy_env`
  - Start backend using `./scripts/dev_backend.sh` from repo root.
- `/agent/suggest` fails
  - Check `.env` keys and restart backend.
- UI state looks stale
  - Hard refresh browser and click `Reset Episode`.
