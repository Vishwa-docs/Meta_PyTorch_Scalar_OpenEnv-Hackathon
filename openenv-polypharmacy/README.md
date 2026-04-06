# PolypharmacyEnv

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement-learning environment that simulates **elderly polypharmacy medication review**. An RL agent acts as a clinical pharmacist assistant, identifying dangerous drug-drug interactions (DDIs), Beers-criteria violations, and proposing safe interventions.

---

## Motivation

Polypharmacy (concurrent use of multiple medications) is extremely common in elderly patients (age >= 65) and carries significant risks:

- **Drug-drug interactions** can cause adverse events, hospitalisation, and death.
- **Beers-criteria violations** flag medications that are inappropriate or require dose adjustments in older adults.
- Stopping critical medications (anticoagulants, insulin) without proper substitution can be equally dangerous.

This environment lets RL and LLM-based agents learn to **balance risk reduction against regimen stability**.

---

## Action Space

Each step, the agent sends a `PolypharmacyAction` with one of three action types:

| `action_type` | Required fields | Description |
|---|---|---|
| `query_ddi` | `drug_id_1`, `drug_id_2` | Query the DDI database for an interaction between two drugs |
| `propose_intervention` | `target_drug_id`, `intervention_type` | Propose changing a medication (`stop`, `dose_reduce`, `substitute`, `add_monitoring`) |
| `finish_review` | — | End the review and trigger final grading |

Optional fields: `proposed_new_drug_id`, `rationale`.

## Observation Space

`PolypharmacyObservation` includes:

- **Patient demographics**: `age`, `sex`, `conditions`, `eGFR_category`, `liver_function_category`
- **Medications**: list of `MedicationEntry` (drug_id, name, class, dose, high-risk flags, Beers flags)
- **History**: `interaction_queries` (past DDI query results), `interventions` (past actions)
- **Budgets**: `remaining_query_budget`, `remaining_intervention_budget`
- **Reward signals**: `shaped_reward`, `done`

## State

`PolypharmacyState`: `episode_id`, `task_id`, `step_count`, `max_steps`, `num_query_actions`, `num_interventions`.

---

## Tasks

| Task ID | Difficulty | Drugs | Query Budget | Intervention Budget | Max Steps | Description |
|---|---|---|---|---|---|---|
| `easy_screening` | Easy | 3-5 | 4 | 2 | 10 | One severe DDI, simple resolution |
| `budgeted_screening` | Medium | 6-10 | 8 | 3 | 20 | Multiple DDIs + Beers issues, limited budgets |
| `complex_tradeoff` | Hard | 10-15 | 12 | 5 | 30 | Critical drugs, trade-off between risk and regimen stability |

---

## Reward Structure

**Per-step shaped rewards:**

| Event | Reward |
|---|---|
| DDI query | -0.01 (cost) + 0.03 bonus if severe DDI discovered |
| Successful intervention | +(previous_risk - new_risk) - 0.02 cost |
| Invalid action | -0.10 penalty |
| Timeout (max steps exceeded) | -0.20 penalty |
| `finish_review` | + grader score (0.0 to 1.0) |

**Terminal grader scoring:**
- **Easy**: 50% risk reduction + 50% targeted intervention flag
- **Medium**: 50% risk reduction + 30% intervention precision + 20% query efficiency
- **Hard**: risk reduction - regimen disruption penalty - critical drug penalty

---

## Setup & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python3 scripts/preprocess_data.py
```

### Run the API server locally

```bash
PYTHONPATH=src uvicorn polypharmacy_env.api.server:app --host 0.0.0.0 --port 7860
```

### Run the heuristic baseline

```bash
PYTHONPATH=src python3 -m polypharmacy_env.baselines.heuristic_agent
```

### Run tests

```bash
PYTHONPATH=src python3 -m pytest src/polypharmacy_env/tests/ -v
```

### Run `inference.py` (LLM baseline)

```bash
# Start the server first, then in another terminal:
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4.1"
export POLYPHARMACY_ENV_URL="http://localhost:7860"
python3 inference.py
```

### Docker

```bash
docker build -t polypharmacy-env .
docker run -p 7860:7860 polypharmacy-env
```

---

## Hugging Face Space

This repo is ready for deployment as a HF Space:

- **Space type**: `docker`
- **Tag**: `openenv`
- The container listens on port 7860 and exposes `/reset`, `/step`, `/state`, `/health`.

---

## Baseline Scores

### Heuristic Agent (deterministic, rule-based)

| Task | Avg Score | Avg Reward |
|---|---|---|
| `easy_screening` | ~0.96 | ~1.30 |
| `budgeted_screening` | ~0.48 | ~0.45 |
| `complex_tradeoff` | ~0.24 | ~0.11 |

*(Scores vary by seed; run `scripts/run_validation.sh` for exact numbers.)*

---

## Project Structure

```
openenv-polypharmacy/
  openenv.yaml              # OpenEnv manifest
  Dockerfile                # Container image
  inference.py              # LLM baseline script
  requirements.txt
  pyproject.toml
  src/polypharmacy_env/
    config.py               # Constants, task configs
    models.py               # Pydantic action/observation/state models
    env_core.py             # PolypharmacyEnv implementation
    tasks.py                # Task selection utilities
    graders.py              # Deterministic graders (3 difficulty levels)
    rewards.py              # Reward shaping logic
    data_loader.py          # CSV data loading
    ddi_simulator.py        # Drug interaction lookup engine
    api/
      server.py             # FastAPI HTTP server
      schemas.py            # Request/response schemas
    baselines/
      heuristic_agent.py    # Rule-based baseline
      random_agent.py       # Random baseline
    tests/
      test_env_core.py
      test_api.py
  data/
    lookups/                # Drug metadata, DDI rules, Beers criteria CSVs
    processed/              # Synthetic patient episodes
  scripts/
    preprocess_data.py      # Synthetic data generator
    run_validation.sh       # Run tests + baseline
```
