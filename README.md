---
title: PolypharmacyEnv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - healthcare
  - polypharmacy
pinned: false
---

# PolypharmacyEnv — Elderly Medication Safety via Reinforcement Learning

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant environment that simulates **elderly polypharmacy medication review**. An RL agent acts as a clinical pharmacist assistant: it queries drug-drug interactions (DDIs), identifies Beers-criteria violations, and proposes safe interventions — all under resource-constrained budgets.

Built for the **PyTorch OpenEnv Hackathon** to demonstrate how clinical decision support for polypharmacy can be framed as a sequential RL problem and served as a reusable environment through the OpenEnv hub.

---

## Why This Matters

Polypharmacy — the simultaneous use of five or more medications — affects the majority of adults over 65. Elderly patients often see multiple specialists who may not be aware of each other's prescriptions, leading to dangerous drug combinations. Studies report that **adverse drug events from polypharmacy contribute to 100,000+ hospitalizations annually** in the US alone.

Current solutions use static risk scoring. PolypharmacyEnv goes further by framing medication review as a **sequential decision problem**, where an RL agent must strategically allocate limited query and intervention budgets to maximize patient safety — exactly the kind of resource-constrained optimization that reinforcement learning excels at.

**Reference**: Larouche, A., Durand, A., Khoury, R. & Sirois, C. (2023). [Neural Bandits for Data Mining: Searching for Dangerous Polypharmacy](https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5). *Advances in Artificial Intelligence*, Springer.

---

## How OpenEnv & RL Power This

### The RL Formulation

PolypharmacyEnv frames medication review as a **Markov Decision Process (MDP)**:

- **State**: Patient profile (age, conditions, organ function) + current medication list + interaction history
- **Action space**: `query_ddi(drug_i, drug_j)` | `propose_intervention(target, type)` | `finish_review`
- **Reward**: Shaped, dense signal at every step (not sparse end-of-episode). Queries cost budget (-0.015), discovering severe DDIs earns bonus (+0.05), successful interventions earn proportional risk reduction minus cost, invalid actions are penalized (-0.15), and `finish_review` triggers a grader that returns a terminal score in [0.0, 1.0].
- **Constraint**: Finite query and intervention budgets, creating a resource-allocation optimization problem.

This MDP is what makes the problem fundamentally different from static risk scoring: the agent must **decide what information to acquire** (which drug pairs to query) and **which interventions to prioritize**, all under budget constraints — a sequential decision problem that RL is designed to solve.

### OpenEnv Interface

PolypharmacyEnv implements the full **OpenEnv standard**:

- **`reset()`** — Generates a new patient scenario (age, conditions, medication list)
- **`step(action)`** — Processes an agent action, updates regimen state, returns shaped reward
- **`state()`** — Returns the current episode snapshot

All models use typed Pydantic classes extending OpenEnv base types (`PolypharmacyAction`, `PolypharmacyObservation`, `PolypharmacyState`).

### What the Environment Enables

The shaped reward function provides continuous signal over the full trajectory, making this environment compatible with standard RL training approaches:

- **Policy gradient methods** (REINFORCE, PPO, GRPO): The per-step reward signal allows policy networks to learn query prioritization and intervention strategies.
- **OpenEnv training pipeline**: Through OpenEnv's `step()`/`reset()` HTTP interface, external RL training loops can connect to this environment and train policies without modification.
- **Neural Bandits (OptimNeuralTS)**: The budget-constrained query selection implements the OptimNeuralTS approach from the reference paper — Neural Thompson Sampling combined with Differential Evolution for efficient search.

### Included Agents

The repository ships with multiple agent implementations spanning rule-based, RL-trained, bandit-based, and LLM-based approaches:

- **OptimNeuralTS bandit** (`train_bandit.py`, `neural_bandits.py`): Implements the paper's core algorithm — Neural Thompson Sampling with Differential Evolution to efficiently search for dangerous drug combinations. Builds an ensemble of models across training steps for high-precision predictions.
- **REINFORCE-trained policy** (`train_rl.py`): A neural network policy trained via REINFORCE with learned baseline against the environment's shaped reward. Demonstrates that the MDP formulation and reward shaping enable genuine policy improvement through RL training.
- **Heuristic agent** (`baselines/heuristic_agent.py`): Deterministic rule-based strategy that queries high-risk drug pairs first, then intervenes on severe DDIs. Serves as a strong domain-knowledge baseline.
- **LLM agent** (`inference.py`): Uses an LLM (Qwen2.5-72B via OpenAI-compatible API) for zero-shot action generation. Demonstrates baseline LLM performance without RL fine-tuning.
- **AI suggestion endpoint** (`/agent/suggest`): LLM-powered action suggestions with rule-based guardrails for the interactive UI.

---

## Repository Structure

```
├── backend/
│   ├── main.py                        # ASGI entrypoint (uvicorn target)
│   ├── requirements.txt               # Python dependencies
│   └── src/polypharmacy_env/
│       ├── env_core.py                # OpenEnv environment: reset/step/state
│       ├── models.py                  # Typed Pydantic models (Action, Observation, State)
│       ├── rewards.py                 # Shaped reward function & regimen risk computation
│       ├── graders.py                 # Deterministic graders for 3 task difficulties
│       ├── tasks.py                   # Task configuration & episode sampling
│       ├── config.py                  # Reward hyperparameters & task parameters
│       ├── data_loader.py            # CSV data loading with caching
│       ├── ddi_simulator.py          # DDI lookup, Beers flags, drug substitution
│       ├── neural_bandits.py         # NeuralTS + Differential Evolution + OptimNeuralTS
│       ├── api/
│       │   ├── app.py                # FastAPI app factory via OpenEnv create_app
│       │   └── routes/agent.py       # POST /agent/suggest (AI-assisted actions)
│       │              bandit.py      # POST /bandit/predict, /bandit/screen
│       ├── baselines/
│       │   ├── heuristic_agent.py    # Deterministic baseline agent
│       │   └── random_agent.py       # Random baseline agent
│       ├── services/
│       │   └── groq_agent.py         # LLM-powered action suggestions
│       └── tests/
│           ├── test_env_core.py      # Environment unit tests
│           └── test_api.py           # HTTP + WebSocket integration tests
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # React control center UI
│   │   └── styles.css                # Production-quality dark theme
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── lookups/                      # drug_metadata.csv, ddi_rules.csv, beers_criteria.csv
│   └── processed/                    # patients_polypharmacy.csv (120 episodes)
├── scripts/
│   ├── preprocess_data.py            # Synthetic data generation
│   ├── dev_backend.sh                # Local backend runner
│   ├── dev_frontend.sh               # Local frontend runner
│   └── run_validation.sh             # Automated test + baseline validation
├── Dockerfile                         # Production multi-stage build (frontend + backend)
├── docker-compose.yml                # Development orchestration
├── inference.py                      # Submission baseline inference script
├── train_rl.py                       # REINFORCE RL training script (PyTorch)
├── train_bandit.py                   # OptimNeuralTS neural bandit training
├── openenv.yaml                      # OpenEnv manifest
└── .env.example                      # Environment variable template
```

---

## Action & Observation Spaces

### Actions

| Action Type | Parameters | Description |
|---|---|---|
| `query_ddi` | `drug_id_1`, `drug_id_2` | Check a drug pair for interactions. Returns severity, recommendation, and risk score. Costs 1 query budget. |
| `propose_intervention` | `target_drug_id`, `intervention_type`, `proposed_new_drug_id` (opt), `rationale` (opt) | Modify the medication regimen. Types: `stop`, `dose_reduce`, `substitute`, `add_monitoring`. Costs 1 intervention budget. |
| `finish_review` | — | End the episode. Triggers grader evaluation and returns final score. |

### Observations

Each observation contains the full patient context:

| Field | Type | Description |
|---|---|---|
| `episode_id` | string | Unique episode identifier |
| `task_id` | string | Current task (easy_screening / budgeted_screening / complex_tradeoff) |
| `age`, `sex` | int, string | Patient demographics |
| `conditions` | list[string] | Active medical conditions |
| `eGFR_category`, `liver_function_category` | string | Organ function status |
| `current_medications` | list[MedicationEntry] | Active drugs with dose, ATC class, Beers flags |
| `interaction_queries` | list[InteractionQueryRecord] | History of DDI queries and results |
| `interventions` | list[InterventionRecord] | History of proposed interventions |
| `remaining_query_budget` | int | Remaining DDI query budget |
| `remaining_intervention_budget` | int | Remaining intervention budget |
| `shaped_reward` | float | Step reward signal |
| `done` | bool | Whether the episode has ended |

---

## Tasks & Difficulty Progression

| Task | Difficulty | Drugs | Query Budget | Intervention Budget | Max Steps | Description |
|---|---|---|---|---|---|---|
| **Easy Screening** | Easy | 3–5 | 4 | 2 | 10 | Small regimen with one severe DDI. Identify and resolve it. |
| **Budgeted Screening** | Medium | 6–10 | 8 | 3 | 20 | Multiple DDIs and Beers issues under tighter budgets. Must prioritize effectively. |
| **Complex Tradeoff** | Hard | 10–15 | 12 | 5 | 30 | Large regimen with critical drugs (warfarin, insulin). Balance risk reduction against regimen disruption. |

### Grading Criteria

- **Easy**: 50% risk reduction + 50% targeted intervention on severe DDI drugs
- **Medium**: 50% risk reduction + 30% intervention precision + 20% query efficiency
- **Hard**: Risk reduction minus penalties for excessive drug changes and stopping critical medications without substitution

All graders are deterministic, producing scores in `[0.0, 1.0]`.

---

## Reward Function Design

The shaped reward provides signal at every step (not just episode end):

| Event | Reward |
|---|---|
| DDI query (any) | -0.015 (budget cost) |
| Discovering a severe DDI | +0.05 bonus |
| Discovering a moderate DDI | +0.02 bonus |
| Successful intervention | +(risk_reduction) - 0.025 cost |
| Invalid action | -0.15 penalty |
| Episode timeout | -0.25 penalty |
| Finish review | +grader_score (0.0–1.0) |

**Regimen risk** aggregates DDI pairwise scores, Beers-criteria violation weights, and high-risk elderly drug penalties, normalized by regimen size and clipped to `[0.0, 1.0]`.

---

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (20+ recommended)
- **Docker** + Docker Compose (for containerized runs)

---

## Setup & Local Development

### 1. Clone and configure

```bash
git clone <repo-url>
cd PolypharmacyEnv
cp .env.example .env
# Edit .env with your API keys if using the AI suggestion feature
```

### 2. Install dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 3. Generate synthetic data (if not already present)

```bash
python scripts/preprocess_data.py
```

### 4. Start services

**Terminal 1 — Backend** (port 7860):
```bash
./scripts/dev_backend.sh
```

**Terminal 2 — Frontend** (port 5173):
```bash
./scripts/dev_frontend.sh
```

### 5. Open the application

- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **Backend health check**: [http://localhost:7860/health](http://localhost:7860/health)

---

## Docker Deployment

### Build and run (single container — production mode)

```bash
docker build -t polypharmacy-env .
docker run -p 7860:7860 polypharmacy-env
```

The UI and API are both served from port 7860.

### Development mode (separate services)

```bash
docker compose up --build
```

- Backend: port 7860
- Frontend: port 5173

---

## Hugging Face Spaces Deployment

### 1. Create a new Space

- Go to [Hugging Face Spaces](https://huggingface.co/new-space)
- Choose **Docker** SDK
- Tag the Space with `openenv`

### 2. Set secrets and variables

In Space Settings → Variables and Secrets:

| Type | Key | Value |
|---|---|---|
| Secret | `HF_TOKEN` | Your Hugging Face API token |
| Variable | `API_BASE_URL` | `https://router.huggingface.co/v1` |
| Variable | `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |

### 3. Push the repository to the Space

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space master
```

### 4. Verify

- Space root URL loads the React UI
- `/health` returns healthy status
- `/reset`, `/step`, `/state` respond to API calls

---

## API Reference

### OpenEnv Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{ "task_id": "easy_screening" }` |
| `POST` | `/step` | Execute an action. Body: `{ "action": { "action_type": "query_ddi", ... } }` |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check |
| `GET` | `/schema` | Action/observation schema |
| `WS` | `/ws` | WebSocket for stateful multi-step sessions |

### Additional Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/agent/suggest` | AI-powered action suggestion. Body: `{ "observation": {...} }` |

---

## Running the Baseline Inference

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"

# Start the environment server (in another terminal)
./scripts/dev_backend.sh

# Run inference
python inference.py
```

The inference script runs all 3 tasks and emits structured `[START]`, `[STEP]`, `[END]` logs for the evaluator.

---

## RL Training (REINFORCE with Learned Baseline)

The repository includes `train_rl.py` — a complete **REINFORCE policy gradient** training loop that trains a neural network policy directly against the environment's shaped reward signal.

### How It Works

| Component | Description |
|---|---|
| **State encoder** | 16-dimensional feature vector: med count, high-risk drug count, Beers-flagged drugs, budget utilization, query outcomes (severe/moderate fractions), step progress, pair coverage |
| **Policy network** | 3-layer MLP (16 → 128 → 128 → 166) with ReLU, outputs masked logits over discrete action space |
| **Value baseline** | 3-layer MLP (16 → 128 → 64 → 1) trained with MSE against discounted returns |
| **Action space** | 166 discrete actions: 105 query_ddi pairs (C(15,2)), 60 interventions (4 types × 15 slots), 1 finish_review |
| **Action masking** | Invalid actions (exhausted budgets, already-queried pairs, empty drug slots) are masked to `-inf` before softmax |
| **Optimization** | REINFORCE with advantage (return - baseline), entropy bonus for exploration, gradient clipping |

### Training

```bash
# Install PyTorch (CPU is sufficient)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Train on easy task (fast, ~30s)
python train_rl.py --task easy_screening --episodes 200

# Train on medium task
python train_rl.py --task budgeted_screening --episodes 500

# Train on hard task (longer episodes)
python train_rl.py --task complex_tradeoff --episodes 500 --batch-size 10

# Full options
python train_rl.py --task easy_screening --episodes 200 \
  --lr 0.0003 --gamma 0.99 --entropy-coeff 0.02 \
  --hidden-dim 128 --batch-size 5 --print-every 10
```

**Outputs:**
- Policy checkpoints: `backend/src/polypharmacy_env/checkpoints/best_{task}.pt` and `final_{task}.pt`
- Training metrics: `training_metrics.json` (per-episode rewards, grader scores, losses)

### Observed Training Results

| Task | Episodes | Greedy Eval (Grader Score) | Stochastic Eval |
|---|---|---|---|
| Easy Screening | 200 | **0.698** | 0.475 |
| Budgeted Screening | 200 | **0.195** | 0.170 |
| Complex Tradeoff | 200 | **0.040** | 0.035 |

The easy task shows clear policy improvement. Medium and hard tasks benefit from more episodes (500+) and hyperparameter tuning — the larger action spaces and longer episodes create a harder credit assignment problem, exactly as designed.

### Integration with OpenEnv Training Pipeline

For production-scale training, this environment is compatible with **TRL's `GRPOTrainer`** via OpenEnv's standard interface:

```python
# Conceptual integration with TRL GRPO
from trl import GRPOTrainer
from openenv import GenericEnvClient

def rollout_func(prompts, trainer):
    env = GenericEnvClient("ws://localhost:7860/ws")
    # ... collect trajectories with token-level logprobs
    # ... return prompt_ids, completion_ids, logprobs, rewards

trainer = GRPOTrainer(model, rollout_function=rollout_func, ...)
trainer.train()
```

The included `train_rl.py` demonstrates the core RL loop with a lightweight MLP policy. For LLM-based policies, connect TRL/veRL/SkyRL to this environment via the WebSocket or HTTP interface.

---

## Neural Bandit Training (OptimNeuralTS)

The repository implements the **OptimNeuralTS** algorithm from the reference paper. This combines Neural Thompson Sampling with Differential Evolution to efficiently search for dangerous drug combinations in a large combinatorial space.

### How OptimNeuralTS Works

| Phase | What Happens |
|---|---|
| **Warm-up** | Randomly sample drug combinations and observe their risk scores to initialize the model's understanding |
| **Neural Thompson Sampling** | A neural network predicts risk for any drug combination, while gradient-based uncertainty drives exploration toward combinations that could be dangerous |
| **Differential Evolution** | Evolves a population of candidate drug combinations, guided by the neural network, to propose new combinations worth investigating |
| **Nearest-neighbor mapping** | Since DE can suggest combinations not in the dataset, we map to the closest real combination using Hamming distance |
| **Ensemble building** | Each training step saves a model snapshot; the final ensemble combines all snapshots for high-precision predictions |

### Key Components (in `neural_bandits.py`)

| Component | Description |
|---|---|
| `RewardNetwork` | Neural network that predicts the Relative Risk (RR) for a multi-hot drug combination vector |
| `NeuralTS` | Thompson Sampling agent using gradient-based uncertainty: `s_t(x) = sqrt(λ · g(x)^T · U^{-1} · g(x))` |
| `differential_evolution()` | DE best/1/bin optimization over multi-hot feature space |
| `OptimNeuralTS` | Full pipeline: warm-up → NeuralTS+DE exploration → ensemble building |

### Training

```bash
# Quick run (small dataset, fast)
python train_bandit.py --total-steps 500 --warmup-steps 100

# Full training (closer to paper settings)
python train_bandit.py --total-steps 3000 --warmup-steps 500 --n-combinations 10000

# Custom DE parameters
python train_bandit.py --de-population 32 --de-steps 16 --de-crossover 0.9

# All options
python train_bandit.py --help
```

**Outputs:**
- Ensemble model: `backend/src/polypharmacy_env/checkpoints/bandit_ensemble.pt`
- Training metrics: `bandit_metrics.json` (precision, recall, patterns detected at each eval step)

### API Endpoints

The trained ensemble is also accessible via API:

| Method | Path | Description |
|---|---|---|
| `POST` | `/bandit/predict` | Predict risk for a single drug combination |
| `POST` | `/bandit/screen` | Screen multiple combinations in bulk |
| `GET` | `/bandit/metrics` | Get current bandit training metrics |

---

## Testing & Validation

```bash
# Unit tests
python -m pytest backend/src/polypharmacy_env/tests -v

# Full validation (tests + heuristic baseline)
./scripts/run_validation.sh

# OpenEnv spec validation
openenv validate
```

---

## Data Sources & Future Plans

### Current Implementation

- **Drug interaction data**: Currently extracted from curated clinical databases and research literature, generating 24 DDI pairs across 33 drugs, 15 Beers criteria entries, and 120 patient episodes across 3 difficulty levels. Data is stored as CSV for deterministic, reproducible evaluation.
- **RL training**: A lightweight REINFORCE policy gradient training loop (`train_rl.py`) trains a neural network policy (MLP) directly against the environment's shaped reward signal. This validates the MDP formulation and demonstrates that the reward shaping enables genuine policy improvement. The trained policy achieves a 0.698 grader score on easy screening after 200 episodes.

### Planned Enhancements

- **Full-scale GRPO training on GPU**: We are provisioning AWS GPU resources (A100/H100 instances) to run full-scale GRPO (Group Relative Policy Optimization) training using TRL's `GRPOTrainer` with LLM-based policies. This will train language models to generate optimal clinical actions by collecting batched rollouts against the environment and computing policy gradient updates on token-level log-probabilities. The OpenEnv WebSocket interface enables high-throughput parallel rollout collection needed for efficient GRPO training.
- **LLM fine-tuning via OpenEnv training pipeline**: Integrate with TRL, veRL, and SkyRL frameworks to fine-tune open-weight LLMs (Llama 3, Qwen 2.5) using the environment's shaped reward as the RL training signal, producing specialized clinical pharmacist agents.
- **Live drug database integration**: Connect directly to established drug interaction databases (DrugBank, RxNorm, FDA Adverse Event Reporting System) for real-time DDI lookup instead of static CSV files, enabling the environment to scale to thousands of drug combinations.
- **EHR integration pipeline**: Develop FHIR-compatible data ingestion so the environment can accept de-identified electronic health record data, making it applicable to real hospital deployments.
- **Multi-agent training**: Extend the environment to support multi-agent scenarios where specialist agents (cardiologist, endocrinologist, etc.) must coordinate on a shared patient regimen.
- **Pharmacogenomics layer**: Incorporate genetic variant data (CYP450 metabolizer status) to personalize drug interaction severity, adding a pharmacogenomics dimension to the RL training signal.

---

## Architecture & Design Decisions

- **OpenEnv compliance**: Full typed Pydantic models for Action, Observation, and State. Environment extends `openenv.core.env_server.interfaces.Environment`.
- **Shaped rewards**: Continuous reward signal at every step to enable efficient RL training (not sparse end-of-episode only).
- **Budget constraints**: Query and intervention budgets create a resource-allocation problem that makes the RL optimization non-trivial.
- **Critical drug handling**: The hard task penalizes stopping critical medications (warfarin, insulin, etc.) without substitution, teaching the agent about real-world clinical constraints.
- **Deterministic graders**: All graders produce reproducible scores for consistent evaluation.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: polypharmacy_env` | Start backend via `./scripts/dev_backend.sh` from repo root |
| `/agent/suggest` returns errors | Check `.env` for valid API keys, restart backend |
| UI shows stale data | Hard refresh browser (Ctrl+Shift+R), click Reset Episode |
| Docker build fails | Ensure Docker has at least 4GB memory allocated |
| WebSocket connection refused | Verify backend is running on port 7860 |

---

## License

MIT
