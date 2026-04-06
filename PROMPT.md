You are an expert Python backend, ML, and infrastructure engineer.
Your task is to implement a complete, production-ready OpenEnv environment called **PolypharmacyEnv** for training and evaluating agentic RL policies that act as an "elderly polypharmacy safety agent" (clinical pharmacist assistant).

The deliverable MUST satisfy all of the following:
- Fully compliant with the OpenEnv spec (typed models, `step()` / `reset()` / `state()`, `openenv.yaml`, HTTP server, Dockerfile).
- Simulates a realistic healthcare workflow around elderly polypharmacy and dangerous drug combinations.
- Defines at least **3 tasks** (easy → medium → hard) with deterministic agent graders producing scores in (0.0, 1.0).
- Provides shaped rewards over the trajectory (not just sparse terminal rewards).
- Includes a baseline LLM-based inference script `inference.py` in the repo root, following the evaluation requirements:
  - Uses the OpenAI Python client.
  - Reads `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment.
  - Emits structured stdout logs in the exact `[START]`, `[STEP]`, `[END]` format from the OpenEnv sample inference script.
- Is containerized and deployable as a **Hugging Face Space** tagged with `openenv` that responds to OpenEnv-style `reset` / `step` / `state` HTTP calls.

Implement everything described below.

=================================================
1. Repository and folder structure
=================================================

Create a Python package repository with this structure (names are important unless clearly labeled as examples):

- `openenv-polypharmacy/`
  - `openenv.yaml`
  - `README.md`
  - `requirements.txt`
  - `Dockerfile`
  - `inference.py`                 # baseline LLM agent per spec
  - `pyproject.toml` or `setup.cfg` (optional but recommended)
  - `src/`
    - `polypharmacy_env/`
      - `__init__.py`
      - `config.py`
      - `models.py`                # Action, Observation, State, helper models
      - `env_core.py`              # PolypharmacyEnv implementation
      - `tasks.py`                 # task setup utilities
      - `graders.py`               # deterministic graders for each task
      - `rewards.py`               # reward shaping logic
      - `data_loader.py`           # load/preprocess patient and lookup data
      - `ddi_simulator.py`         # local DDI / guideline simulator
      - `api/`
        - `__init__.py`
        - `schemas.py`            # HTTP request/response schemas
        - `server.py`             # FastAPI app exposing OpenEnv endpoints
      - `baselines/`
        - `__init__.py`
        - `heuristic_agent.py`    # simple rule-based baseline agent
        - `random_agent.py`       # trivial random baseline (optional)
      - `tests/`
        - `__init__.py`
        - `test_env_core.py`
        - `test_api.py`
  - `data/`
    - `raw/`                      # placeholder for real/synthetic source data
    - `processed/`
    - `lookups/`
      - `ddi_rules.csv`
      - `beers_criteria.csv`
      - `drug_metadata.csv`
  - `scripts/`
    - `preprocess_data.py`
    - `run_validation.sh`         # optional; runs OpenEnv validator, tests, etc.

Use Python 3.10+ with full type hints, and keep the code black/isort-compatible.

=================================================
2. Domain, data, and clinical abstraction
=================================================

2.1. Core scenario

Model an elderly patient (age ≥ 65) with:
- Demographics: age, sex.
- Comorbidities: e.g., hypertension, diabetes, heart failure, CKD, dementia.
- Basic labs: kidney function (eGFR category), liver function category.
- A current medication list (polypharmacy, e.g., 3–15 drugs depending on task).

Each **episode** is one medication-review session where the agent:
- Observes patient info and current meds.
- Optionally **queries** a DDI/guideline tool for specific drug pairs.
- Proposes **interventions**:
  - `stop`: discontinue a drug.
  - `dose_reduce`: lower dose of a drug.
  - `substitute`: swap to a safer alternative.
  - `add_monitoring`: keep the drug but flag extra monitoring.
- Calls `finish_review` when it decides the regimen is acceptable or budgets are exhausted.

No external PHI, EHRs, or online APIs: all data is **synthetic** or de-identified and local to the container (CSV files).

2.2. Data files and CSV schemas

Implement local CSVs under `data/lookups/`:

**`drug_metadata.csv`**
- `drug_id` (string; unique key)
- `generic_name` (string)
- `atc_class` (string)
- `is_high_risk_elderly` (0/1)
- `default_dose_mg` (float)
- `min_dose_mg` (float)
- `max_dose_mg` (float)

**`beers_criteria.csv`**
- `drug_id` (string)
- `criterion_type` (enum string: `avoid`, `caution`, `dose_adjust`, `avoid_in_condition`)
- `condition` (nullable string; e.g., `CKD`, `dementia`)
- `rationale` (brief text)

**`ddi_rules.csv`**
- `drug_id_1` (string; normalized so `drug_id_1 < drug_id_2` lexicographically)
- `drug_id_2` (string)
- `severity` (enum string: `mild`, `moderate`, `severe`)
- `mechanism` (short text)
- `recommendation` (enum string: `avoid_combination`, `monitor_closely`, `dose_adjust`, `no_action`)
- `base_risk_score` (float in [0.0, 1.0])

Implement a synthetic patient-episode dataset under `data/processed/`:

**`patients_polypharmacy.csv`**
- `episode_id` (string)
- `age` (int)
- `sex` (enum: `M`, `F`, `O`)
- `conditions` (semicolon-separated; e.g., `HTN;DM;CKD`)
- `eGFR_category` (enum: `normal`, `mild`, `moderate`, `severe`)
- `liver_function_category` (enum: `normal`, `impaired`)
- `medication_ids` (semicolon-separated list of `drug_id`)
- `baseline_risk_score` (float in [0.0, 1.0])

2.3. Preprocessing script

In `scripts/preprocess_data.py`:
- If real data is not provided, procedurally generate synthetic but plausible data using:
  - Random combinations of conditions and drugs constrained by simple rules (e.g., CKD + renally-cleared drugs).
  - Controlled distribution of high-risk DDIs and Beers violations.
- Explicitly tag episodes as easy/medium/hard (e.g., via number of drugs, number/severity of DDIs, and number of Beers issues).
- Save `patients_polypharmacy.csv` ready for the environment to consume.

=================================================
3. OpenEnv models and environment implementation
=================================================

3.1. Models

In `models.py`, define dataclasses or Pydantic models that extend the appropriate OpenEnv base types (`Action`, `Observation`, `State`) and are JSON-compatible.

Auxiliary models:

**`MedicationEntry`**
- `drug_id: str`
- `generic_name: str`
- `atc_class: str`
- `dose_mg: float`
- `frequency: str`          # e.g., `qd`, `bid`
- `route: str`              # e.g., `po`
- `is_high_risk_elderly: bool`
- `beers_flags: list[str]`  # e.g., `["avoid", "dose_adjust_CKD"]`

**`InteractionQueryRecord`**
- `drug_id_1: str`
- `drug_id_2: str`
- `severity: str | None`
- `recommendation: str | None`
- `risk_score: float | None`
- `step_index: int`

**`InterventionRecord`**
- `target_drug_id: str`
- `action_type: Literal["stop", "dose_reduce", "substitute", "add_monitoring"]`
- `proposed_new_drug_id: str | None`
- `rationale: str`
- `step_index: int`

Core wire models:

**`PolypharmacyObservation`** (extends OpenEnv `Observation`)
- `episode_id: str`
- `task_id: Literal["easy_screening", "budgeted_screening", "complex_tradeoff"]`
- `age: int`
- `sex: str`
- `conditions: list[str]`
- `eGFR_category: str`
- `liver_function_category: str`
- `current_medications: list[MedicationEntry]`
- `interaction_queries: list[InteractionQueryRecord]`
- `interventions: list[InterventionRecord]`
- `step_index: int`
- `remaining_query_budget: int`
- `remaining_intervention_budget: int`
- `shaped_reward: float`  # reward from last step
- `done: bool`

**`PolypharmacyAction`** (extends OpenEnv `Action`)
- `action_type: Literal["query_ddi", "propose_intervention", "finish_review"]`
- `drug_id_1: str | None`        # for DDI queries or some interventions
- `drug_id_2: str | None`        # for DDI queries
- `target_drug_id: str | None`   # for interventions
- `intervention_type: Literal["stop", "dose_reduce", "substitute", "add_monitoring", "none"] | None`
- `proposed_new_drug_id: str | None`
- `rationale: str | None`

**`PolypharmacyState`** (extends OpenEnv `State`)
- `episode_id: str`
- `task_id: str`
- `step_count: int`
- `max_steps: int`
- `num_query_actions: int`
- `num_interventions: int`

3.2. Environment core

In `env_core.py`, implement `PolypharmacyEnv` extending the appropriate OpenEnv environment base class. It must implement:

**`reset(task_id: str | None = None) -> PolypharmacyObservation`**
- If `task_id` is `None`, default to medium (`budgeted_screening`).
- Sample an episode from `patients_polypharmacy.csv` filtered by difficulty.
- Initialize:
  - `episode_id`
  - `step_count = 0`
  - task-specific budgets (query, interventions, max_steps)
  - baseline regime and risk
  - empty `interaction_queries` and `interventions`
- Return the initial `PolypharmacyObservation` with:
  - `step_index = 0`
  - `shaped_reward = 0.0`
  - `done = False`

**`step(action: PolypharmacyAction) -> dict`**
- Validate the action; if invalid:
  - Apply a negative reward.
  - Do not modify regimen, but log error in `info`.
- If `action_type == "query_ddi"`:
  - If query budget exhausted, apply penalty and do not query.
  - Else:
    - Use `ddi_simulator.lookup_ddi(drug_id_1, drug_id_2)` to get severity, recommendation, base_risk_score.
    - Append an `InteractionQueryRecord`.
    - Apply a small negative reward for query cost.
- If `action_type == "propose_intervention"`:
  - If intervention budget exhausted, apply penalty and ignore change.
  - Else:
    - Update `current_medications` according to `intervention_type`:
      - `stop`: remove medication.
      - `dose_reduce`: adjust dose downward within [min_dose_mg, default_dose_mg].
      - `substitute`: replace with a safer alternative from same `atc_class`.
      - `add_monitoring`: keep drug but tag in internal state.
    - Append an `InterventionRecord`.
    - Recompute current regimen risk using the risk model (see 3.3).
    - Compute shaped reward = (previous_risk - new_risk) - small intervention cost.
- If `action_type == "finish_review"`:
  - Mark `done = True`.
  - Call the task’s grader to get episode-level score in [0.0, 1.0].
  - Add this as a terminal bonus to the current step reward.

- In all cases:
  - Increment `step_count`.
  - Check `max_steps`; if exceeded, auto-terminate:
    - `done = True`
    - apply time-out penalty
    - call grader with current trajectory for a final score if appropriate.
  - Construct next `PolypharmacyObservation` with updated fields.
  - Return a dict:
    - `observation`: `PolypharmacyObservation`
    - `reward`: float shaped reward for this step
    - `done`: bool
    - `info`: dict with fields like `current_risk`, `baseline_risk`, `grader_score_if_terminal`, and debug flags.

**`state` property**
- Returns `PolypharmacyState` reflecting the current internal state.

3.3. DDI simulator and risk model

In `ddi_simulator.py`:
- Load `ddi_rules.csv` once via `data_loader`.
- Implement `lookup_ddi(drug_id_1, drug_id_2) -> tuple[severity, recommendation, base_risk_score]`:
  - Normalize the pair ordering.
  - Look up row; if missing, return:
    - severity = `"none"`
    - recommendation = `"no_action"`
    - base_risk_score = 0.0

In `rewards.py` (or a dedicated module), implement:
- `compute_regimen_risk(current_drug_ids, patient_context, ddi_rules, beers_rules, drug_metadata) -> float`
  - Aggregate contributions from:
    - Beers violations (weighted by `criterion_type` and relevant conditions).
    - DDI base risk scores for all present drug pairs.
    - High-risk elderly drugs.
  - Normalize and clip to [0.0, 1.0].

Use this function to compute:
- `baseline_risk` at episode start.
- Risk after each intervention step.

Also implement:
- `compute_shaped_reward(previous_risk, new_risk, action, context, partial_metrics) -> float`
  - Positive component: `previous_risk - new_risk`.
  - Negative components: per-query cost, per-intervention cost, invalid-action penalty, time-out penalty.

=================================================
4. Tasks and graders (3 difficulty levels)
=================================================

Define three task IDs and semantics in `tasks.py` and `graders.py`:

Task IDs:
- `easy_screening`
- `budgeted_screening`
- `complex_tradeoff`

4.1. `easy_screening` (easy)

- Small regimen: 3–5 drugs.
- Exactly one **severe** DDI pair and possibly one simple Beers violation.
- Budgets:
  - query_budget ≈ 4
  - intervention_budget ≈ 2
  - max_steps ≈ 10

Grader:
- Input: full trajectory, baseline risk, final risk, list of interventions.
- Compute:
  - `risk_reduction = max(0.0, baseline_risk - final_risk) / max(baseline_risk, ε)` (normalized).
  - `targeted_intervention_flag = 1.0` if at least one intervention affects one of the drugs in the known severe DDI pair, else 0.0.
- Score:
  - `score = 0.5 * risk_reduction + 0.5 * targeted_intervention_flag`
  - Clip to [0.0, 1.0].

4.2. `budgeted_screening` (medium)

- Medium regimen: 6–10 drugs.
- Multiple DDIs (mild/moderate/severe) and multiple Beers issues.
- Budgets:
  - query_budget ≈ 8
  - intervention_budget ≈ 3
  - max_steps ≈ 20

Grader:
- Compute:
  - `risk_reduction_score` as normalized risk drop.
  - `intervention_precision_score` = fraction of interventions that actually reduce risk or fix guideline violations.
  - `query_efficiency_score` = (number of severe/moderate DDIs discovered) / (number of queries used), normalized.
- Weighted score, for example:
  - `score = 0.5 * risk_reduction_score + 0.3 * intervention_precision_score + 0.2 * query_efficiency_score`
  - Clip to [0.0, 1.0].

4.3. `complex_tradeoff` (hard)

- Larger regimen: 10–15 drugs.
- Some drugs are **clinically critical** (e.g., anticoagulants, insulin analogues) and encoded as such in `drug_metadata` or a small internal map.
- Episodes contain:
  - multiple DDIs and Beers issues, including ones involving critical drugs.
  - safer substitutes for some risky drugs.

Budgets:
- query_budget ≈ 12
- intervention_budget ≈ 5
- max_steps ≈ 30

Grader adds a **regimen disruption penalty** component:
- Metrics:
  - `risk_reduction_score` (as above).
  - `critical_drug_penalty` = penalty if a critical drug is stopped without substitution to another suitable agent.
  - `total_drug_changes` = number of drugs stopped or substituted.
  - `regimen_disruption_penalty` derived from `total_drug_changes` and `critical_drug_penalty`.

Example scoring:
- `base = risk_reduction_score`
- `penalty = α * regimen_disruption_penalty`
- `score = clamp(base - penalty, 0.0, 1.0)`

4.4. Reward shaping

In `rewards.py`, define a consistent shaping scheme:
- On each query:
  - Small negative reward (e.g., −0.01) plus any small bonus if it discovers a severe DDI, if desired.
- On each intervention:
  - Reward ≈ (previous_risk - new_risk) − small intervention cost.
- On invalid actions:
  - Larger negative reward (e.g., −0.1) and no state change.
- On `finish_review`:
  - Add the task-level `score` ∈ [0.0, 1.0] from the corresponding grader to that step’s shaped reward.

Ensure the sum of step rewards per episode remains in a reasonable numeric range (e.g., roughly -5 to +5) while still allowing meaningful differentiation by graders.

=================================================
5. HTTP API server and openenv.yaml
=================================================

5.1. HTTP server (FastAPI)

In `api/server.py`:
- Implement a FastAPI app that maintains a `PolypharmacyEnv` instance (or a multiplexing scheme if needed).
- Endpoints:
  - `POST /reset`:
    - Request body: may include `task_id` (string).
    - Response: serialized `PolypharmacyObservation`.
  - `POST /step`:
    - Request body: serialized `PolypharmacyAction`.
    - Response: dict with:
      - `observation`: `PolypharmacyObservation`
      - `reward`: float
      - `done`: bool
      - `info`: dict
  - `GET /state`:
    - Response: `PolypharmacyState`.

Provide a module-level `app = FastAPI(...)` object for use with uvicorn and Hugging Face Spaces. Ensure the JSON schema is consistent with OpenEnv clients (simple, flat JSON for observation/action/state).

5.2. `openenv.yaml`

At repo root, define `openenv.yaml` consistent with the latest OpenEnv spec. At minimum, include:
- `name`: `polypharmacy_env`
- `version`: e.g., `0.1.0`
- `description`: human-readable description.
- `author`: your details.
- `tags`: e.g., `["healthcare", "polypharmacy", "openenv"]`
- `tasks`:
  - One entry per task:
    - `id`: `"easy_screening"` / `"budgeted_screening"` / `"complex_tradeoff"`
    - `description`: one-line description
    - `difficulty`: `"easy"`, `"medium"`, `"hard"`

Ensure `openenv validate` (or equivalent validator) passes once implemented.

=================================================
6. Baseline heuristic (non-LLM) agent
=================================================

In `baselines/heuristic_agent.py`, implement a simple, deterministic baseline agent that:

For each episode:
- Iterates through all unordered medication pairs within query budget:
  - Calls `query_ddi` via the environment for each pair until the query budget is exhausted or all pairs are examined.
  - Records severe and moderate interactions.
- After querying:
  - For each severe DDI pair:
    - Try `substitute` one of the drugs using `drug_metadata`:
      - Prefer substitute within same `atc_class` that:
        - is not marked high-risk elderly.
        - does not participate in known severe DDIs with the rest of the regimen.
    - If no substitute exists, propose `stop` for the higher-risk drug.
  - Respect intervention budget limits.
- Finally, call `finish_review`.

This baseline should be callable as a simple Python function that interacts with `PolypharmacyEnv` directly (without HTTP).

=================================================
7. Baseline LLM inference script (inference.py)
=================================================

At repo root, create `inference.py` that:

7.1. Uses the OpenAI Python client

- Import and configure the official OpenAI Python client.
- Read environment variables:
  - `OPENAI_API_KEY` (required).
  - `API_BASE_URL` (base URL for LLM; default to OpenAI standard if not set).
  - `MODEL_NAME` (e.g., `gpt-4.1` or similar).
  - `HF_TOKEN` (if needed for HF auth; do not hardcode).
- Read `POLYPHARMACY_ENV_URL` (or similar) for the environment’s HTTP base URL.

7.2. Implements the required logging format

- For each **run** across all tasks:
  - Emit a `[START]` line with a JSON payload exactly matching the evaluation specification:
    - Fields such as `run_id`, `task_id`, `model`, etc., in the same order and naming as the sample OpenEnv inference script.
- For each **step** in an episode:
  - Emit a `[STEP]` line with JSON fields including:
    - `run_id`
    - `task_id`
    - `episode_id`
    - `step_index`
    - `observation_summary` (brief, machine-readable summary)
    - `action_payload` (the action sent to the env)
    - `reward`
    - `done`
- After finishing an episode for a task:
  - Emit an `[END]` line summarizing:
    - `run_id`
    - `task_id`
    - per-episode statistics (e.g., total reward, grader score from last step’s `info`).
- The stdout format MUST follow the sample exactly:
  - Same tags: `[START]`, `[STEP]`, `[END]`.
  - Same JSON field names and ordering as the provided reference.
  - No extra prints except these structured logs (and necessary error messages to stderr).

7.3. LLM agent loop

- For each task (`easy_screening`, `budgeted_screening`, `complex_tradeoff`):
  - Run a fixed small number of episodes (e.g., 5–10 per task) for baseline scoring.
  - For each episode:
    - Call `/reset` with the task id.
    - At each step:
      - Summarize the observation into a concise prompt for the LLM:
        - Include age, sex, conditions, high-risk flags, budgets, and a compressed view of meds and previous actions.
      - Ask the model to output a **strict JSON** representing `PolypharmacyAction` fields.
      - Parse and validate the JSON; if invalid, fall back to a safe default (e.g., `finish_review` or a no-op) and penalize in evaluation.
      - Send this action to `/step` and log `[STEP]`.
    - End when `done=True` or max_steps is reached.
- At the end, print aggregate scores per task and overall.

Make sure runtime < 20 minutes and that the script can run within 2 vCPUs and 8 GB RAM.

=================================================
8. Dockerfile and Hugging Face Space
=================================================

8.1. Dockerfile

Create a `Dockerfile` that:
- Starts from a slim Python image (e.g., `python:3.11-slim`).
- Installs system dependencies as needed (e.g., `build-essential`, `curl`).
- Copies the project into the container.
- Installs Python dependencies from `requirements.txt`.
- Sets appropriate environment variables for the app (e.g., `PORT=7860`).
- Exposes port 7860.
- Uses a `CMD` or `ENTRYPOINT` that runs the FastAPI server, for example:
  - `uvicorn polypharmacy_env.api.server:app --host 0.0.0.0 --port 7860`

8.2. Hugging Face Space

Ensure the repository is ready to be used as a Hugging Face Space:
- Space type: `docker`.
- Tag: `openenv`.
- On container start, the server must listen on the correct port and respond to:
  - `POST /reset`
  - `POST /step`
  - `GET /state`
- The environment must start cleanly with `docker build` + `docker run` locally.

=================================================
9. README and documentation
=================================================

In `README.md`, include:

- **Environment description & motivation**:
  - What PolypharmacyEnv simulates.
  - Why elderly polypharmacy safety matters.
- **Action and observation spaces**:
  - Describe `PolypharmacyAction`, `PolypharmacyObservation`, and `PolypharmacyState` fields and semantics.
- **Task descriptions**:
  - `easy_screening`, `budgeted_screening`, `complex_tradeoff`, their difficulty and goals.
- **Reward structure**:
  - Summarize shaping and terminal rewards.
- **Setup & usage**:
  - How to install dependencies.
  - How to run the API server locally (uvicorn command).
  - How to run the heuristic baseline.
  - How to run `inference.py` with environment variables.
- **Baseline scores**:
  - Document reproducible baseline scores for each task (heuristic agent, and LLM baseline if available).

=================================================
10. Validation and quality gates
=================================================

- Ensure:
  - `openenv.yaml` and the HTTP server pass the OpenEnv validation script.
  - `docker build` and `docker run` work without errors.
  - `inference.py` completes under 20 minutes, within 2 vCPUs / 8 GB RAM.
  - All graders:
    - Are deterministic.
    - Return scores strictly in [0.0, 1.0].
  - No grader returns a constant score irrespective of behavior.

Aim for clean, well-structured, well-documented code with clear separation of concerns between:
- Data loading,
- Environment state & dynamics,
- Reward/grade logic,
- HTTP serving,
- Baseline agents and inference.