"""PolypharmacyEnv – core environment implementing OpenEnv step / reset / state."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from .config import CRITICAL_DRUG_IDS, TaskConfig
from .data_loader import PatientEpisode
from .ddi_simulator import DDISimulator
from .graders import (
    grade_budgeted_screening,
    grade_complex_tradeoff,
    grade_easy_screening,
)
from .models import (
    InteractionQueryRecord,
    InterventionRecord,
    MedicationEntry,
    PolypharmacyAction,
    PolypharmacyObservation,
    PolypharmacyState,
)
from .rewards import compute_regimen_risk, compute_shaped_reward
from .tasks import get_task_config, sample_episode


class PolypharmacyEnv:
    """OpenEnv-compliant environment for elderly polypharmacy medication review."""

    def __init__(self) -> None:
        self._sim = DDISimulator()
        self._task_cfg: Optional[TaskConfig] = None
        self._episode: Optional[PatientEpisode] = None
        self._medications: List[MedicationEntry] = []
        self._interaction_queries: List[InteractionQueryRecord] = []
        self._interventions: List[InterventionRecord] = []
        self._risk_deltas: List[float] = []  # per-intervention risk improvement
        self._step_count: int = 0
        self._done: bool = True
        self._baseline_risk: float = 0.0
        self._current_risk: float = 0.0
        self._remaining_query_budget: int = 0
        self._remaining_intervention_budget: int = 0
        self._severe_moderate_discovered: int = 0
        self._total_drug_changes: int = 0
        self._critical_stopped_without_sub: int = 0
        self._last_reward: float = 0.0

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> PolypharmacyObservation:
        self._task_cfg = get_task_config(task_id)
        self._episode = sample_episode(task_id, seed=seed, episode_id=episode_id)

        # Build medication list
        self._medications = []
        for did in self._episode.medication_ids:
            meta = self._sim.get_drug_meta(did)
            if meta is None:
                continue
            flags = self._sim.get_beers_flags(did, self._episode.conditions)
            self._medications.append(MedicationEntry(
                drug_id=did,
                generic_name=meta.generic_name,
                atc_class=meta.atc_class,
                dose_mg=meta.default_dose_mg,
                is_high_risk_elderly=meta.is_high_risk_elderly,
                beers_flags=flags,
            ))

        self._interaction_queries = []
        self._interventions = []
        self._risk_deltas = []
        self._step_count = 0
        self._done = False
        self._remaining_query_budget = self._task_cfg.query_budget
        self._remaining_intervention_budget = self._task_cfg.intervention_budget
        self._severe_moderate_discovered = 0
        self._total_drug_changes = 0
        self._critical_stopped_without_sub = 0
        self._last_reward = 0.0

        # Compute baseline risk
        self._baseline_risk = self._compute_risk()
        self._current_risk = self._baseline_risk

        return self._make_observation()

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: PolypharmacyAction) -> Dict[str, Any]:
        if self._done:
            return self._terminal_response("Episode already finished.")

        assert self._task_cfg is not None
        assert self._episode is not None

        reward = 0.0
        info: Dict[str, Any] = {}

        # Validate basic action structure
        valid, err = self._validate_action(action)
        if not valid:
            reward = compute_shaped_reward(
                self._current_risk, self._current_risk,
                action.action_type, is_invalid=True,
            )
            info["error"] = err
            self._step_count += 1
            return self._check_timeout_and_respond(reward, info)

        if action.action_type == "query_ddi":
            reward, info = self._handle_query(action)

        elif action.action_type == "propose_intervention":
            reward, info = self._handle_intervention(action)

        elif action.action_type == "finish_review":
            self._done = True
            score = self._run_grader()
            reward = score  # terminal bonus
            info["grader_score"] = score

        self._step_count += 1
        return self._check_timeout_and_respond(reward, info)

    # ── state property ───────────────────────────────────────────────────────

    @property
    def state(self) -> PolypharmacyState:
        return PolypharmacyState(
            episode_id=self._episode.episode_id if self._episode else None,
            task_id=self._task_cfg.task_id if self._task_cfg else "",
            step_count=self._step_count,
            max_steps=self._task_cfg.max_steps if self._task_cfg else 0,
            num_query_actions=len(self._interaction_queries),
            num_interventions=len(self._interventions),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_risk(self) -> float:
        drug_ids = [m.drug_id for m in self._medications]
        return compute_regimen_risk(
            drug_ids,
            self._episode.conditions if self._episode else [],
            self._sim.ddi_rules,
            self._sim.beers_criteria,
            self._sim.drug_metadata,
        )

    def _validate_action(self, action: PolypharmacyAction) -> Tuple[bool, str]:
        if action.action_type == "query_ddi":
            if not action.drug_id_1 or not action.drug_id_2:
                return False, "query_ddi requires drug_id_1 and drug_id_2"
        elif action.action_type == "propose_intervention":
            if not action.target_drug_id:
                return False, "propose_intervention requires target_drug_id"
            if action.intervention_type in (None, "none"):
                return False, "propose_intervention requires a valid intervention_type"
        return True, ""

    def _handle_query(self, action: PolypharmacyAction) -> Tuple[float, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        assert action.drug_id_1 and action.drug_id_2

        if self._remaining_query_budget <= 0:
            reward = compute_shaped_reward(
                self._current_risk, self._current_risk,
                "query_ddi", is_invalid=True,
            )
            info["error"] = "Query budget exhausted"
            return reward, info

        result = self._sim.lookup_ddi(action.drug_id_1, action.drug_id_2)
        self._remaining_query_budget -= 1

        self._interaction_queries.append(InteractionQueryRecord(
            drug_id_1=action.drug_id_1,
            drug_id_2=action.drug_id_2,
            severity=result.severity,
            recommendation=result.recommendation,
            risk_score=result.base_risk_score,
            step_index=self._step_count,
        ))

        discovered_severe = result.severity in ("severe", "moderate")
        if discovered_severe:
            self._severe_moderate_discovered += 1

        reward = compute_shaped_reward(
            self._current_risk, self._current_risk,
            "query_ddi",
            discovered_severe=(result.severity == "severe"),
        )
        info["ddi_result"] = {
            "severity": result.severity,
            "recommendation": result.recommendation,
            "risk_score": result.base_risk_score,
        }
        return reward, info

    def _handle_intervention(self, action: PolypharmacyAction) -> Tuple[float, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        assert action.target_drug_id
        assert action.intervention_type and action.intervention_type != "none"

        if self._remaining_intervention_budget <= 0:
            reward = compute_shaped_reward(
                self._current_risk, self._current_risk,
                "propose_intervention", is_invalid=True,
            )
            info["error"] = "Intervention budget exhausted"
            return reward, info

        # Find target medication
        target_idx: Optional[int] = None
        for i, m in enumerate(self._medications):
            if m.drug_id == action.target_drug_id:
                target_idx = i
                break

        if target_idx is None:
            reward = compute_shaped_reward(
                self._current_risk, self._current_risk,
                "propose_intervention", is_invalid=True,
            )
            info["error"] = f"Drug {action.target_drug_id} not in current medications"
            return reward, info

        previous_risk = self._current_risk
        target_med = self._medications[target_idx]

        if action.intervention_type == "stop":
            self._medications.pop(target_idx)
            self._total_drug_changes += 1
            if action.target_drug_id in CRITICAL_DRUG_IDS:
                self._critical_stopped_without_sub += 1

        elif action.intervention_type == "dose_reduce":
            meta = self._sim.get_drug_meta(action.target_drug_id)
            if meta:
                new_dose = max(meta.min_dose_mg, target_med.dose_mg * 0.5)
                self._medications[target_idx] = target_med.model_copy(
                    update={"dose_mg": new_dose}
                )

        elif action.intervention_type == "substitute":
            new_drug_id = action.proposed_new_drug_id
            if not new_drug_id:
                # Auto-find substitute
                current_ids = [m.drug_id for m in self._medications]
                new_drug_id = self._sim.find_substitute(action.target_drug_id, current_ids)
            if new_drug_id:
                new_meta = self._sim.get_drug_meta(new_drug_id)
                if new_meta:
                    flags = self._sim.get_beers_flags(
                        new_drug_id,
                        self._episode.conditions if self._episode else [],
                    )
                    self._medications[target_idx] = MedicationEntry(
                        drug_id=new_drug_id,
                        generic_name=new_meta.generic_name,
                        atc_class=new_meta.atc_class,
                        dose_mg=new_meta.default_dose_mg,
                        is_high_risk_elderly=new_meta.is_high_risk_elderly,
                        beers_flags=flags,
                    )
                    self._total_drug_changes += 1
                    # If critical drug was substituted, don't penalise
                    if action.target_drug_id in CRITICAL_DRUG_IDS:
                        pass  # substitution is acceptable
                else:
                    info["warning"] = f"Substitute {new_drug_id} not found in metadata"
                    # Don't consume budget for a failed substitute
                    self._remaining_intervention_budget += 1
            else:
                info["warning"] = "No suitable substitute found"
                # Don't consume budget for a failed substitute
                self._remaining_intervention_budget += 1

        elif action.intervention_type == "add_monitoring":
            # Tag in metadata but don't change regimen
            self._medications[target_idx] = target_med.model_copy(
                update={"beers_flags": target_med.beers_flags + ["monitored"]}
            )

        self._remaining_intervention_budget -= 1
        self._current_risk = self._compute_risk()
        risk_delta = previous_risk - self._current_risk
        self._risk_deltas.append(risk_delta)

        self._interventions.append(InterventionRecord(
            target_drug_id=action.target_drug_id,
            action_type=action.intervention_type,
            proposed_new_drug_id=action.proposed_new_drug_id,
            rationale=action.rationale or "",
            step_index=self._step_count,
        ))

        reward = compute_shaped_reward(previous_risk, self._current_risk, "propose_intervention")
        info["risk_delta"] = risk_delta
        return reward, info

    def _run_grader(self) -> float:
        assert self._task_cfg is not None
        tid = self._task_cfg.task_id

        if tid == "easy_screening":
            severe_pairs = self._get_severe_pairs()
            return grade_easy_screening(
                self._baseline_risk,
                self._current_risk,
                self._interventions,
                severe_pairs,
            )
        elif tid == "budgeted_screening":
            return grade_budgeted_screening(
                self._baseline_risk,
                self._current_risk,
                self._interventions,
                self._risk_deltas,
                len(self._interaction_queries),
                self._severe_moderate_discovered,
            )
        elif tid == "complex_tradeoff":
            return grade_complex_tradeoff(
                self._baseline_risk,
                self._current_risk,
                self._interventions,
                self._total_drug_changes,
                self._critical_stopped_without_sub,
            )
        return 0.0

    def _get_severe_pairs(self) -> List[Tuple[str, str]]:
        """Return all severe DDI pairs present in the *initial* medication list."""
        if not self._episode:
            return []
        pairs: List[Tuple[str, str]] = []
        med_ids = self._episode.medication_ids
        for a, b in combinations(sorted(set(med_ids)), 2):
            key = (a, b) if a < b else (b, a)
            rule = self._sim.ddi_rules.get(key)
            if rule and rule.severity == "severe":
                pairs.append(key)
        return pairs

    def _check_timeout_and_respond(
        self, reward: float, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert self._task_cfg is not None

        if not self._done and self._step_count >= self._task_cfg.max_steps:
            self._done = True
            timeout_penalty = compute_shaped_reward(
                self._current_risk, self._current_risk,
                "finish_review", is_timeout=True,
            )
            score = self._run_grader()
            reward += timeout_penalty + score
            info["timeout"] = True
            info["grader_score"] = score

        self._last_reward = reward
        info["current_risk"] = self._current_risk
        info["baseline_risk"] = self._baseline_risk

        obs = self._make_observation(reward=reward)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": self._done,
            "info": info,
        }

    def _terminal_response(self, msg: str) -> Dict[str, Any]:
        obs = self._make_observation()
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": True,
            "info": {"error": msg},
        }

    def _make_observation(self, reward: float = 0.0) -> PolypharmacyObservation:
        ep = self._episode
        cfg = self._task_cfg
        return PolypharmacyObservation(
            episode_id=ep.episode_id if ep else "",
            task_id=cfg.task_id if cfg else "budgeted_screening",
            age=ep.age if ep else 65,
            sex=ep.sex if ep else "M",
            conditions=ep.conditions if ep else [],
            eGFR_category=ep.eGFR_category if ep else "normal",
            liver_function_category=ep.liver_function_category if ep else "normal",
            current_medications=deepcopy(self._medications),
            interaction_queries=deepcopy(self._interaction_queries),
            interventions=deepcopy(self._interventions),
            step_index=self._step_count,
            remaining_query_budget=self._remaining_query_budget,
            remaining_intervention_budget=self._remaining_intervention_budget,
            shaped_reward=reward,
            done=self._done,
            reward=reward,
        )
