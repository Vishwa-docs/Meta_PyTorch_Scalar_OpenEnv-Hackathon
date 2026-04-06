"""Tests for the PolypharmacyEnv core."""

from __future__ import annotations

import pytest

from polypharmacy_env.env_core import PolypharmacyEnv
from polypharmacy_env.models import (
    PolypharmacyAction,
    PolypharmacyObservation,
    PolypharmacyState,
)


class TestReset:
    def test_reset_returns_observation(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="easy_screening", seed=42)
        assert isinstance(obs, PolypharmacyObservation)
        assert obs.done is False
        assert obs.step_index == 0
        assert len(obs.current_medications) >= 3

    def test_reset_medium(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="budgeted_screening", seed=0)
        assert obs.remaining_query_budget == 8
        assert obs.remaining_intervention_budget == 3
        assert len(obs.current_medications) >= 6

    def test_reset_hard(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="complex_tradeoff", seed=0)
        assert obs.remaining_query_budget == 12
        assert obs.remaining_intervention_budget == 5
        assert len(obs.current_medications) >= 10

    def test_default_task(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(seed=0)
        assert obs.task_id == "budgeted_screening"


class TestStep:
    def test_query_ddi(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="easy_screening", seed=42)
        meds = obs.current_medications
        assert len(meds) >= 2

        action = PolypharmacyAction(
            action_type="query_ddi",
            drug_id_1=meds[0].drug_id,
            drug_id_2=meds[1].drug_id,
        )
        obs = env.step(action)
        assert isinstance(obs, PolypharmacyObservation)
        assert obs.done is False
        assert obs.step_index == 1
        assert len(obs.interaction_queries) == 1

    def test_invalid_action_penalised(self) -> None:
        env = PolypharmacyEnv()
        env.reset(task_id="easy_screening", seed=42)

        action = PolypharmacyAction(
            action_type="propose_intervention",
            target_drug_id=None,
            intervention_type=None,
        )
        obs = env.step(action)
        assert obs.reward is not None
        assert obs.reward < 0  # penalty

    def test_finish_review(self) -> None:
        env = PolypharmacyEnv()
        env.reset(task_id="easy_screening", seed=42)

        action = PolypharmacyAction(action_type="finish_review")
        obs = env.step(action)
        assert obs.done is True
        assert "grader_score" in obs.metadata
        score = obs.metadata["grader_score"]
        assert 0.0 <= score <= 1.0

    def test_intervention_stop(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="easy_screening", seed=42)
        target = obs.current_medications[0].drug_id
        n_meds = len(obs.current_medications)

        action = PolypharmacyAction(
            action_type="propose_intervention",
            target_drug_id=target,
            intervention_type="stop",
            rationale="test stop",
        )
        obs = env.step(action)
        assert len(obs.current_medications) == n_meds - 1

    def test_budget_exhaustion(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="easy_screening", seed=42)
        meds = obs.current_medications

        # Exhaust query budget (4 for easy)
        for i in range(4):
            a_idx = i % len(meds)
            b_idx = (i + 1) % len(meds)
            action = PolypharmacyAction(
                action_type="query_ddi",
                drug_id_1=meds[a_idx].drug_id,
                drug_id_2=meds[b_idx].drug_id,
            )
            obs = env.step(action)
            if obs.done:
                break

        if not obs.done:
            assert obs.remaining_query_budget == 0
            # Trying another query should be penalised
            action = PolypharmacyAction(
                action_type="query_ddi",
                drug_id_1=meds[0].drug_id,
                drug_id_2=meds[1].drug_id,
            )
            obs = env.step(action)
            assert obs.reward is not None
            assert obs.reward < 0

    def test_max_steps_timeout(self) -> None:
        env = PolypharmacyEnv()
        obs = env.reset(task_id="easy_screening", seed=42)  # max_steps=10
        meds = obs.current_medications

        # Keep querying until timeout
        for i in range(15):
            if obs.done:
                break
            a = meds[i % len(meds)].drug_id
            b = meds[(i + 1) % len(meds)].drug_id
            action = PolypharmacyAction(
                action_type="query_ddi",
                drug_id_1=a,
                drug_id_2=b,
            )
            obs = env.step(action)

        assert obs.done is True


class TestState:
    def test_state_after_reset(self) -> None:
        env = PolypharmacyEnv()
        env.reset(task_id="easy_screening", seed=42)
        st = env.state
        assert isinstance(st, PolypharmacyState)
        assert st.step_count == 0
        assert st.episode_id is not None


class TestGraderDeterminism:
    def test_same_trajectory_same_score(self) -> None:
        scores = []
        for _ in range(3):
            env = PolypharmacyEnv()
            env.reset(task_id="easy_screening", seed=99)
            obs = env.step(PolypharmacyAction(action_type="finish_review"))
            scores.append(obs.metadata.get("grader_score", 0.0))
        assert all(s == scores[0] for s in scores)

    def test_intervention_changes_score(self) -> None:
        # No intervention
        env = PolypharmacyEnv()
        env.reset(task_id="budgeted_screening", seed=42)
        obs = env.step(PolypharmacyAction(action_type="finish_review"))
        score_noop = obs.metadata.get("grader_score", 0.0)

        # With intervention
        env2 = PolypharmacyEnv()
        obs_init2 = env2.reset(task_id="budgeted_screening", seed=42)
        if obs_init2.current_medications:
            env2.step(PolypharmacyAction(
                action_type="propose_intervention",
                target_drug_id=obs_init2.current_medications[0].drug_id,
                intervention_type="stop",
                rationale="test",
            ))
        obs2 = env2.step(PolypharmacyAction(action_type="finish_review"))
        score_act = obs2.metadata.get("grader_score", 0.0)

        assert score_noop != score_act
