"""Tests for the FastAPI HTTP server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from polypharmacy_env.api.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestReset:
    def test_reset_default(self, client: TestClient) -> None:
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert data["done"] is False

    def test_reset_with_task(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "easy_screening"})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["task_id"] == "easy_screening"


class TestStep:
    def test_step_finish(self, client: TestClient) -> None:
        client.post("/reset", json={"task_id": "easy_screening"})
        resp = client.post("/step", json={"action": {"action_type": "finish_review"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert "info" in data

    def test_step_query(self, client: TestClient) -> None:
        reset_resp = client.post("/reset", json={"task_id": "easy_screening", "seed": 0})
        obs = reset_resp.json()["observation"]
        meds = obs["current_medications"]
        if len(meds) >= 2:
            action = {
                "action_type": "query_ddi",
                "drug_id_1": meds[0]["drug_id"],
                "drug_id_2": meds[1]["drug_id"],
            }
            resp = client.post("/step", json={"action": action})
            assert resp.status_code == 200

    def test_invalid_action(self, client: TestClient) -> None:
        client.post("/reset", json={"task_id": "easy_screening"})
        resp = client.post("/step", json={"action": {"action_type": "invalid_type"}})
        assert resp.status_code == 422


class TestState:
    def test_state(self, client: TestClient) -> None:
        client.post("/reset", json={"task_id": "easy_screening"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "step_count" in data
        assert data["task_id"] == "easy_screening"
