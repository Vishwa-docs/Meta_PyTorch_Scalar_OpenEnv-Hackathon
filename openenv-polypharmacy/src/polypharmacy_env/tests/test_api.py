"""Tests for the FastAPI HTTP server (OpenEnv create_app endpoints).

OpenEnv HTTP endpoints are *stateless*: each /reset and /step creates a
fresh environment instance.  Multi-step sessions only work via WebSocket.
These tests validate single-call behaviour and schema contracts.
"""

from __future__ import annotations

import json

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
        data = resp.json()
        assert data["status"] == "healthy"


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

    def test_reset_observation_has_medications(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"task_id": "easy_screening", "seed": 42})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert len(obs["current_medications"]) >= 3


class TestStep:
    """Test /step endpoint – each call is independent (stateless)."""

    def test_step_finish(self, client: TestClient) -> None:
        resp = client.post(
            "/step",
            json={"action": {"action_type": "finish_review"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    def test_invalid_action_422(self, client: TestClient) -> None:
        resp = client.post(
            "/step",
            json={"action": {"action_type": "invalid_type"}},
        )
        assert resp.status_code == 422


class TestSchema:
    def test_schema(self, client: TestClient) -> None:
        resp = client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        # OpenEnv schema endpoint returns keys: action, observation, state
        assert "action" in data
        assert "observation" in data


class TestWebSocketSession:
    """Test multi-step sessions through the /ws WebSocket endpoint.

    OpenEnv WS protocol:
      Send:  {"type": "reset", "data": {"task_id": "...", "seed": ...}}
      Recv:  {"type": "observation", "data": {"observation": {...}, "reward": ..., "done": ...}}
      Send:  {"type": "step", "data": {"action_type": "...", ...}}
      Recv:  {"type": "observation", "data": {"observation": {...}, ...}}
      Send:  {"type": "state"}
      Recv:  {"type": "state", "data": {...state fields...}}
    """

    def test_ws_reset_step_finish(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            # Reset
            ws.send_json({
                "type": "reset",
                "data": {"task_id": "easy_screening", "seed": 42},
            })
            reset_resp = ws.receive_json()
            assert reset_resp["type"] == "observation"
            reset_data = reset_resp["data"]
            assert reset_data["done"] is False
            obs = reset_data["observation"]
            assert obs["task_id"] == "easy_screening"
            meds = obs["current_medications"]
            assert len(meds) >= 3

            # Step – query DDI
            if len(meds) >= 2:
                ws.send_json({
                    "type": "step",
                    "data": {
                        "action_type": "query_ddi",
                        "drug_id_1": meds[0]["drug_id"],
                        "drug_id_2": meds[1]["drug_id"],
                    },
                })
                step_resp = ws.receive_json()
                assert step_resp["type"] == "observation"
                assert step_resp["data"]["done"] is False

            # Finish
            ws.send_json({
                "type": "step",
                "data": {"action_type": "finish_review"},
            })
            finish_resp = ws.receive_json()
            assert finish_resp["type"] == "observation"
            assert finish_resp["data"]["done"] is True

    def test_ws_state(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "reset",
                "data": {"task_id": "easy_screening", "seed": 0},
            })
            ws.receive_json()  # consume reset response

            ws.send_json({"type": "state"})
            state_resp = ws.receive_json()
            assert state_resp["type"] == "state"
            state_data = state_resp["data"]
            assert state_data["step_count"] == 0
            assert state_data["task_id"] == "easy_screening"
