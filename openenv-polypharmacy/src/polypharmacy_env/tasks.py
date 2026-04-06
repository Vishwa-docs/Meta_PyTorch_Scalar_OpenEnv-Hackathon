"""Task setup utilities: select episodes and configure budgets per difficulty."""

from __future__ import annotations

import random
from typing import Optional

from .config import DEFAULT_TASK, TASK_CONFIGS, TaskConfig
from .data_loader import PatientEpisode, load_patients


# Map OpenEnv difficulty labels to the CSV difficulty tags
_DIFFICULTY_MAP = {
    "easy_screening": "easy",
    "budgeted_screening": "medium",
    "complex_tradeoff": "hard",
}


def get_task_config(task_id: Optional[str] = None) -> TaskConfig:
    tid = task_id or DEFAULT_TASK
    cfg = TASK_CONFIGS.get(tid)
    if cfg is None:
        raise ValueError(f"Unknown task_id {tid!r}. Choose from {list(TASK_CONFIGS)}")
    return cfg


def sample_episode(
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
) -> PatientEpisode:
    """Return a single patient episode appropriate for *task_id*."""
    tid = task_id or DEFAULT_TASK
    difficulty = _DIFFICULTY_MAP.get(tid, "medium")
    episodes = load_patients(difficulty=difficulty)
    if not episodes:
        raise RuntimeError(f"No episodes found for difficulty={difficulty!r}")

    if episode_id:
        for ep in episodes:
            if ep.episode_id == episode_id:
                return ep
        raise ValueError(f"Episode {episode_id!r} not found for difficulty={difficulty!r}")

    rng = random.Random(seed)
    return rng.choice(episodes)
