#!/usr/bin/env python3
"""REINFORCE with Learned Baseline -- RL training for PolypharmacyEnv.

Trains a small neural-network policy to perform medication reviews in the
PolypharmacyEnv environment.  The policy learns to query drug-drug interactions,
propose clinical interventions, and decide when to finalise the review.

Usage examples:
    python train_rl.py --task easy_screening --episodes 200
    python train_rl.py --task budgeted_screening --episodes 500
    python train_rl.py --task complex_tradeoff --episodes 1000
    python train_rl.py --task easy_screening --episodes 300 --lr 5e-4 --batch-size 8

Architecture:
    - Fixed-size state encoding (16-dim global summary features)
    - Fixed 166-dim action space with dynamic validity masking
    - 3-layer MLP policy  (state -> logits over actions)
    - 3-layer MLP value baseline (state -> scalar return estimate)
    - REINFORCE gradient with advantage = (discounted return) - baseline
    - Entropy bonus for sustained exploration
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Environment imports (direct, no HTTP)
# ---------------------------------------------------------------------------
_BACKEND_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backend", "src"
)
sys.path.insert(0, _BACKEND_SRC)

from polypharmacy_env.env_core import PolypharmacyEnv  # noqa: E402
from polypharmacy_env.models import (  # noqa: E402
    PolypharmacyAction,
    PolypharmacyObservation,
)
from polypharmacy_env.config import TASK_CONFIGS, TaskConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants -- action-space geometry
# ---------------------------------------------------------------------------
MAX_MEDS = 15  # upper bound across all task difficulties
INTERVENTION_TYPES: List[str] = [
    "stop",
    "dose_reduce",
    "substitute",
    "add_monitoring",
]
N_INTERVENTION_TYPES = len(INTERVENTION_TYPES)

# Pre-compute the mapping (med_position_i, med_position_j) -> flat action index
# for all possible query_ddi pairs where i < j.
_PAIR_INDEX: Dict[Tuple[int, int], int] = {}
_idx = 0
for _i in range(MAX_MEDS):
    for _j in range(_i + 1, MAX_MEDS):
        _PAIR_INDEX[(_i, _j)] = _idx
        _idx += 1
N_PAIRS = _idx  # C(15,2) = 105
_REVERSE_PAIR: Dict[int, Tuple[int, int]] = {v: k for k, v in _PAIR_INDEX.items()}

N_INTERVENTIONS = MAX_MEDS * N_INTERVENTION_TYPES  # 60
FINISH_IDX = N_PAIRS + N_INTERVENTIONS              # 165
N_ACTIONS = FINISH_IDX + 1                           # 166

# State feature vector length (see encode_state)
STATE_DIM = 16

# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(obs: PolypharmacyObservation, task_cfg: TaskConfig) -> torch.Tensor:
    """Encode the observation into a compact 16-dim feature vector.

    All values are normalised to roughly [0, 1] to help gradient flow.
    """
    meds = obs.current_medications
    n_meds = len(meds)

    n_high_risk = sum(1 for m in meds if m.is_high_risk_elderly)
    n_beers_any = sum(1 for m in meds if m.beers_flags)
    n_beers_avoid = sum(
        1 for m in meds if any("avoid" in f for f in m.beers_flags)
    )

    queries = obs.interaction_queries
    n_queries = len(queries)
    n_severe = sum(1 for q in queries if q.severity == "severe")
    n_moderate = sum(1 for q in queries if q.severity == "moderate")
    n_interventions = len(obs.interventions)

    max_possible_pairs = max(n_meds * (n_meds - 1) // 2, 1)

    # Drugs involved in any discovered severe DDI (among current meds)
    current_ids = {m.drug_id for m in meds}
    drugs_in_severe: Set[str] = set()
    for q in queries:
        if q.severity == "severe":
            if q.drug_id_1 in current_ids:
                drugs_in_severe.add(q.drug_id_1)
            if q.drug_id_2 in current_ids:
                drugs_in_severe.add(q.drug_id_2)

    features = [
        n_meds / MAX_MEDS,
        n_high_risk / max(n_meds, 1),
        n_beers_any / max(n_meds, 1),
        n_beers_avoid / max(n_meds, 1),
        obs.remaining_query_budget / max(task_cfg.query_budget, 1),
        obs.remaining_intervention_budget / max(task_cfg.intervention_budget, 1),
        n_queries / max(task_cfg.query_budget, 1),
        n_severe / max(n_queries, 1),
        n_moderate / max(n_queries, 1),
        n_interventions / max(task_cfg.intervention_budget, 1),
        obs.step_index / max(task_cfg.max_steps, 1),
        n_queries / max_possible_pairs,  # fraction of pairs queried
        float(obs.remaining_query_budget > 0),
        float(obs.remaining_intervention_budget > 0),
        len(drugs_in_severe) / max(n_meds, 1),  # how much of the regimen is "hot"
        float(n_meds <= 2),  # very few meds left -- may be time to finish
    ]
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Action-space helpers
# ---------------------------------------------------------------------------

def get_action_mask(obs: PolypharmacyObservation) -> torch.Tensor:
    """Return a bool tensor of shape (N_ACTIONS,).  True = action is valid."""
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
    meds = obs.current_medications
    n_meds = min(len(meds), MAX_MEDS)

    # Already-queried drug-id pairs (order-invariant)
    queried: Set[frozenset] = set()
    for q in obs.interaction_queries:
        queried.add(frozenset((q.drug_id_1, q.drug_id_2)))

    # --- query_ddi actions ---
    if obs.remaining_query_budget > 0 and n_meds >= 2:
        for i in range(n_meds):
            for j in range(i + 1, n_meds):
                pair_key = frozenset((meds[i].drug_id, meds[j].drug_id))
                if pair_key not in queried:
                    mask[_PAIR_INDEX[(i, j)]] = True

    # --- propose_intervention actions ---
    if obs.remaining_intervention_budget > 0:
        for i in range(n_meds):
            for k in range(N_INTERVENTION_TYPES):
                mask[N_PAIRS + i * N_INTERVENTION_TYPES + k] = True

    # --- finish_review (always valid) ---
    mask[FINISH_IDX] = True
    return mask


def action_idx_to_env_action(
    idx: int,
    meds: list,
) -> PolypharmacyAction:
    """Map a flat action index back to a concrete PolypharmacyAction."""
    if idx == FINISH_IDX:
        return PolypharmacyAction(action_type="finish_review")

    if idx < N_PAIRS:
        i, j = _REVERSE_PAIR[idx]
        return PolypharmacyAction(
            action_type="query_ddi",
            drug_id_1=meds[i].drug_id,
            drug_id_2=meds[j].drug_id,
        )

    # Otherwise it is an intervention action
    rel = idx - N_PAIRS
    med_idx = rel // N_INTERVENTION_TYPES
    type_idx = rel % N_INTERVENTION_TYPES
    return PolypharmacyAction(
        action_type="propose_intervention",
        target_drug_id=meds[med_idx].drug_id,
        intervention_type=INTERVENTION_TYPES[type_idx],
        rationale="rl_policy",
    )


# ---------------------------------------------------------------------------
# Neural-network modules
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """3-layer MLP that maps state features to action logits."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = N_ACTIONS,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> Categorical:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """3-layer MLP baseline that estimates the expected return from a state."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: PolypharmacyEnv,
    task_id: str,
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    task_cfg: TaskConfig,
    seed: Optional[int] = None,
    greedy: bool = False,
) -> Dict[str, Any]:
    """Roll out one full episode, collecting the REINFORCE trajectory.

    When *greedy* is True the policy acts deterministically (argmax) and
    gradients are not recorded.  Used for evaluation.
    """
    obs = env.reset(task_id=task_id, seed=seed)

    states: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    values: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []

    grader_score = 0.0

    while not obs.done:
        state = encode_state(obs, task_cfg)
        mask = get_action_mask(obs)

        # Safety: if somehow no action is valid, force finish
        if not mask.any():
            mask[FINISH_IDX] = True

        if greedy:
            with torch.no_grad():
                dist = policy(state, mask)
                action_idx = dist.probs.argmax()
                value = value_net(state)
        else:
            with torch.no_grad():
                value = value_net(state)
            dist = policy(state, mask)
            action_idx = dist.sample()

        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        states.append(state)
        actions.append(action_idx)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        env_action = action_idx_to_env_action(
            action_idx.item(), obs.current_medications
        )
        obs = env.step(env_action)

        reward = float(obs.reward) if obs.reward is not None else 0.0
        rewards.append(reward)

        if obs.done:
            grader_score = obs.metadata.get("grader_score", 0.0)

    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "values": values,
        "entropies": entropies,
        "grader_score": grader_score,
        "total_reward": sum(rewards),
        "n_steps": len(rewards),
    }


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """Discounted cumulative returns (G_t) for each timestep."""
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return torch.tensor(returns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:  # noqa: C901 (complex but linear)
    task_id: str = args.task
    n_episodes: int = args.episodes
    lr: float = args.lr
    gamma: float = args.gamma
    entropy_coeff: float = args.entropy_coeff
    batch_size: int = args.batch_size
    hidden_dim: int = args.hidden_dim
    print_every: int = args.print_every

    task_cfg = TASK_CONFIGS[task_id]

    # ---- Initialise env & networks ----------------------------------------
    env = PolypharmacyEnv()
    policy = PolicyNetwork(STATE_DIM, N_ACTIONS, hidden=hidden_dim)
    value_net = ValueNetwork(STATE_DIM, hidden=hidden_dim)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=lr * 3)

    # ---- Book-keeping -----------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file)

    episode_rewards: List[float] = []
    episode_grader_scores: List[float] = []
    episode_steps: List[int] = []
    episode_policy_losses: List[float] = []
    episode_value_losses: List[float] = []

    best_avg_score: float = -float("inf")

    print("=" * 72)
    print("REINFORCE Training -- PolypharmacyEnv")
    print("=" * 72)
    print(f"  task            : {task_id}")
    print(f"  episodes        : {n_episodes}")
    print(f"  batch_size      : {batch_size}")
    print(f"  lr              : {lr}")
    print(f"  gamma           : {gamma}")
    print(f"  entropy_coeff   : {entropy_coeff}")
    print(f"  hidden_dim      : {hidden_dim}")
    print(f"  state_dim       : {STATE_DIM}")
    print(f"  action_space    : {N_ACTIONS}")
    print(f"  task budgets    : query={task_cfg.query_budget}  "
          f"intervention={task_cfg.intervention_budget}  "
          f"max_steps={task_cfg.max_steps}")
    print(f"  checkpoint_dir  : {ckpt_dir}")
    print(f"  metrics_file    : {metrics_path}")
    print("=" * 72)
    print()

    t_start = time.time()

    # ---- Main training loop -----------------------------------------------
    # Accumulate a mini-batch of trajectories, then perform one gradient step.
    batch_trajs: List[Dict[str, Any]] = []

    for ep in range(1, n_episodes + 1):
        traj = run_episode(env, task_id, policy, value_net, task_cfg, seed=ep)

        episode_rewards.append(traj["total_reward"])
        episode_grader_scores.append(traj["grader_score"])
        episode_steps.append(traj["n_steps"])

        if traj["n_steps"] == 0:
            # Degenerate episode (should not happen); skip update
            continue

        batch_trajs.append(traj)

        # ---- Gradient step every batch_size episodes ----------------------
        if len(batch_trajs) >= batch_size:
            # Aggregate losses across the batch
            total_policy_loss = torch.tensor(0.0)
            total_value_loss = torch.tensor(0.0)
            total_entropy = torch.tensor(0.0)
            total_steps = 0

            for bt in batch_trajs:
                returns = compute_returns(bt["rewards"], gamma)
                old_values_t = torch.stack(bt["values"])  # detached, from rollout
                log_probs_t = torch.stack(bt["log_probs"])
                entropies_t = torch.stack(bt["entropies"])

                # Advantages use the *detached* rollout values as baseline
                advantages = returns - old_values_t.detach()
                # Per-trajectory advantage normalisation (reduces variance)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # REINFORCE policy gradient (negative because we minimise)
                total_policy_loss = total_policy_loss + (
                    -(log_probs_t * advantages).sum()
                )

                # Recompute value predictions WITH gradients for the value loss
                states_t = torch.stack(bt["states"])
                fresh_values = value_net(states_t)
                total_value_loss = total_value_loss + F.mse_loss(
                    fresh_values, returns, reduction="sum"
                )

                # Entropy (we want to maximise -> subtract from loss)
                total_entropy = total_entropy + entropies_t.sum()
                total_steps += len(bt["rewards"])

            # Normalise by total number of timesteps in the batch
            total_policy_loss = total_policy_loss / total_steps
            total_value_loss = total_value_loss / total_steps
            total_entropy = total_entropy / total_steps

            # Combined policy loss with entropy bonus
            combined_policy_loss = total_policy_loss - entropy_coeff * total_entropy

            policy_optim.zero_grad()
            combined_policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_optim.step()

            value_optim.zero_grad()
            total_value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            value_optim.step()

            episode_policy_losses.append(total_policy_loss.item())
            episode_value_losses.append(total_value_loss.item())

            batch_trajs = []

        # ---- Logging ------------------------------------------------------
        if ep % print_every == 0 or ep == 1:
            window = min(print_every, ep)
            recent_r = episode_rewards[-window:]
            recent_s = episode_grader_scores[-window:]
            recent_st = episode_steps[-window:]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = sum(recent_s) / len(recent_s)
            avg_st = sum(recent_st) / len(recent_st)
            elapsed = time.time() - t_start
            print(
                f"[ep {ep:>4d}/{n_episodes}]  "
                f"avg_reward={avg_r:+.4f}  "
                f"avg_grader={avg_s:.4f}  "
                f"avg_steps={avg_st:.1f}  "
                f"elapsed={elapsed:.1f}s"
            )

            # Save best checkpoint based on rolling grader score
            eval_window = min(30, ep)
            rolling_score = sum(episode_grader_scores[-eval_window:]) / eval_window
            if rolling_score > best_avg_score:
                best_avg_score = rolling_score
                _save_checkpoint(
                    policy, value_net, policy_optim, value_optim,
                    ep, best_avg_score, task_id,
                    ckpt_dir / f"best_{task_id}.pt",
                )

    # ---- Final checkpoint -------------------------------------------------
    _save_checkpoint(
        policy, value_net, policy_optim, value_optim,
        n_episodes, best_avg_score, task_id,
        ckpt_dir / f"final_{task_id}.pt",
    )

    # ---- Save training metrics to JSON ------------------------------------
    metrics = {
        "task_id": task_id,
        "n_episodes": n_episodes,
        "hyperparameters": {
            "lr": lr,
            "gamma": gamma,
            "entropy_coeff": entropy_coeff,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "state_dim": STATE_DIM,
            "action_dim": N_ACTIONS,
        },
        "episode_rewards": episode_rewards,
        "episode_grader_scores": episode_grader_scores,
        "episode_steps": episode_steps,
        "policy_losses": episode_policy_losses,
        "value_losses": episode_value_losses,
        "best_avg_grader_score": best_avg_score,
        "total_training_time_s": time.time() - t_start,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining metrics saved to {metrics_path}")

    # ---- Post-training evaluation -----------------------------------------
    n_eval = 20
    print("\n" + "=" * 72)
    print(f"Post-training evaluation ({n_eval} episodes each mode)")
    print("=" * 72)

    for mode, is_greedy in [("stochastic", False), ("greedy", True)]:
        eval_rewards, eval_scores, eval_steps_list = [], [], []
        for i in range(n_eval):
            traj = run_episode(
                env, task_id, policy, value_net, task_cfg,
                seed=10_000 + i, greedy=is_greedy,
            )
            eval_rewards.append(traj["total_reward"])
            eval_scores.append(traj["grader_score"])
            eval_steps_list.append(traj["n_steps"])
        avg_r = sum(eval_rewards) / len(eval_rewards)
        avg_s = sum(eval_scores) / len(eval_scores)
        avg_st = sum(eval_steps_list) / len(eval_steps_list)
        print(
            f"  [{mode:>10s}]  avg_reward={avg_r:+.4f}  "
            f"avg_grader={avg_s:.4f}  avg_steps={avg_st:.1f}"
        )
        metrics[f"eval_{mode}_avg_reward"] = avg_r
        metrics[f"eval_{mode}_avg_grader_score"] = avg_s
        metrics[f"eval_{mode}_avg_steps"] = avg_st
        metrics[f"eval_{mode}_rewards"] = eval_rewards
        metrics[f"eval_{mode}_grader_scores"] = eval_scores

    print(f"  best training rolling-avg grader: {best_avg_score:.4f}")
    print()

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    policy_optim: torch.optim.Optimizer,
    value_optim: torch.optim.Optimizer,
    episode: int,
    best_score: float,
    task_id: str,
    path: Path,
) -> None:
    torch.save(
        {
            "episode": episode,
            "best_avg_grader_score": best_score,
            "task_id": task_id,
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_net.state_dict(),
            "policy_optim_state_dict": policy_optim.state_dict(),
            "value_optim_state_dict": value_optim.state_dict(),
            "state_dim": STATE_DIM,
            "action_dim": N_ACTIONS,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    hidden_dim: int = 128,
) -> Tuple[PolicyNetwork, ValueNetwork]:
    """Load a trained policy + value net from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu")
    policy = PolicyNetwork(
        ckpt.get("state_dim", STATE_DIM),
        ckpt.get("action_dim", N_ACTIONS),
        hidden=hidden_dim,
    )
    value_net = ValueNetwork(ckpt.get("state_dim", STATE_DIM), hidden=hidden_dim)
    policy.load_state_dict(ckpt["policy_state_dict"])
    value_net.load_state_dict(ckpt["value_state_dict"])
    return policy, value_net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REINFORCE training for PolypharmacyEnv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--task",
        type=str,
        default="easy_screening",
        choices=list(TASK_CONFIGS.keys()),
        help="Task difficulty to train on",
    )
    p.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    p.add_argument("--batch-size", type=int, default=5, help="Episodes per gradient update")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument(
        "--entropy-coeff", type=float, default=0.02,
        help="Entropy bonus coefficient (higher = more exploration)",
    )
    p.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width")
    p.add_argument("--print-every", type=int, default=10, help="Print interval (episodes)")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(_BACKEND_SRC, "polypharmacy_env", "checkpoints"),
        help="Directory to save model checkpoints",
    )
    p.add_argument(
        "--metrics-file",
        type=str,
        default="training_metrics.json",
        help="Path for JSON training metrics",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
