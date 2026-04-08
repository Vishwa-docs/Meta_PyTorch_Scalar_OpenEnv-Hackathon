#!/usr/bin/env python3
"""OptimNeuralTS training -- Neural Bandit search for dangerous polypharmacies.

Implements the training pipeline from:
  Larouche et al., "Neural Bandits for Data Mining: Searching for Dangerous Polypharmacy"
  https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

This script:
  1. Generates a synthetic dataset of drug combinations with simulated Relative Risk (RR)
  2. Runs OptimNeuralTS: warm-up -> NeuralTS+DE exploration -> ensemble building
  3. Evaluates the ensemble's ability to detect Potentially Inappropriate Polypharmacies (PIPs)
  4. Saves the trained ensemble model

Usage:
    python train_bandit.py --total-steps 1000 --warmup-steps 200
    python train_bandit.py --total-steps 3000 --warmup-steps 500 --eval-every 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_BACKEND_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backend", "src"
)
sys.path.insert(0, _BACKEND_SRC)

from polypharmacy_env.neural_bandits import NeuralTS, OptimNeuralTS, nearest_neighbor_hamming  # noqa: E402
from polypharmacy_env.data_loader import load_drug_metadata, load_ddi_rules  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic RR data generation (follows paper Section 4.1)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_drugs: int = 33,
    n_combinations: int = 5000,
    n_dangerous_patterns: int = 10,
    rr_threshold: float = 1.1,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic drug combination data with ground-truth RRs.

    Follows the paper's data generation process:
    - Generate dangerous patterns (binomial)
    - For each combination, compute similarity to nearest pattern
    - Assign RR proportional to similarity (if overlapping) or from N(mu, sigma) if disjoint
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Generate dangerous patterns (multi-hot vectors)
    patterns = []
    for _ in range(n_dangerous_patterns):
        # Each drug has ~30% chance of being in the pattern (smaller patterns)
        p = torch.zeros(n_drugs)
        n_active = rng.randint(2, max(3, n_drugs // 8))
        indices = rng.sample(range(n_drugs), n_active)
        for idx in indices:
            p[idx] = 1.0
        patterns.append(p)

    # Generate distinct drug combinations
    combos = []
    combo_set = set()
    while len(combos) < n_combinations:
        n_active = rng.randint(2, min(8, n_drugs))
        indices = tuple(sorted(rng.sample(range(n_drugs), n_active)))
        if indices not in combo_set:
            combo_set.add(indices)
            vec = torch.zeros(n_drugs)
            for idx in indices:
                vec[idx] = 1.0
            combos.append(vec)

    # Compute RR for each combination based on Hamming distance to nearest pattern
    rrs = []
    nearest_pattern_idx = []
    for combo in combos:
        # Find nearest pattern (Hamming distance)
        min_dist = float("inf")
        best_p_idx = 0
        for p_idx, pattern in enumerate(patterns):
            dist = (combo != pattern).float().sum().item()
            if dist < min_dist:
                min_dist = dist
                best_p_idx = p_idx

        nearest_pattern_idx.append(best_p_idx)
        pattern = patterns[best_p_idx]

        # Check intersection (shared active drugs)
        intersection = (combo * pattern).sum().item()
        if intersection > 0:
            # RR proportional to similarity
            similarity = intersection / max(pattern.sum().item(), 1)
            # Higher similarity -> higher RR
            base_rr = 0.5 + 2.5 * similarity  # range ~[0.5, 3.0]
            noise = rng.gauss(0, 0.15)
            rr = max(0.1, base_rr + noise)
        else:
            # Disjoint: sample from neutral distribution
            rr = max(0.1, rng.gauss(0.85, 0.2))

        rrs.append(rr)

    # Compute pattern RRs (patterns themselves have high RR)
    pattern_rrs = [2.0 + rng.gauss(0, 0.3) for _ in patterns]

    n_pips = sum(1 for rr in rrs if rr > rr_threshold)
    print(f"  Generated {n_combinations} combos, {n_pips} PIPs (RR > {rr_threshold})")
    print(f"  RR range: [{min(rrs):.3f}, {max(rrs):.3f}], mean: {sum(rrs)/len(rrs):.3f}")

    return {
        "combos": combos,
        "rrs": rrs,
        "patterns": patterns,
        "pattern_rrs": pattern_rrs,
        "n_drugs": n_drugs,
        "n_pips": n_pips,
        "rr_threshold": rr_threshold,
        "noise_std": noise_std,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_bandit(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("OptimNeuralTS Training -- Neural Bandits for Polypharmacy")
    print("=" * 72)

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    dataset = generate_synthetic_dataset(
        n_drugs=args.n_drugs,
        n_combinations=args.n_combinations,
        n_dangerous_patterns=args.n_patterns,
        seed=args.seed,
    )
    combos = dataset["combos"]
    rrs = dataset["rrs"]
    patterns = dataset["patterns"]
    pattern_rrs = dataset["pattern_rrs"]
    noise_std = dataset["noise_std"]
    rr_threshold = dataset["rr_threshold"]

    # Initialize OptimNeuralTS
    bandit = OptimNeuralTS(
        input_dim=args.n_drugs,
        hidden=args.hidden_dim,
        reg_lambda=args.reg_lambda,
        exploration_factor=args.exploration_factor,
        lr=args.lr,
        train_epochs=args.train_epochs,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        retrain_every=args.retrain_every,
        de_population=args.de_population,
        de_crossover=args.de_crossover,
        de_weight=args.de_weight,
        de_steps=args.de_steps,
    )

    print(f"\n  n_drugs           : {args.n_drugs}")
    print(f"  n_combinations    : {args.n_combinations}")
    print(f"  total_steps (T)   : {args.total_steps}")
    print(f"  warmup_steps (τ)  : {args.warmup_steps}")
    print(f"  DE population (N) : {args.de_population}")
    print(f"  DE steps (S)      : {args.de_steps}")
    print(f"  retrain_every     : {args.retrain_every}")
    print(f"  hidden_dim        : {args.hidden_dim}")
    print(f"  lr                : {args.lr}")
    print("=" * 72)

    t_start = time.time()

    # Metrics tracking
    step_rewards = []
    pips_found = []
    eval_precisions = []
    eval_recalls = []
    training_dataset_indices = set()

    for t in range(1, args.total_steps + 1):
        # Select action
        idx, info = bandit.select_action(combos)
        training_dataset_indices.add(idx)

        # Observe noisy reward (RR + noise)
        true_rr = rrs[idx]
        noisy_rr = true_rr + random.gauss(0, noise_std)
        reward = noisy_rr

        step_rewards.append(reward)

        # Update bandit
        loss = bandit.observe(combos[idx], reward)

        # Periodic evaluation
        if t % args.eval_every == 0 or t == args.total_steps:
            # Evaluate ensemble on ALL combinations
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for i, combo in enumerate(combos):
                pred = bandit.predict_risk(combo)
                actual_pip = rrs[i] > rr_threshold
                predicted_pip = pred["is_potentially_harmful"]

                if predicted_pip and actual_pip:
                    true_positives += 1
                elif predicted_pip and not actual_pip:
                    false_positives += 1
                elif not predicted_pip and actual_pip:
                    false_negatives += 1
                else:
                    true_negatives += 1

            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            eval_precisions.append(precision)
            eval_recalls.append(recall)

            # Check dangerous pattern detection
            patterns_found = 0
            for p_idx, pattern in enumerate(patterns):
                pred = bandit.predict_risk(pattern)
                if pred["is_potentially_harmful"]:
                    patterns_found += 1
            pattern_ratio = patterns_found / len(patterns)

            # PIPs found outside training data
            pips_outside_train = 0
            total_detected_pips = 0
            for i, combo in enumerate(combos):
                pred = bandit.predict_risk(combo)
                if pred["is_potentially_harmful"]:
                    total_detected_pips += 1
                    if i not in training_dataset_indices:
                        pips_outside_train += 1

            pips_found.append(total_detected_pips)

            elapsed = time.time() - t_start
            phase = info.get("phase", "?")
            n_ens = len(bandit.agent.ensemble_weights)
            print(
                f"[step {t:>5d}/{args.total_steps}] "
                f"phase={phase}  "
                f"precision={precision:.3f}  "
                f"recall={recall:.3f}  "
                f"patterns={pattern_ratio:.2f}  "
                f"PIPs_detected={total_detected_pips}  "
                f"outside_train={pips_outside_train}  "
                f"ensemble={n_ens}  "
                f"elapsed={elapsed:.1f}s"
            )

    # Save metrics
    metrics = {
        "algorithm": "OptimNeuralTS",
        "n_drugs": args.n_drugs,
        "n_combinations": args.n_combinations,
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "n_ensemble_models": len(bandit.agent.ensemble_weights),
        "final_precision": eval_precisions[-1] if eval_precisions else 0,
        "final_recall": eval_recalls[-1] if eval_recalls else 0,
        "eval_precisions": eval_precisions,
        "eval_recalls": eval_recalls,
        "pips_detected": pips_found,
        "step_rewards": step_rewards,
        "total_time_s": time.time() - t_start,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "reg_lambda": args.reg_lambda,
            "exploration_factor": args.exploration_factor,
            "de_population": args.de_population,
            "de_crossover": args.de_crossover,
            "de_weight": args.de_weight,
            "de_steps": args.de_steps,
            "train_epochs": args.train_epochs,
            "retrain_every": args.retrain_every,
        },
    }

    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save model ensemble
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "bandit_ensemble.pt"
    torch.save({
        "ensemble_weights": bandit.agent.ensemble_weights,
        "network_state_dict": bandit.agent.network.state_dict(),
        "U_diag": bandit.agent.U_diag,
        "input_dim": args.n_drugs,
        "hidden_dim": args.hidden_dim,
        "n_steps": args.total_steps,
    }, ckpt_path)
    print(f"Ensemble model saved to {ckpt_path}")

    print(f"\n{'='*72}")
    print("Training complete!")
    print(f"  Ensemble size: {len(bandit.agent.ensemble_weights)} models")
    if eval_precisions:
        print(f"  Final precision: {eval_precisions[-1]:.4f}")
        print(f"  Final recall: {eval_recalls[-1]:.4f}")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OptimNeuralTS training for polypharmacy PIP detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--n-drugs", type=int, default=33, help="Number of possible drugs")
    p.add_argument("--n-combinations", type=int, default=5000, help="Number of distinct drug combinations")
    p.add_argument("--n-patterns", type=int, default=10, help="Number of dangerous patterns")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # OptimNeuralTS
    p.add_argument("--total-steps", type=int, default=1000, help="Total bandit steps T")
    p.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps τ")
    p.add_argument("--retrain-every", type=int, default=10, help="Retrain network every N steps")
    p.add_argument("--hidden-dim", type=int, default=64, help="Network hidden layer size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--reg-lambda", type=float, default=1.0, help="Regularization λ")
    p.add_argument("--exploration-factor", type=float, default=1.0, help="Exploration ν")
    p.add_argument("--train-epochs", type=int, default=50, help="Epochs per retrain")

    # DE
    p.add_argument("--de-population", type=int, default=16, help="DE population size N")
    p.add_argument("--de-crossover", type=float, default=0.9, help="DE crossover rate C")
    p.add_argument("--de-weight", type=float, default=1.0, help="DE differential weight F")
    p.add_argument("--de-steps", type=int, default=8, help="DE optimization steps S")

    # Output
    p.add_argument("--eval-every", type=int, default=100, help="Evaluate every N steps")
    p.add_argument("--metrics-file", type=str, default="bandit_metrics.json", help="Metrics output path")
    p.add_argument(
        "--checkpoint-dir", type=str,
        default=os.path.join(_BACKEND_SRC, "polypharmacy_env", "checkpoints"),
        help="Model checkpoint directory",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bandit(args)
