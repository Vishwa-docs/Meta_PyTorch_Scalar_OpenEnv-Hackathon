"""Neural Thompson Sampling (NeuralTS) with Differential Evolution (DE).

Implements the OptimNeuralTS algorithm from:
  Larouche et al., "Neural Bandits for Data Mining: Searching for Dangerous Polypharmacy"
  https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Key components:
  - NeuralTS: Neural network with gradient-based uncertainty for Thompson Sampling
  - DE (best/1/bin): Differential Evolution to generate candidate drug combinations
  - OptimNeuralTS: Full pipeline combining warm-up, NeuralTS, DE, and ensemble predictions
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reward predictor network (predicts Relative Risk for a drug combination)
# ---------------------------------------------------------------------------


class RewardNetwork(nn.Module):
    """Neural network f(x; theta) that predicts association measure (RR)
    for a multi-hot drug combination vector."""

    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


# ---------------------------------------------------------------------------
# NeuralTS: gradient-based uncertainty estimation
# ---------------------------------------------------------------------------


class NeuralTS:
    """Neural Thompson Sampling agent.

    Uses the neural network gradient to estimate a posterior distribution
    over the predicted reward, enabling exploration via Thompson Sampling.

    At each step t, for an action with features x:
        f_t(x) = network prediction (mean)
        s_t(x) = sqrt(lambda * g(x)^T U_t^{-1} g(x))  (std)
    where g(x) is the gradient of the network output w.r.t. parameters,
    and U_t is the diagonal design matrix accumulated over past actions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        reg_lambda: float = 1.0,
        exploration_factor: float = 1.0,
        lr: float = 0.01,
        train_epochs: int = 100,
    ) -> None:
        self.input_dim = input_dim
        self.reg_lambda = reg_lambda
        self.nu = exploration_factor
        self.lr = lr
        self.train_epochs = train_epochs

        self.network = RewardNetwork(input_dim, hidden)
        self.n_params = sum(p.numel() for p in self.network.parameters())

        # Diagonal approximation of the design matrix U
        self.U_diag = torch.ones(self.n_params) * reg_lambda

        # Training dataset: (context, reward) pairs
        self.contexts: List[torch.Tensor] = []
        self.rewards: List[float] = []

        # Ensemble: store snapshots of model weights at each training step
        self.ensemble_weights: List[Dict[str, torch.Tensor]] = []

    def _get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient g(x; theta) of network output w.r.t. parameters."""
        self.network.zero_grad()
        pred = self.network(x.unsqueeze(0) if x.dim() == 1 else x)
        if pred.dim() > 0:
            pred = pred.sum()
        pred.backward()
        grads = []
        for p in self.network.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """Return (mean, std) for the predicted reward at features x."""
        with torch.no_grad():
            mean = self.network(x.unsqueeze(0) if x.dim() == 1 else x).item()

        g = self._get_gradient(x)
        # s_t(x) = sqrt(lambda * g^T U^{-1} g)  (diagonal approx)
        var = self.reg_lambda * (g ** 2 / self.U_diag).sum().item()
        std = math.sqrt(max(var, 1e-10))
        return mean, std

    def sample_value(self, x: torch.Tensor) -> float:
        """Sample a value from the Thompson Sampling posterior N(f_t, nu * s_t)."""
        mean, std = self.predict(x)
        return random.gauss(mean, self.nu * std)

    def update_design_matrix(self, x: torch.Tensor) -> None:
        """Update U_t with the gradient at x (U_t += g(x) * g(x)^T diagonal)."""
        g = self._get_gradient(x)
        self.U_diag += g ** 2

    def add_observation(self, x: torch.Tensor, reward: float) -> None:
        """Add (context, reward) to training dataset."""
        self.contexts.append(x.detach().clone())
        self.rewards.append(reward)

    def train_network(self) -> float:
        """Train the network on accumulated data. Returns final loss."""
        if not self.contexts:
            return 0.0

        X = torch.stack(self.contexts)
        y = torch.tensor(self.rewards, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        best_loss = float("inf")
        best_state = deepcopy(self.network.state_dict())

        for epoch in range(self.train_epochs):
            optimizer.zero_grad()
            preds = self.network(X)
            loss = F.mse_loss(preds, y)

            # L2 regularization (as in original NeuralTS)
            l2_reg = sum(p.pow(2).sum() for p in self.network.parameters())
            total_loss = loss + self.reg_lambda * 1e-4 * l2_reg

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = deepcopy(self.network.state_dict())

        # Restore best weights (maximizes likelihood)
        self.network.load_state_dict(best_state)

        # Save snapshot for ensemble
        self.ensemble_weights.append(deepcopy(best_state))

        return best_loss

    def ensemble_predict(self, x: torch.Tensor) -> Tuple[float, float, bool]:
        """Predict using ensemble of all intermediate models.

        Returns (mean_pred, lower_bound, is_pip) where:
          - mean_pred: average prediction across ensemble
          - lower_bound: pessimistic estimate (mean - 3*std)
          - is_pip: True if lower_bound > threshold (1.1)
        """
        if not self.ensemble_weights:
            mean, std = self.predict(x)
            lb = mean - 3 * std
            return mean, lb, lb > 1.1

        preds = []
        original_state = deepcopy(self.network.state_dict())

        for state_dict in self.ensemble_weights:
            self.network.load_state_dict(state_dict)
            with torch.no_grad():
                p = self.network(x.unsqueeze(0) if x.dim() == 1 else x).item()
            preds.append(p)

        # Restore current weights
        self.network.load_state_dict(original_state)

        mean_pred = sum(preds) / len(preds)
        # Use ensemble variance for uncertainty
        if len(preds) > 1:
            var = sum((p - mean_pred) ** 2 for p in preds) / (len(preds) - 1)
            std = math.sqrt(var)
        else:
            _, std = self.predict(x)

        lower_bound = mean_pred - 3 * std
        is_pip = lower_bound > 1.1

        return mean_pred, lower_bound, is_pip


# ---------------------------------------------------------------------------
# Differential Evolution (DE best/1/bin)
# ---------------------------------------------------------------------------


def differential_evolution(
    objective_fn,
    dim: int,
    population_size: int = 32,
    crossover_rate: float = 0.9,
    differential_weight: float = 1.0,
    n_steps: int = 16,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """DE best/1/bin optimization on a multi-hot feature space.

    Generates candidate drug combinations by evolving a population and
    evaluating them with the objective function (sampled from NeuralTS).

    Args:
        objective_fn: Maps a feature vector -> scalar value (e.g. Thompson sample)
        dim: Dimensionality of feature vectors (number of possible drugs)
        population_size: N — number of members in population
        crossover_rate: C — probability of component crossover
        differential_weight: F — scaling factor for mutation
        n_steps: S — number of evolution steps

    Returns:
        best_member: The feature vector maximizing the objective
        all_members: All members evaluated during DE (for action set A_t)
    """
    # Initialize population: random multi-hot vectors (each drug has ~20% chance)
    population = []
    for _ in range(population_size):
        member = (torch.rand(dim) > 0.8).float()
        # Ensure at least 2 drugs are present
        if member.sum() < 2:
            indices = random.sample(range(dim), 2)
            member[indices[0]] = 1.0
            member[indices[1]] = 1.0
        population.append(member)

    all_evaluated = list(population)

    for step in range(n_steps):
        # Find best member
        scores = [objective_fn(m) for m in population]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = population[best_idx]

        new_population = []
        for i, w_i in enumerate(population):
            # Random indices (not i)
            candidates = [j for j in range(population_size) if j != i]
            r1, r2 = random.sample(candidates, 2)

            # Mutation: m_i = best + F * (w_r1 - w_r2)
            m_i = best + differential_weight * (population[r1] - population[r2])

            # Crossover: binomial
            l = random.randint(0, dim - 1)  # guaranteed crossover index
            u_i = w_i.clone()
            for j in range(dim):
                if j == l or random.random() <= crossover_rate:
                    u_i[j] = m_i[j]

            # Clamp to [0, 1] and round to get multi-hot
            u_i = torch.clamp(u_i, 0.0, 1.0)
            u_i = (u_i > 0.5).float()

            # Ensure minimum drugs
            if u_i.sum() < 2:
                indices = random.sample(range(dim), 2)
                u_i[indices[0]] = 1.0
                u_i[indices[1]] = 1.0

            # Selection: keep mutant if better
            if objective_fn(u_i) >= objective_fn(w_i):
                new_population.append(u_i)
            else:
                new_population.append(w_i)

            all_evaluated.append(u_i)

        population = new_population

    # Return the best from final population
    final_scores = [objective_fn(m) for m in population]
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    return population[best_idx], all_evaluated


# ---------------------------------------------------------------------------
# Nearest-neighbor mapping (Hamming distance)
# ---------------------------------------------------------------------------


def nearest_neighbor_hamming(
    candidate: torch.Tensor,
    dataset: List[torch.Tensor],
) -> int:
    """Find the index of the nearest neighbor in dataset using Hamming distance."""
    best_dist = float("inf")
    best_idx = 0
    candidate_binary = (candidate > 0.5).float()
    for i, item in enumerate(dataset):
        item_binary = (item > 0.5).float()
        dist = (candidate_binary != item_binary).float().sum().item()
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# OptimNeuralTS: full pipeline
# ---------------------------------------------------------------------------


class OptimNeuralTS:
    """Complete OptimNeuralTS training pipeline.

    Combines NeuralTS with Differential Evolution to efficiently search
    for potentially inappropriate polypharmacies (PIPs) in a large
    combinatorial space of drug combinations.

    The algorithm:
    1. Warm-up: Randomly sample actions for tau steps, collect rewards
    2. Train the neural network on warm-up data
    3. For each subsequent step:
       a. Use DE to find the best candidate action (guided by NeuralTS posterior)
       b. Map candidate to the nearest real drug combination (Hamming distance)
       c. Observe reward (Relative Risk), add to training data
       d. Retrain network periodically
    4. Return ensemble of all intermediate models for prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        reg_lambda: float = 1.0,
        exploration_factor: float = 1.0,
        lr: float = 0.01,
        train_epochs: int = 100,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        retrain_every: int = 10,
        de_population: int = 32,
        de_crossover: float = 0.9,
        de_weight: float = 1.0,
        de_steps: int = 16,
    ) -> None:
        self.agent = NeuralTS(
            input_dim=input_dim,
            hidden=hidden,
            reg_lambda=reg_lambda,
            exploration_factor=exploration_factor,
            lr=lr,
            train_epochs=train_epochs,
        )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.retrain_every = retrain_every
        self.de_population = de_population
        self.de_crossover = de_crossover
        self.de_weight = de_weight
        self.de_steps = de_steps
        self.input_dim = input_dim

        self.step_count = 0
        self.training_log: List[Dict[str, Any]] = []

    def select_action(
        self,
        available_actions: List[torch.Tensor],
    ) -> Tuple[int, Dict[str, Any]]:
        """Select an action from available_actions.

        During warm-up: random selection.
        After warm-up: DE + NeuralTS Thompson Sampling.

        Returns: (index into available_actions, info dict)
        """
        info: Dict[str, Any] = {"phase": "warmup" if self.step_count < self.warmup_steps else "bandit"}

        if self.step_count < self.warmup_steps:
            # Warm-up: random
            idx = random.randint(0, len(available_actions) - 1)
            info["selection"] = "random"
            return idx, info

        # After warm-up: use DE + NeuralTS
        def ts_objective(x: torch.Tensor) -> float:
            return self.agent.sample_value(x)

        # Run DE to find best candidate
        best_candidate, _ = differential_evolution(
            objective_fn=ts_objective,
            dim=self.input_dim,
            population_size=self.de_population,
            crossover_rate=self.de_crossover,
            differential_weight=self.de_weight,
            n_steps=self.de_steps,
        )

        # Update design matrix with DE's recommended action
        self.agent.update_design_matrix(best_candidate)

        # Map to nearest real action (Hamming distance)
        idx = nearest_neighbor_hamming(best_candidate, available_actions)
        info["selection"] = "de_neuralts"

        mean, std = self.agent.predict(available_actions[idx])
        info["predicted_rr"] = mean
        info["uncertainty"] = std

        return idx, info

    def observe(self, x: torch.Tensor, reward: float) -> Optional[float]:
        """Record observation and retrain if needed.

        Returns training loss if retrained, None otherwise.
        """
        self.agent.add_observation(x, reward)
        self.step_count += 1

        loss = None
        # Retrain after warm-up, then every retrain_every steps
        if self.step_count == self.warmup_steps:
            loss = self.agent.train_network()
        elif self.step_count > self.warmup_steps and self.step_count % self.retrain_every == 0:
            loss = self.agent.train_network()

        self.training_log.append({
            "step": self.step_count,
            "reward": reward,
            "loss": loss,
            "n_ensemble": len(self.agent.ensemble_weights),
        })

        return loss

    def predict_risk(self, x: torch.Tensor) -> Dict[str, Any]:
        """Use the ensemble to predict risk for a drug combination.

        Returns dict with mean prediction, lower confidence bound,
        and whether the combination is flagged as a PIP.
        """
        mean, lower_bound, is_pip = self.agent.ensemble_predict(x)
        return {
            "predicted_rr": round(mean, 4),
            "lower_bound": round(lower_bound, 4),
            "is_potentially_harmful": is_pip,
            "n_models_in_ensemble": len(self.agent.ensemble_weights),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics summary."""
        if not self.training_log:
            return {"status": "no_data"}

        rewards = [e["reward"] for e in self.training_log]
        return {
            "total_steps": self.step_count,
            "warmup_steps": self.warmup_steps,
            "n_ensemble_models": len(self.agent.ensemble_weights),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "phase": "warmup" if self.step_count < self.warmup_steps else "bandit",
        }
