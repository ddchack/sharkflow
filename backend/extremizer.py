"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Extremización y Agregación Adaptiva
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
- Satopää extremization (d≈1.73) — 10-25% Brier Score improvement
- Diversity Prediction Theorem tracking
- Vovk's Aggregating Algorithm for adaptive weights
"""
import math
import numpy as np


class Extremizer:
    """
    Satopää et al. (IJF 2014): extremize aggregated forecasts
    in log-odds space by factor d > 1.

    p_ext = logit⁻¹(d * mean(logit(p_i)))

    Optimal d from GJP: 1.161–3.921. Theory (Neyman & Roughgarden):
    d ≈ √3 ≈ 1.73 worst-case optimal.
    """

    @staticmethod
    def logit(p: float) -> float:
        p = max(1e-6, min(1 - 1e-6, p))
        return math.log(p / (1 - p))

    @staticmethod
    def inv_logit(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def extremize(probabilities: list[float], d: float = 1.73) -> float:
        """
        Extremize an aggregate of probability forecasts.
        d > 1: push away from 0.5 (extremize)
        d < 1: push toward 0.5 (anti-extremize)
        d = 1: simple logit average
        """
        if not probabilities:
            return 0.5
        # Filtrar valores inválidos antes de calcular logits
        clean = [p for p in probabilities if isinstance(p, (int, float)) and 0 < p < 1]
        if not clean:
            return 0.5
        logits = [Extremizer.logit(p) for p in clean]
        mean_logit = sum(logits) / len(logits)
        if not math.isfinite(mean_logit):
            return 0.5
        return Extremizer.inv_logit(d * mean_logit)

    @staticmethod
    def weighted_extremize(probs_weights: list[tuple[float, float]],
                            d: float = 1.73) -> float:
        """Extremize with confidence weights."""
        if not probs_weights:
            return 0.5
        total_w = sum(w for _, w in probs_weights)
        if total_w <= 0:
            return 0.5
        weighted_logit = sum(Extremizer.logit(p) * w for p, w in probs_weights) / total_w
        return Extremizer.inv_logit(d * weighted_logit)

    @staticmethod
    def find_optimal_d(predictions_list: list[list[float]],
                        outcomes: list[int]) -> float:
        """
        Find optimal d by minimizing Brier Score on historical data.
        predictions_list: [[p1_model1, p1_model2, ...], [p2_model1, ...], ...]
        """
        if len(predictions_list) < 10:
            return 1.73  # Default

        def brier_at_d(d):
            bs = 0
            for preds, outcome in zip(predictions_list, outcomes):
                ext = Extremizer.extremize(preds, d)
                bs += (ext - outcome) ** 2
            return bs / len(outcomes)

        best_d, best_bs = 1.73, float("inf")
        for d in np.arange(0.5, 4.0, 0.05):
            bs = brier_at_d(d)
            if bs < best_bs:
                best_d, best_bs = d, bs
        return round(best_d, 2)


class DiversityTracker:
    """
    Diversity Prediction Theorem (Scott Page, 2007):
    Collective Error = Avg Individual Error − Diversity

    Track model diversity to ensure ensemble benefits persist.
    """

    @staticmethod
    def compute(model_predictions: list[float], outcome: int) -> dict:
        if not model_predictions:
            return {"collective_error": 0, "avg_individual_error": 0, "diversity": 0}
        n = len(model_predictions)
        avg_pred = sum(model_predictions) / n
        collective_error = (avg_pred - outcome) ** 2
        avg_individual = sum((p - outcome) ** 2 for p in model_predictions) / n
        diversity = sum((p - avg_pred) ** 2 for p in model_predictions) / n
        # Theorem: collective_error = avg_individual - diversity
        return {
            "collective_error": round(collective_error, 6),
            "avg_individual_error": round(avg_individual, 6),
            "diversity": round(diversity, 6),
            "diversity_ratio": round(diversity / max(0.001, avg_individual), 3),
            "n_models": n,
            "ensemble_benefit": round(avg_individual - collective_error, 6),
        }


class AdaptiveAggregator:
    """
    Vovk's Aggregating Algorithm: exponential weight updates.
    w_k ← w_k * exp(-η * loss_k)
    η ≤ 2 for square loss on [0,1]. Optimal: η = sqrt(ln(N)/T).
    """

    def __init__(self, n_models: int, learning_rate: float = 0.5):
        self.n = n_models
        self.eta = learning_rate
        self.weights = np.ones(n_models) / n_models
        self.losses = [[] for _ in range(n_models)]

    def aggregate(self, predictions: list[float]) -> float:
        """Weighted aggregate using current adaptive weights."""
        if len(predictions) != self.n:
            return sum(predictions) / len(predictions)
        return float(np.dot(self.weights, predictions))

    def update(self, predictions: list[float], outcome: int):
        """Update weights after observing outcome."""
        for i, p in enumerate(predictions):
            loss = (p - outcome) ** 2
            self.losses[i].append(loss)
            self.weights[i] *= math.exp(-self.eta * loss)
        total = sum(self.weights)
        if total > 0:
            self.weights = self.weights / total

    def get_weights(self) -> list[float]:
        return [round(w, 4) for w in self.weights]

    def get_model_briers(self) -> list[float]:
        return [round(np.mean(l), 4) if l else 0 for l in self.losses]
