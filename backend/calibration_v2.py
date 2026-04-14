"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Calibración Avanzada (reemplaza Platt scaling)
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
Research-backed upgrades:
- Beta calibration (Kull et al. AISTATS 2017) — supera Platt en 41 datasets
- Isotonic regression — óptima con ≥1000 datos, 42% mejor log-loss
- Temperature scaling (Guo et al. ICML 2017, 4400+ citas) — 1 parámetro
- Venn-ABERS intervals — bounds [p₀,p₁] con garantías finite-sample
- Online recalibration — calibración adaptiva en entornos no estacionarios
"""
import math
import numpy as np
from scipy import optimize
from dataclasses import dataclass


@dataclass
class CalibratedPrediction:
    raw: float
    calibrated: float
    interval_low: float   # Venn-ABERS lower
    interval_high: float  # Venn-ABERS upper
    uncertainty: float     # interval width → Kelly sizing
    method: str


class BetaCalibrator:
    """
    Beta calibration: P_cal = 1/(1 + 1/exp(a*ln(p/(1-p)) + b*ln(p) + c))
    3 params (a,b,c) vs Platt's 2. Includes identity as special case.
    __signature__ = "ddchack"
    """
    def __init__(self):
        self.a, self.b, self.c = 1.0, 0.0, 0.0
        self.fitted = False
        self._history = []

    def add(self, predicted: float, actual: int):
        self._history.append((max(1e-6, min(1-1e-6, predicted)), actual))

    def fit(self) -> bool:
        if len(self._history) < 30:
            return False
        preds = np.array([h[0] for h in self._history])
        outcomes = np.array([h[1] for h in self._history])
        logit_p = np.log(preds / (1 - preds))
        log_p = np.log(preds)

        def nll(params):
            a, b, c = params
            z = a * logit_p + b * log_p + c
            cal = 1.0 / (1.0 + np.exp(-z))
            cal = np.clip(cal, 1e-7, 1 - 1e-7)
            return -np.sum(outcomes * np.log(cal) + (1 - outcomes) * np.log(1 - cal))

        try:
            r = optimize.minimize(nll, [1.0, 0.0, 0.0], method="Nelder-Mead")
            if r.success:
                self.a, self.b, self.c = r.x
                self.fitted = True
                return True
        except Exception:
            pass
        return False

    def calibrate(self, p: float) -> float:
        if not self.fitted:
            return p
        p = max(1e-6, min(1 - 1e-6, p))
        z = self.a * math.log(p / (1 - p)) + self.b * math.log(p) + self.c
        return 1.0 / (1.0 + math.exp(-z))


class IsotonicCalibrator:
    """
    Isotonic regression: non-parametric, monotonic correction.
    Optimal with ≥1000 data points. 42% log-loss reduction documented.
    """
    def __init__(self):
        self._boundaries = []  # sorted (pred, outcome) pairs → step function
        self.fitted = False
        self._history = []

    def add(self, predicted: float, actual: int):
        self._history.append((predicted, actual))

    def fit(self) -> bool:
        if len(self._history) < 50:
            return False
        from sklearn.isotonic import IsotonicRegression
        preds = np.array([h[0] for h in self._history])
        outcomes = np.array([h[1] for h in self._history])
        try:
            self._ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._ir.fit(preds, outcomes)
            self.fitted = True
            return True
        except Exception:
            return False

    def calibrate(self, p: float) -> float:
        if not self.fitted or not hasattr(self, "_ir"):
            return p
        try:
            return float(self._ir.predict([p])[0])
        except Exception:
            return p


class TemperatureScaler:
    """
    Temperature scaling: divide logits by T before softmax.
    Single parameter, reduces ECE from 4-10% to ~0%.
    Guo et al. ICML 2017, 4400+ citations.
    """
    def __init__(self):
        self.T = 1.0
        self.fitted = False
        self._history = []

    def add(self, predicted: float, actual: int):
        self._history.append((max(1e-6, min(1 - 1e-6, predicted)), actual))

    def fit(self) -> bool:
        if len(self._history) < 30:
            return False
        preds = np.array([h[0] for h in self._history])
        outcomes = np.array([h[1] for h in self._history])
        logits = np.log(preds / (1 - preds))

        def nll(T):
            scaled = logits / max(0.01, T[0])
            cal = 1.0 / (1.0 + np.exp(-scaled))
            cal = np.clip(cal, 1e-7, 1 - 1e-7)
            return -np.sum(outcomes * np.log(cal) + (1 - outcomes) * np.log(1 - cal))

        try:
            r = optimize.minimize(nll, [1.0], method="Nelder-Mead", bounds=[(0.01, 10)])
            if r.success:
                self.T = max(0.01, r.x[0])
                self.fitted = True
                return True
        except:
            pass
        return False

    def calibrate(self, p: float) -> float:
        if not self.fitted:
            return p
        p = max(1e-6, min(1 - 1e-6, p))
        logit = math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-logit / self.T))


class VennABERS:
    """
    Produces interval [p0, p1] with finite-sample validity guarantees.
    Wider interval = more epistemic uncertainty = smaller Kelly bet.
    Vovk & Petej (UAI 2014).
    """
    def __init__(self):
        self._history = []
        self._ir_lower = None   # Pre-trained lower-bound isotonic regressor
        self._ir_upper = None   # Pre-trained upper-bound isotonic regressor
        self._venn_fitted = False

    def add(self, predicted: float, actual: int):
        self._history.append((predicted, actual))
        # Invalidate cached regressors — needs re-fit after new data
        self._venn_fitted = False

    def fit(self):
        """
        Pre-train lower and upper bound isotonic regressors so that
        predict_interval() is O(1) instead of O(N log N) per call.
        Must be called explicitly (e.g. after fit_all() in CalibrationSuite).
        """
        if len(self._history) < 20:
            return
        try:
            from sklearn.isotonic import IsotonicRegression
            scores_arr = np.array([h[0] for h in self._history])
            labels_arr = np.array([h[1] for h in self._history])

            # Lower bound regressor: append extra point forcing label=0
            aug_scores_lo = np.append(scores_arr, [np.min(scores_arr) - 1e-6])
            aug_labels_lo = np.append(labels_arr, [0.0])
            idx_lo = np.argsort(aug_scores_lo)
            self._ir_lower = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._ir_lower.fit(aug_scores_lo[idx_lo], aug_labels_lo[idx_lo])

            # Upper bound regressor: append extra point forcing label=1
            aug_scores_hi = np.append(scores_arr, [np.max(scores_arr) + 1e-6])
            aug_labels_hi = np.append(labels_arr, [1.0])
            idx_hi = np.argsort(aug_scores_hi)
            self._ir_upper = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._ir_upper.fit(aug_scores_hi[idx_hi], aug_labels_hi[idx_hi])

            self._venn_fitted = True
        except Exception as e:
            self._venn_fitted = False
            print(f"[VennABERS] Error pre-entrenando regressores: {e}")

    def predict_interval(self, p: float) -> tuple[float, float]:
        """
        Returns (p_low, p_high) interval.
        Uses pre-trained regressors from fit() — O(1) per call.
        Falls back to legacy per-call training if fit() not yet called.
        """
        if self._venn_fitted:
            try:
                s = np.array([p])
                p0 = float(self._ir_lower.predict(s)[0])
                p1 = float(self._ir_upper.predict(s)[0])
                return (
                    max(0.01, min(0.99, min(p0, p1))),
                    max(0.01, min(0.99, max(p0, p1))),
                )
            except Exception:
                return (max(0.05, p - 0.10), min(0.95, p + 0.10))

        # Legacy path: fit() not called yet — train on-the-fly (slow, kept as fallback)
        if len(self._history) < 20:
            return (max(0.05, p - 0.15), min(0.95, p + 0.15))

        scores = np.array([h[0] for h in self._history])
        labels = np.array([h[1] for h in self._history])

        try:
            from sklearn.isotonic import IsotonicRegression
            s0 = np.append(scores, p)
            l0 = np.append(labels, 0)
            ir0 = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            ir0.fit(s0, l0)
            p0 = float(ir0.predict([p])[0])

            s1 = np.append(scores, p)
            l1 = np.append(labels, 1)
            ir1 = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            ir1.fit(s1, l1)
            p1 = float(ir1.predict([p])[0])

            return (min(p0, p1), max(p0, p1))
        except Exception:
            return (max(0.05, p - 0.10), min(0.95, p + 0.10))


class CalibrationSuite:
    """
    Unified calibration system. Runs all methods and picks best.
    Exports uncertainty intervals for Kelly position sizing.
    """
    def __init__(self):
        self.beta = BetaCalibrator()
        self.isotonic = IsotonicCalibrator()
        self.temperature = TemperatureScaler()
        self.venn = VennABERS()
        self._best_method = "beta"

    def add_observation(self, predicted: float, actual: int):
        self.beta.add(predicted, actual)
        self.isotonic.add(predicted, actual)
        self.temperature.add(predicted, actual)
        self.venn.add(predicted, actual)

    def fit_all(self) -> dict:
        results = {}
        results["beta"] = self.beta.fit()
        results["isotonic"] = self.isotonic.fit()
        results["temperature"] = self.temperature.fit()
        self.venn.fit()  # Pre-train isotonic regressors for O(1) predict_interval
        n = len(self.beta._history)

        # Pick best: isotonic if ≥1000, beta if ≥100, temperature otherwise
        if n >= 1000 and results["isotonic"]:
            self._best_method = "isotonic"
        elif n >= 100 and results["beta"]:
            self._best_method = "beta"
        elif results["temperature"]:
            self._best_method = "temperature"
        results["best_method"] = self._best_method
        results["n_observations"] = n
        return results

    def calibrate(self, p: float) -> CalibratedPrediction:
        """Full calibration with Venn-ABERS uncertainty interval."""
        if self._best_method == "isotonic" and self.isotonic.fitted:
            cal = self.isotonic.calibrate(p)
        elif self._best_method == "beta" and self.beta.fitted:
            cal = self.beta.calibrate(p)
        elif self._best_method == "temperature" and self.temperature.fitted:
            cal = self.temperature.calibrate(p)
        else:
            cal = p

        lo, hi = self.venn.predict_interval(p)
        return CalibratedPrediction(
            raw=p, calibrated=cal,
            interval_low=lo, interval_high=hi,
            uncertainty=round(hi - lo, 4),
            method=self._best_method)

    def get_status(self) -> dict:
        return {
            "method": self._best_method,
            "beta_fitted": self.beta.fitted,
            "isotonic_fitted": self.isotonic.fitted,
            "temperature_fitted": self.temperature.fitted,
            "temperature_T": round(self.temperature.T, 4),
            "beta_params": {"a": round(self.beta.a, 4), "b": round(self.beta.b, 4), "c": round(self.beta.c, 4)},
            "n_observations": len(self.beta._history),
        }


# ─────────────────────────────────────────────────────────
# LOGIT CALIBRATION (Political/Geopolitical Markets)
# Based on: arXiv:2602.19520 — Polymarket crowd calibration study
# Slope 1.31 validated for political prediction markets
# ─────────────────────────────────────────────────────────

LOGIT_CALIBRATION_SLOPES: dict[str, float] = {
    "politics": 1.31,
    "geopolitics": 1.31,
    "world": 1.20,
    "elections": 1.31,
    "governance": 1.25,
    "crypto": 1.05,
    "sports": 1.05,
    "finance": 1.10,
    "economics": 1.10,
    "other": 1.15,
}


def logit_calibrate(market_price: float, category: str = "other") -> float:
    """
    Apply logit-space calibration to correct Polymarket crowd bias.

    Formula: P_calibrated = sigmoid(slope * logit(P_market))

    Research finding (arXiv:2602.19520):
    - Polymarket crowds systematically underweight extreme probabilities
    - A 70-cent political contract = ~75% true probability
    - slope=1.31 corrects this bias for political markets

    Args:
        market_price: Raw Polymarket crowd price [0.001, 0.999]
        category: Market category (determines slope)

    Returns:
        Calibrated probability estimate
    """
    slope = LOGIT_CALIBRATION_SLOPES.get(category.lower(), 1.15)
    p = max(0.001, min(0.999, market_price))
    logit_p = math.log(p / (1.0 - p))
    corrected_logit = slope * logit_p
    calibrated = 1.0 / (1.0 + math.exp(-corrected_logit))
    return max(0.01, min(0.99, calibrated))


def logit_calibrate_ensemble(market_price: float, model_prob: float,
                              category: str = "other",
                              model_weight: float = 0.70) -> float:
    """
    Blend logit-calibrated market price with model probability.

    Anti-anchoring blend:
    - market logit-calibrated: 30% (structural correction)
    - model probability: 70% (actual prediction)

    This prevents pure market-copying while still correcting for
    known Polymarket crowd biases.
    """
    market_corrected = logit_calibrate(market_price, category)
    blended = model_prob * model_weight + market_corrected * (1.0 - model_weight)
    return max(0.01, min(0.99, blended))
