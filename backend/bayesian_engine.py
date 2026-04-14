"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Bayesian Probability Engine
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Replaces naive price-based estimation with proper Bayesian updating.
Combines multiple independent signals into a calibrated probability.

Signals:
  - Market price (prior)
  - News sentiment
  - Volume momentum (surge detection)
  - Whale position consensus
  - Price movement trend
  - Spread/liquidity efficiency

Each signal updates the prior via Bayes' theorem iteratively.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─── Calibración logit por categoría ─────────────────────────────────────────
# SharkFlow by Carlos David Donoso Cordero (ddchack)
CALIBRATION_SLOPES = {
    'politics': 1.31,
    'geopolitics': 1.31,
    'world': 1.20,
    'crypto': 1.05,
    'sports': 1.05,
    'finance': 1.10,
    'default': 1.10,
}

def calibrate_probability(raw_prob: float, category: str) -> float:
    """
    Calibración logit por categoría.
    Política/geopolítica tiene mayor slope (más extrema) que crypto/sports.
    """
    p = max(0.01, min(0.99, raw_prob))
    slope = CALIBRATION_SLOPES.get(category.lower(), CALIBRATION_SLOPES['default'])
    logit_p = math.log(p / (1 - p))
    return 1 / (1 + math.exp(-slope * logit_p))


@dataclass
class Signal:
    """A single probability signal with confidence weight."""
    name: str
    value: float       # -1.0 to 1.0 (directional) or 0.0 to 1.0 (probability)
    confidence: float  # 0.0 to 1.0 (how much we trust this signal)
    signal_type: str   # "probability", "directional", "binary"


@dataclass
class BayesianEstimate:
    """Result of Bayesian probability estimation."""
    prior: float
    posterior: float
    signals_used: list
    alpha: float  # Beta distribution α
    beta_param: float  # Beta distribution β
    uncertainty: float  # Width of credible interval
    confidence: float


class BayesianProbabilityEngine:
    """
    Bayesian probability estimator for prediction markets.
    
    Uses Beta-Binomial model:
    - Prior: Beta(α₀, β₀) derived from market price
    - Each signal acts as pseudo-observations updating α, β
    - Posterior: Beta(α₀ + Σαᵢ, β₀ + Σβᵢ)
    
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    # Signal weights (calibrated defaults)
    DEFAULT_WEIGHTS = {
        "market_price": 0.30,
        "news_sentiment": 0.15,
        "volume_momentum": 0.12,
        "whale_consensus": 0.20,
        "price_trend": 0.10,
        "liquidity_quality": 0.08,
        "cross_market": 0.05,
    }

    def __init__(self, prior_strength: float = 10.0, weights: dict = None):
        """
        Args:
            prior_strength: How strongly the market price anchors the prior.
                           Higher = more trust in market, harder to override.
                           10.0 = moderate (good for $100 capital beginner)
            weights: Custom signal weights (must sum to ~1.0)
        """
        self.prior_strength = prior_strength
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._calibration_history = []

    # ─────────────────────────────────────────────────────────
    # BETA DISTRIBUTION PRIOR
    # ─────────────────────────────────────────────────────────

    def market_to_prior(self, market_price: float) -> tuple[float, float]:
        """
        Convert market price to Beta distribution parameters.
        
        If market says YES=0.60, we model prior as Beta(α, β) where:
          α = price × strength
          β = (1 - price) × strength
          
        This gives mean = α/(α+β) = price, with variance controlled by strength.
        Higher strength = tighter distribution = more certain prior.
        """
        # Clampeamos a [0.02, 0.98] para evitar priors degenerados en extremos
        p = max(0.02, min(0.98, market_price))
        alpha = p * self.prior_strength
        beta = (1.0 - p) * self.prior_strength
        return alpha, beta

    @staticmethod
    def beta_mean(alpha: float, beta: float) -> float:
        """Mean of Beta distribution."""
        return alpha / (alpha + beta)

    @staticmethod
    def beta_variance(alpha: float, beta: float) -> float:
        """Variance of Beta distribution."""
        total = alpha + beta
        return (alpha * beta) / (total ** 2 * (total + 1))

    @staticmethod
    def beta_credible_interval(alpha: float, beta: float,
                                width: float = 0.90) -> tuple[float, float]:
        """90% credible interval for Beta distribution."""
        from scipy.stats import beta as beta_dist
        try:
            lower = beta_dist.ppf((1 - width) / 2, alpha, beta)
            upper = beta_dist.ppf(1 - (1 - width) / 2, alpha, beta)
            return (lower, upper)
        except Exception:
            # Fallback without scipy
            mean = alpha / (alpha + beta)
            std = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            z = 1.645  # ~90% CI
            return (max(0, mean - z * std), min(1, mean + z * std))

    # ─────────────────────────────────────────────────────────
    # SIGNAL PROCESSING
    # ─────────────────────────────────────────────────────────

    def signal_to_pseudo_observations(self, signal: Signal,
                                       weight: float) -> tuple[float, float]:
        """
        Convert a signal into pseudo-observations (additional α, β).
        
        A positive signal adds more α (evidence for YES).
        A negative signal adds more β (evidence for NO).
        
        The magnitude depends on signal confidence and weight.
        """
        import math as _math
        # Guard: señal con NaN/Inf no aporta información → retornar neutro
        if not _math.isfinite(signal.value) or not _math.isfinite(signal.confidence):
            return 0.0, 0.0

        # Base observation strength per signal
        obs_strength = 3.0 * weight * signal.confidence

        if signal.signal_type == "probability":
            # Signal directly estimates probability (0 to 1)
            p = max(0.01, min(0.99, signal.value))
            d_alpha = p * obs_strength
            d_beta = (1 - p) * obs_strength

        elif signal.signal_type == "directional":
            # Signal is -1 to +1 (positive = favors YES)
            shift = signal.value * obs_strength
            if shift > 0:
                d_alpha = shift
                d_beta = 0
            else:
                d_alpha = 0
                d_beta = abs(shift)

        elif signal.signal_type == "binary":
            # Signal is 0 or 1
            if signal.value > 0.5:
                d_alpha = obs_strength
                d_beta = 0
            else:
                d_alpha = 0
                d_beta = obs_strength
        else:
            d_alpha = 0
            d_beta = 0

        return d_alpha, d_beta

    # ─────────────────────────────────────────────────────────
    # SIGNAL GENERATORS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def sentiment_signal(sentiment_score: float,
                          num_articles: int) -> Signal:
        """Create signal from news sentiment analysis."""
        # Confidence scales with number of articles
        conf = min(0.8, num_articles / 15.0) * min(1.0, abs(sentiment_score) * 2)
        return Signal(
            name="news_sentiment",
            value=sentiment_score,  # -1 to 1
            confidence=conf,
            signal_type="directional"
        )

    @staticmethod
    def volume_momentum_signal(volume_24h: float,
                                avg_volume_7d: float) -> Signal:
        """
        Create signal from volume surge detection.
        High relative volume suggests information flow → price likely to move.
        """
        if avg_volume_7d <= 0:
            return Signal("volume_momentum", 0, 0, "directional")

        ratio = volume_24h / avg_volume_7d
        # Surge > 2x is significant
        if ratio > 2.0:
            value = min(1.0, (ratio - 1) / 3.0)  # Normalize surge
            conf = min(0.7, ratio / 5.0)
        elif ratio < 0.5:
            value = -0.3  # Declining interest
            conf = 0.2
        else:
            value = 0
            conf = 0.1

        return Signal("volume_momentum", value, conf, "directional")

    @staticmethod
    def whale_consensus_signal(whale_positions: list[dict]) -> Signal:
        """
        Create signal from whale wallet positions.
        whale_positions: [{"side": "YES"/"NO", "size_usd": float}, ...]
        """
        if not whale_positions:
            return Signal("whale_consensus", 0.5, 0, "probability")

        yes_total = sum(w["size_usd"] for w in whale_positions if w["side"] == "YES")
        no_total = sum(w["size_usd"] for w in whale_positions if w["side"] == "NO")
        total = yes_total + no_total

        if total < 100:
            return Signal("whale_consensus", 0.5, 0.1, "probability")

        yes_ratio = yes_total / total
        # Confidence increases with total capital and number of whales
        num_whales = len(whale_positions)
        conf = min(0.85, (math.log10(max(1, total)) / 5.0) * min(1.0, num_whales / 5.0))

        return Signal("whale_consensus", yes_ratio, conf, "probability")

    @staticmethod
    def price_trend_signal(prices_history: list[float]) -> Signal:
        """
        Create signal from recent price movement.
        Uses simple linear regression slope as trend indicator.
        """
        if len(prices_history) < 3:
            return Signal("price_trend", 0, 0, "directional")

        n = len(prices_history)
        x = np.arange(n)
        y = np.array(prices_history)

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / max(1e-10, np.sum((x - x_mean) ** 2))

        # Normalize slope to -1..1 range
        value = max(-1.0, min(1.0, slope * 10))

        # R² as confidence
        y_pred = slope * x + (y_mean - slope * x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / max(1e-10, ss_tot))
        conf = max(0, min(0.6, r_squared))

        return Signal("price_trend", value, conf, "directional")

    @staticmethod
    def liquidity_quality_signal(spread: float, liquidity: float,
                                  volume_24h: float) -> Signal:
        """
        Markets with tight spreads and deep liquidity are more efficiently priced.
        → Our edge is smaller in efficient markets.
        → Loose markets have more mispricing opportunity.
        """
        # Tight spread = efficient (less edge opportunity)
        if spread < 0.02:
            efficiency = 0.9
        elif spread < 0.05:
            efficiency = 0.6
        else:
            efficiency = 0.3

        # Low efficiency = more mispricing = our signals matter more
        # This signal doesn't push YES/NO, it affects confidence of other signals
        value = 0.0  # Neutral directionally
        conf = 1.0 - efficiency  # Higher for inefficient markets

        return Signal("liquidity_quality", value, conf, "directional")

    # ─────────────────────────────────────────────────────────
    # MAIN ESTIMATION
    # ─────────────────────────────────────────────────────────

    def estimate_probability(self, market_price: float,
                              signals: list[Signal] = None,
                              category: str = "default") -> BayesianEstimate:
        """
        Full Bayesian estimation pipeline.

        1. Convert market price to Beta prior
        2. Process each signal into pseudo-observations
        3. Update posterior
        4. Apply logit calibration by category
        5. Return calibrated estimate with uncertainty
        """
        signals = signals or []

        # Step 1: Prior from market
        alpha, beta = self.market_to_prior(market_price)
        prior_mean = self.beta_mean(alpha, beta)

        signals_used = []

        # Step 2: Update with each signal
        for signal in signals:
            weight = self.weights.get(signal.name, 0.10)
            d_alpha, d_beta = self.signal_to_pseudo_observations(signal, weight)
            alpha += d_alpha
            beta += d_beta

            if signal.confidence > 0:
                signals_used.append({
                    "name": signal.name,
                    "value": round(signal.value, 4),
                    "confidence": round(signal.confidence, 3),
                    "impact_alpha": round(d_alpha, 3),
                    "impact_beta": round(d_beta, 3),
                })

        # Step 3: Posterior
        posterior_mean = self.beta_mean(alpha, beta)
        variance = self.beta_variance(alpha, beta)
        uncertainty = math.sqrt(variance) * 2  # ~95% width

        # Step 4: Overall confidence in our estimate
        total_obs = alpha + beta
        base_conf = min(0.9, total_obs / (self.prior_strength + 20))
        signal_conf = np.mean([s.confidence for s in signals]) if signals else 0
        overall_conf = base_conf * 0.6 + signal_conf * 0.4

        # Step 4: Logit calibration by category
        calibrated_posterior = calibrate_probability(posterior_mean, category)

        return BayesianEstimate(
            prior=round(prior_mean, 4),
            posterior=round(max(0.02, min(0.98, calibrated_posterior)), 4),
            signals_used=signals_used,
            alpha=round(alpha, 3),
            beta_param=round(beta, 3),
            uncertainty=round(uncertainty, 4),
            confidence=round(overall_conf, 3),
        )

    def estimate_with_raw_data(self, market_price: float,
                                sentiment_score: float = 0,
                                num_articles: int = 0,
                                volume_24h: float = 0,
                                avg_volume_7d: float = 0,
                                whale_positions: list = None,
                                price_history: list = None,
                                spread: float = 0,
                                liquidity: float = 0,
                                hours_to_resolution: float = None) -> BayesianEstimate:
        """
        Convenience method: build signals from raw data and estimate.
        hours_to_resolution: if provided, applies time-decay — markets close to
        resolution are more efficiently priced, so posterior is pulled toward market.
        Research: price efficiency increases as resolution approaches.
        """
        signals = []

        if num_articles > 0 or abs(sentiment_score) > 0.05:
            signals.append(self.sentiment_signal(sentiment_score, num_articles))

        if volume_24h > 0:
            signals.append(self.volume_momentum_signal(volume_24h, avg_volume_7d))

        if whale_positions:
            signals.append(self.whale_consensus_signal(whale_positions))

        if price_history and len(price_history) >= 3:
            signals.append(self.price_trend_signal(price_history))

        if spread > 0 or liquidity > 0:
            signals.append(self.liquidity_quality_signal(spread, liquidity, volume_24h))

        estimate = self.estimate_probability(market_price, signals)

        # ── Time-decay adjustment: pull toward market price as resolution nears ──
        # Research: prediction markets converge to truth in final hours.
        # <6h: 80% trust market | 6-24h: 60% | 1-7 days: normal | >7 days: no change
        if hours_to_resolution is not None and hours_to_resolution > 0:
            if hours_to_resolution < 6:
                blend = 0.80  # Heavy pull toward market in last hours
            elif hours_to_resolution < 24:
                blend = 0.60
            elif hours_to_resolution < 168:  # 7 days
                blend = 0.30
            else:
                blend = 0.0   # No adjustment for distant markets
            if blend > 0:
                blended = estimate.posterior * (1 - blend) + market_price * blend
                blended = max(0.02, min(0.98, blended))
                # Return modified estimate preserving other fields
                from dataclasses import replace as _dc_replace
                estimate = _dc_replace(estimate, posterior=round(blended, 4))

        return estimate

    # ─────────────────────────────────────────────────────────
    # CALIBRATION
    # ─────────────────────────────────────────────────────────

    def record_prediction(self, market_id: str, predicted_prob: float,
                           actual_outcome: bool):
        """Record a prediction for calibration tracking."""
        self._calibration_history.append({
            "market_id": market_id,
            "predicted": predicted_prob,
            "actual": 1.0 if actual_outcome else 0.0,
        })

    def brier_score(self) -> float:
        """
        Brier Score: mean squared error of probabilistic predictions.
        Lower = better. 0 = perfect, 0.25 = random.
        """
        if not self._calibration_history:
            return 0.25

        errors = [(h["predicted"] - h["actual"]) ** 2
                  for h in self._calibration_history]
        return round(np.mean(errors), 4)

    def calibration_curve(self, bins: int = 10) -> list[dict]:
        """
        Bin predictions and compare predicted vs actual frequencies.
        Perfect calibration: predicted_avg ≈ actual_avg in each bin.
        """
        if len(self._calibration_history) < bins:
            return []

        predictions = sorted(self._calibration_history, key=lambda h: h["predicted"])
        chunk_size = max(1, len(predictions) // bins)
        curve = []

        for i in range(0, len(predictions), chunk_size):
            chunk = predictions[i:i + chunk_size]
            avg_pred = np.mean([c["predicted"] for c in chunk])
            avg_actual = np.mean([c["actual"] for c in chunk])
            curve.append({
                "predicted_avg": round(float(avg_pred), 3),
                "actual_avg": round(float(avg_actual), 3),
                "count": len(chunk),
            })

        return curve
