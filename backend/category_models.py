"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Category-Specific Prediction Models
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Each category uses different signal types and models:

CRYPTO:  Oracle lag detection, spot price correlation, volatility
POLITICS: Poll aggregation (Bayesian), fundamentals, shy voter adj.
SPORTS:  ELO ratings, Poisson scoring, injury adjustment
WEATHER: Forecast model disagreement, climatology baseline
GEOPOLITICS: Escalation models, news velocity, sentiment spike
TECH:    Event-driven, regulatory probability, adoption curves

Based on documented success cases:
- 0x8dxd: $414K via crypto oracle lag (98% WR)
- French Whale: $85M via private polling + Bayesian
- Swarm NBA: $1.49M via ELO + AI ensemble on 3yr data
- Weather traders: 9000%+ ROI via forecast latency
"""

import math
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class MarketCategory(Enum):
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    WEATHER = "weather"
    GEOPOLITICS = "geopolitics"
    TECH = "tech"
    CULTURE = "culture"
    ECONOMICS = "economics"
    OTHER = "other"


@dataclass
class CategorySignal:
    """Output from a category-specific model."""
    category: str
    model_name: str
    estimated_prob: float      # 0-1
    confidence: float          # 0-1
    edge_vs_market: float      # -1 to 1
    signals_used: list         # list of signal names
    reasoning: str
    time_sensitivity: str      # "HIGH", "MEDIUM", "LOW" — how fast edge decays


# ═══════════════════════════════════════════════════════════
# CATEGORY CLASSIFIER
# ═══════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    MarketCategory.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
        "blockchain", "defi", "token", "nft", "binance", "coinbase",
        "altcoin", "dogecoin", "xrp", "cardano", "polygon", "matic",
    ],
    MarketCategory.POLITICS: [
        "election", "president", "congress", "senate", "vote", "poll",
        "democrat", "republican", "trump", "biden", "harris", "governor",
        "primary", "nominee", "cabinet", "impeach", "legislation",
    ],
    MarketCategory.SPORTS: [
        "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
        "championship", "playoff", "super bowl", "world cup", "olympics",
        "match", "tournament", "finals", "mvp", "scoring",
        # Patrones clave que capturan casi todos los mercados deportivos
        " vs ", " vs. ", "win on", "beat ", "beats ", "cover the spread",
        "over/under", "moneyline", "point spread",
        # Ligas y competiciones
        "premier league", "la liga", "serie a", "bundesliga", "champions league",
        "ncaa", "march madness", "playoff", "stanley cup", "world series",
        "wimbledon", "masters", "pga", "formula 1", "f1 race",
        # Equipos/deportistas comunes en Polymarket
        "lakers", "celtics", "warriors", "bulls", "nets", "heat", "bucks",
        "pistons", "wizards", "hornets", "magic", "jazz", "blazers",
        "hawks", "spurs", "suns", "nuggets", "clippers", "knicks",
        "patriots", "chiefs", "eagles", "cowboys", "49ers", "ravens",
        "yankees", "dodgers", "cubs", "red sox", "astros",
        "blackhawks", "penguins", "rangers", "maple leafs",
        "real madrid", "barcelona", "manchester", "liverpool", "arsenal",
        "chelsea", "juventus", "bayern", "psg", "atletico",
        "roma", "milan", "inter", "napoli", "sevilla",
        "hofstra", "akron", "zips", "crimson tide", "bruins", "wildcats",
        # Torneos de fútbol y eventos internacionales
        "euro 20", "euros 20", "copa america", "copa del mundo", "gold cup",
        "concacaf", "conmebol", "afcon", "afc cup", "asian cup",
        "coupe du monde", "nations league", "world rugby",
        # Países en contexto deportivo (cuando van con "win"/"beat"/"qualify")
        "qualify for", "advance to", "win the", "cup final", "league title",
    ],
    MarketCategory.WEATHER: [
        "temperature", "weather", "rain", "snow", "hurricane", "celsius",
        "fahrenheit", "forecast", "climate", "cold", "storm",
    ],
    MarketCategory.GEOPOLITICS: [
        "war", "conflict", "sanctions", "nato", "china", "russia",
        "ukraine", "iran", "strike", "military", "ceasefire", "treaty",
        "invasion", "nuclear", "missile", "peace", "diplomacy",
        # Líderes mundiales y eventos geopolíticos
        "netanyahu", "putin", "zelensky", "xi jinping", "modi", "kim jong",
        "hamas", "hezbollah", "houthi", "isis", "taliban", "gaza",
        "west bank", "israel", "palestine", "taiwan", "south china sea",
        "north korea", "south korea", "pakistan", "india border",
        "resign", "resignation", "removed from", "out by", "ousted",
        "coup", "assassination", "prime minister", "chancellor", "minister",
    ],
    MarketCategory.TECH: [
        "ai", "artificial intelligence", "openai", "google", "apple",
        "microsoft", "meta", "spacex", "launch", "release",
        "product", "regulation", "chip", "semiconductor", "agI",
        "chatgpt", "gemini", "grok", "claude", "llm", "model",
        "iphone", "android", "app store", "antitrust tech",
    ],
    MarketCategory.ECONOMICS: [
        "fed", "interest rate", "inflation", "gdp", "recession",
        "unemployment", "tariff", "trade", "s&p", "nasdaq",
        "yield", "bond", "treasury", "cpi", "jobs", "dow jones",
        "stock price", "stock market", "shares", "earnings",
        # Empresas cotizadas — precio de acción es economía, no tech
        "tesla stock", "apple stock", "nvidia stock", "amazon stock",
        "google stock", "microsoft stock", "meta stock",
        "ipo", "market cap", "revenue", "quarterly", "annual report",
    ],
}


def classify_market(question: str) -> MarketCategory:
    """Classify a market question into a category."""
    import re
    q_lower = question.lower()

    # Construye función de match: keywords cortos (≤3 chars) usan word boundaries
    # para evitar falsos positivos como "ai" en "spain", "eth" en "method", etc.
    def _kw_match(kw: str, text: str) -> bool:
        if len(kw) <= 3:
            return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))
        return kw in text

    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if _kw_match(kw, q_lower))
        if score > 0:
            scores[cat] = score

    # "vs"/"beat" es señal débil de deportes — solo suma peso, no fuerza la categoría
    if re.search(r'\bvs\.?\s', q_lower) or " beat " in q_lower or " beats " in q_lower:
        scores[MarketCategory.SPORTS] = scores.get(MarketCategory.SPORTS, 0) + 1

    # Bonus: "stock/price" → economics, PERO no si ya hay alguna señal crypto explícita
    if re.search(r'\b(stock|shares|ticker|price)\b', q_lower):
        _crypto_score = scores.get(MarketCategory.CRYPTO, 0)
        if _crypto_score == 0:  # Solo añadir si NO hay señal crypto (btc/eth/sol/etc.)
            scores[MarketCategory.ECONOMICS] = scores.get(MarketCategory.ECONOMICS, 0) + 2

    # Bonus: patrón "<nombre> out by" / "resign" sin contexto deportivo → geopolítica/política
    if re.search(r'\b(resign|out by|ousted|removed|fired|impeach)\b', q_lower):
        if MarketCategory.SPORTS not in scores or scores.get(MarketCategory.SPORTS, 0) < 2:
            scores[MarketCategory.GEOPOLITICS] = scores.get(MarketCategory.GEOPOLITICS, 0) + 2

    # Bonus: países o líderes con "will X happen" sin keyword explícita
    _geopolitical_names = [
        "netanyahu", "zelensky", "putin", "xi ", "modi ", "kim jong",
        "hamas", "hezbollah", "houthi", "gaza", "ukraine", "taiwan",
        "israel", "iran nuclear", "north korea",
    ]
    if any(name in q_lower for name in _geopolitical_names):
        scores[MarketCategory.GEOPOLITICS] = scores.get(MarketCategory.GEOPOLITICS, 0) + 3

    if not scores:
        return MarketCategory.OTHER
    return max(scores, key=scores.get)


# ═══════════════════════════════════════════════════════════
# CRYPTO MODEL — Oracle Lag + Spot Correlation
# ═══════════════════════════════════════════════════════════
# Research: 0x8dxd made $414K exploiting 2-15s Chainlink lag.
# Strategy: compare real-time spot price vs Polymarket implied.
# Edge decays in 2.7 seconds average (2026 data).

class CryptoModel:
    """
    __signature__ = "ddchack"
    Crypto market prediction using spot price correlation.
    """

    @staticmethod
    def oracle_lag_signal(spot_price: float, oracle_price: float,
                          threshold_pct: float = 0.3) -> CategorySignal:
        """
        Detect oracle lag for short-term crypto markets.
        When spot has already moved but oracle hasn't updated.

        Research basis: $313→$414K bot (98% WR, 6615 trades)
        """
        if oracle_price <= 0:
            return CategorySignal("crypto", "oracle_lag", 0.5, 0, 0, [], "", "HIGH")

        pct_diff = (spot_price - oracle_price) / oracle_price * 100

        if abs(pct_diff) < threshold_pct:
            return CategorySignal("crypto", "oracle_lag", 0.5, 0.1, 0,
                                   ["spot_price"], "No significant lag detected", "HIGH")

        # If spot is higher than oracle → price going UP
        if pct_diff > 0:
            prob_up = min(0.95, 0.5 + pct_diff / 2.0 * 0.1)
            edge = prob_up - 0.5
            return CategorySignal("crypto", "oracle_lag", prob_up,
                                   min(0.9, abs(pct_diff) / 1.0),
                                   edge, ["spot_price", "oracle_price"],
                                   f"Spot {pct_diff:+.2f}% above oracle. UP likely.",
                                   "HIGH")
        else:
            prob_down = min(0.95, 0.5 + abs(pct_diff) / 2.0 * 0.1)
            edge = prob_down - 0.5
            return CategorySignal("crypto", "oracle_lag", 1 - prob_down,
                                   min(0.9, abs(pct_diff) / 1.0),
                                   -edge, ["spot_price", "oracle_price"],
                                   f"Spot {pct_diff:+.2f}% below oracle. DOWN likely.",
                                   "HIGH")

    @staticmethod
    def volatility_signal(prices_1h: list[float]) -> CategorySignal:
        """
        Volatility-based signal. High volatility → wider spreads, more
        opportunity but also more risk.
        """
        if len(prices_1h) < 5:
            return CategorySignal("crypto", "volatility", 0.5, 0, 0, [], "", "MEDIUM")

        returns = np.diff(np.log(np.array(prices_1h)))
        vol = float(np.std(returns))
        vol_annualized = vol * math.sqrt(365 * 24)

        # High vol = uncertain → reduce confidence
        conf_penalty = min(0.5, vol * 10)
        return CategorySignal("crypto", "volatility", 0.5, 0.3,
                               0, ["price_history"],
                               f"1h vol: {vol*100:.2f}%, annualized: {vol_annualized*100:.0f}%",
                               "MEDIUM")


# ═══════════════════════════════════════════════════════════
# POLITICS MODEL — Poll Aggregation + Shy Voter Adjustment
# ═══════════════════════════════════════════════════════════
# Research: French Whale used private YouGov "neighbor polling"
# to find shy Trump voter effect. Polymarket was marginally
# more accurate than polls in 2024. UCLA model: 72.6% Trump.

class PoliticsModel:
    """Bayesian poll aggregation for election markets."""

    @staticmethod
    def aggregate_polls(polls: list[dict], market_price: float) -> CategorySignal:
        """
        Bayesian poll aggregation.
        polls: [{"candidate_pct": float, "sample_size": int,
                 "days_old": int, "pollster_rating": float}]

        Uses inverse-variance weighting with recency decay.
        """
        if not polls:
            return CategorySignal("politics", "poll_agg", market_price, 0.1, 0,
                                   [], "No polls available", "LOW")

        weighted_sum = 0.0
        weight_total = 0.0

        for poll in polls:
            # Weight = sample_size / (1 + days_old) * pollster_rating
            recency = 1.0 / (1.0 + poll.get("days_old", 30) / 7.0)
            rating = poll.get("pollster_rating", 0.5)
            n = poll.get("sample_size", 500)
            # Inverse variance weight: proportional to sqrt(n)
            w = math.sqrt(n) * recency * rating
            weighted_sum += poll["candidate_pct"] / 100.0 * w
            weight_total += w

        if weight_total <= 0:
            return CategorySignal("politics", "poll_agg", market_price, 0.1, 0,
                                   [], "Insufficient poll data", "LOW")

        agg_prob = weighted_sum / weight_total

        # Shy voter adjustment: polls historically underestimate
        # certain candidates by 1-3 points
        # This is configurable — set to 0 by default
        shy_adjustment = 0.0  # User can configure per-candidate
        agg_prob = max(0.02, min(0.98, agg_prob + shy_adjustment))

        edge = agg_prob - market_price
        conf = min(0.8, math.sqrt(weight_total) / 20.0)

        return CategorySignal("politics", "poll_aggregation", agg_prob,
                               conf, edge,
                               [f"{len(polls)} polls"],
                               f"Aggregated {len(polls)} polls → {agg_prob*100:.1f}% "
                               f"(market: {market_price*100:.1f}%)",
                               "LOW")

    @staticmethod
    def fundamentals_signal(incumbency: bool, economy_good: bool,
                             approval_rating: float) -> CategorySignal:
        """
        Simple fundamentals model for elections.
        Research: fundamentals models have ~55-60% accuracy alone.
        """
        base = 0.50
        if incumbency:
            base += 0.05
        if economy_good:
            base += 0.08
        # Approval rating effect
        base += (approval_rating - 45) * 0.01
        base = max(0.20, min(0.80, base))

        return CategorySignal("politics", "fundamentals", base, 0.3,
                               0, ["incumbency", "economy", "approval"],
                               f"Fundamentals model: {base*100:.0f}%",
                               "LOW")


# ═══════════════════════════════════════════════════════════
# SPORTS MODEL — ELO + Poisson
# ═══════════════════════════════════════════════════════════
# Research: ELO equivalent to logistic regression with SGD.
# Weighted ELO produced 3.56% ROI in tennis.
# Poisson for goal/score prediction (Karlis & Ntzoufras 2003).
# Swarm NBA model: $1.49M using 3yr data + 4096-agent ensemble.

class SportsModel:
    """ELO rating system + Poisson scoring model for sports markets."""

    @staticmethod
    def elo_win_probability(rating_a: float, rating_b: float) -> float:
        """
        P(A wins) = 1 / (1 + 10^(-(Ra - Rb)/400))

        Standard ELO formula. Research: equivalent to logistic
        regression with SGD on match outcomes.
        """
        return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))

    @staticmethod
    def elo_update(rating: float, expected: float, actual: float,
                    k: float = 32.0) -> float:
        """Update ELO rating after a match. K=32 default, 16 for established."""
        return rating + k * (actual - expected)

    @staticmethod
    def elo_signal(rating_a: float, rating_b: float,
                    market_price_a: float) -> CategorySignal:
        """Generate signal from ELO ratings vs market price."""
        elo_prob = SportsModel.elo_win_probability(rating_a, rating_b)
        edge = elo_prob - market_price_a

        return CategorySignal("sports", "elo_rating", elo_prob,
                               0.5,  # ELO alone is ~55% accurate
                               edge,
                               ["elo_a", "elo_b"],
                               f"ELO: {elo_prob*100:.1f}% vs market {market_price_a*100:.1f}%",
                               "MEDIUM")

    @staticmethod
    def poisson_probability(lambda_a: float, lambda_b: float,
                             target: str = "a_wins") -> float:
        """
        Poisson model for score-based sports (soccer, hockey, etc.)

        lambda_a = expected goals for team A
        lambda_b = expected goals for team B

        P(X=k) = (λ^k * e^-λ) / k!

        Research: Karlis & Ntzoufras (2003) bivariate Poisson.
        """
        from scipy.stats import poisson

        max_goals = 10
        prob = 0.0

        if target == "a_wins":
            for a in range(max_goals):
                for b in range(a):  # b < a → A wins
                    prob += poisson.pmf(a, lambda_a) * poisson.pmf(b, lambda_b)
        elif target == "draw":
            for g in range(max_goals):
                prob += poisson.pmf(g, lambda_a) * poisson.pmf(g, lambda_b)
        elif target == "b_wins":
            for b in range(max_goals):
                for a in range(b):
                    prob += poisson.pmf(a, lambda_a) * poisson.pmf(b, lambda_b)
        elif target == "over_2.5":
            for a in range(max_goals):
                for b in range(max_goals):
                    if a + b > 2:
                        prob += poisson.pmf(a, lambda_a) * poisson.pmf(b, lambda_b)

        return prob

    @staticmethod
    def poisson_signal(lambda_a: float, lambda_b: float,
                        market_price: float,
                        target: str = "a_wins") -> CategorySignal:
        """Generate signal from Poisson model vs market price."""
        prob = SportsModel.poisson_probability(lambda_a, lambda_b, target)
        edge = prob - market_price

        return CategorySignal("sports", "poisson_model", prob,
                               0.45, edge,
                               ["expected_goals_a", "expected_goals_b"],
                               f"Poisson({target}): {prob*100:.1f}%",
                               "MEDIUM")


# ═══════════════════════════════════════════════════════════
# WEATHER MODEL — Forecast Disagreement + Climatology
# ═══════════════════════════════════════════════════════════
# Research: weather traders made 9000%+ ROI via forecast latency.
# Models (GFS, ECMWF) update every 6 hours.
# Edge: new model run data available immediately but slow to
# reflect in Polymarket prices.

class WeatherModel:
    """Weather prediction using forecast data and climatology."""

    @staticmethod
    def forecast_signal(forecast_value: float, resolution_threshold: float,
                         market_price_yes: float,
                         forecast_uncertainty: float = 2.0) -> CategorySignal:
        """
        Compare weather forecast to market resolution criteria.

        E.g., market: "Will NYC temp exceed 90°F?"
        forecast: 88°F, threshold: 90°F, uncertainty: ±2°F

        Uses normal distribution around forecast to estimate probability.
        """
        from scipy.stats import norm

        # P(actual > threshold) given forecast ± uncertainty
        z = (resolution_threshold - forecast_value) / max(0.1, forecast_uncertainty)
        prob_exceeds = 1.0 - norm.cdf(z)
        prob_exceeds = max(0.02, min(0.98, prob_exceeds))

        edge = prob_exceeds - market_price_yes

        return CategorySignal("weather", "forecast_model", prob_exceeds,
                               min(0.8, 1.0 / (1.0 + forecast_uncertainty)),
                               edge,
                               ["forecast", "threshold", "uncertainty"],
                               f"Forecast {forecast_value}° vs threshold {resolution_threshold}°. "
                               f"P(exceed)={prob_exceeds*100:.1f}%, market={market_price_yes*100:.1f}%",
                               "MEDIUM")

    @staticmethod
    def model_disagreement_signal(gfs_forecast: float,
                                    ecmwf_forecast: float) -> float:
        """
        When GFS and ECMWF disagree significantly, uncertainty is high.
        Returns disagreement score 0-1. High = reduce position sizes.
        """
        diff = abs(gfs_forecast - ecmwf_forecast)
        # Normalize: 5°F difference = high disagreement
        return min(1.0, diff / 5.0)


# ═══════════════════════════════════════════════════════════
# GEOPOLITICS MODEL — Escalation + News Velocity
# ═══════════════════════════════════════════════════════════
# Research: 29.7% activity rate (fastest growing). Markets move
# 10-15 min before traditional media. Iran strike contract
# hit $188M volume. Correlation with defense stocks (RTX).

class GeopoliticsModel:
    """Geopolitical event probability estimation."""

    @staticmethod
    def escalation_signal(current_events: list[str],
                          market_price: float) -> CategorySignal:
        """
        Simple escalation scoring based on event keywords.
        In production, this would use an LLM for deeper analysis.
        """
        escalation_keywords = {
            "high": ["attack", "strike", "invasion", "nuclear", "war declared",
                     "troops deployed", "missiles launched", "bombing"],
            "medium": ["sanctions", "threat", "mobilization", "border",
                       "ultimatum", "military buildup", "embargo"],
            "low": ["talks", "negotiation", "ceasefire", "diplomacy",
                    "summit", "agreement", "withdrawal"],
        }

        score = 0
        events_text = " ".join(current_events).lower()

        for kw in escalation_keywords["high"]:
            if kw in events_text:
                score += 3
        for kw in escalation_keywords["medium"]:
            if kw in events_text:
                score += 1
        for kw in escalation_keywords["low"]:
            if kw in events_text:
                score -= 1

        # Normalize to probability shift
        shift = max(-0.15, min(0.15, score * 0.03))
        estimated = max(0.05, min(0.95, market_price + shift))
        edge = estimated - market_price

        return CategorySignal("geopolitics", "escalation_model",
                               estimated, min(0.6, abs(score) / 10),
                               edge, [f"{len(current_events)} events"],
                               f"Escalation score: {score}. "
                               f"Estimated: {estimated*100:.1f}%",
                               "HIGH")


# ═══════════════════════════════════════════════════════════
# HMM REGIME DETECTOR
# ═══════════════════════════════════════════════════════════
# Research: Wang, Lin & Mikhelson (2020) showed HMM beats
# individual factor models for regime detection.
# 2-3 states: Bull/Low-vol, Bear/High-vol, Sideways.
# Action: reduce position sizes in high-vol regime.

class RegimeDetector:
    """
    Simple regime detection using price returns.
    Classifies market into volatility regimes for position sizing.
    """

    @staticmethod
    def detect_regime(returns: list[float]) -> dict:
        """
        Classify current market regime based on recent returns.

        Returns: {"regime": str, "vol_multiplier": float,
                  "confidence": float, "stats": dict}
        """
        if len(returns) < 10:
            return {"regime": "UNKNOWN", "vol_multiplier": 0.5,
                    "confidence": 0, "stats": {}}

        r = np.array(returns)
        mean_r = float(np.mean(r))
        std_r = float(np.std(r))
        skew = float(np.mean(((r - mean_r) / max(1e-6, std_r)) ** 3))

        # Simple regime classification
        if std_r < 0.02 and mean_r > 0:
            regime = "BULL_LOW_VOL"
            multiplier = 1.2  # Increase positions slightly
        elif std_r > 0.05 and mean_r < 0:
            regime = "BEAR_HIGH_VOL"
            multiplier = 0.4  # Significantly reduce
        elif std_r > 0.04:
            regime = "HIGH_VOL"
            multiplier = 0.6  # Reduce positions
        elif abs(mean_r) < 0.005:
            regime = "SIDEWAYS"
            multiplier = 0.8  # Slight reduction (mean reversion)
        else:
            regime = "TRENDING"
            multiplier = 1.0

        return {
            "regime": regime,
            "vol_multiplier": round(multiplier, 2),
            "confidence": min(0.8, len(returns) / 50),
            "stats": {
                "mean_return": round(mean_r, 6),
                "std_return": round(std_r, 6),
                "skewness": round(skew, 4),
                "n_observations": len(returns),
            },
        }


# ═══════════════════════════════════════════════════════════
# MASTER CATEGORY EVALUATOR
# ═══════════════════════════════════════════════════════════

class CategoryEvaluator:
    """
    Master evaluator that selects and runs the right model
    based on market category.
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self):
        self.crypto = CryptoModel()
        self.politics = PoliticsModel()
        self.sports = SportsModel()
        self.weather = WeatherModel()
        self.geopolitics = GeopoliticsModel()
        self.regime = RegimeDetector()

    def evaluate(self, question: str, market_price: float,
                 context: dict = None) -> CategorySignal:
        """
        Evaluate a market using the appropriate category model.

        context: optional dict with category-specific data like:
          - spot_price, oracle_price (crypto)
          - polls (politics)
          - elo_a, elo_b (sports)
          - forecast, threshold (weather)
          - events (geopolitics)
        """
        ctx = context or {}
        category = classify_market(question)

        if category == MarketCategory.CRYPTO:
            spot = ctx.get("spot_price", 0)
            oracle = ctx.get("oracle_price", 0)
            if spot > 0 and oracle > 0:
                return self.crypto.oracle_lag_signal(spot, oracle)
            return CategorySignal("crypto", "default", market_price,
                                   0.2, 0, [], "No spot/oracle data", "LOW")

        elif category == MarketCategory.POLITICS:
            polls = ctx.get("polls", [])
            if polls:
                return self.politics.aggregate_polls(polls, market_price)
            return CategorySignal("politics", "default", market_price,
                                   0.2, 0, [], "No poll data", "LOW")

        elif category == MarketCategory.SPORTS:
            # Intentar cargar contexto desde SportsIntelligence si está disponible
            _si_ctx: dict = {}
            try:
                from sports_intel import get_sports_intel
                _si = get_sports_intel()
                if _si is not None and _si._initialized:
                    _si_ctx = _si.get_context(question)
            except Exception:
                pass

            # Merge con el contexto manual que pase el caller (tiene precedencia)
            elo_a = ctx.get("elo_a", 0) or _si_ctx.get("elo_a", 0)
            elo_b = ctx.get("elo_b", 0) or _si_ctx.get("elo_b", 0)

            if elo_a > 0 and elo_b > 0:
                # Si SportsIntel ya calculó elo_prob con injury+altitude adj, usarlo
                elo_prob = _si_ctx.get("elo_prob", 0)
                if elo_prob > 0:
                    edge = elo_prob - market_price
                    signals = ["elo_a", "elo_b"]
                    if _si_ctx.get("injury_adj", 0) != 0:
                        signals.append("injury_adj")
                    if _si_ctx.get("altitude_adj", 0) != 0:
                        signals.append("altitude_adj")
                    inj_info = ""
                    if _si_ctx.get("injuries_a") or _si_ctx.get("injuries_b"):
                        ia = len(_si_ctx.get("injuries_a", []))
                        ib = len(_si_ctx.get("injuries_b", []))
                        inj_info = f" | Lesiones: {_si_ctx['team_a']}={ia} {_si_ctx['team_b']}={ib}"
                    return CategorySignal(
                        "sports", "elo_intel", elo_prob,
                        0.55,  # ELO+injuries+altitude: confianza moderada
                        edge, signals,
                        f"ELO: {elo_a:.0f} vs {elo_b:.0f} → {elo_prob*100:.1f}% "
                        f"(market: {market_price*100:.1f}%){inj_info}",
                        "MEDIUM"
                    )
                return self.sports.elo_signal(elo_a, elo_b, market_price)

            lambda_a = ctx.get("expected_goals_a", 0)
            lambda_b = ctx.get("expected_goals_b", 0)
            if lambda_a > 0 and lambda_b > 0:
                return self.sports.poisson_signal(lambda_a, lambda_b,
                                                   market_price)
            # Sin ELO ni Poisson: no hay señal real sobre el mercado.
            # Confidence=0.05 (mínimo) para que el ensemble ignore este modelo
            # y el precio de mercado (70% de peso) domine la estimación.
            return CategorySignal("sports", "no_signal", market_price,
                                   0.05, 0, [],
                                   "Sin datos ELO/Poisson — usando precio de mercado como referencia",
                                   "LOW")

        elif category == MarketCategory.WEATHER:
            forecast = ctx.get("forecast_value", 0)
            threshold = ctx.get("threshold", 0)
            if forecast > 0 and threshold > 0:
                return self.weather.forecast_signal(
                    forecast, threshold, market_price,
                    ctx.get("uncertainty", 2.0))
            return CategorySignal("weather", "default", market_price,
                                   0.2, 0, [], "No forecast data", "LOW")

        elif category == MarketCategory.GEOPOLITICS:
            events = ctx.get("events", [])
            return self.geopolitics.escalation_signal(events, market_price)

        else:
            # For tech, economics, culture, other — use default
            return CategorySignal(category.value if isinstance(category, MarketCategory) else "other",
                                   "default", market_price,
                                   0.2, 0, [], "Using market price as baseline",
                                   "LOW")

    @staticmethod
    def get_category_allocation() -> dict:
        """
        Recommended portfolio allocation by category.
        Research: optimal diversification is 5-10 simultaneous positions.
        """
        return {
            "politics": {"min_pct": 15, "max_pct": 30, "description": "Elections, legislation"},
            "economics": {"min_pct": 15, "max_pct": 25, "description": "Fed, inflation, GDP"},
            "geopolitics": {"min_pct": 10, "max_pct": 20, "description": "Conflicts, sanctions"},
            "crypto": {"min_pct": 10, "max_pct": 20, "description": "BTC/ETH price, regulation"},
            "sports": {"min_pct": 5, "max_pct": 15, "description": "NBA, NFL, soccer"},
            "weather": {"min_pct": 5, "max_pct": 10, "description": "Temperature, storms"},
            "tech": {"min_pct": 5, "max_pct": 15, "description": "AI, launches, regulation"},
        }
