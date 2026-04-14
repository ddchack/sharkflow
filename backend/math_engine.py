"""
══════════════════════════════════════════════════════════════
Polymarket Bot - Mathematical Engine
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Implements:
- Kelly Criterion (full & fractional)
- Expected Value (EV) calculation
- Edge detection vs market odds
- Implied probability extraction
- Risk-adjusted position sizing
- Sharpe-like ratio for prediction markets
- Compound growth simulator
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────────────────
# POLYMARKET REAL FEE RATES (by category)
# Source: polymarket.com fee schedule (taker rates)
# ─────────────────────────────────────────────────────────
CATEGORY_FEE_RATES: dict[str, float] = {
    "crypto": 0.072,        # 7.2% taker
    "sports": 0.030,        # 3.0% taker
    "politics": 0.040,      # 4.0% taker
    "economics": 0.040,     # 4.0% taker
    "finance": 0.040,       # 4.0% taker
    "technology": 0.040,    # 4.0% taker
    "geopolitics": 0.000,   # 0% fee-free
    "world": 0.040,         # 4.0% taker
    "science": 0.040,       # 4.0% taker
    "entertainment": 0.040, # 4.0% taker
    "other": 0.040,         # 4.0% default taker
}

MAKER_REBATE_RATES: dict[str, float] = {
    "crypto": 0.20,         # 20% rebate if maker order
    "sports": 0.25,
    "politics": 0.25,
    "geopolitics": 0.00,
    "other": 0.25,
}

def get_polymarket_fee(price: float, category: str = "other", maker: bool = False) -> float:
    """
    Real Polymarket fee: feeRate * price * (1 - price) per share.
    Maker orders receive a rebate instead of paying a fee.
    Returns cost per dollar wagered (positive = cost, negative = rebate received).
    """
    rate = CATEGORY_FEE_RATES.get(category.lower(), 0.040)
    if maker:
        rebate = MAKER_REBATE_RATES.get(category.lower(), 0.25)
        return -rate * rebate * price * (1 - price)  # negative = income
    return rate * price * (1 - price)

# Legacy aliases kept for backward compatibility
FEE_RATES = {
    'crypto': 0.072,
    'sports': 0.03,
    'politics': 0.0,
    'geopolitics': 0.0,
    'finance': 0.02,
    'science': 0.02,
    'default': 0.02,
}

def calculate_fee(price: float, size: float, category: str, is_maker: bool = False) -> float:
    """Fee real de Polymarket según categoría. Makers reciben rebate (0%)."""
    rate = FEE_RATES.get(category.lower(), FEE_RATES['default'])
    if is_maker:
        rate *= 0.0  # makers get rebate on most categories
    return rate * price * (1 - price) * size


def apply_edge_skepticism(kelly_fraction: float, edge: float) -> float:
    """
    Descuenta Kelly cuando el edge reportado es sospechosamente alto.
    Edges >12% se penalizan progresivamente (real histórico: 1-3%).
    """
    if edge > 0.12:
        discount = 1.0 - 0.5 * min(1.0, (edge - 0.12) / (0.30 - 0.12))
        kelly_fraction *= max(0.5, discount)
    return kelly_fraction


@dataclass
class MarketOdds:
    """Represents a binary market's odds structure."""
    yes_price: float  # Market price for YES (0.01 - 0.99)
    no_price: float   # Market price for NO (0.01 - 0.99)
    volume_24h: float = 0.0
    liquidity: float = 0.0
    spread: float = 0.0

    @property
    def implied_yes_prob(self) -> float:
        return self.yes_price

    @property
    def implied_no_prob(self) -> float:
        return self.no_price

    @property
    def overround(self) -> float:
        """Market overround (vig). 1.0 = fair, >1.0 = bookmaker edge."""
        return self.yes_price + self.no_price


@dataclass
class BetRecommendation:
    """Output of the analysis engine for a single market."""
    market_id: str
    market_question: str
    side: str  # "YES" or "NO"
    market_price: float
    estimated_true_prob: float
    edge_percent: float
    expected_value: float
    kelly_fraction: float
    recommended_bet_usd: float
    confidence_score: float  # 0-100
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    reasoning: list = field(default_factory=list)
    # Audit trail for post-hoc Kelly analysis
    kelly_raw: float = 0.0           # Full Kelly before multiplier
    kelly_skepticism: float = 1.0    # Skepticism discount applied
    fee_cost_per_dollar: float = 0.0 # Real fee paid per dollar
    category: str = "other"
    maker_order: bool = False


class MathEngine:
    """
    Core mathematical engine for evaluating prediction market bets.
    __signature__ = "ddchack"
    """

    def __init__(self, max_capital: float = 100.0, kelly_multiplier: float = 0.25,
                 min_edge: float = 0.03, min_ev: float = 0.05):
        self.max_capital = max_capital
        self.kelly_multiplier = kelly_multiplier  # Fractional Kelly (0.25 = quarter Kelly)
        self.min_edge = min_edge  # Minimum edge to consider a bet (3%)
        self.min_ev = min_ev  # Minimum EV to consider a bet (5 cents per dollar)

    # ─────────────────────────────────────────────────────────
    # CORE FORMULAS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def implied_probability(price: float) -> float:
        """Extract implied probability from market price."""
        return max(0.01, min(0.99, price))

    @staticmethod
    def decimal_odds(price: float) -> float:
        """Convert market price to decimal odds. E.g., price=0.40 → odds=2.50"""
        if price <= 0:
            return float('inf')
        return 1.0 / price

    def expected_value(self, true_prob: float, price: float,
                       category: str = "other", maker: bool = False) -> float:
        """
        Calculate Expected Value per $1 bet.

        EV = (p_true * payout) - (1 - p_true) * stake - fee_cost
        Where payout = (1/price) - 1 (net profit if win)

        Example: true_prob=0.60, price=0.40
          EV = 0.60 * (1/0.40 - 1) - 0.40 * 1 = 0.60 * 1.5 - 0.40 = 0.50
        Fee is subtracted to reflect real Polymarket taker/maker costs.
        """
        if price <= 0 or price >= 1:
            return 0.0
        net_profit = (1.0 / price) - 1.0
        fee_cost = get_polymarket_fee(price, category, maker)
        ev = (true_prob * net_profit) - (1.0 - true_prob) - fee_cost
        return round(ev, 6)

    def edge(self, true_prob: float, market_prob: float) -> float:
        """
        Calculate edge: difference between our estimate and the market's.
        Positive = we think market is wrong in our favor.
        """
        return true_prob - market_prob

    def kelly_criterion(self, true_prob: float, price: float,
                        category: str = "other", maker: bool = False) -> float:
        """
        Full Kelly Criterion for binary outcome.

        f* = (b_net*p - q) / b_net
        Where:
          b = net decimal odds = (1/price) - 1
          b_net = b adjusted for real Polymarket fees
          p = true probability of winning
          q = 1 - p

        Returns fraction of bankroll to bet (0.0 to 1.0).
        Negative values mean don't bet.
        """
        if price <= 0 or price >= 1:
            return 0.0
        b = (1.0 / price) - 1.0
        if b <= 0:
            return 0.0
        p = true_prob
        q = 1.0 - p
        fee_cost = get_polymarket_fee(price, category, maker)
        # Adjust net odds for fee: fee reduces the net profit
        b_net = b - fee_cost * (1 + b)  # fee reduces effective payout
        if b_net <= 0:
            return 0.0
        f_star = (b_net * p - q) / b_net
        return max(0.0, f_star)

    def fractional_kelly(self, true_prob: float, price: float) -> float:
        """
        Fractional Kelly: more conservative position sizing.
        Uses self.kelly_multiplier (default 0.25 = quarter Kelly).
        Applies edge skepticism to discount suspiciously high edges.
        """
        full_kelly = self.kelly_criterion(true_prob, price)
        frac = full_kelly * self.kelly_multiplier
        # Calcular edge para aplicar escepticismo
        edge_val = abs(true_prob - price)
        frac = apply_edge_skepticism(frac, edge_val)
        return frac

    def position_size_usd(self, true_prob: float, price: float,
                          available_capital: float = None) -> float:
        """Calculate recommended bet size in USD."""
        capital = available_capital or self.max_capital
        frac = self.fractional_kelly(true_prob, price)
        raw_size = frac * capital
        # Cap at 25% of capital per single bet for safety
        max_single = capital * 0.25
        return round(min(raw_size, max_single), 2)

    # ─────────────────────────────────────────────────────────
    # ADVANCED METRICS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def sharpe_ratio_estimate(ev_per_dollar: float, win_prob: float) -> float:
        """
        Estimate a Sharpe-like ratio for a prediction market bet.
        Higher = better risk-adjusted return.
        
        SR ≈ EV / σ where σ = sqrt(p * (1-p)) * payout_range
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        sigma = math.sqrt(win_prob * (1.0 - win_prob))
        if sigma == 0:
            return 0.0
        return ev_per_dollar / sigma

    @staticmethod
    def breakeven_probability(price: float) -> float:
        """The minimum true probability needed to break even at this price."""
        return price

    def compound_growth(self, bets: list[dict]) -> dict:
        """
        Simulate compound growth over a series of bets.
        Each bet: {"true_prob": float, "price": float, "won": bool}
        Returns growth trajectory.
        """
        capital = self.max_capital
        trajectory = [capital]
        wins = 0
        losses = 0

        for bet in bets:
            size = self.position_size_usd(bet["true_prob"], bet["price"], capital)
            if bet["won"]:
                profit = size * ((1.0 / bet["price"]) - 1.0)
                capital += profit
                wins += 1
            else:
                capital -= size
                losses += 1
            trajectory.append(round(capital, 2))

        return {
            "final_capital": round(capital, 2),
            "total_return_pct": round(((capital - self.max_capital) / self.max_capital) * 100, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(1, wins + losses) * 100, 1),
            "trajectory": trajectory
        }

    # ─────────────────────────────────────────────────────────
    # CONFIDENCE & SCORING
    # ─────────────────────────────────────────────────────────

    def confidence_score(self, edge: float, ev: float, volume_24h: float,
                         liquidity: float, sentiment_score: float = 0.0,
                         spread: float = 0.0) -> float:
        """
        Composite confidence score (0-100) combining multiple signals.
        
        Weights:
          - Edge magnitude: 30%
          - EV magnitude: 25%
          - Volume (liquidity proxy): 15%
          - Liquidity depth: 15%
          - News sentiment alignment: 10%
          - Spread tightness: 5%
        """
        # Normalize edge (0-1 scale, where 0.20+ = max)
        edge_score = min(1.0, abs(edge) / 0.20)

        # Normalize EV (0-1 scale, where 0.50+ = max)
        ev_score = min(1.0, max(0, ev) / 0.50)

        # Volume score (log scale, $100K+ = great)
        vol_score = min(1.0, math.log10(max(1, volume_24h)) / 5.0)

        # Liquidity score (log scale, $50K+ = great)
        liq_score = min(1.0, math.log10(max(1, liquidity)) / 4.7)

        # Sentiment alignment (-1 to 1 → 0 to 1)
        sent_score = (sentiment_score + 1.0) / 2.0

        # Spread tightness (smaller = better, 0.02 = tight)
        spread_score = max(0, 1.0 - (spread / 0.10))

        composite = (
            edge_score * 0.30 +
            ev_score * 0.25 +
            vol_score * 0.15 +
            liq_score * 0.15 +
            sent_score * 0.10 +
            spread_score * 0.05
        )

        return round(composite * 100, 1)

    def risk_level(self, kelly_frac: float, spread: float, volume: float) -> str:
        """Categorize risk based on multiple factors."""
        if kelly_frac > 0.15 or spread > 0.08 or volume < 1000:
            return "HIGH"
        elif kelly_frac > 0.08 or spread > 0.04 or volume < 10000:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def edge_skepticism_discount(edge_fraction: float) -> float:
        """
        Apply a discount to oversized edges (LLM miscalibration prevention).
        Edges > 12% get a linear discount: 1.0 at 12% → 0.5 at 30%+
        Based on: polybot research showing LLM edges > 12% are often noise.
        """
        if edge_fraction <= 0.12:
            return 1.0
        if edge_fraction >= 0.30:
            return 0.5
        return 1.0 - 0.5 * (edge_fraction - 0.12) / (0.30 - 0.12)

    # ─────────────────────────────────────────────────────────
    # MAIN EVALUATION
    # ─────────────────────────────────────────────────────────

    def evaluate_market(self, market_id: str, question: str,
                        odds: MarketOdds, estimated_true_prob: float,
                        sentiment_score: float = 0.0,
                        available_capital: float = None,
                        min_edge_override: float = None) -> Optional[BetRecommendation]:
        """
        Full evaluation pipeline for a single market.
        Returns BetRecommendation if the bet meets thresholds, else None.
        """
        capital = available_capital or self.max_capital
        reasoning = []

        # Determine which side to bet
        yes_edge = self.edge(estimated_true_prob, odds.implied_yes_prob)
        no_edge = self.edge(1.0 - estimated_true_prob, odds.implied_no_prob)

        if yes_edge > no_edge and yes_edge > 0:
            side = "YES"
            price = odds.yes_price
            true_p = estimated_true_prob
            edge_val = yes_edge
        elif no_edge > 0:
            side = "NO"
            price = odds.no_price
            true_p = 1.0 - estimated_true_prob
            edge_val = no_edge
        else:
            return None  # No edge on either side

        # Check minimum edge threshold (override permite bajar el filtro desde la UI)
        effective_min_edge = min_edge_override if min_edge_override is not None else self.min_edge
        if edge_val < effective_min_edge:
            return None

        # Calculate metrics
        ev = self.expected_value(true_p, price)
        if ev < self.min_ev:
            return None

        kelly_f = self.fractional_kelly(true_p, price)
        # Apply edge skepticism discount (prevents oversizing on possibly miscalibrated signals)
        skepticism = self.edge_skepticism_discount(edge_val)
        kelly_f = kelly_f * skepticism
        bet_size = self.position_size_usd(true_p, price, capital)
        bet_size = bet_size * skepticism
        confidence = self.confidence_score(
            edge_val, ev, odds.volume_24h, odds.liquidity,
            sentiment_score, odds.spread
        )
        risk = self.risk_level(kelly_f, odds.spread, odds.volume_24h)

        # Build reasoning
        reasoning.append(f"Edge: {edge_val*100:.1f}% over market implied probability")
        reasoning.append(f"EV: +${ev:.4f} per $1 wagered")
        reasoning.append(f"Kelly suggests {kelly_f*100:.1f}% of bankroll")
        if sentiment_score > 0.2:
            reasoning.append(f"News sentiment POSITIVE ({sentiment_score:.2f}) supports {side}")
        elif sentiment_score < -0.2:
            reasoning.append(f"News sentiment NEGATIVE ({sentiment_score:.2f}) caution on {side}")
        if odds.volume_24h > 50000:
            reasoning.append("High volume market - good liquidity")
        if odds.spread < 0.03:
            reasoning.append("Tight spread - efficient market")

        return BetRecommendation(
            market_id=market_id,
            market_question=question,
            side=side,
            market_price=price,
            estimated_true_prob=true_p,
            edge_percent=round(edge_val * 100, 2),
            expected_value=round(ev, 4),
            kelly_fraction=round(kelly_f, 4),
            recommended_bet_usd=bet_size,
            confidence_score=confidence,
            risk_level=risk,
            reasoning=reasoning,
            kelly_raw=round(self.kelly_criterion(true_p, price), 4),
            kelly_skepticism=round(self.edge_skepticism_discount(edge_val), 4),
            fee_cost_per_dollar=round(get_polymarket_fee(price, "other"), 6),
        )
