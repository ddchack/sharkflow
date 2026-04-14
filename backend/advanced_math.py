"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Advanced Mathematical Algorithms
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Based on documented successful strategies:
- $313→$414K bot (temporal arbitrage, 98% win rate)
- ilovecircle $2.2M AI ensemble (74% accuracy)
- $40M arbitrage ecosystem (86M transactions analyzed)
- French Whale $85M (private polling + Bayesian updating)
- IMDEA paper "Unravelling the Probabilistic Forest"

Implements:
1. KL Divergence for mispricing detection & opportunity ranking
2. Multi-Kelly for simultaneous bets (numerical optimization)
3. LMSR (Logarithmic Market Scoring Rule) fair price calculator
4. Brier Score decomposition (REL, RES, UNC)
5. Monte Carlo portfolio simulation
6. Gaussian Copula for correlated market modeling
7. Binary Sharpe Ratio (exact formula for prediction markets)
8. Effective Exposure with correlation adjustment
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import optimize, stats


# ═══════════════════════════════════════════════════════════
# 1. KL DIVERGENCE — Mispricing Detection & Opportunity Ranking
# ═══════════════════════════════════════════════════════════
# Key insight from research: expected profit for a log-utility
# investor equals the KL divergence between their beliefs and
# the market price. Higher D_KL = bigger edge = higher priority.

class KLDivergence:
    """
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    @staticmethod
    def binary_kl(p_model: float, p_market: float) -> float:
        """
        KL divergence for binary outcomes.
        D_KL(p_model || p_market)

        This equals the expected log-return for a growth-optimal investor.
        Higher = more profitable opportunity.

        Research basis: Robin Hanson's LMSR theory proves trader profit
        is proportional to KL divergence between beliefs and market.
        """
        p = max(1e-6, min(1 - 1e-6, p_model))
        m = max(1e-6, min(1 - 1e-6, p_market))
        return p * math.log(p / m) + (1 - p) * math.log((1 - p) / (1 - m))

    @staticmethod
    def rank_opportunities(markets: list[dict]) -> list[dict]:
        """
        Rank markets by KL divergence (expected profitability).
        Each market dict needs: 'p_model', 'p_market', plus any metadata.

        Returns sorted list with 'kl_divergence' and 'rank' added.
        Research: D_KL > 0.05 is actionable threshold.
        """
        for m in markets:
            m["kl_divergence"] = KLDivergence.binary_kl(m["p_model"], m["p_market"])
        ranked = sorted(markets, key=lambda x: x["kl_divergence"], reverse=True)
        for i, m in enumerate(ranked):
            m["rank"] = i + 1
        return ranked

    @staticmethod
    def is_actionable(p_model: float, p_market: float, threshold: float = 0.05) -> bool:
        """Check if KL divergence exceeds actionable threshold."""
        return KLDivergence.binary_kl(p_model, p_market) >= threshold


# ═══════════════════════════════════════════════════════════
# 2. MULTI-KELLY — Simultaneous Bet Optimization
# ═══════════════════════════════════════════════════════════
# Research: optimal simultaneous bets are SMALLER than individual
# Kelly fractions. Full Kelly has 33% chance of halving bankroll
# before doubling. Quarter-Kelly recommended for $100 capital.

class MultiKelly:
    """
    Numerically solve Kelly Criterion for multiple simultaneous
    independent binary bets.

    Maximize: E[ln(W)] = sum over all 2^N outcome combinations of
    probability(combo) * ln(1 + sum of gains/losses for that combo)

    Source: vegapit.com + Kelly criterion Wikipedia formalization
    """

    @staticmethod
    def single_kelly(p: float, price: float) -> float:
        """Single bet Kelly: f* = (p - m) / (1 - m)"""
        if price <= 0 or price >= 1 or p <= price:
            return 0.0
        return (p - price) / (1 - price)

    @staticmethod
    def multi_kelly_optimize(bets: list[dict], fraction: float = 0.25) -> list[float]:
        """
        Optimize Kelly fractions for N simultaneous independent bets.

        bets: [{"p": true_prob, "price": market_price}, ...]
        fraction: Kelly fraction multiplier (0.25 = quarter Kelly)

        Returns: list of optimal fraction of bankroll for each bet.

        Method: maximize expected log-wealth over all 2^N outcome states.
        Uses scipy.optimize.minimize with SLSQP.
        """
        n = len(bets)
        if n == 0:
            return []
        if n == 1:
            k = MultiKelly.single_kelly(bets[0]["p"], bets[0]["price"])
            return [k * fraction]
        if n > 12:
            # For >12 bets, use individual Kelly (2^N too expensive)
            return [MultiKelly.single_kelly(b["p"], b["price"]) * fraction for b in bets]

        probs = [b["p"] for b in bets]
        odds = [(1.0 / b["price"]) - 1.0 for b in bets]  # net payout per $1

        def neg_expected_log_wealth(fracs):
            """Negative expected log wealth (to minimize)."""
            total = 0.0
            # Iterate over all 2^N outcome combinations
            for state in range(2 ** n):
                combo_prob = 1.0
                wealth_change = 0.0
                for i in range(n):
                    if state & (1 << i):  # bet i wins
                        combo_prob *= probs[i]
                        wealth_change += fracs[i] * odds[i]
                    else:  # bet i loses
                        combo_prob *= (1 - probs[i])
                        wealth_change -= fracs[i]
                # 1e-6 en lugar de 1e-10: ruina total (−100%) ≈ log(1e-6)=−13.8
                # Previene penalización infinita que distorsiona gradiente del optimizador
                log_w = math.log(max(1e-6, 1.0 + wealth_change))
                total += combo_prob * log_w
            return -total

        # Constraints: each fraction >= 0, sum of fractions <= 1
        bounds = [(0, 0.5)] * n  # Cap each at 50%
        constraints = [{"type": "ineq", "fun": lambda f: 1.0 - sum(f)}]

        x0 = np.array([0.01] * n)
        try:
            result = optimize.minimize(
                neg_expected_log_wealth, x0,
                method="SLSQP", bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200}
            )
            if result.success:
                return [max(0, f) * fraction for f in result.x]
        except Exception:
            pass

        # Fallback: individual Kelly * shrinkage
        shrinkage = 1.0 / math.sqrt(n)  # reduce as bets increase
        return [MultiKelly.single_kelly(b["p"], b["price"]) * fraction * shrinkage
                for b in bets]


# ═══════════════════════════════════════════════════════════
# 3. LMSR — Logarithmic Market Scoring Rule
# ═══════════════════════════════════════════════════════════
# Robin Hanson (2003/2007). Foundation of prediction market AMMs.
# Cost function: C(q) = b * ln(sum(exp(qi/b)))
# Price function (softmax): pi = exp(qi/b) / sum(exp(qj/b))

class LMSR:
    """
    Logarithmic Market Scoring Rule implementation.
    Used to compute fair prices and detect mispricing vs. CLOB.
    """

    @staticmethod
    def cost(quantities: list[float], b: float) -> float:
        """Cost function: C(q) = b * ln(sum(exp(qi/b)))"""
        max_q = max(quantities)
        # Numerically stable version
        exp_sum = sum(math.exp((q - max_q) / b) for q in quantities)
        return b * (max_q / b + math.log(exp_sum))

    @staticmethod
    def prices(quantities: list[float], b: float) -> list[float]:
        """Price function (softmax): pi = exp(qi/b) / sum(exp(qj/b))"""
        max_q = max(quantities)
        exps = [math.exp((q - max_q) / b) for q in quantities]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def cost_to_buy(quantities: list[float], b: float,
                     outcome: int, shares: float) -> float:
        """Cost to buy `shares` of `outcome`."""
        new_q = quantities.copy()
        new_q[outcome] += shares
        return LMSR.cost(new_q, b) - LMSR.cost(quantities, b)

    @staticmethod
    def fair_price_binary(yes_quantity: float, no_quantity: float,
                          b: float) -> tuple[float, float]:
        """Get fair YES/NO prices for a binary market."""
        prices = LMSR.prices([yes_quantity, no_quantity], b)
        return prices[0], prices[1]

    @staticmethod
    def max_market_maker_loss(b: float, n_outcomes: int) -> float:
        """Maximum loss for the market maker: b * ln(n)"""
        return b * math.log(n_outcomes)


# ═══════════════════════════════════════════════════════════
# 4. BRIER SCORE DECOMPOSITION
# ═══════════════════════════════════════════════════════════
# BS = REL - RES + UNC (Murphy 1973)
# REL = calibration (0 = perfect)
# RES = discrimination (higher = better)
# UNC = inherent difficulty

class BrierScoreAnalyzer:
    """
    Full Brier Score analysis with Murphy decomposition.
    Essential for evaluating and improving model calibration.
    """

    @staticmethod
    def brier_score(predictions: list[float], outcomes: list[int]) -> float:
        """BS = (1/N) * sum((fi - oi)^2)"""
        n = len(predictions)
        if n == 0:
            return 0.25
        return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n

    @staticmethod
    def decompose(predictions: list[float], outcomes: list[int],
                   n_bins: int = 10) -> dict:
        """
        Murphy decomposition: BS = REL - RES + UNC

        Returns: {reliability, resolution, uncertainty, brier_score,
                  calibration_curve, skill_score}
        """
        n = len(predictions)
        if n < n_bins:
            return {"error": "Not enough data"}

        o_bar = sum(outcomes) / n  # base rate
        unc = o_bar * (1 - o_bar)

        # Bin predictions
        bins = {}
        for p, o in zip(predictions, outcomes):
            b = min(int(p * n_bins), n_bins - 1)
            if b not in bins:
                bins[b] = {"preds": [], "outcomes": []}
            bins[b]["preds"].append(p)
            bins[b]["outcomes"].append(o)

        rel = 0.0
        res = 0.0
        cal_curve = []

        for b_idx in sorted(bins.keys()):
            b = bins[b_idx]
            nk = len(b["preds"])
            fk = sum(b["preds"]) / nk  # avg prediction in bin
            ok = sum(b["outcomes"]) / nk  # avg outcome in bin

            rel += nk * (fk - ok) ** 2
            res += nk * (ok - o_bar) ** 2

            cal_curve.append({
                "bin": b_idx / n_bins,
                "predicted": round(fk, 4),
                "actual": round(ok, 4),
                "count": nk,
            })

        rel /= n
        res /= n
        bs = rel - res + unc

        # Skill score: 1 - BS/BS_ref (BS_ref = always predict base rate)
        bs_ref = unc
        skill = 1 - (bs / bs_ref) if bs_ref > 0 else 0

        return {
            "brier_score": round(bs, 6),
            "reliability": round(rel, 6),
            "resolution": round(res, 6),
            "uncertainty": round(unc, 6),
            "skill_score": round(skill, 4),
            "base_rate": round(o_bar, 4),
            "calibration_curve": cal_curve,
            "diagnosis": (
                "WELL_CALIBRATED" if rel < 0.01
                else "OVERCONFIDENT" if rel > 0.03
                else "SLIGHTLY_MISCALIBRATED"
            ),
        }


# ═══════════════════════════════════════════════════════════
# 5. MONTE CARLO PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════
# Research: proportional staking produces 230% more profit
# than fixed staking over 5000 bets.

class MonteCarloSimulator:
    """
    Simulate portfolio of prediction market bets.
    Models path-dependent outcomes with proper compounding.
    """

    @staticmethod
    def simulate_portfolio(bets: list[dict], initial_capital: float = 100.0,
                            n_simulations: int = 1000,
                            kelly_fraction: float = 0.25) -> dict:
        """
        Monte Carlo simulation of a portfolio of bets.

        bets: [{"p": true_prob, "price": market_price, "category": str}, ...]
        Returns: distribution of final wealth, VaR, expected growth, etc.

        Based on research showing proportional staking >> fixed staking.
        """
        n_bets = len(bets)
        if n_bets == 0:
            return {"error": "No bets"}

        final_capitals = []
        max_drawdowns = []
        win_rates = []

        for _ in range(n_simulations):
            capital = initial_capital
            peak = capital
            max_dd = 0.0
            wins = 0

            # Shuffle bet order (order matters for compounding)
            order = np.random.permutation(n_bets)

            for idx in order:
                bet = bets[idx]
                p, price = bet["p"], bet["price"]

                # Kelly sizing with current capital (proportional staking)
                k = MultiKelly.single_kelly(p, price) * kelly_fraction
                bet_size = k * capital
                bet_size = min(bet_size, capital * 0.25)  # Cap at 25%

                if bet_size < 0.5 or capital < 1:
                    continue

                # Simulate outcome
                won = np.random.random() < p
                if won:
                    profit = bet_size * ((1.0 / price) - 1.0)
                    capital += profit
                    wins += 1
                else:
                    capital -= bet_size

                capital = max(0, capital)
                if capital > peak:
                    peak = capital
                dd = (peak - capital) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_capitals.append(capital)
            max_drawdowns.append(max_dd)
            win_rates.append(wins / max(1, n_bets))

        fc = np.array(final_capitals)
        return {
            "mean_final": round(float(np.mean(fc)), 2),
            "median_final": round(float(np.median(fc)), 2),
            "std_final": round(float(np.std(fc)), 2),
            "min_final": round(float(np.min(fc)), 2),
            "max_final": round(float(np.max(fc)), 2),
            "p5_final": round(float(np.percentile(fc, 5)), 2),
            "p95_final": round(float(np.percentile(fc, 95)), 2),
            "prob_profit": round(float(np.mean(fc > initial_capital)), 4),
            "prob_double": round(float(np.mean(fc > initial_capital * 2)), 4),
            "prob_ruin": round(float(np.mean(fc < 10)), 4),  # <$10 = ruin
            "expected_return_pct": round(float(np.mean(fc) / initial_capital - 1) * 100, 2),
            "var_5pct": round(float(initial_capital - np.percentile(fc, 5)), 2),
            "avg_max_drawdown": round(float(np.mean(max_drawdowns)) * 100, 2),
            "avg_win_rate": round(float(np.mean(win_rates)) * 100, 1),
            "sharpe": round(float(np.mean(fc - initial_capital) / max(1, np.std(fc))), 3),
            "n_simulations": n_simulations,
            "n_bets": n_bets,
        }


# ═══════════════════════════════════════════════════════════
# 6. GAUSSIAN COPULA — Correlated Market Modeling
# ═══════════════════════════════════════════════════════════
# For modeling correlation between related markets:
# e.g., "Trump wins state A" and "Trump wins state B"
# Research: must count correlated positions as single trade.

class GaussianCopula:
    """
    Model joint probabilities for correlated binary markets.
    Used for portfolio correlation and effective exposure.
    """

    @staticmethod
    def joint_probability(p1: float, p2: float, rho: float) -> dict:
        """
        Joint probabilities for two correlated binary outcomes
        using Gaussian Copula.

        Returns: P(both yes), P(1 yes 2 no), P(1 no 2 yes), P(both no)
        """
        from scipy.stats import norm, multivariate_normal

        z1 = norm.ppf(max(1e-6, min(1 - 1e-6, p1)))
        z2 = norm.ppf(max(1e-6, min(1 - 1e-6, p2)))

        cov = [[1, rho], [rho, 1]]
        mvn = multivariate_normal(mean=[0, 0], cov=cov)

        # P(both yes) = P(Z1 < z1, Z2 < z2)
        p_both_yes = mvn.cdf([z1, z2])
        p_1yes_2no = p1 - p_both_yes
        p_1no_2yes = p2 - p_both_yes
        p_both_no = 1 - p1 - p2 + p_both_yes

        return {
            "both_yes": round(p_both_yes, 6),
            "yes_no": round(max(0, p_1yes_2no), 6),
            "no_yes": round(max(0, p_1no_2yes), 6),
            "both_no": round(max(0, p_both_no), 6),
        }

    @staticmethod
    def effective_exposure(positions: list[dict]) -> float:
        """
        Calculate correlation-adjusted effective exposure.

        positions: [{"size": usd_amount, "category": str, "market_id": str}]

        Research formula:
        Eff_Exp = sum(size_i * sqrt(1 + sum_j(rho_ij * size_j/size_i)))

        Category correlations (empirical estimates from Polymarket data):
        """
        CATEGORY_CORRELATIONS = {
            ("politics", "politics"): 0.6,
            ("crypto", "crypto"): 0.7,
            ("sports", "sports"): 0.2,  # Low: different games
            ("macro", "macro"): 0.5,
            ("geopolitics", "geopolitics"): 0.4,
            ("weather", "weather"): 0.3,
            ("tech", "tech"): 0.3,
            ("politics", "macro"): 0.4,
            ("politics", "geopolitics"): 0.3,
            ("crypto", "macro"): 0.5,
            ("crypto", "tech"): 0.3,
            ("geopolitics", "macro"): 0.3,
        }

        total_eff = 0.0
        for i, pos_i in enumerate(positions):
            size_i = pos_i.get("size", 0)
            if size_i == 0:
                continue  # Skip zero-size positions to avoid division by zero
            corr_sum = 0.0
            for j, pos_j in enumerate(positions):
                if i == j:
                    continue
                cat_pair = tuple(sorted([pos_i.get("category", "other"),
                                          pos_j.get("category", "other")]))
                rho = CATEGORY_CORRELATIONS.get(cat_pair, 0.1)
                corr_sum += rho * pos_j.get("size", 0) / size_i

            total_eff += size_i * math.sqrt(1 + corr_sum)

        return round(total_eff, 2)


# ═══════════════════════════════════════════════════════════
# 7. BINARY SHARPE RATIO
# ═══════════════════════════════════════════════════════════
# Exact formula for prediction markets:
# SR = (p - m) / sqrt(m(1-m))  — denominator is market-implied volatility
# Portfolio SR with independence: uses weight-adjusted formula

class BinarySharpe:
    """Sharpe ratio calculations specific to binary prediction markets."""

    @staticmethod
    def single_bet(p: float, m: float) -> float:
        """SR = (p - m) / sqrt(m(1-m)) for a single binary bet.
        Denominator uses market price m (not model p) to represent
        market-implied volatility — measures edge relative to market risk."""
        if p <= 0 or p >= 1 or m <= 0 or m >= 1:
            return 0.0
        return (p - m) / math.sqrt(m * (1 - m))

    @staticmethod
    def portfolio(bets: list[dict]) -> float:
        """
        Portfolio Sharpe for N independent binary bets.
        bets: [{"p": true_prob, "m": market_price, "w": weight}, ...]

        SR_portfolio = E[R] / std(R)
        E[R] = sum(wi * (pi - mi) / mi)
        std(R) = sqrt(sum(wi^2 * mi*(1-mi) / mi^2))
        Denominator uses market price mi (not model pi) — market-implied volatility.
        """
        if not bets:
            return 0.0

        er = sum(b["w"] * (b["p"] - b["m"]) / b["m"] for b in bets)
        var = sum(b["w"] ** 2 * b["m"] * (1 - b["m"]) / b["m"] ** 2 for b in bets)

        if var <= 0:
            return 0.0
        return er / math.sqrt(var)

    @staticmethod
    def kelly_sharpe_relationship(sharpe: float) -> float:
        """
        Kelly growth rate ≈ SR²/2
        Research: Sharpe > 1 is very good, > 2 is exceptional
        """
        return sharpe ** 2 / 2


# ═══════════════════════════════════════════════════════════
# 8. PLATT SCALING — Probability Calibration
# ═══════════════════════════════════════════════════════════
# P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
# Research: with >1000 calibration points, isotonic matches Platt.

class PlattCalibrator:
    """
    Calibrate raw probability estimates using Platt scaling.
    Learns A, B parameters from historical predictions vs outcomes.
    """

    def __init__(self):
        self.A = -1.0  # Default: identity-like mapping
        self.B = 0.0
        self.is_fitted = False
        self._history = []

    def add_observation(self, predicted: float, actual: int):
        """Add a prediction-outcome pair for calibration."""
        self._history.append((predicted, actual))

    def fit(self):
        """Fit Platt scaling parameters A, B using MLE."""
        if len(self._history) < 30:
            return False

        preds = np.array([h[0] for h in self._history])
        outcomes = np.array([h[1] for h in self._history])

        def neg_log_likelihood(params):
            a, b = params
            # Calibrated probabilities
            cal = 1.0 / (1.0 + np.exp(a * preds + b))
            cal = np.clip(cal, 1e-7, 1 - 1e-7)
            # NLL
            return -np.sum(outcomes * np.log(cal) + (1 - outcomes) * np.log(1 - cal))

        try:
            result = optimize.minimize(neg_log_likelihood, [self.A, self.B],
                                        method="Nelder-Mead")
            if result.success:
                self.A, self.B = result.x
                # Post-fit: in 1/(1+exp(A*f+B)), A must be negative so that
                # higher raw scores map to higher calibrated probabilities.
                # If A >= 0, the curve is inverted — reset to safe defaults.
                if self.A >= 0:
                    print(f"[PlattCalibrator] ADVERTENCIA: A={self.A:.4f} >= 0, "
                          f"curva invertida detectada. Reseteando a defaults.")
                    self.A = -1.0
                    self.B = 0.0
                self.is_fitted = True
                return True
        except Exception:
            pass
        return False

    def calibrate(self, raw_prob: float) -> float:
        """Apply Platt scaling to a raw probability."""
        if not self.is_fitted:
            return raw_prob
        return 1.0 / (1.0 + math.exp(self.A * raw_prob + self.B))

    def get_params(self) -> dict:
        return {"A": self.A, "B": self.B, "fitted": self.is_fitted,
                "n_observations": len(self._history)}
