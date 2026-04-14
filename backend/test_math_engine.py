"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para MathEngine: Kelly, EV, edge, Sharpe, confidence, compound growth.
"""
import unittest
import math
from math_engine import MathEngine, MarketOdds, BetRecommendation


class TestImpliedProbability(unittest.TestCase):

    def test_price_passthrough(self):
        self.assertAlmostEqual(MathEngine.implied_probability(0.6), 0.6)

    def test_clamp_above_1(self):
        self.assertEqual(MathEngine.implied_probability(1.5), 0.99)

    def test_clamp_below_0(self):
        self.assertEqual(MathEngine.implied_probability(-0.1), 0.01)

    def test_boundary_0(self):
        self.assertEqual(MathEngine.implied_probability(0.0), 0.01)

    def test_boundary_1(self):
        self.assertEqual(MathEngine.implied_probability(1.0), 0.99)


class TestDecimalOdds(unittest.TestCase):

    def test_50pct_returns_2x(self):
        self.assertAlmostEqual(MathEngine.decimal_odds(0.5), 2.0)

    def test_25pct_returns_4x(self):
        self.assertAlmostEqual(MathEngine.decimal_odds(0.25), 4.0)

    def test_zero_price_returns_inf(self):
        self.assertEqual(MathEngine.decimal_odds(0.0), float('inf'))


class TestExpectedValue(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine()

    def test_ev_zero_when_no_edge(self):
        ev = self.eng.expected_value(0.5, 0.5)
        self.assertAlmostEqual(ev, 0.0, places=5)

    def test_ev_positive_with_edge(self):
        """true_prob=0.6 vs price=0.5: positive edge → positive EV"""
        ev = self.eng.expected_value(0.6, 0.5)
        self.assertGreater(ev, 0)

    def test_ev_negative_without_edge(self):
        ev = self.eng.expected_value(0.4, 0.5)
        self.assertLess(ev, 0)

    def test_ev_boundary_inputs(self):
        self.assertEqual(self.eng.expected_value(0.6, 0.0), 0.0)
        self.assertEqual(self.eng.expected_value(0.6, 1.0), 0.0)

    def test_ev_formula_correctness(self):
        """true_prob=0.60, price=0.40: EV = 0.60*(1/0.40-1) - 0.40 = 0.50"""
        ev = self.eng.expected_value(0.60, 0.40)
        self.assertAlmostEqual(ev, 0.50, places=4)


class TestKellyCriterion(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine()

    def test_kelly_positive_edge(self):
        k = self.eng.kelly_criterion(0.7, 0.5)
        self.assertGreater(k, 0)

    def test_kelly_no_edge_zero(self):
        k = self.eng.kelly_criterion(0.5, 0.5)
        self.assertAlmostEqual(k, 0.0, places=5)

    def test_kelly_negative_edge_zero(self):
        k = self.eng.kelly_criterion(0.3, 0.5)
        self.assertEqual(k, 0.0)

    def test_kelly_formula_correctness(self):
        """p=0.6, price=0.5: b=1, f*=(1*0.6-0.4)/1=0.2"""
        k = self.eng.kelly_criterion(0.6, 0.5)
        self.assertAlmostEqual(k, 0.2, places=5)

    def test_fractional_kelly_is_fraction(self):
        full = self.eng.kelly_criterion(0.7, 0.5)
        frac = self.eng.fractional_kelly(0.7, 0.5)
        self.assertAlmostEqual(frac, full * self.eng.kelly_multiplier, places=5)

    def test_kelly_boundary_price(self):
        self.assertEqual(self.eng.kelly_criterion(0.7, 0.0), 0.0)
        self.assertEqual(self.eng.kelly_criterion(0.7, 1.0), 0.0)

    def test_kelly_never_negative(self):
        for p, m in [(0.1, 0.9), (0.0, 0.5), (0.4, 0.6)]:
            k = self.eng.kelly_criterion(p, m)
            self.assertGreaterEqual(k, 0.0)


class TestEdge(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine()

    def test_positive_edge(self):
        self.assertGreater(self.eng.edge(0.7, 0.5), 0)

    def test_negative_edge(self):
        self.assertLess(self.eng.edge(0.3, 0.5), 0)

    def test_zero_edge(self):
        self.assertAlmostEqual(self.eng.edge(0.5, 0.5), 0.0)

    def test_edge_magnitude(self):
        self.assertAlmostEqual(self.eng.edge(0.7, 0.5), 0.2, places=5)


class TestPositionSize(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine(max_capital=100.0)

    def test_size_positive_with_edge(self):
        size = self.eng.position_size_usd(0.7, 0.5)
        self.assertGreater(size, 0)

    def test_size_zero_without_edge(self):
        size = self.eng.position_size_usd(0.4, 0.6)
        self.assertEqual(size, 0.0)

    def test_size_capped_at_25pct_capital(self):
        """Even with huge edge, should not bet >25% of capital"""
        size = self.eng.position_size_usd(0.99, 0.01, available_capital=100.0)
        self.assertLessEqual(size, 25.0)

    def test_size_respects_available_capital(self):
        s50  = self.eng.position_size_usd(0.7, 0.5, available_capital=50.0)
        s100 = self.eng.position_size_usd(0.7, 0.5, available_capital=100.0)
        self.assertLess(s50, s100)


class TestSharpeRatio(unittest.TestCase):

    def test_positive_ev_positive_sharpe(self):
        sr = MathEngine.sharpe_ratio_estimate(ev_per_dollar=0.3, win_prob=0.6)
        self.assertGreater(sr, 0)

    def test_zero_ev_zero_sharpe(self):
        sr = MathEngine.sharpe_ratio_estimate(ev_per_dollar=0.0, win_prob=0.5)
        self.assertAlmostEqual(sr, 0.0, places=5)

    def test_boundary_winprob(self):
        self.assertEqual(MathEngine.sharpe_ratio_estimate(0.3, 0.0), 0.0)
        self.assertEqual(MathEngine.sharpe_ratio_estimate(0.3, 1.0), 0.0)


class TestConfidenceScore(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine()

    def test_score_in_range(self):
        score = self.eng.confidence_score(
            edge=0.15, ev=0.3, volume_24h=10000, liquidity=5000)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_more_liquidity_higher_score(self):
        s_low  = self.eng.confidence_score(0.15, 0.3, 100,    500)
        s_high = self.eng.confidence_score(0.15, 0.3, 100000, 50000)
        self.assertGreater(s_high, s_low)

    def test_higher_edge_higher_score(self):
        s_small = self.eng.confidence_score(0.05, 0.1, 1000, 500)
        s_large = self.eng.confidence_score(0.20, 0.5, 1000, 500)
        self.assertGreater(s_large, s_small)


class TestCompoundGrowth(unittest.TestCase):

    def setUp(self):
        self.eng = MathEngine(max_capital=100.0)

    def test_no_bets_returns_initial_capital(self):
        result = self.eng.compound_growth([])
        self.assertEqual(result["final_capital"], 100.0)
        self.assertEqual(result["wins"], 0)
        self.assertEqual(result["losses"], 0)

    def test_all_wins_capital_grows(self):
        bets = [{"true_prob": 0.7, "price": 0.5, "won": True}] * 5
        result = self.eng.compound_growth(bets)
        self.assertGreater(result["final_capital"], 100.0)

    def test_all_losses_capital_shrinks(self):
        bets = [{"true_prob": 0.7, "price": 0.5, "won": False}] * 5
        result = self.eng.compound_growth(bets)
        self.assertLess(result["final_capital"], 100.0)

    def test_win_rate_calculated(self):
        bets = [
            {"true_prob": 0.7, "price": 0.5, "won": True},
            {"true_prob": 0.7, "price": 0.5, "won": False},
            {"true_prob": 0.7, "price": 0.5, "won": True},
        ]
        result = self.eng.compound_growth(bets)
        self.assertAlmostEqual(result["win_rate"], 66.7, delta=0.5)

    def test_trajectory_length(self):
        bets = [{"true_prob": 0.6, "price": 0.5, "won": True}] * 3
        result = self.eng.compound_growth(bets)
        self.assertEqual(len(result["trajectory"]), 4)  # initial + 3 bets


class TestMarketOdds(unittest.TestCase):

    def test_overround_fair_market(self):
        odds = MarketOdds(yes_price=0.5, no_price=0.5)
        self.assertAlmostEqual(odds.overround, 1.0)

    def test_overround_with_vig(self):
        odds = MarketOdds(yes_price=0.52, no_price=0.50)
        self.assertGreater(odds.overround, 1.0)

    def test_implied_probs(self):
        odds = MarketOdds(yes_price=0.7, no_price=0.3)
        self.assertAlmostEqual(odds.implied_yes_prob, 0.7)
        self.assertAlmostEqual(odds.implied_no_prob, 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
