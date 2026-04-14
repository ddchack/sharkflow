"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para BacktestEngine._compute_category_metrics y Extremizer.find_optimal_d
"""
import unittest
from dataclasses import dataclass, field
from extremizer import Extremizer, DiversityTracker, AdaptiveAggregator


# ──────────────────────────────────────────────────────────
# Extremizer tests
# ──────────────────────────────────────────────────────────

class TestExtremizer(unittest.TestCase):

    def test_extremize_pushes_away_from_half(self):
        """d>1 debe alejar la probabilidad de 0.5"""
        p = Extremizer.extremize([0.6, 0.6], d=2.0)
        self.assertGreater(p, 0.6)

    def test_extremize_toward_half_when_d_lt_1(self):
        """d<1 debe acercar hacia 0.5"""
        p = Extremizer.extremize([0.7, 0.7], d=0.5)
        self.assertLess(p, 0.7)

    def test_extremize_d1_equals_logit_mean(self):
        """d=1 debe ser equivalente al promedio en espacio logit"""
        p1 = Extremizer.extremize([0.3, 0.7], d=1.0)
        self.assertAlmostEqual(p1, 0.5, places=3)

    def test_weighted_extremize_single_prob(self):
        """Con una sola prob, el resultado debe ser esa misma prob (d=1)"""
        p = Extremizer.weighted_extremize([(0.7, 1.0)], d=1.0)
        self.assertAlmostEqual(p, 0.7, places=3)

    def test_weighted_extremize_output_in_range(self):
        """Resultado siempre entre 0 y 1"""
        p = Extremizer.weighted_extremize([(0.1, 0.3), (0.9, 0.7)], d=2.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_find_optimal_d_returns_default_when_few_trades(self):
        """Con menos de 10 datos debe retornar 1.73"""
        d = Extremizer.find_optimal_d([[0.6], [0.7]], [1, 0])
        self.assertEqual(d, 1.73)

    def test_find_optimal_d_perfect_predictor(self):
        """Con predicciones perfectas, d alto debería ganar (no importa el valor exacto)"""
        import random
        random.seed(42)
        preds = [[0.9] for _ in range(20)]
        outcomes = [1] * 20
        d = Extremizer.find_optimal_d(preds, outcomes)
        self.assertIsInstance(d, float)
        self.assertGreater(d, 0.0)

    def test_find_optimal_d_range(self):
        """d siempre entre 0.5 y 4.0"""
        import numpy as np
        np.random.seed(0)
        preds = [[np.random.random()] for _ in range(30)]
        outcomes = [int(np.random.random() > 0.5) for _ in range(30)]
        d = Extremizer.find_optimal_d(preds, outcomes)
        self.assertGreaterEqual(d, 0.5)
        self.assertLessEqual(d, 4.0)


# ──────────────────────────────────────────────────────────
# DiversityTracker tests
# ──────────────────────────────────────────────────────────

class TestDiversityTracker(unittest.TestCase):

    def test_perfect_agreement_zero_diversity(self):
        """Todos predicen lo mismo → diversidad = 0"""
        r = DiversityTracker.compute([0.7, 0.7, 0.7], outcome=1)
        self.assertAlmostEqual(r["diversity"], 0.0, places=5)

    def test_theorem_holds(self):
        """Collective Error = Avg Individual - Diversity (DPT)"""
        r = DiversityTracker.compute([0.4, 0.6, 0.8], outcome=1)
        lhs = round(r["collective_error"], 4)
        rhs = round(r["avg_individual_error"] - r["diversity"], 4)
        self.assertAlmostEqual(lhs, rhs, places=3)

    def test_empty_returns_zeros(self):
        r = DiversityTracker.compute([], outcome=1)
        self.assertEqual(r["collective_error"], 0)


# ──────────────────────────────────────────────────────────
# AdaptiveAggregator tests
# ──────────────────────────────────────────────────────────

class TestAdaptiveAggregator(unittest.TestCase):

    def test_uniform_initial_weights(self):
        agg = AdaptiveAggregator(n_models=3, learning_rate=0.5)  # type: ignore[call-arg]
        w = agg.get_weights()
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(sum(w), 1.0, places=3)

    def test_weights_sum_to_one_after_update(self):
        agg = AdaptiveAggregator(n_models=2, learning_rate=0.5)
        agg.update([0.8, 0.3], outcome=1)
        self.assertAlmostEqual(sum(agg.get_weights()), 1.0, places=5)

    def test_better_model_gets_higher_weight(self):
        """El modelo con menor error acumulado debe tener más peso"""
        agg = AdaptiveAggregator(n_models=2, learning_rate=1.0)
        for _ in range(10):
            # modelo 0 predice bien, modelo 1 predice mal
            agg.update([0.9, 0.1], outcome=1)
        w = agg.get_weights()
        self.assertGreater(w[0], w[1])

    def test_aggregate_returns_scalar(self):
        agg = AdaptiveAggregator(n_models=3)
        result = agg.aggregate([0.3, 0.5, 0.7])
        self.assertIsInstance(result, float)


# ──────────────────────────────────────────────────────────
# BacktestEngine._compute_category_metrics unit test
# ──────────────────────────────────────────────────────────

class TestCategoryMetrics(unittest.TestCase):

    def _make_trade(self, cat, won, pnl, bet=1.0, edge=5.0):
        """Crea un mock BacktestTrade."""
        from backtest_engine import BacktestTrade
        return BacktestTrade(
            market_id="test_id",
            question="Test question?",
            category=cat,
            side="YES",
            entry_price=0.5,
            estimated_prob=0.6,
            bet_usd=bet,
            edge_pct=edge,
            kelly=0.1,
            resolved_outcome="YES" if won else "NO",
            won=won,
            pnl_usd=pnl,
            confidence=0.7,
        )

    def test_single_category(self):
        from backtest_engine import BacktestEngine
        engine = BacktestEngine()
        trades = [
            self._make_trade("MarketCategory.SPORTS", True,  1.5, 1.0),
            self._make_trade("MarketCategory.SPORTS", True,  0.8, 1.0),
            self._make_trade("MarketCategory.SPORTS", False, -1.0, 1.0),
        ]
        metrics = engine._compute_category_metrics(trades)
        self.assertIn("MarketCategory.SPORTS", metrics)
        s = metrics["MarketCategory.SPORTS"]
        self.assertEqual(s["trades"], 3)
        self.assertAlmostEqual(s["win_rate"], 66.7, places=0)
        self.assertAlmostEqual(s["total_pnl"], 1.3, places=1)

    def test_multiple_categories(self):
        from backtest_engine import BacktestEngine
        engine = BacktestEngine()
        trades = [
            self._make_trade("sports",   True,  1.0),
            self._make_trade("politics", False, -1.0),
            self._make_trade("politics", True,   0.5),
        ]
        metrics = engine._compute_category_metrics(trades)
        self.assertEqual(len(metrics), 2)
        self.assertIn("sports", metrics)
        self.assertIn("politics", metrics)

    def test_empty_trades(self):
        from backtest_engine import BacktestEngine
        engine = BacktestEngine()
        metrics = engine._compute_category_metrics([])
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
