"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para advanced_math: KLDivergence, MultiKelly, BinarySharpe.
"""
import unittest
import math
from advanced_math import KLDivergence, MultiKelly, BinarySharpe


class TestKLDivergence(unittest.TestCase):

    def test_identical_distributions_zero(self):
        """D_KL(p||p) = 0"""
        self.assertAlmostEqual(KLDivergence.binary_kl(0.6, 0.6), 0.0, places=6)

    def test_symmetric_model_higher_than_market(self):
        kl1 = KLDivergence.binary_kl(0.7, 0.5)
        kl2 = KLDivergence.binary_kl(0.3, 0.5)
        self.assertAlmostEqual(kl1, kl2, places=6)

    def test_always_nonnegative(self):
        """KL divergence is always ≥ 0"""
        for p, m in [(0.6, 0.5), (0.4, 0.5), (0.8, 0.2), (0.2, 0.8)]:
            self.assertGreaterEqual(KLDivergence.binary_kl(p, m), 0)

    def test_larger_divergence_more_profitable(self):
        kl_small = KLDivergence.binary_kl(0.55, 0.5)
        kl_large = KLDivergence.binary_kl(0.80, 0.5)
        self.assertGreater(kl_large, kl_small)

    def test_boundary_inputs_no_crash(self):
        # Should handle extremes gracefully
        _ = KLDivergence.binary_kl(0.001, 0.999)
        _ = KLDivergence.binary_kl(0.999, 0.001)

    def test_is_actionable_threshold(self):
        self.assertTrue(KLDivergence.is_actionable(0.8, 0.5, threshold=0.05))
        self.assertFalse(KLDivergence.is_actionable(0.51, 0.5, threshold=0.05))

    def test_rank_opportunities_sorted(self):
        markets = [
            {"p_model": 0.6, "p_market": 0.5},
            {"p_model": 0.9, "p_market": 0.5},
            {"p_model": 0.55, "p_market": 0.5},
        ]
        ranked = KLDivergence.rank_opportunities(markets)
        kls = [m["kl_divergence"] for m in ranked]
        self.assertEqual(kls, sorted(kls, reverse=True))

    def test_rank_adds_rank_field(self):
        markets = [{"p_model": 0.7, "p_market": 0.5}]
        ranked = KLDivergence.rank_opportunities(markets)
        self.assertIn("rank", ranked[0])
        self.assertEqual(ranked[0]["rank"], 1)


class TestMultiKelly(unittest.TestCase):

    def test_single_kelly_positive_edge(self):
        """Si p > price, Kelly > 0"""
        k = MultiKelly.single_kelly(0.6, 0.5)
        self.assertGreater(k, 0)

    def test_single_kelly_no_edge_zero(self):
        """Si p ≤ price, Kelly = 0"""
        k = MultiKelly.single_kelly(0.5, 0.6)
        self.assertEqual(k, 0.0)

    def test_single_kelly_at_boundary(self):
        # p=0 → p <= price (0 <= 0.5) → 0
        k = MultiKelly.single_kelly(0.0, 0.5)
        self.assertEqual(k, 0.0)
        # p=1.0 → Kelly = (1-0.5)/(1-0.5) = 1.0 (certeza total)
        k2 = MultiKelly.single_kelly(1.0, 0.5)
        self.assertAlmostEqual(k2, 1.0, places=5)

    def test_single_kelly_fraction_applied(self):
        """Quarter Kelly (0.25) should return 25% of full Kelly"""
        full = MultiKelly.single_kelly(0.7, 0.5)
        fracs = MultiKelly.multi_kelly_optimize([{"p": 0.7, "price": 0.5}], fraction=0.25)
        self.assertAlmostEqual(fracs[0], full * 0.25, places=4)

    def test_multi_kelly_empty_input(self):
        result = MultiKelly.multi_kelly_optimize([])
        self.assertEqual(result, [])

    def test_multi_kelly_returns_n_fractions(self):
        bets = [{"p": 0.6, "price": 0.5}, {"p": 0.7, "price": 0.4}]
        fracs = MultiKelly.multi_kelly_optimize(bets, fraction=0.25)
        self.assertEqual(len(fracs), 2)

    def test_multi_kelly_fractions_nonnegative(self):
        bets = [{"p": 0.6, "price": 0.5}, {"p": 0.65, "price": 0.45}]
        fracs = MultiKelly.multi_kelly_optimize(bets, fraction=0.25)
        for f in fracs:
            self.assertGreaterEqual(f, 0)

    def test_multi_kelly_high_n_uses_individual(self):
        """N > 12 should use individual Kelly (no 2^N enumeration)"""
        bets = [{"p": 0.6, "price": 0.5}] * 15
        fracs = MultiKelly.multi_kelly_optimize(bets, fraction=0.25)
        self.assertEqual(len(fracs), 15)

    def test_multi_kelly_no_edge_all_zero(self):
        bets = [{"p": 0.4, "price": 0.5}, {"p": 0.3, "price": 0.6}]
        fracs = MultiKelly.multi_kelly_optimize(bets, fraction=0.25)
        for f in fracs:
            self.assertAlmostEqual(f, 0.0, places=5)


class TestBinarySharpe(unittest.TestCase):

    def test_positive_edge_positive_sharpe(self):
        sr = BinarySharpe.single_bet(0.7, 0.5)
        self.assertGreater(sr, 0)

    def test_negative_edge_negative_sharpe(self):
        sr = BinarySharpe.single_bet(0.3, 0.5)
        self.assertLess(sr, 0)

    def test_zero_edge_zero_sharpe(self):
        sr = BinarySharpe.single_bet(0.5, 0.5)
        self.assertAlmostEqual(sr, 0.0, places=5)

    def test_boundary_zero(self):
        self.assertEqual(BinarySharpe.single_bet(0.0, 0.5), 0.0)
        self.assertEqual(BinarySharpe.single_bet(1.0, 0.5), 0.0)

    def test_portfolio_empty(self):
        self.assertEqual(BinarySharpe.portfolio([]), 0.0)

    def test_portfolio_multiple_bets(self):
        bets = [
            {"p": 0.7, "m": 0.5, "w": 0.5},
            {"p": 0.6, "m": 0.4, "w": 0.5},
        ]
        sr = BinarySharpe.portfolio(bets)
        self.assertGreater(sr, 0)

    def test_kelly_sharpe_relationship(self):
        """Kelly growth rate ≈ SR² / 2"""
        sr = 2.0
        growth = BinarySharpe.kelly_sharpe_relationship(sr)
        self.assertAlmostEqual(growth, 2.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
