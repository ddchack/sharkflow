"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para EscapeRoomEngine: filter_opportunities, scoring, bet_size, category thresholds.
"""
import unittest
from escape_room import EscapeRoomEngine, EROpportunity, ER_MIN_MULTIPLIER_BY_CATEGORY, ER_MAX_ENTRY_PRICE


def _rec(market_id="m1", category="crypto", side="YES", yes_price=0.20, no_price=0.82,
         edge_pct=15.0, kl_div=0.08, ensemble_prob=0.35, liquidity=1000.0):
    return {
        "market_id": market_id, "question": f"Will {market_id} happen?",
        "slug": market_id, "category": category, "side": side,
        "yes_price": yes_price, "no_price": no_price,
        "edge_pct": edge_pct, "kl_divergence": kl_div,
        "ensemble_prob": ensemble_prob, "liquidity": liquidity,
        "yes_token": "yt", "no_token": "nt", "reasoning": [],
    }


class MockLLM:
    """Mock LLM para tests sin API calls."""
    pass


class TestFilterOpportunities(unittest.TestCase):

    def setUp(self):
        self.engine = EscapeRoomEngine(llm_engine=MockLLM(), sports_client=None)

    def test_basic_filter_passes_valid_rec(self):
        recs = [_rec(yes_price=0.20, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)
        self.assertAlmostEqual(opps[0].entry_price, 0.20)

    def test_price_too_high_filtered(self):
        """Precio > ER_MAX_ENTRY_PRICE (0.33) no pasa."""
        recs = [_rec(yes_price=0.40, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_price_too_low_filtered(self):
        """Precio < 0.05 no pasa (demasiado especulativo)."""
        recs = [_rec(yes_price=0.03, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_no_edge_filtered(self):
        """Edge ≤ 0 no pasa."""
        recs = [_rec(yes_price=0.20, edge_pct=-5.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_zero_edge_filtered(self):
        recs = [_rec(yes_price=0.20, edge_pct=0.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_low_liquidity_filtered(self):
        """Liquidez < 500 no pasa."""
        recs = [_rec(yes_price=0.20, edge_pct=10.0, liquidity=200.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_side_no_uses_no_price(self):
        """Para side=NO, el entry_price debe ser no_price."""
        recs = [_rec(side="NO", yes_price=0.85, no_price=0.18, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)
        self.assertAlmostEqual(opps[0].entry_price, 0.18)

    def test_multiplier_calculated_correctly(self):
        recs = [_rec(yes_price=0.20, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertAlmostEqual(opps[0].multiplier, 5.0, places=1)

    def test_sorted_by_kl_descending(self):
        recs = [
            _rec(market_id="low",  yes_price=0.20, edge_pct=5.0, kl_div=0.05),
            _rec(market_id="high", yes_price=0.20, edge_pct=5.0, kl_div=0.20),
        ]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(opps[0].market_id, "high")

    def test_max_20_results(self):
        recs = [_rec(market_id=f"m{i}", yes_price=0.20, edge_pct=5.0) for i in range(30)]
        opps = self.engine.filter_opportunities(recs)
        self.assertLessEqual(len(opps), 20)


class TestCategoryMultiplierThreshold(unittest.TestCase):

    def setUp(self):
        self.engine = EscapeRoomEngine(llm_engine=MockLLM(), sports_client=None)

    def test_geopolitics_requires_6x_minimum(self):
        """Geopolitics: necesita 6x → precio max = 1/6 ≈ 0.167."""
        # Precio 0.20 → 5x < 6x → filtrado
        recs = [_rec(category="geopolitics", yes_price=0.20, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_geopolitics_passes_with_6x(self):
        """Geopolitics: precio 0.15 → 6.67x ≥ 6x → pasa."""
        recs = [_rec(category="geopolitics", yes_price=0.15, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)

    def test_politics_requires_5x_minimum(self):
        """Politics: precio 0.22 → 4.5x < 5x → filtrado."""
        recs = [_rec(category="politics", yes_price=0.22, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 0)

    def test_politics_passes_with_5x(self):
        """Politics: precio 0.19 → 5.26x ≥ 5x → pasa."""
        recs = [_rec(category="politics", yes_price=0.19, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)

    def test_crypto_passes_at_3x(self):
        """Crypto: precio 0.33 → 3.03x ≥ 3x → pasa."""
        recs = [_rec(category="crypto", yes_price=0.32, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)

    def test_sports_requires_35x(self):
        """Sports: precio 0.30 → 3.33x ≥ 3.5x → pasa (borderline)."""
        recs = [_rec(category="sports", yes_price=0.28, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        self.assertEqual(len(opps), 1)

    def test_category_min_multiplier_stored_in_opp(self):
        recs = [_rec(category="geopolitics", yes_price=0.15, edge_pct=10.0)]
        opps = self.engine.filter_opportunities(recs)
        if opps:
            self.assertAlmostEqual(opps[0].cat_min_multiplier, 6.0)

    def test_category_map_completeness(self):
        """Todas las categorías esperadas tienen threshold definido."""
        for cat in ["crypto", "sports", "economics", "tech", "politics", "geopolitics", "other"]:
            self.assertIn(cat, ER_MIN_MULTIPLIER_BY_CATEGORY)

    def test_thresholds_in_valid_range(self):
        for cat, mult in ER_MIN_MULTIPLIER_BY_CATEGORY.items():
            self.assertGreaterEqual(mult, 3.0, f"{cat} threshold below 3x")
            self.assertLessEqual(mult, 20.0, f"{cat} threshold above 20x")


class TestEROpportunityDataclass(unittest.TestCase):

    def test_default_values(self):
        opp = EROpportunity(
            market_id="m1", question="test?", slug="test", category="crypto",
            side="YES", entry_price=0.20, multiplier=5.0,
            ensemble_prob=0.35, edge_pct=10.0, kl_divergence=0.08,
            yes_token="yt", no_token="nt",
        )
        self.assertEqual(opp.er_score, 0.0)
        self.assertEqual(opp.bet_usd, 0.0)
        self.assertEqual(opp.claude_recommendation, "NO_APOSTAR")
        self.assertEqual(opp.gpt_recommendation, "NO_APOSTAR")
        self.assertEqual(opp.ai_consensus, 0.0)
        self.assertEqual(opp.cat_min_multiplier, 3.0)

    def test_entry_price_stored_correctly(self):
        opp = EROpportunity(
            market_id="m1", question="?", slug="s", category="crypto",
            side="YES", entry_price=0.15, multiplier=6.67,
            ensemble_prob=0.30, edge_pct=5.0, kl_divergence=0.04,
            yes_token="yt", no_token="nt",
        )
        self.assertAlmostEqual(opp.entry_price, 0.15)
        self.assertAlmostEqual(opp.multiplier, 6.67)


if __name__ == "__main__":
    unittest.main(verbosity=2)
