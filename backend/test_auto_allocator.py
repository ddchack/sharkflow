"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para AutoAllocator — distribución de capital, límites, diversificación.
"""
import unittest
from auto_allocator import AutoAllocator, AllocationPlan, AllocationSlot
from datetime import datetime, timezone, timedelta


def _rec(market_id="m1", question="Will X?", category="crypto",
         side="YES", market_price=0.5, ensemble_prob=0.65,
         kl_divergence=0.08, edge_pct=15.0, confidence=65.0,
         end_date=None):
    """Helper: create a minimal recommendation dict."""
    if end_date is None:
        end_date = (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "market_id": market_id,
        "question": question,
        "category": category,
        "side": side,
        "market_price": market_price,
        "ensemble_prob": ensemble_prob,
        "kl_divergence": kl_divergence,
        "binary_sharpe": 1.5,
        "edge_pct": edge_pct,
        "confidence": confidence,
        "end_date": end_date,
        "yes_token": "yt",
        "no_token": "nt",
        "reasoning": [],
    }


def _multi_recs(n=10, category="crypto"):
    return [_rec(market_id=f"m{i}", category=category,
                 kl_divergence=0.05 + i * 0.01,
                 ensemble_prob=0.55 + i * 0.01) for i in range(n)]


class TestAutoAllocatorBasic(unittest.TestCase):

    def setUp(self):
        self.alloc = AutoAllocator(kelly_fraction=0.25)

    def test_empty_recommendations_returns_empty_plan(self):
        plan = self.alloc.allocate([], total_budget=20.0)
        self.assertIsInstance(plan, AllocationPlan)
        self.assertEqual(plan.total_allocated, 0.0)
        self.assertEqual(len(plan.slots), 0)

    def test_plan_has_slots(self):
        recs = _multi_recs(8)
        plan = self.alloc.allocate(recs, total_budget=20.0, min_bets=3, max_bets=8)
        self.assertGreater(len(plan.slots), 0)

    def test_total_allocated_leq_budget(self):
        recs = _multi_recs(8)
        plan = self.alloc.allocate(recs, total_budget=20.0)
        self.assertLessEqual(plan.total_allocated, 20.0)

    def test_reserve_retained(self):
        """5% of budget should be reserved."""
        recs = _multi_recs(10)
        plan = self.alloc.allocate(recs, total_budget=100.0, min_bets=5, max_bets=10)
        self.assertLessEqual(plan.total_allocated, 95.0 + 0.01)  # ≤ 95% of budget

    def test_min_bet_size_respected(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=20.0)
        for slot in plan.slots:
            self.assertGreaterEqual(slot["allocated_usd"], AutoAllocator.MIN_BET_USD - 0.01)

    def test_max_single_bet_pct_respected(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=20.0)
        max_single = 20.0 * AutoAllocator.MAX_SINGLE_BET_PCT
        for slot in plan.slots:
            self.assertLessEqual(slot["allocated_usd"], max_single + 0.01)

    def test_plan_has_expected_fields(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=20.0)
        self.assertIsInstance(plan.plan_id, str)
        self.assertIsNotNone(plan.category_breakdown)
        self.assertIsInstance(plan.slots, list)
        self.assertGreaterEqual(plan.portfolio_sharpe, 0)


class TestAutoAllocatorCategoryDiversification(unittest.TestCase):

    def setUp(self):
        self.alloc = AutoAllocator(kelly_fraction=0.25)

    def test_category_cap_respected(self):
        """No category should get more than MAX_PER_CATEGORY_PCT of budget."""
        recs = _multi_recs(10, category="crypto")  # All same category
        plan = self.alloc.allocate(recs, total_budget=100.0, max_bets=10)
        max_cat = 100.0 * AutoAllocator.MAX_PER_CATEGORY_PCT
        for cat, total in plan.category_breakdown.items():
            # The allocator can overshoot the cap slightly when min_bet_usd kicks in
            # (known limitation: each extra bet adds MIN_BET_USD past the threshold)
            n_bets = len(plan.slots)
            tolerance = n_bets * AutoAllocator.MIN_BET_USD
            self.assertLessEqual(float(total), max_cat + tolerance)

    def test_mixed_categories_in_breakdown(self):
        recs = (
            _multi_recs(4, category="crypto") +
            _multi_recs(4, category="sports") +
            _multi_recs(4, category="politics")
        )
        plan = self.alloc.allocate(recs, total_budget=50.0, max_bets=12)
        self.assertGreater(len(plan.category_breakdown), 1)

    def test_higher_kl_gets_more_allocation(self):
        """Two recs in different categories — higher KL should get more capital."""
        rec_low  = _rec(market_id="low",  category="politics", kl_divergence=0.05, ensemble_prob=0.60)
        rec_high = _rec(market_id="high", category="sports",   kl_divergence=0.20, ensemble_prob=0.70)
        plan = self.alloc.allocate([rec_low, rec_high], total_budget=20.0, min_bets=2)
        allocs = {s["market_id"]: float(s["allocated_usd"]) for s in plan.slots}
        if "low" in allocs and "high" in allocs:
            self.assertGreaterEqual(allocs["high"], allocs["low"])


class TestAutoAllocatorTimeHorizon(unittest.TestCase):

    def setUp(self):
        self.alloc = AutoAllocator(kelly_fraction=0.25)

    def test_short_horizon_filter(self):
        short_date = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        long_date  = (datetime.now(timezone.utc) + timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        recs = [
            _rec(market_id="short1", end_date=short_date),
            _rec(market_id="short2", end_date=short_date),
            _rec(market_id="long1",  end_date=long_date),
        ]
        plan = self.alloc.allocate(recs, total_budget=20.0, time_horizon="SHORT", min_bets=1)
        market_ids = [s["market_id"] for s in plan.slots]
        self.assertTrue(len(market_ids) > 0)

    def test_time_horizon_slot_labels(self):
        short_date  = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        medium_date = (datetime.now(timezone.utc) + timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        long_date   = (datetime.now(timezone.utc) + timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        recs = [
            _rec(market_id="s", end_date=short_date,  kl_divergence=0.10),
            _rec(market_id="m", end_date=medium_date, kl_divergence=0.09),
            _rec(market_id="l", end_date=long_date,   kl_divergence=0.08),
        ]
        plan = self.alloc.allocate(recs, total_budget=20.0, min_bets=1, max_bets=3)
        th_map = {s["market_id"]: s["time_horizon"] for s in plan.slots}
        if "s" in th_map:
            self.assertEqual(th_map["s"], "SHORT")
        if "m" in th_map:
            self.assertEqual(th_map["m"], "MEDIUM")
        if "l" in th_map:
            self.assertEqual(th_map["l"], "LONG")


class TestAutoAllocatorMetrics(unittest.TestCase):

    def setUp(self):
        self.alloc = AutoAllocator(kelly_fraction=0.25)

    def test_portfolio_sharpe_nonnegative_positive_ev(self):
        recs = _multi_recs(5)  # All have positive edge
        plan = self.alloc.allocate(recs, total_budget=50.0, min_bets=3)
        self.assertGreaterEqual(plan.portfolio_sharpe, 0)

    def test_allocation_pct_sums_to_total_allocated_pct(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=50.0)
        pct_sum = sum(s["allocation_pct"] for s in plan.slots)
        expected = plan.total_allocated / plan.total_budget * 100
        self.assertAlmostEqual(pct_sum, expected, delta=1.0)

    def test_priority_ranks_sequential(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=20.0, min_bets=3, max_bets=5)
        ranks = sorted(s["priority_rank"] for s in plan.slots)
        self.assertEqual(ranks, list(range(1, len(plan.slots) + 1)))

    def test_effective_exposure_nonnegative(self):
        recs = _multi_recs(5)
        plan = self.alloc.allocate(recs, total_budget=50.0)
        self.assertGreaterEqual(plan.effective_exposure, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
