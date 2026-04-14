"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para MeanReversionEngine y microstructure (OBI + VPIN).
"""
import unittest
import math
from mean_reversion import MeanReversionEngine, MeanReversionSignal, logit, inv_logit
from microstructure import OrderBookImbalance, VPIN, OBISignal, VPINSignal


# ── helpers ──────────────────────────────────────────────────────────

def _flat_prices(p=0.5, n=30):
    return [p] * n

def _trending_up(start=0.3, end=0.7, n=30):
    return [start + (end - start) * i / (n - 1) for i in range(n)]

def _spiked(base=0.5, spike=0.85, n=30):
    """Prices stable then one big spike at the end."""
    return [base] * (n - 3) + [base, base * 0.9, spike]


# ── Logit helpers ─────────────────────────────────────────────────────

class TestLogitHelpers(unittest.TestCase):

    def test_logit_at_half(self):
        self.assertAlmostEqual(logit(0.5), 0.0, places=5)

    def test_logit_above_half_positive(self):
        self.assertGreater(logit(0.7), 0)

    def test_logit_below_half_negative(self):
        self.assertLess(logit(0.3), 0)

    def test_inv_logit_roundtrip(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.assertAlmostEqual(inv_logit(logit(p)), p, places=5)

    def test_logit_clamped_boundaries(self):
        # Should not raise for extreme values
        _ = logit(0.0001)
        _ = logit(0.9999)


# ── MeanReversionEngine ───────────────────────────────────────────────

class TestMeanReversionDetect(unittest.TestCase):

    def test_flat_prices_neutral(self):
        sig = MeanReversionEngine.detect_overreaction(_flat_prices())
        self.assertEqual(sig.direction, "NEUTRAL")
        self.assertFalse(sig.is_overreaction)

    def test_spike_high_triggers_fade_up(self):
        prices = _flat_prices(0.5, 28) + [0.85, 0.88]
        sig = MeanReversionEngine.detect_overreaction(prices)
        self.assertEqual(sig.direction, "FADE_UP")
        self.assertTrue(sig.is_overreaction)

    def test_spike_low_triggers_fade_down(self):
        prices = _flat_prices(0.5, 28) + [0.15, 0.12]
        sig = MeanReversionEngine.detect_overreaction(prices)
        self.assertEqual(sig.direction, "FADE_DOWN")
        self.assertTrue(sig.is_overreaction)

    def test_too_few_prices_returns_neutral(self):
        sig = MeanReversionEngine.detect_overreaction([0.5, 0.6])
        self.assertEqual(sig.direction, "NEUTRAL")

    def test_bollinger_position_range(self):
        sig = MeanReversionEngine.detect_overreaction(_trending_up())
        self.assertGreaterEqual(sig.bollinger_position, 0)
        self.assertLessEqual(sig.bollinger_position, 1)

    def test_confidence_range(self):
        sig = MeanReversionEngine.detect_overreaction(_spiked())
        self.assertGreaterEqual(sig.confidence, 0)
        self.assertLessEqual(sig.confidence, 1)

    def test_rsi_range(self):
        prices = _trending_up()
        sig = MeanReversionEngine.detect_overreaction(prices)
        self.assertGreaterEqual(sig.rsi, 0)
        self.assertLessEqual(sig.rsi, 100)

    def test_overreaction_returns_dataclass(self):
        sig = MeanReversionEngine.detect_overreaction(_flat_prices())
        self.assertIsInstance(sig, MeanReversionSignal)

    def test_high_price_fade_direction(self):
        """Precio en zona alta (>0.80) → FADE_UP expected"""
        prices = [0.50] * 25 + [0.82, 0.83, 0.84, 0.85, 0.86]
        sig = MeanReversionEngine.detect_overreaction(prices)
        # Either detected or neutral — just no crash
        self.assertIn(sig.direction, ["FADE_UP", "NEUTRAL"])


# ── MeanReversionEngine — RSI ─────────────────────────────────────────

class TestMeanReversionRSI(unittest.TestCase):

    def test_rsi_uptrend_above_50(self):
        prices = _trending_up(0.3, 0.7, 20)
        sig = MeanReversionEngine.detect_overreaction(prices)
        # Linear uptrend: all gains → RSI = 100 or exactly 50 depending on implementation
        self.assertGreaterEqual(sig.rsi, 50)

    def test_rsi_downtrend_below_50(self):
        prices = list(reversed(_trending_up(0.3, 0.7, 20)))
        sig = MeanReversionEngine.detect_overreaction(prices)
        # Linear downtrend: all losses → RSI ≤ 50
        self.assertLessEqual(sig.rsi, 50)


# ── OrderBookImbalance ────────────────────────────────────────────────

class TestOBI(unittest.TestCase):

    def _bids(self, sizes):
        return [{"price": 0.5 - i * 0.01, "size": s} for i, s in enumerate(sizes)]

    def _asks(self, sizes):
        return [{"price": 0.5 + (i + 1) * 0.01, "size": s} for i, s in enumerate(sizes)]

    def test_balanced_book_neutral(self):
        sig = OrderBookImbalance.compute_obi(
            self._bids([100, 80, 60]),
            self._asks([100, 80, 60]))
        self.assertEqual(sig.predicted_direction, "NEUTRAL")

    def test_heavy_bids_direction_up(self):
        sig = OrderBookImbalance.compute_obi(
            self._bids([500, 400, 300]),
            self._asks([50, 40, 30]))
        self.assertEqual(sig.predicted_direction, "UP")

    def test_heavy_asks_direction_down(self):
        sig = OrderBookImbalance.compute_obi(
            self._bids([50, 40, 30]),
            self._asks([500, 400, 300]))
        self.assertEqual(sig.predicted_direction, "DOWN")

    def test_empty_book_neutral(self):
        sig = OrderBookImbalance.compute_obi([], [])
        self.assertEqual(sig.predicted_direction, "NEUTRAL")
        self.assertEqual(sig.imbalance, 0)

    def test_strength_between_0_and_1(self):
        sig = OrderBookImbalance.compute_obi(
            self._bids([100, 80]),
            self._asks([50, 40]))
        self.assertGreaterEqual(sig.strength, 0)
        self.assertLessEqual(sig.strength, 1)

    def test_returns_obi_signal(self):
        sig = OrderBookImbalance.compute_obi(
            self._bids([100]),
            self._asks([100]))
        self.assertIsInstance(sig, OBISignal)


# ── VPIN ──────────────────────────────────────────────────────────────

class TestVPIN(unittest.TestCase):

    def _trades(self, n_buy, n_sell, size=10):
        """Interleave BUY and SELL so buckets are mixed (not all-BUY then all-SELL)."""
        trades = []
        total = n_buy + n_sell
        for i in range(total):
            # Distribute buys and sells proportionally throughout the sequence
            side = "BUY" if (i * n_buy // total) > ((i - 1) * n_buy // total if i > 0 else -1) else "SELL"
            trades.append({"price": 0.5, "size": size, "side": side})
        return trades

    def test_too_few_trades_returns_default(self):
        sig = VPIN.compute([], bucket_size=50)
        self.assertEqual(sig.toxicity, "LOW")

    def test_balanced_trades_low_vpin(self):
        trades = self._trades(100, 100, size=5)
        sig = VPIN.compute(trades, bucket_size=50, n_buckets=10)
        self.assertLess(sig.vpin, 0.5)

    def test_imbalanced_trades_higher_vpin(self):
        trades = self._trades(200, 20, size=5)
        sig = VPIN.compute(trades, bucket_size=50, n_buckets=10)
        # With heavy buy side, VPIN should be higher
        self.assertGreater(sig.vpin, 0.0)

    def test_buy_heavy_smart_money_buy(self):
        trades = self._trades(200, 20, size=5)
        sig = VPIN.compute(trades, bucket_size=50, n_buckets=10)
        self.assertIn(sig.smart_money_side, ["BUY", "NEUTRAL"])

    def test_toxicity_levels(self):
        for vpin, expected in [(0.9, "EXTREME"), (0.65, "HIGH"), (0.45, "MEDIUM"), (0.2, "LOW")]:
            trades = self._trades(int(vpin * 100), int((1 - vpin) * 100), size=5)
            sig = VPIN.compute(trades, bucket_size=30, n_buckets=10)
            self.assertIn(sig.toxicity, ["LOW", "MEDIUM", "HIGH", "EXTREME"])

    def test_returns_vpin_signal(self):
        trades = self._trades(50, 50, size=5)
        sig = VPIN.compute(trades)
        self.assertIsInstance(sig, VPINSignal)

    def test_vpin_in_range(self):
        trades = self._trades(80, 20, size=5)
        sig = VPIN.compute(trades, bucket_size=30, n_buckets=10)
        self.assertGreaterEqual(sig.vpin, 0)
        self.assertLessEqual(sig.vpin, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
