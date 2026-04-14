"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para RiskManager: validaciones, circuit breakers, drawdown, PnL.
"""
import unittest
import time
from risk_manager import RiskManager, RiskState, RiskLimits


def _rm(initial_capital=100.0, **limit_overrides):
    limits = RiskLimits(**limit_overrides)
    return RiskManager(initial_capital=initial_capital, limits=limits)


def _valid_trade(rm, bet_usd=5.0, confidence=70.0, liquidity=10000.0, spread=0.02, market_id="mkt1"):
    return rm.validate_trade(bet_usd, confidence, liquidity, spread, market_id)


class TestBasicValidation(unittest.TestCase):

    def test_valid_trade_approved(self):
        rm = _rm(min_time_between_trades=0.0)
        result = _valid_trade(rm)
        self.assertTrue(result["approved"])

    def test_low_confidence_rejected(self):
        rm = _rm(min_time_between_trades=0.0, min_confidence=70.0)
        result = rm.validate_trade(5.0, confidence=30.0, market_liquidity=10000, spread=0.02)
        self.assertFalse(result["approved"])

    def test_low_liquidity_rejected(self):
        rm = _rm(min_time_between_trades=0.0, min_liquidity_usd=5000.0)
        result = rm.validate_trade(5.0, confidence=70.0, market_liquidity=100, spread=0.02)
        self.assertFalse(result["approved"])

    def test_high_spread_rejected(self):
        rm = _rm(min_time_between_trades=0.0, max_spread_pct=5.0)
        result = rm.validate_trade(5.0, confidence=70.0, market_liquidity=10000, spread=0.20)  # 20%
        self.assertFalse(result["approved"])

    def test_bet_capped_at_25pct_capital(self):
        rm = _rm(initial_capital=100.0, min_time_between_trades=0.0, max_single_bet_pct=25.0)
        result = rm.validate_trade(50.0, confidence=70.0, market_liquidity=10000, spread=0.02)
        self.assertLessEqual(result["adjusted_size"], 25.01)

    def test_duplicate_position_blocked(self):
        rm = _rm(min_time_between_trades=0.0)
        # First trade at max for this market
        rm.state.current_capital = 100.0
        rm._position_tracker["mkt1"] = 25.0  # Already at max (25% of 100)
        result = rm.validate_trade(5.0, confidence=70.0, market_liquidity=10000, spread=0.02, market_id="mkt1")
        self.assertFalse(result["approved"])

    def test_kill_switch_blocks_all_trades(self):
        rm = _rm(min_time_between_trades=0.0)
        rm.state.is_active = False
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])

    def test_paused_blocks_all_trades(self):
        rm = _rm(min_time_between_trades=0.0)
        rm.state.is_paused = True
        rm.state.pause_reason = "test pause"
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])

    def test_daily_trade_limit(self):
        rm = _rm(min_time_between_trades=0.0, max_daily_trades=3)
        rm.state.daily_trades = 3
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])


class TestCircuitBreakers(unittest.TestCase):

    def test_3_consecutive_losses_pauses_bot(self):
        rm = _rm(min_time_between_trades=0.0, max_consecutive_losses=3)
        rm.state.consecutive_losses = 3
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])
        self.assertTrue(rm.state.is_paused)

    def test_2_consecutive_losses_reduces_size(self):
        rm = _rm(min_time_between_trades=0.0, max_consecutive_losses=3, loss_reduction_factor=0.5)
        rm.state.consecutive_losses = 2
        result = rm.validate_trade(20.0, confidence=70.0, market_liquidity=10000, spread=0.02)
        # With consecutive_losses=2: reduction = 0.5^(2-1) = 0.5
        self.assertLessEqual(result["adjusted_size"], 10.5)  # 20 * 0.5 = 10

    def test_daily_loss_limit_triggers_pause(self):
        rm = _rm(min_time_between_trades=0.0, max_daily_loss_pct=10.0)
        rm.state.daily_pnl = -12.0  # -12% loss
        rm._daily_start_capital = 100.0
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])
        self.assertTrue(rm.state.is_paused)

    def test_max_drawdown_kills_bot(self):
        rm = _rm(min_time_between_trades=0.0, max_drawdown_pct=20.0)
        rm.state.current_drawdown_pct = 21.0
        result = _valid_trade(rm)
        self.assertFalse(result["approved"])
        self.assertFalse(rm.state.is_active)


class TestTradeRecording(unittest.TestCase):

    def test_record_win_updates_capital(self):
        rm = _rm()
        rm.record_trade("mkt1", 5.0, pnl=2.5, won=True)
        self.assertGreater(rm.state.current_capital, 100.0)

    def test_record_loss_decreases_capital(self):
        rm = _rm()
        rm.record_trade("mkt1", 5.0, pnl=-5.0, won=False)
        self.assertLess(rm.state.current_capital, 100.0)

    def test_consecutive_losses_tracked(self):
        rm = _rm()
        rm.record_trade("mkt1", 5.0, pnl=-5.0, won=False)
        rm.record_trade("mkt2", 5.0, pnl=-5.0, won=False)
        self.assertEqual(rm.state.consecutive_losses, 2)

    def test_win_resets_consecutive_losses(self):
        rm = _rm()
        rm.state.consecutive_losses = 2
        rm.record_trade("mkt1", 5.0, pnl=2.5, won=True)
        self.assertEqual(rm.state.consecutive_losses, 0)

    def test_exposure_tracked(self):
        rm = _rm()
        rm.record_trade("mkt1", 10.0)
        self.assertGreater(rm.state.total_exposure_usd, 0)

    def test_exposure_reduced_on_close(self):
        rm = _rm()
        rm.record_trade("mkt1", 10.0)
        exp_before = rm.state.total_exposure_usd
        rm.close_position("mkt1", pnl=5.0)
        self.assertLessEqual(rm.state.total_exposure_usd, exp_before)


class TestRiskReport(unittest.TestCase):

    def test_report_has_required_fields(self):
        rm = _rm()
        report = rm.get_risk_report()
        self.assertIn("capital", report)
        self.assertIn("risk_metrics", report)
        self.assertIn("limits", report)

    def test_drawdown_tracked(self):
        rm = _rm(initial_capital=100.0)
        rm.record_trade("mkt1", 5.0, pnl=-15.0, won=False)
        report = rm.get_risk_report()
        self.assertGreater(report["risk_metrics"]["current_drawdown_pct"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
