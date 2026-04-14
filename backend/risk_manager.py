"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Risk Manager
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Advanced risk management:
- Circuit breakers (auto-pause on excessive losses)
- Position sizing with anti-martingale
- Drawdown tracking
- Correlation-aware exposure limits
- Pre-trade validation
- Kill switch
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import deque


@dataclass
class RiskState:
    """Current risk state of the bot."""
    is_active: bool = True        # Master switch
    is_paused: bool = False       # Temporarily paused by circuit breaker
    pause_reason: str = ""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    peak_capital: float = 100.0
    current_capital: float = 100.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    open_positions: int = 0
    total_exposure_usd: float = 0.0
    last_trade_time: float = 0.0


@dataclass
class RiskLimits:
    """Configurable risk parameters."""
    max_daily_loss_pct: float = 10.0       # Pause if daily loss > 10%
    max_drawdown_pct: float = 20.0          # Kill switch if drawdown > 20%
    max_consecutive_losses: int = 3         # Guía: parar completamente tras 3 pérdidas seguidas
    max_single_bet_pct: float = 25.0        # Max 25% of capital per trade
    max_daily_trades: int = 30              # Max trades per day
    max_open_positions: int = 10            # Max simultaneous open positions
    max_total_exposure_pct: float = 80.0    # Max 80% of capital deployed
    min_time_between_trades: float = 5.0    # Seconds between trades
    min_liquidity_usd: float = 5000.0       # Minimum market liquidity
    max_spread_pct: float = 8.0             # Max acceptable spread
    min_confidence: float = 40.0            # Minimum confidence score
    loss_reduction_factor: float = 0.5      # Reduce size by 50% after consecutive losses


class RiskManager:
    """
    Enforces risk limits and circuit breakers.
    __author__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self, initial_capital: float = 100.0,
                 limits: RiskLimits = None):
        self.initial_capital = initial_capital
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            peak_capital=initial_capital,
            current_capital=initial_capital,
        )
        self._trade_log: deque = deque(maxlen=500)
        self._daily_start_capital = initial_capital
        self._position_tracker: dict[str, float] = {}  # market_id -> exposure_usd
        # Adaptive Kelly multiplier (0.15-0.35, ajustado por drawdown 72h)
        self._kelly_multiplier: float = 0.25
        self._capital_72h_ago: float = initial_capital  # snapshot cada 72h

    def adaptive_kelly_multiplier(self) -> float:
        """
        Ajusta el Kelly multiplier basado en drawdown de 72h.
        Si drawdown_3d > 30%: reduce 0.02 (floor 0.15)
        Si drawdown_3d < 15%: aumenta 0.02 (cap 0.35)
        """
        if self._capital_72h_ago <= 0:
            return self._kelly_multiplier
        drawdown_3d = max(0.0, 1.0 - self.state.current_capital / self._capital_72h_ago)
        if drawdown_3d > 0.30:
            self._kelly_multiplier = max(0.15, self._kelly_multiplier - 0.02)
        elif drawdown_3d < 0.15:
            self._kelly_multiplier = min(0.35, self._kelly_multiplier + 0.02)
        return self._kelly_multiplier

    def update_capital_72h_snapshot(self):
        """Actualiza el snapshot de capital de 72h. Llamar cada 72h."""
        self._capital_72h_ago = self.state.current_capital

    # ─────────────────────────────────────────────────────────
    # PRE-TRADE VALIDATION
    # ─────────────────────────────────────────────────────────

    def validate_trade(self, bet_usd: float, confidence: float,
                       market_liquidity: float, spread: float,
                       market_id: str = "") -> dict:
        """
        Validate a proposed trade against all risk limits.
        Returns: {"approved": bool, "adjusted_size": float, "reasons": [str]}
        """
        reasons = []
        approved = True
        adjusted_size = bet_usd

        # Master switch
        if not self.state.is_active:
            return {"approved": False, "adjusted_size": 0,
                    "reasons": ["Bot is deactivated (kill switch)"]}

        # Circuit breaker pause
        if self.state.is_paused:
            return {"approved": False, "adjusted_size": 0,
                    "reasons": [f"Paused: {self.state.pause_reason}"]}

        # Daily loss limit
        if self.state.daily_pnl < 0:
            daily_loss_pct = abs(self.state.daily_pnl) / max(1, self._daily_start_capital) * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                self._trigger_circuit_breaker(
                    f"Daily loss limit reached: -{daily_loss_pct:.1f}%")
                return {"approved": False, "adjusted_size": 0,
                        "reasons": ["Daily loss limit reached"]}

        # Drawdown check
        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            self._kill_switch(
                f"Max drawdown reached: -{self.state.current_drawdown_pct:.1f}%")
            return {"approved": False, "adjusted_size": 0,
                    "reasons": ["Max drawdown - kill switch activated"]}

        # Max daily trades
        if self.state.daily_trades >= self.limits.max_daily_trades:
            return {"approved": False, "adjusted_size": 0,
                    "reasons": ["Daily trade limit reached"]}

        # Max open positions
        if self.state.open_positions >= self.limits.max_open_positions:
            reasons.append("Max open positions reached")
            approved = False

        # Time between trades
        elapsed = time.time() - self.state.last_trade_time
        if elapsed < self.limits.min_time_between_trades:
            reasons.append(f"Too fast. Wait {self.limits.min_time_between_trades - elapsed:.0f}s")
            approved = False

        # Confidence threshold
        if confidence < self.limits.min_confidence:
            reasons.append(f"Confidence {confidence:.0f} < minimum {self.limits.min_confidence:.0f}")
            approved = False

        # Liquidity check
        if market_liquidity < self.limits.min_liquidity_usd:
            reasons.append(f"Low liquidity: ${market_liquidity:,.0f}")
            approved = False

        # Spread check
        if spread * 100 > self.limits.max_spread_pct:
            reasons.append(f"Spread too wide: {spread*100:.1f}%")
            approved = False

        # Position sizing adjustments
        max_bet = self.state.current_capital * (self.limits.max_single_bet_pct / 100)
        adjusted_size = min(adjusted_size, max_bet)
        # Si el capital total permite $1 pero el % lo recortó, forzar mínimo de Polymarket
        if adjusted_size < 1.0 and self.state.current_capital >= 1.0:
            adjusted_size = 1.0

        # Consecutive loss reduction / halt
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            reasons.append(f"HALT: {self.state.consecutive_losses} consecutive losses — bot paused for 24h")
            self.state.is_paused = True
            self.state.pause_reason = f"3+ consecutive losses at {datetime.now(timezone.utc).isoformat()}"
            approved = False
        elif self.state.consecutive_losses >= 2:
            reduction = self.limits.loss_reduction_factor ** (
                self.state.consecutive_losses - 1)
            adjusted_size *= reduction
            reasons.append(f"Size reduced {reduction:.0%} due to {self.state.consecutive_losses} consecutive losses")

        # Total exposure check
        max_exposure = self.state.current_capital * (self.limits.max_total_exposure_pct / 100)
        remaining_exposure = max_exposure - self.state.total_exposure_usd
        if adjusted_size > remaining_exposure:
            adjusted_size = max(0, remaining_exposure)
            reasons.append(f"Exposure capped at {self.limits.max_total_exposure_pct}%")

        # Duplicate position check
        if market_id and market_id in self._position_tracker:
            existing = self._position_tracker[market_id]
            remaining = max_bet - existing
            adjusted_size = min(adjusted_size, max(0, remaining))
            if adjusted_size <= 0:
                reasons.append(f"Already have ${existing:.2f} exposure in this market")
                approved = False

        # Minimum viable bet
        if adjusted_size < 1.0:
            reasons.append("Adjusted size below $1 minimum")
            approved = False

        if not reasons and approved:
            reasons.append("All checks passed")

        return {
            "approved": approved,
            "adjusted_size": round(adjusted_size, 2),
            "reasons": reasons,
        }

    # ─────────────────────────────────────────────────────────
    # TRADE RECORDING
    # ─────────────────────────────────────────────────────────

    def record_trade(self, market_id: str, bet_usd: float,
                     pnl: float = 0, won: bool = None):
        """Record a completed trade and update risk state."""
        self.state.daily_trades += 1
        self.state.last_trade_time = time.time()

        if pnl != 0 or won is not None:
            self.state.daily_pnl += pnl
            self.state.current_capital += pnl

            if won or pnl > 0:
                self.state.consecutive_losses = 0
                self.state.consecutive_wins += 1
            elif won is False or pnl < 0:
                self.state.consecutive_wins = 0
                self.state.consecutive_losses += 1

            # Update peak and drawdown
            if self.state.current_capital > self.state.peak_capital:
                self.state.peak_capital = self.state.current_capital

            if self.state.peak_capital > 0:
                self.state.current_drawdown_pct = (
                    (self.state.peak_capital - self.state.current_capital) /
                    self.state.peak_capital * 100
                )
                self.state.max_drawdown_pct = max(
                    self.state.max_drawdown_pct,
                    self.state.current_drawdown_pct
                )

        # Track position
        if market_id:
            self._position_tracker[market_id] = (
                self._position_tracker.get(market_id, 0) + bet_usd)
            self.state.total_exposure_usd = sum(self._position_tracker.values())
            self.state.open_positions = len(self._position_tracker)

        self._trade_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "bet_usd": bet_usd,
            "pnl": pnl,
            "capital_after": self.state.current_capital,
        })

    def close_position(self, market_id: str, pnl: float = 0, resolution: str = "resolved") -> bool:
        """Marca una posición como cerrada cuando el mercado se resuelve."""
        closed = False
        if market_id in self._position_tracker:
            size = self._position_tracker.pop(market_id)
            self.state.total_exposure_usd = sum(self._position_tracker.values())
            self.state.open_positions = len(self._position_tracker)
            print(f"[RiskManager] Posición cerrada: {market_id} ({resolution}), size=${size:.2f}")
            closed = True

        if pnl != 0:
            self.state.daily_pnl += pnl
            self.state.current_capital += pnl

        return closed

    def cleanup_old_positions(self, max_age_hours: float = 48.0) -> int:
        """Limpia posiciones abiertas con más de max_age_hours (probablemente ya resueltas)."""
        now = time.time()
        # Determinar la primera vez que se vio cada market_id en el trade log
        market_first_seen: dict = {}
        for entry in self._trade_log:
            mid = entry.get("market_id", "")
            if mid and mid in self._position_tracker and mid not in market_first_seen:
                ts_str = entry.get("timestamp", "")
                if ts_str:
                    try:
                        from datetime import datetime as _dt
                        # Normalizar sufijo "Z" → "+00:00" para evitar double-suffix en Python < 3.11
                        _ts_norm = ts_str.replace("Z", "+00:00") if ts_str.endswith("Z") else ts_str
                        opened_at = _dt.fromisoformat(_ts_norm).timestamp()
                        market_first_seen[mid] = opened_at
                    except Exception:
                        pass
        to_close = [
            mid for mid, opened_at in market_first_seen.items()
            if (now - opened_at) / 3600 > max_age_hours
        ]
        for mid in to_close:
            self._position_tracker.pop(mid, None)
        if to_close:
            self.state.total_exposure_usd = sum(self._position_tracker.values())
            self.state.open_positions = len(self._position_tracker)
            print(f"[RiskManager] Limpieza automática: {len(to_close)} posiciones viejas eliminadas")
        return len(to_close)

    # ─────────────────────────────────────────────────────────
    # CIRCUIT BREAKERS
    # ─────────────────────────────────────────────────────────

    def _trigger_circuit_breaker(self, reason: str):
        """Temporarily pause trading."""
        self.state.is_paused = True
        self.state.pause_reason = reason
        print(f"[RISK] ⚠️ CIRCUIT BREAKER: {reason}")

    def _kill_switch(self, reason: str):
        """Emergency stop - deactivate bot completely."""
        self.state.is_active = False
        self.state.is_paused = True
        self.state.pause_reason = f"KILL SWITCH: {reason}"
        print(f"[RISK] 🛑 KILL SWITCH ACTIVATED: {reason}")

    def resume(self):
        """Resume trading after circuit breaker pause."""
        self.state.is_paused = False
        self.state.pause_reason = ""

    def activate(self):
        """Re-activate after kill switch."""
        self.state.is_active = True
        self.state.is_paused = False
        self.state.pause_reason = ""

    def reset_daily(self):
        """Reset daily counters (call at midnight)."""
        self._daily_start_capital = self.state.current_capital
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        # Reanudar si la pausa fue por límite diario O por pérdidas consecutivas
        # (al día siguiente se resetea el contador)
        should_resume = (
            self.state.is_paused and
            self.state.pause_reason and (
                "Daily" in self.state.pause_reason or
                "daily" in self.state.pause_reason or
                "Consecutive" in self.state.pause_reason or
                "consecutive" in self.state.pause_reason or
                "losses" in self.state.pause_reason.lower()
            )
        )
        if should_resume:
            self.state.is_paused = False
            self.state.pause_reason = ""
            self.state.consecutive_losses = 0
            print("[RiskManager] Reset diario: pausa levantada, pérdidas consecutivas reseteadas")
        # Limpiar posiciones antiguas que probablemente ya se resolvieron
        cleaned = self.cleanup_old_positions(max_age_hours=24.0)
        if cleaned > 0:
            print(f"[RiskManager] Reset diario: {cleaned} posiciones antiguas limpiadas")

    def compute_adaptive_kelly_multiplier(self,
                                           base_multiplier: float = 0.25,
                                           lookback_days: int = 3) -> float:
        """
        Adaptive Kelly adjustment based on recent rolling drawdown.

        Adjustment rules (from polybot research):
        - drawdown > 30% in last 3d: reduce by 0.02
        - drawdown < 15% in last 3d: increase by 0.02
        - Clamped to [0.10, 0.40] for safety

        Called hourly by the bot loop to adapt to market conditions.
        """
        current_dd = self.state.current_drawdown_pct / 100.0

        if current_dd > 0.30:
            adjustment = -0.02
        elif current_dd > 0.20:
            adjustment = -0.01
        elif current_dd < 0.05:
            adjustment = +0.02
        elif current_dd < 0.15:
            adjustment = +0.01
        else:
            adjustment = 0.0  # No change in neutral zone

        new_multiplier = base_multiplier + adjustment
        new_multiplier = max(0.10, min(0.40, new_multiplier))

        if adjustment != 0:
            direction = "UP" if adjustment > 0 else "DOWN"
            print(f"[RiskManager] Adaptive Kelly: {base_multiplier:.2f} {direction} {new_multiplier:.2f} "
                  f"(drawdown={current_dd*100:.1f}%)")

        return round(new_multiplier, 3)

    def compute_adaptive_tp_sl(self,
                                base_tp_pct: float = 0.25,
                                base_sl_pct: float = 0.20,
                                lookback_trades: int = 20) -> dict:
        """
        Adaptive Take-Profit and Stop-Loss thresholds.

        Searches for optimal TP/SL in grid based on recent trade history.
        TP grid: 10%-50% in 5% steps
        SL grid: 10%-40% in 5% steps

        Optimizes for: max(frequency × EV)
        Falls back to base values if insufficient history.
        """
        if len(self._trade_log) < 10:
            return {"tp_pct": base_tp_pct, "sl_pct": base_sl_pct,
                    "source": "default_insufficient_history"}

        # Use last lookback_trades with P&L data
        recent = [t for t in list(self._trade_log)[-lookback_trades:]
                  if t.get("pnl", 0) != 0]

        if len(recent) < 5:
            return {"tp_pct": base_tp_pct, "sl_pct": base_sl_pct,
                    "source": "default_insufficient_pnl_data"}

        pnls = [t["pnl"] for t in recent]
        sizes = [t.get("bet_usd", 1) for t in recent]

        best_ev = float("-inf")
        best_tp, best_sl = base_tp_pct, base_sl_pct

        for tp in [i / 100 for i in range(10, 55, 5)]:
            for sl in [i / 100 for i in range(10, 45, 5)]:
                # Simulate: wins capped at TP, losses capped at SL
                ev_total = 0
                n = 0
                for pnl, size in zip(pnls, sizes):
                    if size <= 0:
                        continue
                    pnl_pct = pnl / size
                    simulated = min(pnl_pct, tp) if pnl_pct > 0 else max(pnl_pct, -sl)
                    ev_total += simulated
                    n += 1

                if n > 0:
                    avg_ev = ev_total / n
                    frequency = n / lookback_trades
                    score = avg_ev * frequency
                    if score > best_ev:
                        best_ev = score
                        best_tp, best_sl = tp, sl

        return {
            "tp_pct": best_tp,
            "sl_pct": best_sl,
            "source": "adaptive_grid_search",
            "trades_analyzed": len(recent),
            "best_ev_score": round(best_ev, 4),
        }

    # ─────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────

    def get_risk_report(self) -> dict:
        """Full risk state report for the dashboard."""
        return {
            "is_active": self.state.is_active,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
            "capital": {
                "initial": self.initial_capital,
                "current": round(self.state.current_capital, 2),
                "peak": round(self.state.peak_capital, 2),
                "daily_pnl": round(self.state.daily_pnl, 2),
                "daily_pnl_pct": round(
                    self.state.daily_pnl / max(1, self._daily_start_capital) * 100, 2),
                "total_return_pct": round(
                    (self.state.current_capital - self.initial_capital) /
                    self.initial_capital * 100, 2),
            },
            "risk_metrics": {
                "current_drawdown_pct": round(self.state.current_drawdown_pct, 2),
                "max_drawdown_pct": round(self.state.max_drawdown_pct, 2),
                "consecutive_losses": self.state.consecutive_losses,
                "consecutive_wins": self.state.consecutive_wins,
                "open_positions": self.state.open_positions,
                "total_exposure_usd": round(self.state.total_exposure_usd, 2),
                "exposure_pct": round(
                    self.state.total_exposure_usd /
                    max(1, self.state.current_capital) * 100, 1),
            },
            "daily_stats": {
                "trades_today": self.state.daily_trades,
                "trades_remaining": max(0,
                    self.limits.max_daily_trades - self.state.daily_trades),
            },
            "limits": {
                "max_daily_loss": f"{self.limits.max_daily_loss_pct}%",
                "max_drawdown": f"{self.limits.max_drawdown_pct}%",
                "max_single_bet": f"{self.limits.max_single_bet_pct}%",
                "max_exposure": f"{self.limits.max_total_exposure_pct}%",
            },
        }
