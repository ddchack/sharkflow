"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Mean Reversion Engine
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
Research: 58% of Polymarket markets show negative autocorrelation.
Price spikes revert 30-50% within 72 hours. (Ng et al. 2025, 6058 markets)
Mean reversion is THE dominant pattern, not momentum.

Implements:
- Overreaction detection (>10pp moves → fade signal)
- Bollinger Bands in logit space (avoids [0,1] boundary issues)
- RSI adapted for binary contracts (reversal detector, not momentum)
- Autocorrelation analysis
- 72-hour correction estimator
"""
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class MeanReversionSignal:
    is_overreaction: bool
    direction: str          # "FADE_UP" (sell), "FADE_DOWN" (buy), "NEUTRAL"
    deviation_pct: float    # How far from mean
    expected_reversion_pct: float  # Expected 30-50% of move
    confidence: float
    bollinger_position: float  # 0=lower band, 0.5=middle, 1=upper
    rsi: float
    autocorrelation: float  # Negative = mean reverting


def logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


class MeanReversionEngine:
    """
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    @staticmethod
    def detect_overreaction(prices: list[float], window: int = 24) -> MeanReversionSignal:
        """
        Detect overreaction in price history.
        Research: moves >10pp in a day typically revert 30-50% within 72h.
        """
        if len(prices) < max(5, window):
            return MeanReversionSignal(False, "NEUTRAL", 0, 0, 0, 0.5, 50, 0)

        current = prices[-1]
        recent = prices[-min(window, len(prices)):]
        logit_prices = [logit(p) for p in recent]

        # Moving average in logit space
        ma = np.mean(logit_prices)
        std = np.std(logit_prices) if len(logit_prices) > 2 else 0.1

        current_logit = logit(current)
        z_score = (current_logit - ma) / max(0.01, std)

        # Bollinger position (0=lower, 1=upper)
        upper = ma + 2 * std
        lower = ma - 2 * std
        bb_range = upper - lower
        bb_pos = (current_logit - lower) / max(0.01, bb_range)
        bb_pos = max(0, min(1, bb_pos))

        # RSI in price space
        rsi = MeanReversionEngine._compute_rsi(prices, min(14, len(prices) - 1))

        # Autocorrelation (lag-1)
        if len(prices) > 5:
            returns = np.diff(prices)
            if len(returns) > 2:
                ac = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
            else:
                ac = 0
        else:
            ac = 0

        # 24h price change (porcentual relativo, no absoluto)
        if len(prices) > 1:
            _base = prices[max(0, len(prices) - 24)]
            change_24h = (current / _base - 1.0) * 100 if _base > 0 else 0.0
        else:
            change_24h = 0

        # Overreaction detection
        is_over = abs(z_score) > 1.5 or abs(change_24h) > 10
        if is_over and z_score > 1.5:
            direction = "FADE_UP"
            expected_rev = abs(change_24h) * 0.4  # 30-50% reversion
        elif is_over and z_score < -1.5:
            direction = "FADE_DOWN"
            expected_rev = abs(change_24h) * 0.4
        else:
            direction = "NEUTRAL"
            expected_rev = 0

        conf = min(0.9, abs(z_score) / 3.0) if is_over else 0.1

        return MeanReversionSignal(
            is_overreaction=is_over,
            direction=direction,
            deviation_pct=round(abs(z_score) * 100, 1) / 100,
            expected_reversion_pct=round(expected_rev, 1),
            confidence=round(conf, 3),
            bollinger_position=round(bb_pos, 3),
            rsi=round(rsi, 1),
            autocorrelation=round(ac, 4))

    @staticmethod
    def _compute_rsi(prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        changes = np.diff(prices[-period - 1:])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_signal(prices: list[float], window: int = 20,
                          num_std: float = 2.0) -> dict:
        """Bollinger Bands in logit space."""
        if len(prices) < window:
            return {"signal": "NEUTRAL", "squeeze": False, "width": 0}

        logit_p = [logit(p) for p in prices[-window:]]
        ma = np.mean(logit_p)
        std = np.std(logit_p)
        upper = ma + num_std * std
        lower = ma - num_std * std
        current = logit_p[-1]
        width = upper - lower

        # Squeeze detection (low volatility → expansion coming)
        historical_widths = []
        for i in range(max(0, len(prices) - window * 3), len(prices) - window + 1):
            lp = [logit(p) for p in prices[i:i + window]]
            historical_widths.append(np.std(lp) * 2 * num_std)
        squeeze = width < np.percentile(historical_widths, 20) if historical_widths else False

        if current > upper:
            signal = "OVERBOUGHT"  # Fade → sell
        elif current < lower:
            signal = "OVERSOLD"    # Fade → buy
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "squeeze": squeeze,
            "width": round(width, 4),
            "ma_price": round(inv_logit(ma), 4),
            "upper_price": round(inv_logit(upper), 4),
            "lower_price": round(inv_logit(lower), 4),
        }

    @staticmethod
    def fade_score(prices: list[float]) -> dict:
        """
        Combined mean reversion score: should you fade this market?
        Returns score 0-1 where 1 = strong fade opportunity.
        """
        sig = MeanReversionEngine.detect_overreaction(prices)
        bb = MeanReversionEngine.bollinger_signal(prices)

        score = 0.0
        reasons = []

        if sig.is_overreaction:
            score += 0.35
            reasons.append(f"Sobrerreacción detectada: z={sig.deviation_pct}")

        if sig.rsi > 80 or sig.rsi < 20:
            score += 0.25
            reasons.append(f"RSI extremo: {sig.rsi}")

        if bb["signal"] in ("OVERBOUGHT", "OVERSOLD"):
            score += 0.25
            reasons.append(f"Fuera de Bollinger: {bb['signal']}")

        if sig.autocorrelation < -0.2:
            score += 0.15
            reasons.append(f"Autocorrelación negativa: {sig.autocorrelation}")

        side = "COMPRAR" if sig.direction == "FADE_DOWN" or (bb["signal"] == "OVERSOLD") else "VENDER" if sig.direction == "FADE_UP" or (bb["signal"] == "OVERBOUGHT") else "ESPERAR"

        return {
            "fade_score": round(min(1.0, score), 3),
            "side": side,
            "reasons": reasons,
            "signal": sig.__dict__ if hasattr(sig, '__dict__') else {},
            "bollinger": bb,
        }
