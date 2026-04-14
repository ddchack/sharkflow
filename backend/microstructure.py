"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Microestructura de Mercado (OBI + VPIN)
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
- OBI: Order Book Imbalance — explica 65% de movimientos de precio
  (Cont, Kukanov & Stoikov 2014)
- VPIN: Volume-Synchronized Probability of Informed Trading
  (Easley, López de Prado & O'Hara 2012) — detecta smart money
- Trade Flow Toxicity — 3.88x mayor en DeFi que CeFi
"""
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class OBISignal:
    imbalance: float       # -1 (sell pressure) to +1 (buy pressure)
    multi_level: float     # Multi-level OBI (MLOFI)
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    strength: float        # 0-1
    bid_total: float
    ask_total: float


@dataclass
class VPINSignal:
    vpin: float            # 0-1, >0.7 = high informed trading
    toxicity: str          # "LOW", "MEDIUM", "HIGH", "EXTREME"
    smart_money_side: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float


class OrderBookImbalance:
    """
    OBI explains ~65% of short-term price movements (R²).
    Multi-level OBI (MLOFI) combines all book levels.
    __signature__ = "ddchack"
    """

    @staticmethod
    def compute_obi(bids: list[dict], asks: list[dict],
                     levels: int = 5) -> OBISignal:
        """
        Compute OBI from orderbook data.
        bids: [{"price": float, "size": float}, ...] best first
        asks: [{"price": float, "size": float}, ...] best first
        """
        if not bids or not asks:
            return OBISignal(0, 0, "NEUTRAL", 0, 0, 0)

        # Level 1 (best bid/ask)
        bid_vol = bids[0].get("size", 0)
        ask_vol = asks[0].get("size", 0)
        total = bid_vol + ask_vol
        obi_1 = (bid_vol - ask_vol) / total if total > 0 else 0

        # Multi-level OBI with exponential decay
        bid_total, ask_total = 0, 0
        for i, b in enumerate(bids[:levels]):
            weight = math.exp(-0.5 * i)  # Decay: closer levels matter more
            bid_total += b.get("size", 0) * weight
        for i, a in enumerate(asks[:levels]):
            weight = math.exp(-0.5 * i)
            ask_total += a.get("size", 0) * weight

        ml_total = bid_total + ask_total
        mlobi = (bid_total - ask_total) / ml_total if ml_total > 0 else 0

        if mlobi > 0.2:
            direction = "UP"
        elif mlobi < -0.2:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return OBISignal(
            imbalance=round(obi_1, 4),
            multi_level=round(mlobi, 4),
            predicted_direction=direction,
            strength=round(min(1.0, abs(mlobi)), 3),
            bid_total=round(bid_total, 2),
            ask_total=round(ask_total, 2))


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.
    Easley, López de Prado & O'Hara (2010-2012).
    VPIN > 0.7 = high informed trading probability.
    Predicted Flash Crash 2010 hours in advance.
    """

    @staticmethod
    def compute(trades: list[dict], bucket_size: float = 50.0,
                n_buckets: int = 50) -> VPINSignal:
        """
        trades: [{"price": float, "size": float, "side": "BUY"|"SELL"}, ...]
        bucket_size: volume per bucket
        """
        if not trades or len(trades) < 20:
            return VPINSignal(0.3, "LOW", "NEUTRAL", 0.1)

        # Classify trades and fill buckets
        buckets_buy = []
        buckets_sell = []
        current_buy, current_sell, current_vol = 0, 0, 0

        for t in trades:
            if not isinstance(t, dict):
                continue
            vol = float(t.get("size", 0) or 0)
            if vol <= 0:
                continue
            if t.get("side", "").upper() == "BUY":
                current_buy += vol
            else:
                current_sell += vol
            current_vol += vol

            if current_vol >= bucket_size:
                buckets_buy.append(current_buy)
                buckets_sell.append(current_sell)
                current_buy, current_sell, current_vol = 0, 0, 0

        if len(buckets_buy) < 5:
            return VPINSignal(0.3, "LOW", "NEUTRAL", 0.2)

        # VPIN = mean(|V_buy - V_sell|) / bucket_size over last n_buckets
        recent_b = buckets_buy[-n_buckets:]
        recent_s = buckets_sell[-n_buckets:]
        imbalances = [abs(b - s) for b, s in zip(recent_b, recent_s)]
        vpin = sum(imbalances) / (len(imbalances) * bucket_size)
        vpin = min(1.0, vpin)

        # Determine smart money direction
        total_buy = sum(recent_b)
        total_sell = sum(recent_s)
        if total_buy > total_sell * 1.3:
            sm_side = "BUY"
        elif total_sell > total_buy * 1.3:
            sm_side = "SELL"
        else:
            sm_side = "NEUTRAL"

        if vpin > 0.8:
            tox = "EXTREME"
        elif vpin > 0.6:
            tox = "HIGH"
        elif vpin > 0.4:
            tox = "MEDIUM"
        else:
            tox = "LOW"

        return VPINSignal(
            vpin=round(vpin, 4),
            toxicity=tox,
            smart_money_side=sm_side,
            confidence=round(min(0.9, len(buckets_buy) / 100), 3))

    @staticmethod
    def from_orderbook_snapshots(snapshots: list[dict]) -> VPINSignal:
        """
        Approximate VPIN from orderbook snapshots when trade data unavailable.
        Infer buy/sell pressure from bid/ask changes.
        """
        if len(snapshots) < 10:
            return VPINSignal(0.3, "LOW", "NEUTRAL", 0.1)

        buy_pressure, sell_pressure = 0, 0
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            bid_change = curr.get("best_bid", 0) - prev.get("best_bid", 0)
            ask_change = curr.get("best_ask", 0) - prev.get("best_ask", 0)
            if bid_change > 0:
                buy_pressure += abs(bid_change)
            if ask_change < 0:
                sell_pressure += abs(ask_change)

        total = buy_pressure + sell_pressure
        vpin_approx = abs(buy_pressure - sell_pressure) / total if total > 0 else 0.3
        sm = "BUY" if buy_pressure > sell_pressure * 1.3 else "SELL" if sell_pressure > buy_pressure * 1.3 else "NEUTRAL"

        return VPINSignal(
            vpin=round(min(1.0, vpin_approx), 4),
            toxicity="HIGH" if vpin_approx > 0.6 else "MEDIUM" if vpin_approx > 0.4 else "LOW",
            smart_money_side=sm,
            confidence=round(min(0.7, len(snapshots) / 50), 3))


class MicrostructureAnalyzer:
    """Combines OBI + VPIN into unified signal."""

    @staticmethod
    def analyze(bids: list, asks: list, trades: list = None,
                snapshots: list = None) -> dict:
        obi = OrderBookImbalance.compute_obi(bids, asks)

        if trades:
            vpin_sig = VPIN.compute(trades)
        elif snapshots:
            vpin_sig = VPIN.from_orderbook_snapshots(snapshots)
        else:
            vpin_sig = VPINSignal(0.3, "LOW", "NEUTRAL", 0.1)

        # Combined signal
        signals_agree = (obi.predicted_direction == "UP" and vpin_sig.smart_money_side == "BUY") or \
                        (obi.predicted_direction == "DOWN" and vpin_sig.smart_money_side == "SELL")

        return {
            "obi": {"imbalance": obi.imbalance, "multi_level": obi.multi_level,
                     "direction": obi.predicted_direction, "strength": obi.strength},
            "vpin": {"score": vpin_sig.vpin, "toxicity": vpin_sig.toxicity,
                     "smart_money": vpin_sig.smart_money_side, "confidence": vpin_sig.confidence},
            "combined": {
                "signals_agree": signals_agree,
                "confidence": round((obi.strength + vpin_sig.confidence) / 2, 3),
                "suggested_action": obi.predicted_direction if signals_agree else "CAUTELA",
            },
        }
