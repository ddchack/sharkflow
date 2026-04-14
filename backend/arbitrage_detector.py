"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Arbitrage Detector
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Detects risk-free (or near risk-free) profit opportunities:

1. Intra-Market: YES + NO < $1.00 → buy both = guaranteed profit
2. Cross-Event: Mutually exclusive outcomes summing to != $1.00
3. Stale Price: Market price lagging behind confirmed external data

Polymarket fee: 2% on winning outcome profits.
Minimum profitable spread after fees: ~2.5-3%
"""

import json
import httpx
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    arb_type: str          # "INTRA_MARKET", "CROSS_EVENT", "STALE_PRICE"
    market_id: str
    question: str
    yes_price: float
    no_price: float
    combined_cost: float   # YES + NO price
    gross_profit: float    # $1.00 - combined_cost
    net_profit: float      # After 2% fee
    net_profit_pct: float
    fee_estimate: float
    recommended_size: float
    confidence: str        # "HIGH", "MEDIUM", "LOW"
    tokens: dict           # {"yes_token": "...", "no_token": "..."}
    details: str
    detected_at: str


class ArbitrageDetector:
    """
    Scans Polymarket for arbitrage opportunities.
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    POLYMARKET_FEE = 0.02  # 2% on profitable outcomes
    MIN_NET_PROFIT_PCT = 0.001  # 0.1% minimum net profit to flag

    def __init__(self, max_capital: float = 100.0):
        self.max_capital = max_capital
        self.client = httpx.AsyncClient(timeout=20.0)
        self._opportunities: list[ArbitrageOpportunity] = []

    async def close(self):
        await self.client.aclose()

    # ─────────────────────────────────────────────────────────
    # INTRA-MARKET ARBITRAGE
    # ─────────────────────────────────────────────────────────

    async def scan_intra_market(self, markets: list[dict] = None,
                                 limit: int = 200) -> list[ArbitrageOpportunity]:
        """
        Scan for markets where YES + NO < $1.00.
        
        If you can buy YES@0.45 and NO@0.52 = $0.97 total,
        one must pay out $1.00, so profit = $0.03 - fees.
        
        Fee: 2% on the winning side's profit.
        - Buy YES@0.45, NO@0.52. Total cost: $0.97
        - If YES wins: get $1.00, profit on YES = $0.55
          Fee = 0.55 * 0.02 = $0.011. Net = $0.03 - $0.011 = $0.019
        - If NO wins: get $1.00, profit on NO = $0.48
          Fee = 0.48 * 0.02 = $0.0096. Net = $0.03 - $0.0096 = $0.020
        """
        if not markets:
            markets = await self._fetch_active_markets(limit)

        opportunities = []

        for mkt in markets:
            arb = self._check_intra_market(mkt)
            if arb:
                opportunities.append(arb)

        # Sort by net profit descending
        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)
        self._opportunities = opportunities
        return opportunities

    def _check_intra_market(self, market: dict) -> Optional[ArbitrageOpportunity]:
        """Check a single market for intra-market arbitrage."""
        try:
            prices_str = market.get("outcomePrices", "")
            if isinstance(prices_str, str):
                prices = json.loads(prices_str) if prices_str else []
            else:
                prices = prices_str

            if len(prices) < 2:
                return None

            yes_price = float(prices[0])
            no_price = float(prices[1])

            # Skip invalid prices
            if yes_price <= 0.01 or no_price <= 0.01:
                return None
            if yes_price >= 0.99 or no_price >= 0.99:
                return None

            combined = yes_price + no_price
            gross_profit = 1.0 - combined

            if gross_profit <= 0:
                return None  # No arb (prices sum to >= $1)

            # Calculate fee: 2% on the winning side's profit
            # Worst case fee (highest fee scenario)
            yes_win_profit = 1.0 - yes_price  # Profit if YES wins
            no_win_profit = 1.0 - no_price    # Profit if NO wins
            max_fee = max(yes_win_profit, no_win_profit) * self.POLYMARKET_FEE

            net_profit = gross_profit - max_fee
            if net_profit <= 0:
                return None

            net_profit_pct = net_profit / combined

            if net_profit_pct < self.MIN_NET_PROFIT_PCT:
                return None

            # Extract tokens
            tokens_str = market.get("clobTokenIds", "")
            if isinstance(tokens_str, str):
                tokens = json.loads(tokens_str) if tokens_str else []
            else:
                tokens = tokens_str

            # Recommended size: buy as many pairs as capital allows
            pairs_affordable = min(
                self.max_capital * 0.5 / combined,  # Max 50% of capital
                100  # Max 100 pairs
            )
            recommended_size = round(pairs_affordable * combined, 2)

            # Confidence based on spread magnitude
            if net_profit_pct > 0.02:
                confidence = "HIGH"
            elif net_profit_pct > 0.01:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            return ArbitrageOpportunity(
                arb_type="INTRA_MARKET",
                market_id=market.get("conditionId", market.get("id", "")),
                question=market.get("question", ""),
                yes_price=yes_price,
                no_price=no_price,
                combined_cost=round(combined, 4),
                gross_profit=round(gross_profit, 4),
                net_profit=round(net_profit, 4),
                net_profit_pct=round(net_profit_pct * 100, 3),
                fee_estimate=round(max_fee, 4),
                recommended_size=recommended_size,
                confidence=confidence,
                tokens={
                    "yes_token": tokens[0] if len(tokens) > 0 else "",
                    "no_token": tokens[1] if len(tokens) > 1 else "",
                },
                details=f"Buy YES@{yes_price:.2f} + NO@{no_price:.2f} = ${combined:.4f}. "
                        f"Guaranteed payout $1.00. Net profit: ${net_profit:.4f} ({net_profit_pct*100:.2f}%)",
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    # ─────────────────────────────────────────────────────────
    # CROSS-EVENT ARBITRAGE
    # ─────────────────────────────────────────────────────────

    async def scan_cross_event(self, limit: int = 50) -> list[ArbitrageOpportunity]:
        """
        Scan events where mutually exclusive markets don't sum to $1.00.
        
        Example: "How many Fed rate cuts?" with options 0,1,2,...,8+
        If sum of all YES prices < $1.00, buy all = guaranteed profit.
        """
        opportunities = []

        try:
            resp = await self.client.get(
                f"{GAMMA_API}/events",
                params={"limit": limit, "active": "true", "closed": "false"}
            )
            resp.raise_for_status()
            events = resp.json()

            for event in events:
                markets = event.get("markets", [])
                if len(markets) < 2:
                    continue

                arb = self._check_cross_event(event, markets)
                if arb:
                    opportunities.append(arb)

        except Exception as e:
            print(f"[Arb] Error scanning cross-event: {e}")

        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)
        return opportunities

    def _check_cross_event(self, event: dict,
                            markets: list[dict]) -> Optional[ArbitrageOpportunity]:
        """Check if an event's mutually exclusive markets sum to != $1.00."""
        try:
            total_yes = 0.0
            valid_markets = 0

            for mkt in markets:
                prices_str = mkt.get("outcomePrices", "")
                if isinstance(prices_str, str):
                    prices = json.loads(prices_str) if prices_str else []
                else:
                    prices = prices_str

                if len(prices) >= 1:
                    yes_price = float(prices[0])
                    if 0.01 < yes_price < 0.99:
                        total_yes += yes_price
                        valid_markets += 1

            if valid_markets < 2:
                return None

            # For mutually exclusive outcomes, sum should be ~$1.00
            deviation = 1.0 - total_yes
            gross_profit = abs(deviation) if total_yes < 1.0 else 0

            if gross_profit < 0.01:
                return None

            # Estimate fees
            avg_price = total_yes / valid_markets
            fee = (1.0 - avg_price) * self.POLYMARKET_FEE
            net_profit = gross_profit - fee

            if net_profit <= 0:
                return None

            return ArbitrageOpportunity(
                arb_type="CROSS_EVENT",
                market_id=str(event.get("id", "")),
                question=event.get("title", event.get("slug", "")),
                yes_price=total_yes,
                no_price=0,
                combined_cost=round(total_yes, 4),
                gross_profit=round(gross_profit, 4),
                net_profit=round(net_profit, 4),
                net_profit_pct=round((net_profit / total_yes) * 100, 3) if total_yes > 0 else 0,
                fee_estimate=round(fee, 4),
                recommended_size=round(min(self.max_capital * 0.3, 50), 2),
                confidence="MEDIUM",
                tokens={},
                details=f"{valid_markets} mutually exclusive markets sum to ${total_yes:.4f}. "
                        f"Should be $1.00. Gap: ${gross_profit:.4f}",
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    async def _fetch_active_markets(self, limit: int = 200) -> list[dict]:
        """Fetch active markets from Gamma API."""
        try:
            resp = await self.client.get(
                f"{GAMMA_API}/markets",
                params={
                    "limit": limit,
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                }
            )
            resp.raise_for_status()
            return resp.json() if isinstance(resp.json(), list) else []
        except Exception as e:
            print(f"[Arb] Error fetching markets: {e}")
            return []

    async def full_scan(self) -> dict:
        """Run all arbitrage detection strategies."""
        markets = await self._fetch_active_markets(200)

        intra = await self.scan_intra_market(markets)
        cross = await self.scan_cross_event()

        return {
            "intra_market": [
                {
                    "type": o.arb_type,
                    "question": o.question[:80],
                    "yes_price": o.yes_price,
                    "no_price": o.no_price,
                    "combined": o.combined_cost,
                    "net_profit": o.net_profit,
                    "net_profit_pct": o.net_profit_pct,
                    "confidence": o.confidence,
                    "size_usd": o.recommended_size,
                    "tokens": o.tokens,
                    "details": o.details,
                } for o in intra
            ],
            "cross_event": [
                {
                    "type": o.arb_type,
                    "question": o.question[:80],
                    "total_yes": o.combined_cost,
                    "net_profit": o.net_profit,
                    "net_profit_pct": o.net_profit_pct,
                    "confidence": o.confidence,
                    "details": o.details,
                } for o in cross
            ],
            "total_opportunities": len(intra) + len(cross),
            "scanned_markets": len(markets),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
