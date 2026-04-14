"""
══════════════════════════════════════════════════════════════
Polymarket Bot - Market Scanner
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Scans Polymarket for opportunities using Gamma + CLOB APIs.
"""

import httpx
import asyncio
import json
import re as _re
from datetime import datetime, timezone
from typing import Optional
from math_engine import MathEngine, MarketOdds, BetRecommendation


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


class MarketScanner:
    """
    Fetches markets, orderbook data, and identifies trading opportunities.
    __author__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self, math_engine: MathEngine = None):
        self.math = math_engine or MathEngine()
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {}
        self._cache_ts = {}

    async def close(self):
        await self.client.aclose()

    # ─────────────────────────────────────────────────────────
    # GAMMA API - Market Discovery
    # ─────────────────────────────────────────────────────────

    async def fetch_active_markets(self, limit: int = 100, offset: int = 0,
                                    order: str = "volume24hr",
                                    ascending: bool = False,
                                    tag: str = None) -> list[dict]:
        """Fetch active markets from Gamma API sorted by volume."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag:
            params["tag_id"] = tag

        try:
            resp = await self.client.get(f"{GAMMA_API}/markets", params=params)
            resp.raise_for_status()
            markets = resp.json()
            return markets if isinstance(markets, list) else []
        except Exception as e:
            print(f"[Scanner] Error fetching markets: {e}")
            return []

    async def fetch_events(self, limit: int = 50, active: bool = True) -> list[dict]:
        """Fetch events (groups of related markets)."""
        params = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": "false",
        }
        try:
            resp = await self.client.get(f"{GAMMA_API}/events", params=params)
            resp.raise_for_status()
            return resp.json() if isinstance(resp.json(), list) else []
        except Exception as e:
            print(f"[Scanner] Error fetching events: {e}")
            return []

    async def fetch_exhaustive_groups(self, limit: int = 200) -> list[dict]:
        """
        Fetch exhaustive market groups via /events API.
        This finds ~400+ mutually-exclusive groups vs ~3 via group_slug.
        Essential for cross-event arbitrage detection.

        Each event contains multiple markets where exactly one will resolve YES.
        Example: "NHL Hart Trophy" has 118 candidate markets.
        """
        groups = []
        try:
            events = await self.fetch_events(limit=limit)
            for event in events:
                markets_in_event = event.get("markets", [])
                if not isinstance(markets_in_event, list) or len(markets_in_event) < 2:
                    continue

                # Build group entry with market prices
                market_prices = []
                for m in markets_in_event:
                    prices_str = m.get("outcomePrices", "[]")
                    try:
                        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                        yes_price = float(prices[0]) if prices else 0.5
                    except (json.JSONDecodeError, ValueError, IndexError):
                        yes_price = 0.5
                    market_prices.append({
                        "id": m.get("conditionId", m.get("id", "")),
                        "question": m.get("question", ""),
                        "yes_price": yes_price,
                    })

                if len(market_prices) < 2:
                    continue

                total_prob = sum(m["yes_price"] for m in market_prices)

                groups.append({
                    "event_slug": event.get("slug", ""),
                    "event_title": event.get("title", ""),
                    "market_count": len(market_prices),
                    "markets": market_prices,
                    "total_prob": round(total_prob, 4),
                    "has_arb_opportunity": total_prob < 0.97,  # Small buffer for fees
                    "arb_type": "buy_all_yes" if total_prob < 0.97 else (
                        "buy_all_no" if total_prob > 1.05 else "none"),
                })
        except Exception as e:
            print(f"[Scanner] Error fetching exhaustive groups: {e}")

        return groups

    async def scan_exhaustive_arbitrage(self, limit: int = 200) -> list[dict]:
        """
        Scan for arbitrage in exhaustive event groups.
        Returns groups where sum of YES prices < 1.0 (risk-free profit).
        """
        groups = await self.fetch_exhaustive_groups(limit=limit)
        opportunities = [g for g in groups if g.get("has_arb_opportunity", False)]
        opportunities.sort(key=lambda g: g["total_prob"])  # Lowest sum first = best arb
        return opportunities

    async def fetch_market_by_id(self, condition_id: str) -> Optional[dict]:
        """Fetch a single market by condition ID."""
        try:
            resp = await self.client.get(f"{GAMMA_API}/markets/{condition_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Scanner] Error fetching market {condition_id}: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # CLOB API - Orderbook & Pricing
    # ─────────────────────────────────────────────────────────

    async def fetch_orderbook(self, token_id: str) -> Optional[dict]:
        """Fetch orderbook for a token."""
        try:
            resp = await self.client.get(f"{CLOB_API}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Scanner] Error fetching orderbook: {e}")
            return None

    async def fetch_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        try:
            resp = await self.client.get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception as e:
            print(f"[Scanner] Error fetching midpoint: {e}")
            return None

    async def fetch_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a token on a given side."""
        try:
            resp = await self.client.get(
                f"{CLOB_API}/price",
                params={"token_id": token_id, "side": side}
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            print(f"[Scanner] Error fetching price: {e}")
            return None

    async def fetch_spread(self, token_id: str) -> Optional[dict]:
        """Get spread for a token."""
        try:
            resp = await self.client.get(f"{CLOB_API}/spread", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return None

    async def fetch_price_history(self, token_id: str, interval: str = "1d") -> list:
        """Fetch price history. interval: 1h, 6h, 1d, 1w, 1m"""
        try:
            resp = await self.client.get(
                f"{CLOB_API}/prices-history",
                params={"market": token_id, "interval": interval}
            )
            resp.raise_for_status()
            return resp.json().get("history", [])
        except Exception as e:
            print(f"[Scanner] Error fetching price history: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # MARKET ANALYSIS
    # ─────────────────────────────────────────────────────────

    def parse_market_odds(self, market: dict) -> Optional[MarketOdds]:
        """Extract odds structure from a Gamma market object."""
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

            # Clamp to valid range
            yes_price = max(0.01, min(0.99, yes_price))
            no_price = max(0.01, min(0.99, no_price))

            volume = float(market.get("volume24hr", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)
            spread = float(market.get("spread", 0) or 0)

            return MarketOdds(
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=volume,
                liquidity=liquidity,
                spread=spread
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None

    def extract_token_ids(self, market: dict) -> tuple[str, str]:
        """Extract YES and NO token IDs from market data."""
        tokens_str = market.get("clobTokenIds", "")
        if isinstance(tokens_str, str):
            try:
                tokens = json.loads(tokens_str) if tokens_str else []
            except json.JSONDecodeError:
                tokens = []
        else:
            tokens = tokens_str
        yes_token = tokens[0] if len(tokens) > 0 else ""
        no_token = tokens[1] if len(tokens) > 1 else ""
        return yes_token, no_token

    async def scan_opportunities(self, sentiment_scores: dict = None,
                                  limit: int = 50,
                                  available_capital: float = None) -> list[dict]:
        """
        Full scan pipeline:
        1. Fetch top markets by volume
        2. Analyze odds and calculate metrics
        3. Filter by edge/EV thresholds
        4. Return sorted recommendations
        """
        sentiments = sentiment_scores or {}
        markets = await self.fetch_active_markets(limit=limit)
        recommendations = []

        for mkt in markets:
            odds = self.parse_market_odds(mkt)
            if not odds:
                continue

            question = mkt.get("question", "Unknown")
            condition_id = mkt.get("conditionId", mkt.get("id", ""))
            slug = mkt.get("slug", "")

            # Get sentiment score para este mercado usando word boundaries
            # (evita falsos positivos: "bit" no debe matchear "bitcoin")
            sentiment = 0.0
            _q_lower = question.lower()
            for keyword, score in sentiments.items():
                _pattern = r'\b' + _re.escape(keyword.lower()) + r'\b'
                if _re.search(_pattern, _q_lower):
                    sentiment = score
                    break

            # For now, estimate true prob as a slight adjustment from market price
            # In production, this would come from your model or news analysis
            # The sentiment shifts our estimate slightly
            market_yes_prob = odds.implied_yes_prob
            adjustment = sentiment * 0.05  # Sentiment can shift up to 5%
            estimated_true_prob = max(0.05, min(0.95, market_yes_prob + adjustment))

            rec = self.math.evaluate_market(
                market_id=condition_id,
                question=question,
                odds=odds,
                estimated_true_prob=estimated_true_prob,
                sentiment_score=sentiment,
                available_capital=available_capital
            )

            if rec:
                yes_token, no_token = self.extract_token_ids(mkt)
                rec_dict = {
                    "market_id": rec.market_id,
                    "question": rec.market_question,
                    "slug": slug,
                    "side": rec.side,
                    "market_price": rec.market_price,
                    "estimated_prob": rec.estimated_true_prob,
                    "edge_pct": rec.edge_percent,
                    "ev": rec.expected_value,
                    "kelly": rec.kelly_fraction,
                    "bet_usd": rec.recommended_bet_usd,
                    "confidence": rec.confidence_score,
                    "risk": rec.risk_level,
                    "reasoning": rec.reasoning,
                    "yes_token_id": yes_token,
                    "no_token_id": no_token,
                    "volume_24h": odds.volume_24h,
                    "liquidity": odds.liquidity,
                    "outcomes": json.loads(mkt.get("outcomes", "[]")) if isinstance(mkt.get("outcomes"), str) else mkt.get("outcomes", []),
                    "end_date": mkt.get("endDate", ""),
                    "image": mkt.get("image", ""),
                }
                recommendations.append(rec_dict)

        # Sort by confidence score descending
        recommendations.sort(key=lambda r: r["confidence"], reverse=True)
        return recommendations

    async def get_market_detail(self, market: dict) -> dict:
        """Get enriched market data with orderbook info."""
        yes_token, no_token = self.extract_token_ids(market)
        detail = {
            "market": market,
            "yes_orderbook": None,
            "no_orderbook": None,
            "yes_midpoint": None,
        }

        if yes_token:
            detail["yes_orderbook"] = await self.fetch_orderbook(yes_token)
            detail["yes_midpoint"] = await self.fetch_midpoint(yes_token)

        return detail
