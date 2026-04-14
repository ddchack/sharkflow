"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Whale Tracker
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Tracks large wallet (whale) activity using Polymarket Data API.
Generates whale consensus signals for the Bayesian engine.

Data API endpoints used:
- GET /activity?user={address} — recent trades
- GET /positions?user={address} — current positions
"""

import httpx
import asyncio
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

DATA_API = "https://data-api.polymarket.com"


@dataclass
class WhaleProfile:
    address: str
    alias: str
    win_rate: float
    total_volume: float
    total_pnl: float
    active_positions: int
    last_trade: str
    trust_score: float  # 0-1 based on track record


@dataclass
class WhaleMove:
    whale_address: str
    whale_alias: str
    market_id: str
    question: str
    side: str  # "YES" or "NO"
    size_usd: float
    price: float
    timestamp: str
    trust_score: float


class WhaleTracker:
    """
    Monitors whale wallet activity for copy-trading signals.
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    # Known profitable wallets (placeholder — user configures these)
    DEFAULT_WHALES = [
        # These are example addresses; user should add real ones
        # from Polywhaler, PolyTrack leaderboards, etc.
    ]

    def __init__(self, whale_addresses: list[str] = None,
                 min_trade_usd: float = 500.0):
        self.whales = whale_addresses or self.DEFAULT_WHALES
        self.min_trade_usd = min_trade_usd
        self.client = httpx.AsyncClient(timeout=20.0)
        self._profiles: dict[str, WhaleProfile] = {}
        self._recent_moves: list[WhaleMove] = []
        self._last_seen: dict[str, str] = {}  # address -> last trade ID

    async def close(self):
        await self.client.aclose()

    def add_whale(self, address: str, alias: str = ""):
        """Add a whale wallet to track."""
        if address not in self.whales:
            self.whales.append(address)
            self._profiles[address] = WhaleProfile(
                address=address, alias=alias or address[:8],
                win_rate=0, total_volume=0, total_pnl=0,
                active_positions=0, last_trade="", trust_score=0.5,
            )

    def remove_whale(self, address: str):
        """Remove a whale from tracking."""
        if address in self.whales:
            self.whales.remove(address)
            self._profiles.pop(address, None)

    # ─────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────

    async def fetch_whale_activity(self, address: str,
                                    limit: int = 20) -> list[dict]:
        """Fetch recent trades for a wallet via Data API."""
        try:
            resp = await self.client.get(
                f"{DATA_API}/activity",
                params={
                    "user": address,
                    "limit": limit,
                    "sortBy": "TIMESTAMP",
                    "sortOrder": "DESC",
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            print(f"[Whale] Error fetching activity for {address[:10]}: {e}")
            return []

    async def fetch_whale_positions(self, address: str) -> list[dict]:
        """Fetch current positions for a wallet."""
        try:
            resp = await self.client.get(
                f"{DATA_API}/positions",
                params={
                    "user": address,
                    "sizeThreshold": 10,  # Min $10 positions
                    "limit": 50,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            print(f"[Whale] Error fetching positions for {address[:10]}: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────

    async def _scan_single_whale(self, address: str,
                                   max_age_days: int = 7) -> tuple:
        """Escanea una wallet y retorna (movimientos, stats_diagnostico)."""
        import time as _time
        moves = []
        stats = {"fetched": 0, "skip_type": 0, "skip_age": 0, "skip_size": 0, "skip_side": 0}

        activity = await self.fetch_whale_activity(address, limit=50)
        if not activity:
            return moves, stats

        stats["fetched"] = len(activity)
        cutoff_ts = _time.time() - max_age_days * 86400
        last_seen_tx = self._last_seen.get(address, "")

        profile = self._profiles.get(address)
        trust = max(0.0, min(1.0, profile.trust_score)) if profile else 0.5
        alias = profile.alias if profile else address[:8]

        for trade in activity:
            tx_hash = trade.get("transactionHash", "")
            if tx_hash and tx_hash == last_seen_tx:
                break

            # Solo procesar TRADE (no REDEEM ni MERGE)
            t_type = trade.get("type", trade.get("action", "")).upper()
            if t_type and t_type != "TRADE":
                stats["skip_type"] += 1
                continue

            # Filtrar por antigüedad
            ts = trade.get("timestamp", 0)
            try:
                ts_float = float(ts) if ts else 0.0
                # Timestamps en milisegundos → convertir a segundos
                if ts_float > 1e12:
                    ts_float /= 1000.0
            except (TypeError, ValueError):
                ts_float = 0.0
            if ts_float and ts_float < cutoff_ts:
                stats["skip_age"] += 1
                continue

            size = float(trade.get("usdcSize", 0) or trade.get("size", 0) or 0)
            if size < self.min_trade_usd:
                stats["skip_size"] += 1
                continue

            side = trade.get("side", trade.get("outcome", ""))
            if not side:
                stats["skip_side"] += 1
                continue

            side_upper = side.upper()
            if side_upper in ("BUY", "YES", "LONG"):
                side_norm = "YES"
            elif side_upper in ("SELL", "NO", "SHORT"):
                side_norm = "NO"
            else:
                stats["skip_side"] += 1
                continue

            moves.append(WhaleMove(
                whale_address=address,
                whale_alias=alias,
                market_id=trade.get("conditionId", ""),
                question=trade.get("title", trade.get("question", ""))[:100],
                side=side_norm,
                size_usd=size,
                price=float(trade.get("price", 0) or 0),
                timestamp=str(int(ts_float)) if ts_float else "",
                trust_score=trust,
            ))

        first_tx = activity[0].get("transactionHash", "")
        if first_tx:
            self._last_seen[address] = first_tx

        return moves, stats

    async def scan_whale_moves(self, max_age_days: int = 7) -> tuple:
        """Escanea todas las wallets en paralelo. Retorna (moves, aggregate_stats)."""
        results = await asyncio.gather(
            *[self._scan_single_whale(addr, max_age_days=max_age_days) for addr in self.whales],
            return_exceptions=True,
        )
        new_moves = []
        agg = {"fetched": 0, "skip_type": 0, "skip_age": 0, "skip_size": 0, "skip_side": 0}
        for r in results:
            if isinstance(r, tuple):
                moves, stats = r
                new_moves.extend(moves)
                for k in agg:
                    agg[k] += stats.get(k, 0)

        new_moves.sort(key=lambda m: m.size_usd, reverse=True)
        self._recent_moves = new_moves
        return new_moves, agg

    def get_whale_consensus(self, market_id: str) -> list[dict]:
        """
        Get whale consensus for a specific market.
        Returns list of whale positions for Bayesian engine.
        """
        positions = []
        for move in self._recent_moves:
            if move.market_id == market_id:
                positions.append({
                    "side": move.side,
                    "size_usd": move.size_usd,
                    "trust_score": move.trust_score,
                    "whale": move.whale_alias,
                })
        return positions

    async def get_market_whale_positions(self, market_id: str) -> list[dict]:
        """
        Get all whale positions for a specific market by checking each wallet.
        More thorough but slower than scan_whale_moves.
        """
        positions = []
        for address in self.whales:
            wallet_positions = await self.fetch_whale_positions(address)
            for pos in wallet_positions:
                pos_market = pos.get("conditionId", pos.get("market", ""))
                if pos_market == market_id:
                    size = float(pos.get("size", 0) or 0)
                    side = pos.get("outcome", pos.get("side", "YES"))
                    if size > 0:
                        profile = self._profiles.get(address)
                        positions.append({
                            "side": side.upper(),
                            "size_usd": size,
                            "trust_score": profile.trust_score if profile else 0.5,
                            "whale": profile.alias if profile else address[:8],
                        })
        return positions

    # ─────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────

    def get_recent_moves_report(self, limit: int = 20) -> list[dict]:
        """Get recent whale moves for dashboard display."""
        return [
            {
                "whale": m.whale_alias,
                "market": m.question[:60],
                "side": m.side,
                "size": m.size_usd,
                "price": m.price,
                "trust": m.trust_score,
                "time": m.timestamp,
            }
            for m in self._recent_moves[:limit]
        ]

    def get_tracked_whales(self) -> list[dict]:
        """Get list of tracked whale wallets."""
        return [
            {
                "address": addr,
                "alias": self._profiles.get(addr, WhaleProfile(
                    addr, addr[:8], 0, 0, 0, 0, "", 0.5)).alias,
                "trust_score": self._profiles.get(addr, WhaleProfile(
                    addr, addr[:8], 0, 0, 0, 0, "", 0.5)).trust_score,
            }
            for addr in self.whales
        ]
