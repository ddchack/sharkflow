"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Real-Time WebSocket Client
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Connects to Polymarket CLOB WebSocket for real-time:
- Orderbook updates (book events)
- Price changes
- Last trade prices
- Tick size changes

Replaces REST polling with sub-second event streaming.
"""

import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass, field

import websockets


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10  # seconds


@dataclass
class OrderbookState:
    """Local orderbook state maintained from WebSocket updates."""
    token_id: str
    bids: list = field(default_factory=list)  # [[price, size], ...]
    asks: list = field(default_factory=list)
    midpoint: float = 0.0
    spread: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_trade_price: float = 0.0
    last_update: float = 0.0


class RealtimeClient:
    """
    WebSocket client for real-time Polymarket data.
    __author__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self):
        self.ws = None
        self.subscribed_assets: list[str] = []
        self.orderbooks: dict[str, OrderbookState] = {}
        self._running = False
        self._callbacks: dict[str, list[Callable]] = {
            "book": [],
            "price_change": [],
            "last_trade_price": [],
            "tick_size_change": [],
            "connected": [],
            "disconnected": [],
            "error": [],
        }
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    # ─────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable):
        """Register callback for an event type."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, data: dict):
        """Emit event to all registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    task = asyncio.create_task(cb(data))
                    # Capturar excepciones de tasks async para que no queden silenciosas
                    task.add_done_callback(
                        lambda t: t.exception() and
                        print(f"[WS] Async callback error for {event}: {t.exception()}")
                        if not t.cancelled() and t.exception() else None
                    )
                else:
                    cb(data)
            except Exception as e:
                print(f"[WS] Callback error for {event}: {e}")

    # ─────────────────────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────

    async def connect(self, asset_ids: list[str]):
        """Connect and subscribe to market data for given asset IDs."""
        self.subscribed_assets = asset_ids
        self._running = True

        for aid in asset_ids:
            if aid not in self.orderbooks:
                self.orderbooks[aid] = OrderbookState(token_id=aid)

        while self._running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    self.ws = ws
                    self._reconnect_delay = 1.0

                    # Subscribe
                    sub_msg = {
                        "assets_ids": self.subscribed_assets,
                        "type": "market",
                    }
                    await ws.send(json.dumps(sub_msg))
                    print(f"[WS] Connected. Subscribed to {len(self.subscribed_assets)} assets")
                    self._emit("connected", {"assets": len(self.subscribed_assets)})

                    # Listen loop
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            if message == "PONG":
                                continue

            except websockets.ConnectionClosed as e:
                print(f"[WS] Connection closed: {e}")
                self._emit("disconnected", {"reason": str(e)})
            except Exception as e:
                print(f"[WS] Error: {e}")
                self._emit("error", {"error": str(e)})

            if self._running:
                print(f"[WS] Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def disconnect(self):
        """Gracefully close connection."""
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def add_subscription(self, asset_id: str):
        """Add a new asset subscription to existing connection."""
        if asset_id not in self.subscribed_assets:
            self.subscribed_assets.append(asset_id)
            self.orderbooks[asset_id] = OrderbookState(token_id=asset_id)

            if self.ws and self.ws.open:
                sub_msg = {
                    "assets_ids": [asset_id],
                    "type": "market",
                }
                await self.ws.send(json.dumps(sub_msg))

    # ─────────────────────────────────────────────────────────
    # MESSAGE HANDLING
    # ─────────────────────────────────────────────────────────

    async def _handle_message(self, data: dict):
        """Route incoming WebSocket messages to handlers."""
        event_type = data.get("event_type", "")

        if event_type == "book":
            self._handle_book(data)
            self._emit("book", data)

        elif event_type == "price_change":
            self._handle_price_change(data)
            self._emit("price_change", data)

        elif event_type == "last_trade_price":
            self._handle_last_trade(data)
            self._emit("last_trade_price", data)

        elif event_type == "tick_size_change":
            self._emit("tick_size_change", data)

    def _handle_book(self, data: dict):
        """Update local orderbook from book snapshot."""
        asset_id = data.get("asset_id", "")
        if asset_id not in self.orderbooks:
            self.orderbooks[asset_id] = OrderbookState(token_id=asset_id)

        ob = self.orderbooks[asset_id]
        ob.bids = data.get("bids", [])
        ob.asks = data.get("asks", [])
        ob.last_update = time.time()

        # Calculate derived fields
        if ob.bids:
            ob.best_bid = float(ob.bids[0][0]) if isinstance(ob.bids[0], list) else float(ob.bids[0].get("price", 0))
        if ob.asks:
            ob.best_ask = float(ob.asks[0][0]) if isinstance(ob.asks[0], list) else float(ob.asks[0].get("price", 0))

        if ob.best_bid > 0 and ob.best_ask > 0:
            ob.midpoint = (ob.best_bid + ob.best_ask) / 2
            ob.spread = ob.best_ask - ob.best_bid

    def _handle_price_change(self, data: dict):
        """Handle price change events."""
        asset_id = data.get("asset_id", "")
        if asset_id in self.orderbooks:
            price = data.get("price")
            if price:
                self.orderbooks[asset_id].midpoint = float(price)
                self.orderbooks[asset_id].last_update = time.time()

    def _handle_last_trade(self, data: dict):
        """Handle last trade price events."""
        asset_id = data.get("asset_id", "")
        if asset_id in self.orderbooks:
            price = data.get("price")
            if price:
                self.orderbooks[asset_id].last_trade_price = float(price)

    # ─────────────────────────────────────────────────────────
    # DATA ACCESS
    # ─────────────────────────────────────────────────────────

    def get_orderbook(self, asset_id: str) -> Optional[OrderbookState]:
        """Get current orderbook state for an asset."""
        return self.orderbooks.get(asset_id)

    def get_midpoint(self, asset_id: str) -> float:
        """Get current midpoint price."""
        ob = self.orderbooks.get(asset_id)
        return ob.midpoint if ob else 0.0

    def get_spread(self, asset_id: str) -> float:
        """Get current bid-ask spread."""
        ob = self.orderbooks.get(asset_id)
        return ob.spread if ob else 0.0

    def get_all_prices(self) -> dict[str, float]:
        """Get all current midpoints."""
        return {
            aid: ob.midpoint
            for aid, ob in self.orderbooks.items()
            if ob.midpoint > 0
        }

    def get_stale_assets(self, max_age_seconds: float = 60) -> list[str]:
        """Find assets that haven't been updated recently."""
        now = time.time()
        return [
            aid for aid, ob in self.orderbooks.items()
            if now - ob.last_update > max_age_seconds
        ]
