"""
SharkFlow Dump & Hedge — Feeds WebSocket en Tiempo Real
========================================================
Conexión a Binance WebSocket para precios de BTC/ETH/SOL/XRP en tiempo real.
Detecta movimientos bruscos que preceden a dumps en Polymarket (lag ~55s).
Integra con DumpHedgeEngine para señalizar oportunidades automáticamente.

Desarrollado por: Carlos David Donoso Cordero (ddchack)
"""
# <!-- author: Carlos David Donoso Cordero (ddchack) | module: dump_hedge_ws | version: 1.0 -->

import asyncio
import json
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# URLs de WebSocket de Binance
BINANCE_WS_MULTI = (
    "wss://stream.binance.com:9443/stream?streams="
    "btcusdt@aggTrade/ethusdt@aggTrade/solusdt@aggTrade/xrpusdt@aggTrade"
)

# Mapeo símbolo Binance → asset SharkFlow
SYMBOL_MAP = {
    "btcusdt": "btc",
    "ethusdt": "eth",
    "solusdt": "sol",
    "xrpusdt": "xrp",
}


class BinancePriceFeed:
    """
    Feed de precios en tiempo real desde Binance WebSocket.

    Suscribe a aggTrade de BTC, ETH, SOL y XRP simultáneamente.
    Emite callbacks cuando llega un nuevo precio.
    Incluye reconexión automática con backoff exponencial.

    Desarrollado por: Carlos David Donoso Cordero (ddchack)
    """

    def __init__(self):
        self._prices: dict[str, float] = {}   # asset -> último precio
        self._prev:   dict[str, float] = {}   # asset -> precio anterior ciclo
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_price: Optional[Callable] = None  # callback(asset, price, pct)

    @property
    def prices(self) -> dict:
        return self._prices.copy()

    def get_price(self, asset: str) -> Optional[float]:
        return self._prices.get(asset.lower())

    def get_change_pct(self, asset: str) -> Optional[float]:
        """% cambio desde el ciclo anterior."""
        a = asset.lower()
        prev = self._prev.get(a, 0)
        curr = self._prices.get(a, 0)
        if not prev or not curr:
            return None
        return (curr - prev) / prev * 100.0

    def start(self, on_price: Optional[Callable] = None):
        """Inicia la conexión WebSocket a Binance."""
        if self._running:
            return
        self._running  = True
        self._on_price = on_price
        self._task = asyncio.create_task(self._connect_loop())

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _connect_loop(self):
        """Mantiene la conexión con reconexión automática."""
        retry = 0
        while self._running:
            try:
                # Importar websockets de forma lazy para no romper si no está instalado
                try:
                    import websockets
                except ImportError:
                    logger.warning("[D&H WS] websockets no instalado — feed Binance deshabilitado")
                    await asyncio.sleep(3600)
                    continue

                async with websockets.connect(
                    BINANCE_WS_MULTI,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("[D&H WS] Binance conectado")
                    retry = 0
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg  = json.loads(raw)
                            data = msg.get("data", msg)
                            sym  = (data.get("s") or "").lower()
                            asset = SYMBOL_MAP.get(sym)
                            if not asset:
                                continue

                            price = float(data.get("p", 0))  # aggTrade price
                            if price <= 0:
                                continue

                            # Solo actualizar _prev si ya teníamos un precio previo
                            # (evitar change_pct=0 falso en la primera lectura)
                            if asset in self._prices:
                                self._prev[asset] = self._prices[asset]
                            self._prices[asset] = price

                            if self._on_price:
                                pct = self.get_change_pct(asset) or 0.0
                                # Callback asíncrono si procede
                                if asyncio.iscoroutinefunction(self._on_price):
                                    asyncio.create_task(
                                        self._on_price(asset, price, pct)
                                    )
                                else:
                                    self._on_price(asset, price, pct)

                        except Exception as e:
                            logger.debug(f"[D&H WS] parse error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    delay = min(2 ** retry, 60)
                    logger.warning(
                        f"[D&H WS] Binance desconectado: {e}. Reconectando en {delay}s"
                    )
                    await asyncio.sleep(delay)
                    retry += 1

        logger.info("[D&H WS] Binance feed detenido")


class DumpDetector:
    """
    Detecta dumps/pumps en tiempo real comparando precios Binance
    vs precios del orderbook de Polymarket.

    El CLOB de Polymarket tarda ~55 segundos en ajustar cuando BTC
    se mueve bruscamente, creando la ventana de oportunidad D&H.

    Desarrollado por: Carlos David Donoso Cordero (ddchack)
    """

    def __init__(
        self,
        dump_threshold_pct: float = 0.5,
        signal_cooldown_s: float = 30.0,
    ):
        """
        dump_threshold_pct: % mínimo de movimiento para señalizar.
            0.5% es conservador y ya supera el lag de ajuste de PM.
        signal_cooldown_s: segundos mínimos entre señales del mismo activo.
        """
        self.threshold  = dump_threshold_pct
        self.cooldown   = signal_cooldown_s
        self._feed      = BinancePriceFeed()
        self._on_dump:  Optional[Callable] = None
        self._last_sig: dict[str, float] = {}  # asset -> ts última señal

    @property
    def prices(self) -> dict:
        return self._feed.prices

    def get_price(self, asset: str) -> Optional[float]:
        return self._feed.get_price(asset)

    def start(self, on_dump: Optional[Callable] = None):
        """Inicia el feed y la detección de dumps."""
        self._on_dump = on_dump
        self._feed.start(on_price=self._on_price_update)

    def stop(self):
        self._feed.stop()

    async def _on_price_update(self, asset: str, price: float, pct: float):
        """Callback: se llama en cada nuevo precio de Binance."""
        if abs(pct) < self.threshold:
            return

        now  = time.time()
        last = self._last_sig.get(asset, 0)
        if now - last < self.cooldown:
            return  # En cooldown para este activo

        self._last_sig[asset] = now
        direction = "DOWN" if pct < 0 else "UP"

        logger.info(
            f"[D&H Detector] ⚡ {asset.upper()} movimiento brusco: "
            f"{pct:+.3f}% @ ${price:,.2f} → señal {direction}"
        )

        if self._on_dump:
            if asyncio.iscoroutinefunction(self._on_dump):
                asyncio.create_task(
                    self._on_dump(
                        asset=asset,
                        price=price,
                        pct_change=pct,
                        direction=direction,
                    )
                )
            else:
                self._on_dump(
                    asset=asset,
                    price=price,
                    pct_change=pct,
                    direction=direction,
                )
