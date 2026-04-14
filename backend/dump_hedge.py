"""
SharkFlow Dump & Hedge Engine — Arbitraje Temporal de Mercados Crypto
=====================================================================
Estrategia: detecta cuando un lado (UP/DOWN) de un mercado BTC/ETH/SOL/XRP
de 5 o 15 minutos colapsa en precio por movimiento brusco.
Compra ese lado barato (Pierna 1), luego espera que el lado opuesto
también caiga por la reversión. Si UP+DOWN < 1.0, cualquier pago
supera la inversión total: GANANCIA GARANTIZADA pase lo que pase.

Piramidación: una vez asegurada la ganancia base, reinvierte en la
siguiente oscilación dentro de la misma ventana para amplificar el retorno.

Desarrollado por: Carlos David Donoso Cordero (ddchack)
"""
# <!-- author: Carlos David Donoso Cordero (ddchack) | module: dump_hedge | version: 1.0 -->

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
import httpx

logger = logging.getLogger(__name__)

# ── Configuración por defecto ────────────────────────────────────────
DH_DEFAULT_CONFIG: dict = {
    "assets": ["btc", "eth", "sol", "xrp"],
    "timeframes": ["5m", "15m"],
    "dump_threshold": 0.15,       # Comprar cuando precio ≤ 15¢
    "max_combined_price": 0.95,   # Solo hedge si UP+DOWN ≤ 0.95
    "bet_per_leg": 1.0,           # USD por pierna (monto base para dutching)
    "max_legs_per_window": 2,     # Máx piernas por ventana de tiempo
    "max_pyramid_layers": 3,      # Máx capas de piramidación
    "leg2_wait_timeout": 120,     # Segundos esperando Pierna 2
    "stop_loss_on_orphan": True,  # Vender Pierna 1 si se cumple timeout
    "use_maker_orders": True,     # Órdenes maker (0 fees) vs taker
    "dry_run": True,              # SIEMPRE demo por defecto
    "scan_interval": 30,          # Segundos entre escaneos automáticos
}

GAMMA_API   = "https://gamma-api.polymarket.com"
CLOB_API    = "https://clob.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3"

# Mapeo asset → símbolo Binance
BINANCE_SYMBOLS: dict[str, str] = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

# ── Fee taker (fórmula Polymarket desde 30-mar-2026) ────────────────
def taker_fee(price: float) -> float:
    """Fee taker: 0.072 × p² × (1−p). Para maker = 0."""
    p = max(0.01, min(0.99, float(price)))
    return 0.072 * p * p * (1.0 - p)

# Mínimo de shares para poder vender en CLOB
DH_MIN_SHARES = 5.0
# Capital máximo total en riesgo en cualquier momento
DH_MAX_RISK_USD = 100.0
# Circuit breaker: pausar tras N piernas huérfanas consecutivas
DH_CIRCUIT_BREAKER_ORPHANS = 3

# ── Premium Ineficiency Tiers ────────────────────────────────────────────────
# Cuando la ineficiencia es mayor (combined más bajo), el ROI garantizado es mayor.
# El sistema amplifica el presupuesto en las oportunidades con mayor retorno potencial.
# Principio: "cuando hay más diferencia, el retorno es superior" (Asymmetric Return Premium)
DH_TIERS: dict[str, dict] = {
    "extreme":  {"combined_max": 0.75, "budget_mult": 2.0, "min_score": 55},  # ROI >33%
    "premium":  {"combined_max": 0.85, "budget_mult": 1.5, "min_score": 60},  # ROI 18-33%
    "standard": {"combined_max": 0.95, "budget_mult": 1.0, "min_score": 60},  # ROI 5-18%
}


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class DHLeg:
    """Una pierna de la operación Dump & Hedge."""
    leg_id: str
    asset: str
    timeframe: str
    market_id: str
    question: str
    side: str           # "UP" o "DOWN"
    price: float        # precio de compra
    amount_usd: float   # USD invertidos
    shares: float       # shares recibidas (amount/price)
    token_id: str
    timestamp: float = field(default_factory=time.time)
    is_demo: bool = True
    fee_usd: float = 0.0


@dataclass
class DHHedge:
    """Hedge completado (Pierna 1 + Pierna 2 con ganancia garantizada)."""
    hedge_id: str
    asset: str
    timeframe: str
    leg1: DHLeg
    leg2: DHLeg
    total_invested: float
    guaranteed_payout: float
    guaranteed_profit: float    # Ganancia mínima garantizada después de fees
    guaranteed_roi_pct: float
    layer: int = 1              # Capa de piramidación (1 = base)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DHOpportunity:
    """Oportunidad D&H detectada en el escaneo de mercados."""
    asset: str
    timeframe: str
    market_id: str
    question: str
    up_token: str
    down_token: str
    up_price: float
    down_price: float
    combined_price: float   # UP + DOWN (< 1.0 = hay ineficiencia explotable)
    dump_side: str          # "UP" o "DOWN" (el lado barato / que dumpeó)
    dump_price: float       # precio del lado dumpeado
    potential_profit_usd: float
    potential_roi_pct: float
    net_profit_after_fees: float
    # ── Campos de precisión (v2) ─────────────────────────────────
    confidence_score: float = 0.0     # 0-100: confianza en la detección
    precision_signals: list = field(default_factory=list)  # señales confirmadoras
    ineficiency_depth: float = 0.0    # 1 - combined_price (margen disponible)
    premium_tier: str = "standard"           # "extreme" / "premium" / "standard"
    optimal_budget_usd: float = 0.0          # Presupuesto óptimo según tier
    expected_value_per_dollar: float = 0.0   # ROI garantizada × confianza / 100
    guaranteed_roi_pct: float = 0.0          # ROI garantizada neta (= guaranteed_profit / total_invested × 100)
    timestamp: float = field(default_factory=time.time)


# ── Motor principal ──────────────────────────────────────────────────

class DumpHedgeEngine:
    """
    Motor del módulo Dump & Hedge de SharkFlow.

    Monitorea mercados crypto UP/DOWN en Polymarket, detecta cuando
    uno de los lados colapsa (dump), compra ese lado (Pierna 1) y
    espera la reversión para comprar el opuesto (Pierna 2).

    Con dutching óptimo (monto proporcional al precio):
        payout = total_budget / (p_up + p_down)
    Si p_up + p_down < 1.0 → payout > total_budget → ganancia garantizada.

    Desarrollado por: Carlos David Donoso Cordero (ddchack)
    """

    def __init__(self, trading_client=None, scanner=None):
        self.trader = trading_client
        self.scanner = scanner
        self._http = httpx.AsyncClient(timeout=8.0)
        self.config = DH_DEFAULT_CONFIG.copy()

        # Estado en tiempo real
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._opportunities: list[DHOpportunity] = []
        self._active_legs: list[DHLeg] = []
        self._completed_hedges: list[DHHedge] = []
        self._orphaned_legs: list[DHLeg] = []

        # P&L de sesión
        self._session_pnl: float = 0.0
        self._total_invested: float = 0.0
        self._orphan_streak: int = 0
        self._is_paused: bool = False
        self._pause_reason: str = ""

        # Cache de mercados (TTL = 60s)
        self._markets_cache: list[dict] = []
        self._markets_cache_ts: float = 0.0
        self._CACHE_TTL: float = 60.0

        # Datos de precio Binance inyectados por dump_hedge_ws
        self.binance_prices: dict[str, float] = {}

    async def close(self):
        await self._http.aclose()

    # ════════════════════════════════════════════════════════════════
    # MERCADOS
    # ════════════════════════════════════════════════════════════════

    def _current_window_ts(self, timeframe: str) -> int:
        """Retorna el timestamp de inicio de la ventana actual."""
        seconds = 300 if timeframe == "5m" else 900
        return (int(time.time()) // seconds) * seconds

    def _build_slugs(self) -> list[tuple[str, str, str]]:
        """
        Genera la lista de slugs a buscar: (asset, timeframe, slug).
        Polymarket usa el formato: {asset}-updown-{tf}-{window_ts}
        """
        slugs = []
        for asset in self.config["assets"]:
            for tf in self.config["timeframes"]:
                wts  = self._current_window_ts(tf)
                slug = f"{asset}-updown-{tf}-{wts}"
                slugs.append((asset, tf, slug))
        return slugs

    async def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        """Busca un mercado exacto por slug en Gamma API."""
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict) and data.get("id"):
                return data
        except Exception as e:
            logger.debug(f"[D&H] fetch slug {slug}: {e}")
        return None

    async def _fetch_markets(self) -> list[dict]:
        """
        Obtiene mercados UP/DOWN para todos los activos configurados.
        Estrategia dual:
          1. Búsqueda directa por slug exacto (rápida, confiable).
          2. Fallback: búsqueda textual amplia en Gamma API.
        """
        now = time.time()
        if self._markets_cache and (now - self._markets_cache_ts) < self._CACHE_TTL:
            return self._markets_cache

        found: list[dict] = []

        # ── Paso 1: slug exacto (mismo mecanismo que Rush Mode) ──────
        slug_tasks = self._build_slugs()
        tasks = [self._fetch_market_by_slug(slug) for _, _, slug in slug_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (asset, tf, slug), result in zip(slug_tasks, results):
            if isinstance(result, dict) and result:
                result["_asset"] = asset
                result["_tf"]    = tf
                found.append(result)
                logger.debug(f"[D&H] ✓ slug exacto {slug}")

        # ── Paso 2: fallback si no encontramos suficientes mercados ──
        if len(found) < len(self.config["assets"]):
            missing_assets = {a for a in self.config["assets"]
                              if not any(m["_asset"] == a for m in found)}
            if missing_assets:
                try:
                    resp = await self._http.get(
                        f"{GAMMA_API}/markets",
                        params={"active": True, "tag_slug": "crypto", "limit": 300},
                        timeout=8.0,
                    )
                    resp.raise_for_status()
                    raw = resp.json()
                    all_markets = raw if isinstance(raw, list) else raw.get("markets", [])

                    for m in all_markets:
                        q    = (m.get("question") or "").lower()
                        slug = (m.get("slug") or "").lower()
                        is_updown = ("up" in q and "down" in q) or "updown" in slug
                        is_5m  = "5 min" in q or "5-min" in slug or "-5m-" in slug
                        is_15m = "15 min" in q or "15-min" in slug or "-15m-" in slug
                        if not is_updown or not (is_5m or is_15m):
                            continue
                        for asset in missing_assets:
                            if asset in q or asset in slug:
                                tf = "5m" if is_5m else "15m"
                                if tf in self.config["timeframes"]:
                                    if not any(mm["_asset"] == asset and mm["_tf"] == tf
                                               for mm in found):
                                        m["_asset"] = asset
                                        m["_tf"]    = tf
                                        found.append(m)
                                        break
                except Exception as e:
                    logger.warning(f"[D&H] Fallback search error: {e}")

        self._markets_cache    = found
        self._markets_cache_ts = now
        logger.info(
            f"[D&H] {len(found)} mercados UP/DOWN encontrados: "
            + ", ".join(f"{m['_asset']}/{m['_tf']}" for m in found)
        )
        return found

    # ════════════════════════════════════════════════════════════════
    # PRECIOS BINANCE REST (para el dashboard sin WS activo)
    # ════════════════════════════════════════════════════════════════

    async def get_binance_prices_rest(self) -> dict[str, float]:
        """
        Obtiene precios de BTC/ETH/SOL/XRP desde Binance REST API.
        Usado cuando el WS feed no está activo (ej: solo consultar el dashboard).
        """
        symbols = [BINANCE_SYMBOLS[a] for a in self.config["assets"]
                   if a in BINANCE_SYMBOLS]
        prices: dict[str, float] = {}
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/ticker/price",
                params={"symbols": json.dumps(symbols)},
                timeout=4.0,
            )
            resp.raise_for_status()
            for entry in resp.json():
                sym = entry.get("symbol", "")
                for asset, bsym in BINANCE_SYMBOLS.items():
                    if bsym == sym:
                        prices[asset] = float(entry["price"])
        except Exception as e:
            logger.debug(f"[D&H] Binance REST prices error: {e}")
        return prices

    # ════════════════════════════════════════════════════════════════
    # PRECIOS EN TIEMPO REAL
    # ════════════════════════════════════════════════════════════════

    async def _get_clob_best_ask(self, token_id: str) -> Optional[float]:
        """Mejor ask del CLOB para un token (precio de compra / taker)."""
        if not token_id:
            return None
        try:
            resp = await self._http.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
                timeout=4.0,
            )
            resp.raise_for_status()
            asks = resp.json().get("asks", [])
            if asks:
                return float(asks[0]["price"])
        except Exception as e:
            logger.debug(f"[D&H] CLOB ask error {str(token_id)[:12]}…: {e}")
        return None

    def _extract_token_ids(self, market: dict) -> tuple[str, str]:
        """
        Extrae los token IDs del mercado.
        Polymarket Gamma API puede usar 'clobTokenIds', 'tokens', o 'clob_token_ids'.
        Siempre retorna (up_token, dn_token) — puede ser "" si no está disponible.
        """
        # Campo principal en Gamma API v2
        for field in ("clobTokenIds", "clob_token_ids", "tokens"):
            raw = market.get(field)
            if not raw:
                continue
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    continue
            if isinstance(raw, list) and len(raw) >= 2:
                t0, t1 = raw[0], raw[1]
                if isinstance(t0, dict):
                    return t0.get("token_id", ""), t1.get("token_id", "")
                return str(t0), str(t1)
        return "", ""

    async def _get_market_prices(self, market: dict) -> Optional[tuple]:
        """
        Retorna (up_price, dn_price, up_token, dn_token).
        Intenta CLOB best-ask primero; fallback a outcomePrices de Gamma.
        """
        up_token, dn_token = self._extract_token_ids(market)

        if up_token and dn_token:
            up_p, dn_p = await asyncio.gather(
                self._get_clob_best_ask(up_token),
                self._get_clob_best_ask(dn_token),
            )
            if up_p and dn_p:
                return up_p, dn_p, up_token, dn_token

        # Fallback: outcomePrices de Gamma (mid-price, sin spread)
        try:
            raw = market.get("outcomePrices", "[]")
            prices = json.loads(raw) if isinstance(raw, str) else (raw or [])
            if len(prices) >= 2:
                return float(prices[0]), float(prices[1]), up_token, dn_token
        except Exception:
            pass
        return None

    # ════════════════════════════════════════════════════════════════
    # CÁLCULO DE GANANCIA GARANTIZADA
    # ════════════════════════════════════════════════════════════════

    def calculate_guaranteed_profit(
        self,
        leg1_price: float,
        leg1_amount: float,
        leg2_price: float,
        leg2_amount: float,
    ) -> dict:
        """
        Calcula la ganancia garantizada de un hedge dado (con fees taker).

        Con dutching óptimo:
            payout = total_budget / (p1 + p2)
            Si p1+p2 < 1.0 → payout > total_budget → profit > 0

        Incluye cálculo de fees taker (0 si use_maker_orders=True).
        """
        total_invested = leg1_amount + leg2_amount
        payout_leg1 = leg1_amount / leg1_price if leg1_price > 0 else 0.0
        payout_leg2 = leg2_amount / leg2_price if leg2_price > 0 else 0.0
        min_payout  = min(payout_leg1, payout_leg2)
        max_payout  = max(payout_leg1, payout_leg2)

        # Fees (cero si usa maker)
        if self.config.get("use_maker_orders", True):
            fee1 = fee2 = 0.0
        else:
            fee1 = leg1_amount * taker_fee(leg1_price)
            fee2 = leg2_amount * taker_fee(leg2_price)
        total_fees = fee1 + fee2

        min_profit = min_payout - total_invested - total_fees
        max_profit = max_payout - total_invested - total_fees
        roi        = (min_profit / total_invested * 100) if total_invested > 0 else 0.0

        return {
            "total_invested":       round(total_invested, 4),
            "payout_if_leg1_wins":  round(payout_leg1, 4),
            "payout_if_leg2_wins":  round(payout_leg2, 4),
            "min_payout":           round(min_payout, 4),
            "max_payout":           round(max_payout, 4),
            "total_fees_taker":     round(total_fees, 4),
            "min_profit_after_fees": round(min_profit, 4),
            "max_profit_after_fees": round(max_profit, 4),
            "min_roi_pct":          round(roi, 2),
            "is_profitable":        min_profit > 0,
            "combined_price":       round(leg1_price + leg2_price, 4),
        }

    # ════════════════════════════════════════════════════════════════
    # DETECCIÓN DE OPORTUNIDADES
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def classify_tier(combined_price: float) -> str:
        """
        Clasifica la ineficiencia del mercado en un tier de retorno.
        Cuanto más bajo el combined, mayor el ROI garantizado:
          extreme  : combined < 0.75  → ROI garantizada > 33%  → presupuesto 2x
          premium  : combined 0.75-0.85 → ROI garantizada 18-33% → presupuesto 1.5x
          standard : combined 0.85-0.95 → ROI garantizada 5-18%  → presupuesto 1x
          reject   : combined ≥ 0.95   → ROI garantizada < 5%   → no apostar
        """
        if combined_price < 0.75:   return "extreme"
        elif combined_price < 0.85: return "premium"
        elif combined_price < 0.95: return "standard"
        else: return "reject"

    async def scan_opportunities(
        self,
        broadcast_fn: Optional[Callable] = None,
        lax_max_combined: float = 0.0,
    ) -> list["DHOpportunity"]:
        """
        Escanea todos los mercados configurados buscando oportunidades D&H.

        broadcast_fn: función async para enviar logs al dashboard (opcional).
        lax_max_combined: si > 0, sobreescribe max_combined_price del config.
        """
        markets = await self._fetch_markets()
        opps: list[DHOpportunity] = []

        # ── Parámetros efectivos (lax override) ──────────────────────────
        effective_max_comb = lax_max_combined if lax_max_combined > 0 else self.config["max_combined_price"]
        threshold  = self.config["dump_threshold"]

        async def _bcast(msg: str):
            if broadcast_fn:
                try:
                    await broadcast_fn("dh_log", {"line": msg})
                except Exception:
                    pass

        # Guardar precios anteriores ANTES de refrescar (S6: comparación entre scans)
        for asset_key, cur_price in list(self.binance_prices.items()):
            setattr(self, f"_prev_{asset_key}", cur_price)
        # Refrescar precios spot si el WS no los ha actualizado recientemente
        if not self.binance_prices:
            try:
                self.binance_prices = await self.get_binance_prices_rest()
                # Primera vez: prev = actual (sin cambio este scan, comparará en el siguiente)
                for asset_key, cur_price in self.binance_prices.items():
                    setattr(self, f"_prev_{asset_key}", cur_price)
            except Exception:
                pass

        n_scanned = 0
        n_candidates = 0
        skip_reasons: dict[str, int] = {}

        for market in markets:
            n_scanned += 1
            asset_label = f"{market.get('_asset','?')}/{market.get('_tf','?')}"
            try:
                prices = await self._get_market_prices(market)
                if not prices:
                    skip_reasons["sin_precios"] = skip_reasons.get("sin_precios", 0) + 1
                    continue
                up_p, dn_p, up_tok, dn_tok = prices
                if up_p <= 0 or dn_p <= 0:
                    skip_reasons["precio_cero"] = skip_reasons.get("precio_cero", 0) + 1
                    continue

                combined = up_p + dn_p

                if combined >= effective_max_comb:
                    msg = (f"[D&H] {asset_label}: combined={combined:.3f} ≥ max={effective_max_comb:.3f} "
                           f"— no arbitraje (UP={up_p:.3f} DN={dn_p:.3f})")
                    logger.debug(msg)
                    if broadcast_fn:
                        await _bcast(msg)
                    skip_reasons["combined_alto"] = skip_reasons.get("combined_alto", 0) + 1
                    continue

                dump_side = None
                if up_p <= threshold:
                    dump_side = "UP"
                elif dn_p <= threshold:
                    dump_side = "DOWN"

                if not dump_side:
                    msg = (f"[D&H] {asset_label}: sin dump (UP={up_p:.3f} DN={dn_p:.3f}, "
                           f"necesita ≤{threshold:.3f} en algún lado)")
                    logger.debug(msg)
                    if broadcast_fn:
                        await _bcast(msg)
                    skip_reasons["sin_dump"] = skip_reasons.get("sin_dump", 0) + 1
                    continue

                n_candidates += 1
                opp = self._build_opportunity(
                    market, up_p, dn_p, up_tok, dn_tok, dump_side
                )

                if not opp:
                    msg = f"[D&H] {asset_label}: dump={dump_side} @ {up_p if dump_side=='UP' else dn_p:.3f} — shares insuficientes"
                    logger.debug(msg)
                    if broadcast_fn:
                        await _bcast(msg)
                    skip_reasons["shares_insuf"] = skip_reasons.get("shares_insuf", 0) + 1
                    continue

                if opp.net_profit_after_fees <= 0:
                    msg = (f"[D&H] {asset_label}: profit negativo después de fees "
                           f"(net={opp.net_profit_after_fees:.4f}, roi={opp.guaranteed_roi_pct:.1f}%)")
                    logger.debug(msg)
                    if broadcast_fn:
                        await _bcast(msg)
                    skip_reasons["profit_negativo"] = skip_reasons.get("profit_negativo", 0) + 1
                    continue

                if opp.premium_tier == "reject":
                    msg = (f"[D&H] {asset_label}: tier=reject (combined={combined:.3f} ≥ 0.95)")
                    logger.debug(msg)
                    skip_reasons["tier_reject"] = skip_reasons.get("tier_reject", 0) + 1
                    continue

                min_score = DH_TIERS.get(opp.premium_tier, DH_TIERS["standard"])["min_score"]
                if opp.confidence_score < min_score:
                    msg = (f"[D&H] {asset_label}: score={opp.confidence_score:.0f} < mín={min_score} "
                           f"(tier={opp.premium_tier}, dump={dump_side} @ {opp.dump_price:.3f})")
                    logger.debug(msg)
                    if broadcast_fn:
                        await _bcast(msg)
                    skip_reasons["score_bajo"] = skip_reasons.get("score_bajo", 0) + 1
                    continue

                opps.append(opp)

            except Exception as e:
                logger.debug(f"[D&H] scan error {asset_label}: {e}")

        # ── Heartbeat de diagnóstico ────────────────────────────────────────
        skip_summary = ", ".join(f"{k}={v}" for k, v in skip_reasons.items()) if skip_reasons else "ninguno"
        heartbeat_msg = (
            f"[D&H] Escaneados {n_scanned} mercados — {n_candidates} candidatos — "
            f"{len(opps)} aprobados | Rechazados: {skip_summary}"
        )
        logger.info(heartbeat_msg)
        if broadcast_fn:
            await _bcast(heartbeat_msg)

        # Ordenar por EV por dólar: ROI garantizada × confianza (Asymmetric Return Premium)
        # Las oportunidades "extreme" con alta confianza tienen mayor EV por dólar
        self._opportunities = sorted(
            opps,
            key=lambda x: x.expected_value_per_dollar,
            reverse=True
        )
        return self._opportunities

    def _build_opportunity(
        self, market, up_p, dn_p, up_tok, dn_tok, dump_side
    ) -> Optional[DHOpportunity]:
        """Construye DHOpportunity con cálculo de profit usando dutching."""
        combined = up_p + dn_p
        budget   = self.config["bet_per_leg"] * 2
        payout   = budget / combined if combined > 0 else 0

        leg1_p   = up_p if dump_side == "UP" else dn_p
        leg2_p   = dn_p if dump_side == "UP" else up_p
        leg1_amt = budget * leg1_p / combined if combined > 0 else budget / 2
        leg2_amt = budget - leg1_amt

        if self.config.get("use_maker_orders", True):
            fee1 = fee2 = 0.0
        else:
            fee1 = leg1_amt * taker_fee(leg1_p)
            fee2 = leg2_amt * taker_fee(leg2_p)

        gross = payout - budget
        net   = gross - fee1 - fee2
        roi   = (net / budget * 100) if budget > 0 else 0.0

        # Verificar mínimo de shares en Pierna 1
        shares_leg1 = leg1_amt / leg1_p if leg1_p > 0 else 0
        if shares_leg1 < DH_MIN_SHARES:
            return None

        opp = DHOpportunity(
            asset=market.get("_asset", "btc"),
            timeframe=market.get("_tf", "5m"),
            market_id=market.get("id") or market.get("market_id") or "",
            question=market.get("question", ""),
            up_token=up_tok,
            down_token=dn_tok,
            up_price=up_p,
            down_price=dn_p,
            combined_price=round(combined, 4),
            dump_side=dump_side,
            dump_price=leg1_p,
            potential_profit_usd=round(gross, 4),
            potential_roi_pct=round(roi, 2),
            net_profit_after_fees=round(net, 4),
            ineficiency_depth=round(1.0 - combined, 4),
        )
        # Clasificar tier y calcular presupuesto óptimo
        tier = DumpHedgeEngine.classify_tier(combined)
        budget_mult = DH_TIERS.get(tier, DH_TIERS["standard"])["budget_mult"]
        opp.premium_tier   = tier
        opp.optimal_budget_usd = round(budget * budget_mult, 4)
        opp.guaranteed_roi_pct = round(net / budget * 100, 2) if budget > 0 else 0.0
        # Scoring de precisión: calcula confianza con múltiples señales
        opp.confidence_score, opp.precision_signals = self._score_opportunity(opp)
        # EV por dólar: cuánto retorno garantizado por dólar, ponderado por confianza
        opp.expected_value_per_dollar = round(
            (opp.guaranteed_roi_pct / 100.0) * (opp.confidence_score / 100.0), 4
        )
        return opp

    def _score_opportunity(self, opp: DHOpportunity) -> tuple[float, list]:
        """
        Scoring de precisión para oportunidades D&H.
        Objetivo: ≥80% precision en detección.

        Señales evaluadas:
          S1 — Profundidad de ineficiencia (combined < 0.90 = fuerte)
          S2 — Precio del dump muy bajo (≤ 0.10 = confirmación de pánico)
          S3 — Asimetría: dump_price << lado_opuesto (captura real unilateral)
          S4 — Timeframe corto (5m mejor que 15m: reversión más rápida)
          S5 — Ratio precio_opuesto / dump_price > 3 (el mercado espera corrección)
        """
        signals: list[str] = []
        score = 0.0

        combined    = opp.combined_price
        dump_p      = opp.dump_price
        other_p     = opp.down_price if opp.dump_side == "UP" else opp.up_price
        inef_depth  = 1.0 - combined

        # S1: Ineficiencia profunda
        if inef_depth >= 0.15:
            score += 30
            signals.append(f"S1:ineficiencia_profunda({inef_depth*100:.1f}%)")
        elif inef_depth >= 0.08:
            score += 15
            signals.append(f"S1:ineficiencia_moderada({inef_depth*100:.1f}%)")

        # S2: Precio del dump muy bajo (pánico real)
        if dump_p <= 0.08:
            score += 25
            signals.append(f"S2:dump_pánico({dump_p*100:.1f}¢)")
        elif dump_p <= 0.12:
            score += 15
            signals.append(f"S2:dump_fuerte({dump_p*100:.1f}¢)")
        elif dump_p <= 0.15:
            score += 8
            signals.append(f"S2:dump_leve({dump_p*100:.1f}¢)")

        # S3: Asimetría fuerte (un lado dumpeó, el otro todavía alto)
        if other_p >= 0.75:
            score += 20
            signals.append(f"S3:asimetria_fuerte(otro={other_p*100:.1f}¢)")
        elif other_p >= 0.60:
            score += 10
            signals.append(f"S3:asimetria_moderada(otro={other_p*100:.1f}¢)")

        # S4: Timeframe corto → reversión más rápida
        if opp.timeframe == "5m":
            score += 15
            signals.append("S4:timeframe_5m")
        elif opp.timeframe == "15m":
            score += 8
            signals.append("S4:timeframe_15m")

        # S5: Ratio lado_opuesto / dump_price (mercado confía en corrección)
        if dump_p > 0:
            ratio = other_p / dump_p
            if ratio >= 5.0:
                score += 10
                signals.append(f"S5:ratio_alto({ratio:.1f}x)")
            elif ratio >= 3.0:
                score += 5
                signals.append(f"S5:ratio_medio({ratio:.1f}x)")

        # S6: Confirmación de movimiento brusco en Binance (correlación precio spot)
        # Si tenemos precio spot y el dump_side es "DOWN" → precio cayó → confirma pánico
        # Si dump_side es "UP" → precio subió fuerte → confirma euforia temporal
        asset_key = getattr(opp, "asset", "").lower()
        binance_p = self.binance_prices.get(asset_key, 0.0)
        if binance_p > 0:
            # Comparar con precio anterior si está disponible
            prev_key = f"_prev_{asset_key}"
            prev_p = getattr(self, prev_key, 0.0)
            if prev_p > 0:
                pct_change = (binance_p - prev_p) / prev_p
                if opp.dump_side == "DOWN" and pct_change <= -0.005:   # ≥0.5% caída
                    score += 20
                    signals.append(f"S6:binance_confirma_caída({pct_change*100:.2f}%)")
                elif opp.dump_side == "UP" and pct_change >= 0.005:    # ≥0.5% subida
                    score += 20
                    signals.append(f"S6:binance_confirma_subida({pct_change*100:.2f}%)")
                elif abs(pct_change) < 0.001:
                    # Precio spot plano → puede ser falsa señal
                    score -= 10
                    signals.append(f"S6:binance_spot_plano({pct_change*100:.3f}%)")
            else:
                # Precio disponible pero sin histórico → señal neutral, sólo loguear
                signals.append(f"S6:binance_spot={binance_p:.2f}(sin prev)")

        # S7: Velocidad del movimiento en Binance (tasa de cambio rápida = mayor probabilidad de reversión)
        # Una caída/subida rápida tiende a rebotar. Una caída lenta puede continuar.
        asset_key2 = getattr(opp, "asset", "").lower()
        binance_p2 = self.binance_prices.get(asset_key2, 0.0)
        if binance_p2 > 0:
            prev_key2 = f"_prev_{asset_key2}"
            prev_p2 = getattr(self, prev_key2, 0.0)
            if prev_p2 > 0:
                abs_chg = abs(binance_p2 - prev_p2) / prev_p2
                if abs_chg >= 0.02:    # ≥2% movimiento en un scan = extremo rápido
                    score += 15
                    signals.append(f"S7:velocidad_extrema({abs_chg*100:.2f}%/scan)")
                elif abs_chg >= 0.01:  # 1-2%
                    score += 8
                    signals.append(f"S7:velocidad_alta({abs_chg*100:.2f}%/scan)")

        # S8: Profit/Risk ratio — garantía de ganancia con margen de seguridad
        # Si el retorno neto después de fees supera el 2% del capital invertido, es sólido
        budget = (self.config.get("bet_per_leg", 1.0)) * 2
        net_roi = opp.net_profit_after_fees / budget * 100 if budget > 0 else 0
        if net_roi >= 4.0:
            score += 15
            signals.append(f"S8:profit_ratio_excelente({net_roi:.1f}%)")
        elif net_roi >= 2.0:
            score += 8
            signals.append(f"S8:profit_ratio_bueno({net_roi:.1f}%)")
        elif net_roi >= 0.5:
            score += 3
            signals.append(f"S8:profit_ratio_marginal({net_roi:.1f}%)")

        # S9: Extreme leg — cuando dump_price ≤ 8¢, el retorno individual es 12x-25x+
        # Esto activa el "Asymmetric Return Premium": más diferencia = retorno superior
        dump_p = opp.dump_price
        if dump_p <= 0.05:
            score += 20
            signals.append(f"S9:extreme_dump({dump_p*100:.1f}¢→{1/dump_p:.0f}x retorno)")
        elif dump_p <= 0.08:
            score += 10
            signals.append(f"S9:high_return({dump_p*100:.1f}¢→{1/dump_p:.0f}x retorno)")

        return round(min(100.0, score), 1), signals

    # ════════════════════════════════════════════════════════════════
    # EJECUCIÓN DE PIERNAS
    # ════════════════════════════════════════════════════════════════

    async def execute_leg1(self, opp: DHOpportunity) -> Optional[DHLeg]:
        """
        Pierna 1: compra el lado barato (el que dumpeó).
        En demo: simula la compra sin mover fondos reales.
        """
        combined = opp.combined_price
        budget   = self.config["bet_per_leg"] * 2
        price    = opp.dump_price
        token    = opp.up_token if opp.dump_side == "UP" else opp.down_token
        amount   = budget * price / combined if combined > 0 else self.config["bet_per_leg"]
        shares   = amount / price if price > 0 else 0
        fee      = amount * taker_fee(price) if not self.config.get("use_maker_orders") else 0.0

        if not self.config["dry_run"] and self.trader:
            try:
                result = await self.trader.buy_token(
                    token_id=token,
                    amount_usd=amount,
                    use_limit=self.config.get("use_maker_orders", True),
                )
                if not result or getattr(result, "status", None) != "filled":
                    logger.warning(f"[D&H] Leg1 rechazada por el CLOB: {result}")
                    return None
            except Exception as e:
                logger.error(f"[D&H] Error ejecutando Leg1: {e}")
                return None

        leg = DHLeg(
            leg_id=f"DH-L1-{opp.asset.upper()}-{int(time.time())}",
            asset=opp.asset,
            timeframe=opp.timeframe,
            market_id=opp.market_id,
            question=opp.question,
            side=opp.dump_side,
            price=price,
            amount_usd=round(amount, 4),
            shares=round(shares, 2),
            token_id=token,
            is_demo=self.config["dry_run"],
            fee_usd=round(fee, 4),
        )
        self._active_legs.append(leg)
        self._total_invested += amount
        logger.info(
            f"[D&H] ✓ Leg1 {opp.dump_side} @ {price:.3f} | "
            f"{shares:.1f} shares | ${amount:.2f} | demo={leg.is_demo}"
        )
        return leg

    async def execute_leg2(
        self,
        leg1: DHLeg,
        price2: float,
        token2: str,
        layer: int = 1,
    ) -> Optional[DHLeg]:
        """
        Pierna 2: compra el lado opuesto (hedge / cobertura).
        Completa el hedge y registra la ganancia garantizada.
        """
        combined = leg1.price + price2
        budget   = self.config["bet_per_leg"] * 2
        amount   = budget * price2 / combined if combined > 0 else self.config["bet_per_leg"]
        shares   = amount / price2 if price2 > 0 else 0
        fee      = amount * taker_fee(price2) if not self.config.get("use_maker_orders") else 0.0
        side2    = "DOWN" if leg1.side == "UP" else "UP"

        if not self.config["dry_run"] and self.trader:
            try:
                result = await self.trader.buy_token(
                    token_id=token2,
                    amount_usd=amount,
                    use_limit=self.config.get("use_maker_orders", True),
                )
                if not result or getattr(result, "status", None) != "filled":
                    logger.warning(f"[D&H] Leg2 rechazada por el CLOB: {result}")
                    return None
            except Exception as e:
                logger.error(f"[D&H] Error ejecutando Leg2: {e}")
                return None

        leg2 = DHLeg(
            leg_id=f"DH-L2-{leg1.asset.upper()}-{int(time.time())}",
            asset=leg1.asset,
            timeframe=leg1.timeframe,
            market_id=leg1.market_id,
            question=leg1.question,
            side=side2,
            price=price2,
            amount_usd=round(amount, 4),
            shares=round(shares, 2),
            token_id=token2,
            is_demo=self.config["dry_run"],
            fee_usd=round(fee, 4),
        )
        self._total_invested += amount

        # Registrar hedge completado
        result = self.calculate_guaranteed_profit(
            leg1.price, leg1.amount_usd, price2, amount
        )
        if result["is_profitable"]:
            hedge = DHHedge(
                hedge_id=f"DH-H-{leg1.asset.upper()}-{int(time.time())}",
                asset=leg1.asset,
                timeframe=leg1.timeframe,
                leg1=leg1,
                leg2=leg2,
                total_invested=result["total_invested"],
                guaranteed_payout=result["min_payout"],
                guaranteed_profit=result["min_profit_after_fees"],
                guaranteed_roi_pct=result["min_roi_pct"],
                layer=layer,
            )
            self._completed_hedges.append(hedge)
            self._session_pnl += hedge.guaranteed_profit
            self._orphan_streak = 0
            logger.info(
                f"[D&H] ✓ HEDGE COMPLETO capa={layer} "
                f"+${hedge.guaranteed_profit:.3f} ROI={hedge.guaranteed_roi_pct:.1f}%"
            )

        self._active_legs = [l for l in self._active_legs if l.leg_id != leg1.leg_id]
        return leg2

    # ════════════════════════════════════════════════════════════════
    # PIRAMIDACIÓN
    # ════════════════════════════════════════════════════════════════

    def evaluate_pyramid(self, layer: int) -> dict:
        """
        Evalúa si hay suficiente ganancia asegurada para arriesgar
        una capa adicional de piramidación.
        """
        if layer > self.config["max_pyramid_layers"]:
            return {"can_pyramid": False, "reason": "max_layers_reached"}
        if not self._completed_hedges:
            return {"can_pyramid": False, "reason": "no_base_hedge"}

        secured  = sum(h.guaranteed_profit for h in self._completed_hedges)
        risk_next = self.config["bet_per_leg"] * 2

        if secured > risk_next:
            return {
                "can_pyramid": True,
                "layer": layer,
                "secured_profit": round(secured, 4),
                "risk_next_layer": round(risk_next, 4),
                "net_floor": round(secured - risk_next, 4),
            }
        return {
            "can_pyramid": False,
            "reason": "insufficient_secured_profit",
            "secured_profit": round(secured, 4),
            "risk_next_layer": round(risk_next, 4),
        }

    # ════════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER Y STOP-LOSS
    # ════════════════════════════════════════════════════════════════

    def register_orphan(self, leg1: DHLeg) -> str:
        """
        Registra una pierna huérfana (Leg2 no llegó en timeout).
        Si se alcanzan N huérfanas seguidas: activa circuit breaker.
        """
        self._orphaned_legs.append(leg1)
        self._active_legs = [l for l in self._active_legs if l.leg_id != leg1.leg_id]
        self._orphan_streak += 1

        if self._orphan_streak >= DH_CIRCUIT_BREAKER_ORPHANS:
            self._is_paused   = True
            self._pause_reason = (
                f"Circuit breaker: {DH_CIRCUIT_BREAKER_ORPHANS} "
                "piernas huérfanas consecutivas"
            )
            logger.warning(f"[D&H] ⚠️ {self._pause_reason}")
            return "circuit_breaker"
        return "orphan"

    def resume(self):
        """Reanuda el bot después de un circuit breaker."""
        self._is_paused    = False
        self._pause_reason = ""
        self._orphan_streak = 0
        logger.info("[D&H] Bot reanudado manualmente")

    # ════════════════════════════════════════════════════════════════
    # WORKER AUTOMÁTICO
    # ════════════════════════════════════════════════════════════════

    async def _worker(self, broadcast_fn: Optional[Callable] = None):
        """Loop principal del bot autónomo D&H."""
        logger.info("[D&H] ⚡ Worker iniciado")

        while self._running:
            try:
                if self._is_paused:
                    await asyncio.sleep(30)
                    continue

                # Escanear mercados (broadcast_fn pasa logs detallados al dashboard)
                opps = await self.scan_opportunities(broadcast_fn=broadcast_fn)
                if broadcast_fn and opps:
                    await broadcast_fn(
                        "dh_opportunities",
                        [self._opp_to_dict(o) for o in opps[:8]]
                    )

                # Ejecutar la mejor oportunidad (si hay capital disponible)
                for opp in opps[:1]:
                    if self._total_invested >= DH_MAX_RISK_USD:
                        logger.info("[D&H] Capital máximo en riesgo alcanzado")
                        break

                    leg1 = await self.execute_leg1(opp)
                    if not leg1:
                        continue
                    if broadcast_fn:
                        await broadcast_fn("dh_leg1", self._leg_to_dict(leg1))

                    # Esperar Pierna 2 (reversión)
                    timeout = self.config["leg2_wait_timeout"]
                    t0      = time.time()
                    hedged  = False
                    layer   = 1

                    while time.time() - t0 < timeout:
                        await asyncio.sleep(5)
                        if not self._running:
                            break

                        # Volver a pedir precios
                        m_dummy = {
                            "id": opp.market_id,
                            "tokens": [
                                {"token_id": opp.up_token},
                                {"token_id": opp.down_token},
                            ],
                        }
                        prices = await self._get_market_prices(m_dummy)
                        if not prices:
                            continue

                        up_p, dn_p, _, _ = prices
                        price2  = dn_p if opp.dump_side == "UP" else up_p
                        token2  = opp.down_token if opp.dump_side == "UP" else opp.up_token

                        chk = self.calculate_guaranteed_profit(
                            leg1.price, leg1.amount_usd, price2,
                            self.config["bet_per_leg"]
                        )
                        if chk["is_profitable"]:
                            leg2 = await self.execute_leg2(leg1, price2, token2, layer)
                            if leg2 and broadcast_fn and self._completed_hedges:
                                last_h = self._completed_hedges[-1]
                                await broadcast_fn("dh_hedge_complete", {
                                    "hedge": self._hedge_to_dict(last_h),
                                    "session_pnl": round(self._session_pnl, 4),
                                })
                                # Evaluar piramidación
                                py = self.evaluate_pyramid(layer + 1)
                                if py.get("can_pyramid"):
                                    await broadcast_fn("dh_pyramid_available", py)
                            hedged = True
                            break

                    if not hedged:
                        action = self.register_orphan(leg1)
                        if broadcast_fn:
                            await broadcast_fn("dh_orphan", {
                                "leg": self._leg_to_dict(leg1),
                                "streak": self._orphan_streak,
                                "circuit_breaker": action == "circuit_breaker",
                            })

                await asyncio.sleep(self.config.get("scan_interval", 30))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[D&H] Worker error: {e}")
                await asyncio.sleep(10)

        logger.info("[D&H] Worker detenido")

    def start(self, broadcast_fn: Optional[Callable] = None) -> bool:
        """Inicia el bot en modo automático."""
        if self._running:
            return False
        self._running     = True
        self._is_paused   = False
        self._orphan_streak = 0
        self._task = asyncio.create_task(self._worker(broadcast_fn))
        logger.info(f"[D&H] Bot iniciado (dry_run={self.config['dry_run']})")
        return True

    def stop(self):
        """Detiene el bot."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("[D&H] Bot detenido")

    # ════════════════════════════════════════════════════════════════
    # SERIALIZACIÓN Y ESTADO
    # ════════════════════════════════════════════════════════════════

    def _opp_to_dict(self, o: DHOpportunity) -> dict:
        return {
            "asset": o.asset,
            "timeframe": o.timeframe,
            "market_id": o.market_id,
            "question": (o.question or "")[:90],
            "up_price": o.up_price,
            "down_price": o.down_price,
            "combined_price": o.combined_price,
            "dump_side": o.dump_side,
            "dump_price": o.dump_price,
            "potential_profit_usd": o.potential_profit_usd,
            "potential_roi_pct": o.potential_roi_pct,
            "net_profit_after_fees": o.net_profit_after_fees,
            "confidence_score": o.confidence_score,
            "precision_signals": o.precision_signals,
            "ineficiency_depth": o.ineficiency_depth,
            "timestamp": o.timestamp,
        }

    def _leg_to_dict(self, l: DHLeg) -> dict:
        return {
            "leg_id": l.leg_id,
            "asset": l.asset,
            "timeframe": l.timeframe,
            "side": l.side,
            "price": l.price,
            "amount_usd": l.amount_usd,
            "shares": l.shares,
            "fee_usd": l.fee_usd,
            "is_demo": l.is_demo,
            "timestamp": l.timestamp,
            "market_id": l.market_id,
        }

    def _hedge_to_dict(self, h: DHHedge) -> dict:
        return {
            "hedge_id": h.hedge_id,
            "asset": h.asset,
            "timeframe": h.timeframe,
            "total_invested": h.total_invested,
            "guaranteed_payout": h.guaranteed_payout,
            "guaranteed_profit": h.guaranteed_profit,
            "guaranteed_roi_pct": h.guaranteed_roi_pct,
            "layer": h.layer,
            "timestamp": h.timestamp,
            "leg1": self._leg_to_dict(h.leg1),
            "leg2": self._leg_to_dict(h.leg2),
        }

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "is_paused": self._is_paused,
            "pause_reason": self._pause_reason,
            "dry_run": self.config["dry_run"],
            "session_pnl": round(self._session_pnl, 4),
            "active_legs": len(self._active_legs),
            "active_legs_detail": [self._leg_to_dict(l) for l in self._active_legs],
            "completed_hedges": len(self._completed_hedges),
            "orphaned_legs": len(self._orphaned_legs),
            "orphan_streak": self._orphan_streak,
            "circuit_breaker_threshold": DH_CIRCUIT_BREAKER_ORPHANS,
            "opportunities_cached": len(self._opportunities),
            "total_invested": round(self._total_invested, 4),
            "max_risk_usd": DH_MAX_RISK_USD,
            "config": self.config,
        }

    def get_opportunities(self) -> list[dict]:
        return [self._opp_to_dict(o) for o in self._opportunities]

    def get_history(self, limit: int = 50) -> list[dict]:
        return [self._hedge_to_dict(h) for h in self._completed_hedges[-limit:]]

    def get_stats(self) -> dict:
        total_ops = len(self._completed_hedges) + len(self._orphaned_legs)
        win_rate  = (len(self._completed_hedges) / total_ops * 100) if total_ops > 0 else 0.0
        avg_profit = (
            self._session_pnl / len(self._completed_hedges)
            if self._completed_hedges else 0.0
        )
        return {
            "total_hedges_completed": len(self._completed_hedges),
            "total_orphaned_legs": len(self._orphaned_legs),
            "win_rate_pct": round(win_rate, 1),
            "session_pnl": round(self._session_pnl, 4),
            "avg_profit_per_hedge": round(avg_profit, 4),
            "total_invested_session": round(self._total_invested, 4),
        }
