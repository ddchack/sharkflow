"""
SharkFlow Rush Mode — Bitcoin 5-Minutos
Señal combinada: BTC momentum (Binance 5m klines) + consenso de mercado (Polymarket).
Solo apuesta cuando AMBAS señales coinciden, en ventana 25-40s antes del cierre.
"""
import httpx
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Ventana de apuesta ──────────────────────────────────────────
RUSH_MIN_SECONDS     = 27     # v2: ventana óptima empieza a 27s (era 25s)
RUSH_MAX_SECONDS     = 55     # ventana de entrada hasta 55s antes del cierre
RUSH_SCAN_INTERVAL   = 5      # escanear cada 5s
# v2: timing zones (Chainlink lag compensation — oracle lag ≈ 3.5s)
RUSH_OPTIMAL_MIN     = 27     # 27-35s = "zona óptima" (oracle puede confirmar dentro de ventana)
RUSH_OPTIMAL_MAX     = 35     # >35s = "early" (demasiado tiempo para reversión)
RUSH_LATE_THRESHOLD  = 20     # <20s = "critical" (no-fill casi seguro en CLOB)

# ── Umbrales de señal ───────────────────────────────────────────
RUSH_MARKET_HIGH     = 0.52   # Poly dice UP si precio Up > 52% (umbral bajo — confirmación ligera)
RUSH_MARKET_LOW      = 0.48   # Poly dice DOWN si precio Up < 48%
RUSH_BTC_MOMENTUM    = 0.0006  # Momentum mínimo BTC (≥0.06% — umbral bajo, BTC es señal primaria)
RUSH_BTC_STRONG      = 0.0018  # Momentum fuerte (≥0.18%, ~4σ) — señal de alta confianza para ETH/SOL
RUSH_TIGHT_WINDOW    = 25     # señal débil: entrar en ventana completa (era 35 — bloqueaba 55s de 65s disponibles)

# ── Multi-asset: ETH y SOL disponibles (correlación BTC-ETH=0.87, BTC-SOL=0.83) ─
RUSH_ETH_ENABLED        = True    # escanear mercados ETH además de BTC
RUSH_ETH_CORRELATION    = 0.87    # factor de correlación para ajustar confianza ETH
RUSH_ETH_MIN_CONFIDENCE = 0.40    # confianza mínima para apostar ETH (via señal BTC)

RUSH_SOL_ENABLED        = True    # v3: escanear mercados SOL (correlación 0.83)
RUSH_SOL_CORRELATION    = 0.83    # BTC-SOL correlación histórica (menor que ETH)
RUSH_SOL_MIN_CONFIDENCE = 0.38    # umbral ligeramente inferior al de ETH

# ── Multi-period momentum multipliers ───────────────────────────────────────
# Investigación: momentum sostenido en 2 candles 1-min tiene win rate ~72% vs ~63% single.
RUSH_MOMENTUM_MULTIPLIERS: dict[str, float] = {
    "ACCELERATING_UP":   1.35,   # ambos candles UP y acelerando → señal más fuerte
    "ACCELERATING_DOWN": 1.35,
    "SUSTAINED_UP":      1.20,   # ambos candles en misma dirección → confirmado
    "SUSTAINED_DOWN":    1.20,
    "SINGLE_UP":         1.00,   # solo un candle cumple el umbral → señal estándar
    "SINGLE_DOWN":       1.00,
    "MIXED":             0.60,   # candles contradictorios → reducir tamaño
}

# ── Zonas de retorno asimétrico ──────────────────────────────────────────────
# La zona "premium" (precio 0.20-0.40) tiene retorno 2.5x-5x y Kelly mayor (+30%).
# La zona "marginal" (0.60-0.78) se acepta con Kelly reducido (-30%).
# Precio > 0.78: rechazar — retorno < 1.28x no compensa el riesgo de ejecución.
RUSH_RETURN_ZONES: list[tuple] = [
    # (nombre, precio_min, precio_max, modificador_kelly)
    ("ultra",    0.10, 0.20, 1.50),   # 5x-10x retorno, Kelly +50% — alta asimetría
    ("premium",  0.20, 0.40, 1.30),   # 2.5x-5x retorno, Kelly +30%
    ("good",     0.40, 0.60, 1.00),   # 1.7x-2.5x retorno, Kelly normal
    ("marginal", 0.60, 0.78, 0.70),   # 1.3x-1.7x retorno, Kelly -30%
]
RUSH_MAX_SIGNAL_PRICE = 0.78   # rechaza precios > 78¢ (retorno < 1.28x — sin edge)
RUSH_MIN_RETURN       = 1.28   # = 1/0.78
RUSH_KELLY_CAP        = 0.35   # fracción Kelly máxima permitida (35% del capital)
RUSH_KELLY_MIN        = 0.02   # fracción mínima para que valga ejecutar (2%, era 5% — bloqueaba señales débiles)

GAMMA_API   = "https://gamma-api.polymarket.com"
BINANCE_API = "https://api.binance.com/api/v3"

# ── Capital configurable (BUG 1 fix) ─────────────────────────────────────────
# Valor por defecto: $3.0. api_server.py puede modificar esta variable global
# al iniciar Rush Mode para reflejar el capital real configurado por el usuario.
_rush_max_capital: float = 3.0


@dataclass
class RushOpportunity:
    market_id: str
    question: str
    slug: str
    window_ts: int           # timestamp de inicio de la ventana de 5 min
    up_price: float
    down_price: float
    side: str                # "UP" o "DOWN"
    signal_price: float      # precio del lado apostado en Polymarket
    signal_strength: float   # 0.0-1.0 combinado
    seconds_remaining: float
    token: str               # token a comprar en CLOB
    potential_return: float
    btc_current: float       # precio BTC ahora
    btc_open: float          # precio BTC al inicio de ventana
    btc_change_pct: float    # % cambio desde inicio de ventana
    poly_signal: str         # "UP" / "DOWN"
    btc_signal: str          # "UP" / "DOWN"
    btc_volume_ratio: float = 1.0    # volumen actual / promedio ventana (>1.5 = aceleración)
    signal_quality: str = "normal"   # "strong" / "normal" / "weak"
    return_zone: str = "good"        # "premium" / "good" / "marginal"
    kelly_fraction: float = 0.0      # Kelly óptimo para esta apuesta (0-1)
    kelly_bet_usd: float = 0.0       # Monto sugerido en USD
    model_prob: float = 0.0          # Probabilidad estimada por el modelo
    # v2: campos multi-period momentum
    momentum_type: str = "SINGLE_UP"        # ACCELERATING/SUSTAINED/SINGLE/MIXED
    momentum_multiplier: float = 1.0        # multiplicador Kelly según momentum type
    timing_quality: str = "optimal"         # optimal/early/late/critical
    # v2: asset field (BTC o ETH)
    asset: str = "BTC"                      # activo del mercado


class RushEngine:
    def __init__(self, scanner=None):
        self.scanner = scanner
        self._http = httpx.AsyncClient(timeout=6.0)
        self._btc_open_cache: OrderedDict[int, float] = OrderedDict()  # window_ts -> open price (LRU, max 10)
        self._dynamic_correlations: dict = {}
        self._last_corr_update: float = 0.0  # BUG 2 fix: throttle correlaciones

    async def close(self):
        await self._http.aclose()

    def _cache_set(self, key: int, value: float) -> None:
        """Guarda precio de apertura BTC en caché LRU con límite de tamaño."""
        self._btc_open_cache[key] = value
        while len(self._btc_open_cache) > 10:
            self._btc_open_cache.popitem(last=False)

    # ── Precio BTC actual ──────────────────────────────────────

    async def get_btc_price(self) -> Optional[float]:
        """Precio BTC en tiempo real desde Binance."""
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/ticker/price",
                params={"symbol": "BTCUSDT"}, timeout=3.0)
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception as e:
            logger.debug(f"[Rush] BTC price error: {e}")
            return None

    # ── v2: Multi-period momentum (últimos 2 candles 1-min) ──────

    async def get_btc_1min_momentum(self) -> dict:
        """
        v2: Analiza momentum BTC en los últimos 2 candles de 1 minuto.
        Si ambos van en la misma dirección → momentum sostenido (más predictivo).
        Si acelera (c2 > c1) → señal más fuerte.
        Returns: {"type": str, "c1_pct": float, "c2_pct": float, "multiplier": float}
        """
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": 3},
                timeout=4.0,
            )
            resp.raise_for_status()
            klines = resp.json()
            if len(klines) < 3:
                return {"type": "SINGLE_UP", "c1_pct": 0.0, "c2_pct": 0.0, "multiplier": 1.0}

            # klines[-1] = vela actual (incompleta), klines[-2] = penúltima, klines[-3] = antepenúltima
            c1_open  = float(klines[-3][1]); c1_close = float(klines[-3][4])
            c2_open  = float(klines[-2][1]); c2_close = float(klines[-2][4])
            c1_pct   = (c1_close - c1_open) / max(c1_open, 0.01)
            c2_pct   = (c2_close - c2_open) / max(c2_open, 0.01)
            threshold = RUSH_BTC_MOMENTUM

            up1 = c1_pct >= threshold; down1 = c1_pct <= -threshold
            up2 = c2_pct >= threshold; down2 = c2_pct <= -threshold

            if up1 and up2:
                mtype = "ACCELERATING_UP" if abs(c2_pct) > abs(c1_pct) else "SUSTAINED_UP"
            elif down1 and down2:
                mtype = "ACCELERATING_DOWN" if abs(c2_pct) > abs(c1_pct) else "SUSTAINED_DOWN"
            elif (up1 and down2) or (down1 and up2):
                mtype = "MIXED"   # FIX: debe ir ANTES de SINGLE para que no sea dead code
            elif up1 or up2:
                mtype = "SINGLE_UP"
            elif down1 or down2:
                mtype = "SINGLE_DOWN"
            else:
                mtype = "SINGLE_UP"  # debajo del umbral → tratar como single

            mult = RUSH_MOMENTUM_MULTIPLIERS.get(mtype, 1.0)
            return {"type": mtype, "c1_pct": round(c1_pct * 100, 4),
                    "c2_pct": round(c2_pct * 100, 4), "multiplier": mult}
        except Exception as e:
            logger.debug(f"[Rush] 1-min momentum error: {e}")
            return {"type": "SINGLE_UP", "c1_pct": 0.0, "c2_pct": 0.0, "multiplier": 1.0}

    @staticmethod
    def classify_timing_quality(seconds_remaining: float) -> str:
        """
        v2: Clasifica la calidad del timing de entrada con compensación de lag Chainlink.
        El oracle Chainlink tiene lag ≈ 2-5s (promedio 3.5s).
        optimal: 27-35s → señal definida + tiempo suficiente para fill + oracle alcanza
        early:   35-90s → señal puede revertirse antes de cierre
        late:    20-27s → riesgo de no-fill en CLOB (makers se retiran)
        critical: <20s  → no entrar
        """
        if seconds_remaining < RUSH_LATE_THRESHOLD:
            return "critical"
        elif seconds_remaining <= RUSH_OPTIMAL_MIN:
            return "late"
        elif seconds_remaining <= RUSH_OPTIMAL_MAX:
            return "optimal"
        else:
            return "early"

    async def _fetch_eth_market(self) -> Optional[dict]:
        """
        v2: Busca mercado ETH 5-min de la ventana actual.
        Misma lógica que BTC pero con slug eth-updown-5m-{window_ts}.
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        window_ts = (now_ts // 300) * 300
        slug = f"eth-updown-5m-{window_ts}"
        return await self._fetch_market_by_slug(slug)

    async def _fetch_sol_market(self) -> Optional[dict]:
        """
        v3: Busca mercado SOL 5-min de la ventana actual.
        Correlación BTC-SOL = 0.83 (menor que ETH 0.87).
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        window_ts = (now_ts // 300) * 300
        slug = f"sol-updown-5m-{window_ts}"
        return await self._fetch_market_by_slug(slug)

    async def get_btc_volume_ratio(self, window_ts: int) -> float:
        """
        Ratio volumen_actual / volumen_promedio_últimas_6_ventanas.
        Ratio > 1.5 indica aceleración de volumen → señal más fuerte.
        Retorna 1.0 si no puede calcular (fail-safe).
        """
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": "5m",
                        "endTime": (window_ts + 300) * 1000, "limit": 7},
                timeout=4.0,
            )
            resp.raise_for_status()
            klines = resp.json()
            if len(klines) < 2:
                return 1.0
            # Índice 5 = quote asset volume (más estable que base volume)
            volumes = [float(k[5]) for k in klines if len(k) > 5]
            if len(volumes) < 2:
                return 1.0
            current_vol = volumes[-1]           # ventana actual
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1])  # promedio anteriores
            return round(current_vol / max(1.0, avg_vol), 2)
        except Exception as e:
            logger.debug(f"[Rush] volume ratio error: {e}")
            return 1.0

    # ── Precio BTC al inicio de la ventana ────────────────────

    async def get_btc_window_open(self, window_ts: int) -> Optional[float]:
        """
        Precio de apertura de BTC al inicio de la ventana de 5 min.
        Usa klines de 5 minutos de Binance; cachea por window_ts.
        """
        if window_ts in self._btc_open_cache:
            return self._btc_open_cache[window_ts]
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "5m",
                    "startTime": window_ts * 1000,
                    "limit": 1,
                },
                timeout=4.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and len(data[0]) > 1:
                open_price = float(data[0][1])  # índice 1 = open
                self._cache_set(window_ts, open_price)
                return open_price
        except Exception as e:
            logger.debug(f"[Rush] BTC window open error: {e}")
        return None

    # ── Precio de cierre para resolución ──────────────────────

    async def check_btc_resolution(
        self, window_ts: int, side: str, btc_open: float
    ) -> Optional[str]:
        """
        Verifica si un bet ganó o perdió comparando el precio de cierre
        de la vela de 5 min vs el precio de apertura.
        Returns: "won", "lost", o None si aún no cerró.
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        window_end = window_ts + 300
        if now_ts < window_end + 5:   # dar 5s extra para que cierre la vela
            return None

        try:
            resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "5m",
                    "startTime": window_ts * 1000,
                    "limit": 1,
                },
                timeout=4.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and len(data[0]) > 4:
                close_price = float(data[0][4])   # índice 4 = close
                if close_price == btc_open:
                    return None   # empate exacto → esperar resolución del oracle
                if side == "UP":
                    return "won" if close_price > btc_open else "lost"
                else:  # DOWN
                    return "won" if close_price < btc_open else "lost"
        except Exception as e:
            logger.debug(f"[Rush] Resolution check error: {e}")
        return None

    # ── Mercado por slug ───────────────────────────────────────

    async def _fetch_market_by_slug(self, slug: str) -> Optional[dict]:
        """Busca mercado por slug exacto, con fallback por búsqueda de texto."""
        # 1. Intento con slug exacto
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
            logger.debug(f"[Rush] fetch slug exacto {slug}: {e}")

        # 2. Fallback: buscar mercados BTC/ETH/SOL 5-min que cierren pronto
        # (por si el slug tiene formato diferente o timestamp desfasado)
        try:
            asset = "btc" if "btc" in slug else ("eth" if "eth" in slug else "sol")
            now_ts = int(datetime.now(timezone.utc).timestamp())
            window_ts_local = (now_ts // 300) * 300
            target_end_ts = window_ts_local + 300

            resp2 = await self._http.get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "limit": 50, "order": "end_date_min"},
                timeout=6.0,
            )
            resp2.raise_for_status()
            data2 = resp2.json()
            markets = data2 if isinstance(data2, list) else data2.get("markets", [])

            best_match = None
            best_diff  = 999
            for m in markets:
                m_slug = (m.get("slug") or "").lower()
                end_str = m.get("endDate") or m.get("end_date_min", "")
                if asset not in m_slug:
                    continue
                if not ("updown" in m_slug or "up-down" in m_slug):
                    continue
                if not ("5m" in m_slug or "5min" in m_slug or "5-min" in m_slug):
                    continue
                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    diff = abs(end_dt.timestamp() - target_end_ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = m
                except Exception:
                    pass

            if best_match and best_diff < 120:  # dentro de 2 minutos del target
                logger.info(f"[Rush] Fallback match: slug={best_match.get('slug')} diff={best_diff:.0f}s")
                return best_match
        except Exception as e2:
            logger.debug(f"[Rush] fallback market search error: {e2}")

        return None

    def _current_window(self) -> tuple[str, int]:
        """
        Retorna (slug, window_ts) de la ventana de 5 min actual.
        Los mercados BTC 5-min de Polymarket usan como slug:
        btc-updown-5m-{timestamp_inicio_ventana}
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        window_ts = (now_ts // 300) * 300
        return f"btc-updown-5m-{window_ts}", window_ts

    # ── Asymmetric Return Optimization ────────────────────────

    @staticmethod
    def classify_return_zone(signal_price: float) -> str:
        """
        Clasifica el precio del lado apostado en una zona de retorno asimétrico.
        La zona "premium" (0.20-0.42) tiene retorno 2.4x-5x y Kelly ampliado.
        Precios > 0.72 se rechazan: retorno < 1.4x no compensa ejecución.
        """
        if signal_price < 0.10:
            return "rejected"   # < 10¢: sin liquidez en CLOB, fill imposible
        elif signal_price < 0.20:
            return "ultra"      # 5x-10x retorno, Kelly +50% (alta asimetría)
        elif signal_price <= 0.40:
            return "premium"    # 2.5x-5x retorno, Kelly +30%
        elif signal_price <= 0.60:
            return "good"       # 1.7x-2.5x retorno, Kelly normal
        elif signal_price <= 0.78:
            return "marginal"   # 1.3x-1.7x retorno, Kelly -30%
        else:
            return "rejected"   # > 78¢: retorno < 1.28x, sin edge

    @staticmethod
    def compute_rush_kelly(signal_price: float, signal_strength: float,
                           rush_capital: float = 3.0) -> tuple:
        """
        Calcula el tamaño óptimo de apuesta usando Kelly Criterion asimétrico.

        En mercados binarios:
          b = (1/price - 1)   ← ganancia neta por dólar apostado (retorno bruto - 1)
          p = model_probability ← estimado desde signal_strength (rango 0.52-0.70)
          f = (p*b - q) / b   ← fracción Kelly pura

        Ajustes:
          - Cap en RUSH_KELLY_CAP = 35% para evitar ruina
          - Modificador por zona: premium +20%, good ×1, marginal -25%
          - Mínimo RUSH_KELLY_MIN = 5% para que valga la ejecución

        Returns: (kelly_fraction, kelly_bet_usd, model_prob)
        """
        # BUG 3 fix: evitar phantom edge cuando la señal es casi inexistente.
        # Antes: signal_strength=0 → p=0.52 (2% edge artificial garantizado).
        # Ahora: si signal_strength < 0.05, no hay edge real → Kelly = 0.
        if signal_strength < 0.05:
            # Señal casi nula: usar signal_price como probabilidad → Kelly cero
            p = signal_price
        else:
            # Mapeo lineal real: [0.05, 1.0] → edge verdadero sobre precio de mercado
            # Mínimo edge real 1% sobre signal_price, máximo al 85%
            p = signal_price + signal_strength * 0.18
            p = max(signal_price + 0.01, min(0.85, p))
        # Probabilidad estimada del modelo: signal_strength 0→1 mapea a prob 0.52→0.70 (legado)
        # p   = 0.52 + signal_strength * 0.18  ← REEMPLAZADO (ver arriba)
        q   = 1.0 - p
        b   = max(0.01, (1.0 / max(0.01, signal_price)) - 1.0)
        raw = (p * b - q) / b

        if raw <= 0:
            return 0.0, 0.0, round(p, 4)

        zone     = RushEngine.classify_return_zone(signal_price)
        zone_mod = {"ultra": 1.50, "premium": 1.30, "good": 1.00, "marginal": 0.70}.get(zone, 0.0)

        if zone_mod == 0.0:
            return 0.0, 0.0, round(p, 4)

        kelly_frac = min(raw, RUSH_KELLY_CAP) * zone_mod
        if kelly_frac < RUSH_KELLY_MIN:
            return 0.0, 0.0, round(p, 4)

        kelly_usd = round(kelly_frac * rush_capital, 4)
        return round(kelly_frac, 4), kelly_usd, round(p, 4)

    async def _build_derived_opp(
        self,
        asset: str,
        fetch_fn,
        correlation: float,
        min_confidence: float,
        agreed_side: str,
        combined_strength: float,
        window_ts: int,
        seconds_left: float,
        btc_current: float,
        btc_open: float,
        btc_change_pct: float,
        poly_signal: str,
        btc_signal: str,
        vol_ratio: float,
        momentum_type: str,
        timing_q: str,
        rush_capital: float,
    ) -> "Optional[RushOpportunity]":
        """Genera oportunidad para un activo derivado (ETH/SOL) basado en señal BTC."""
        try:
            mkt = await fetch_fn()
            if not mkt:
                return None
            prices_raw = mkt.get("outcomePrices", "")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if not prices or len(prices) < 2:
                return None
            up_p = max(0.01, min(0.99, float(prices[0])))
            dn_p = max(0.01, min(0.99, float(prices[1])))
            sig = up_p if agreed_side == "UP" else dn_p
            conf = round(combined_strength * correlation, 3)
            if conf < min_confidence:
                return None
            zone = self.classify_return_zone(sig)
            if zone == "rejected":
                return None
            tokens_raw = mkt.get("clobTokenIds", "")
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            tok = ""
            if isinstance(tokens, list) and len(tokens) >= 2:
                _raw_tok = tokens[0] if agreed_side == "UP" else tokens[1]
                tok = str(_raw_tok.get("token_id") or _raw_tok.get("tokenId") or "") if isinstance(_raw_tok, dict) else str(_raw_tok or "")
            kf, ku, mp = self.compute_rush_kelly(sig, conf, rush_capital)
            cid = mkt.get("conditionId", mkt.get("id", ""))
            slug_prefix = {"ETH": "eth-updown-5m", "SOL": "sol-updown-5m"}.get(asset, f"{asset.lower()}-updown-5m")
            return RushOpportunity(
                market_id=cid,
                question=mkt.get("question", f"{asset} Up or Down 5min?"),
                slug=f"{slug_prefix}-{window_ts}",
                window_ts=window_ts,
                up_price=round(up_p, 4),
                down_price=round(dn_p, 4),
                side=agreed_side,
                signal_price=round(sig, 4),
                signal_strength=conf,
                seconds_remaining=round(seconds_left, 1),
                token=tok,
                potential_return=round(1.0 / max(0.01, sig), 3),
                btc_current=round(btc_current, 2),
                btc_open=round(btc_open, 2),
                btc_change_pct=round(btc_change_pct * 100, 4),
                poly_signal=poly_signal,
                btc_signal=btc_signal,
                btc_volume_ratio=vol_ratio,
                signal_quality="normal",
                return_zone=zone,
                kelly_fraction=kf,
                kelly_bet_usd=ku,
                model_prob=mp,
                momentum_type=momentum_type,
                momentum_multiplier=round(correlation, 3),
                timing_quality=timing_q,
                asset=asset,
            )
        except Exception as e:
            logger.debug(f"[Rush] {asset} derived opp error: {e}")
            return None

    # ── Pipeline principal ─────────────────────────────────────

    async def find_rush_opportunities(
        self,
        min_seconds: int = RUSH_MIN_SECONDS,
        max_seconds: int = RUSH_MAX_SECONDS,
        optimize_timing: bool = True,
        lax_mode: bool = False,
        btc_momentum_override: float = 0.0,
        poly_high_override: float = 0.0,
        poly_low_override: float = 0.0,
        zone_filter_disabled: bool = False,
    ) -> tuple[list["RushOpportunity"], str]:
        """
        Busca el mercado BTC 5-min actual. Solo apuesta si:
        1. Faltan entre min_seconds y max_seconds para cerrar.
        2. Polymarket crowd dice UP (>60%) o DOWN (<40%).
        3. BTC momentum desde apertura de ventana coincide con la señal.

        lax_mode: umbrales más bajos, señal única permitida.
        btc_momentum_override: umbral BTC personalizado (0=usar default/lax).
        poly_high_override: umbral Poly alto personalizado.
        poly_low_override: umbral Poly bajo personalizado.
        zone_filter_disabled: deshabilitar filtro de zona de retorno.

        Returns: (lista de oportunidades, mensaje diagnóstico para el log)
        """
        # Determinar umbrales efectivos
        _btc_momentum_thr = (btc_momentum_override if btc_momentum_override > 0
                              else (0.0005 if lax_mode else RUSH_BTC_MOMENTUM))
        _poly_high = (poly_high_override if poly_high_override > 0
                      else (0.55 if lax_mode else RUSH_MARKET_HIGH))
        _poly_low  = (poly_low_override if poly_low_override > 0
                      else (0.45 if lax_mode else RUSH_MARKET_LOW))

        now = datetime.now(timezone.utc)
        slug, window_ts = self._current_window()

        mkt = await self._fetch_market_by_slug(slug)
        if not mkt:
            return [], f"Mercado no encontrado: slug={slug} (verificar formato Polymarket)"

        # ── Tiempo restante ──────────────────────────────────
        end_date_str = mkt.get("endDate", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                end_dt = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc)
        else:
            end_dt = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc)

        seconds_left = (end_dt - now).total_seconds()

        if not (min_seconds <= seconds_left <= max_seconds):
            if seconds_left > max_seconds:
                return [], f"Esperando ventana ({seconds_left:.0f}s restantes, entra a {max_seconds}s)"
            else:
                return [], f"Ventana cerrada ({seconds_left:.0f}s — ya pasó)"

        # ── Precios Polymarket (Up = índice 0, Down = índice 1) ──
        prices_raw = mkt.get("outcomePrices", "")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        except (json.JSONDecodeError, TypeError):
            return [], "Error parseando precios de Polymarket"

        if not prices or len(prices) < 2:
            return [], "Sin precios en el mercado"

        try:
            up_price   = max(0.01, min(0.99, float(prices[0])))
            down_price = max(0.01, min(0.99, float(prices[1])))
        except (ValueError, TypeError):
            return [], "Precios inválidos en el mercado"

        # ── Señal 1: BTC momentum (SEÑAL PRIMARIA) ──────────────
        btc_current = await self.get_btc_price()
        btc_open    = await self.get_btc_window_open(window_ts)

        if not btc_current or not btc_open or btc_open <= 0:
            return [], "Sin datos BTC (error Binance)"

        btc_change_pct = (btc_current - btc_open) / btc_open
        btc_abs = abs(btc_change_pct)

        if btc_change_pct >= _btc_momentum_thr:
            btc_signal = "UP"
        elif btc_change_pct <= -_btc_momentum_thr:
            btc_signal = "DOWN"
        else:
            btc_signal = "NEUTRAL"

        # BTC fuerte: umbral para entrar sin necesidad de confirmación Poly
        _btc_strong_thr = RUSH_BTC_STRONG  # 0.18% — señal BTC muy clara
        btc_is_strong = btc_abs >= _btc_strong_thr

        # ── Señal 2: Polymarket crowd (CONFIRMACIÓN) ────────────
        if up_price > _poly_high:
            poly_signal = "UP"
        elif up_price < _poly_low:
            poly_signal = "DOWN"
        else:
            poly_signal = "NEUTRAL"

        # Sesgo Poly: qué tan lejos está del 50% (0.0 = neutral, 1.0 = certeza)
        poly_bias = abs(up_price - 0.5) / 0.5
        # Señal Poly débil: precio tiene ligero sesgo aunque no cruce umbral
        _poly_weak_thr = 0.04  # ≥4% de sesgo = 52%/48% — señal muy débil pero detectable
        poly_has_weak_bias = (
            poly_bias >= _poly_weak_thr and
            ((up_price > 0.5 and btc_signal == "UP") or
             (up_price < 0.5 and btc_signal == "DOWN"))
        )

        # ── Lógica de combinación (BTC primario, Poly confirmación) ──
        poly_strength = poly_bias
        signal_is_strong = False
        agreed_side = None

        if btc_is_strong:
            # BTC mueve ≥0.18%: entrar en la dirección de BTC siempre
            agreed_side = btc_signal
            signal_is_strong = True
            logger.debug(f"[Rush] BTC fuerte ({btc_change_pct*100:+.3f}%) → entrando directo")

        elif btc_signal != "NEUTRAL" and poly_signal != "NEUTRAL":
            if btc_signal == poly_signal:
                # Ambas coinciden → señal fuerte
                agreed_side = btc_signal
                signal_is_strong = True
            else:
                # Contradicción: en modo lax usar la más fuerte
                if lax_mode:
                    _btc_str = min(1.0, btc_abs / 0.003)
                    agreed_side = btc_signal if _btc_str >= poly_bias else poly_signal
                    signal_is_strong = False
                else:
                    return [], (
                        f"Señales contradictorias: BTC={btc_signal} ({btc_change_pct*100:+.3f}%) "
                        f"vs Poly={poly_signal} (Up={up_price*100:.1f}%)"
                    )

        elif btc_signal != "NEUTRAL":
            # Solo BTC tiene señal, Poly neutral
            if lax_mode or poly_has_weak_bias:
                # Poly tiene ligero sesgo en la misma dirección → suficiente
                agreed_side = btc_signal
                signal_is_strong = False
                logger.debug(f"[Rush] BTC+Poly-débil: BTC={btc_change_pct*100:+.3f}% Up={up_price*100:.1f}%")
            else:
                # Sin confirmación Poly — aceptar si BTC supera umbral medio (0.10%)
                _btc_medium_thr = max(_btc_momentum_thr * 1.5, 0.001)  # 1.5× el mínimo
                if btc_abs >= _btc_medium_thr:
                    agreed_side = btc_signal
                    signal_is_strong = False
                    logger.debug(f"[Rush] Solo BTC medio ({btc_change_pct*100:+.3f}% ≥ {_btc_medium_thr*100:.3f}%)")
                else:
                    return [], (
                        f"BTC ({btc_change_pct*100:+.3f}%) insuficiente sin Poly "
                        f"(Up={up_price*100:.1f}%, necesita ≥{_btc_medium_thr*100:.3f}% o Poly>{_poly_high*100:.0f}%)"
                    )

        elif poly_signal != "NEUTRAL":
            # Solo Poly tiene señal, BTC neutral.
            # Contradicción real: BTC debe moverse al menos 50% del umbral en dirección opuesta.
            # Movimientos menores son ruido y no deben cancelar una señal Poly clara.
            # Ejemplo: BTC=-0.037% con threshold=0.10% → no contradice (0.037 < 0.05).
            #          BTC=-0.08% con threshold=0.10% → sí contradice (0.08 >= 0.05).
            _contra_thr = _btc_momentum_thr * 0.5
            _btc_contradicts_poly = (
                (btc_change_pct < -_contra_thr and poly_signal == "UP") or
                (btc_change_pct >  _contra_thr and poly_signal == "DOWN")
            )
            if _btc_contradicts_poly and not lax_mode:
                return [], (
                    f"BTC contradice a Poly: BTC={btc_change_pct*100:+.3f}% apunta "
                    f"{'DOWN' if btc_change_pct < 0 else 'UP'} pero Poly={poly_signal} "
                    f"(Up={up_price*100:.1f}%) — sin señal limpia"
                )
            if lax_mode:
                agreed_side = poly_signal
                signal_is_strong = False
            elif poly_bias >= 0.12:  # Poly muy claro (≥56% ó ≤44%)
                agreed_side = poly_signal
                signal_is_strong = False
                logger.debug(f"[Rush] Solo Poly fuerte: Up={up_price*100:.1f}% bias={poly_bias:.3f}")
            else:
                return [], (
                    f"Solo Poly ({poly_signal} Up={up_price*100:.1f}%) — BTC plano ({btc_change_pct*100:+.3f}%) "
                    f"y consenso Poly insuficiente (bias={poly_bias:.2f}<0.12)"
                )

        else:
            # Ambos neutrales
            if lax_mode and btc_abs > 0.0003:
                agreed_side = "UP" if btc_change_pct > 0 else "DOWN"
                signal_is_strong = False
            else:
                return [], (
                    f"Sin señal: BTC={btc_change_pct*100:+.3f}% (necesita ≥±{_btc_momentum_thr*100:.3f}%) "
                    f"Poly Up={up_price*100:.1f}% (necesita >{_poly_high*100:.0f}% o <{_poly_low*100:.0f}%)"
                )

        if agreed_side is None:
            return [], "Sin lado acordado — señales insuficientes"

        # ── Ventana ajustada (solo en modo estricto con señal débil) ─
        if optimize_timing and not signal_is_strong and not lax_mode:
            tight = max(min_seconds, RUSH_TIGHT_WINDOW)
            if seconds_left > tight:
                return [], (
                    f"Señal débil — confirmando en {seconds_left:.0f}s→{tight}s "
                    f"(BTC={btc_change_pct*100:+.3f}%, Up={up_price*100:.1f}%)"
                )

        # ── Tokens ──────────────────────────────────────────
        tokens_raw = mkt.get("clobTokenIds", "")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
        except (json.JSONDecodeError, TypeError):
            tokens = []

        def _extract_token(t) -> str:
            """Extrae token_id como string — Polymarket devuelve string o dict con token_id."""
            if isinstance(t, dict):
                return str(t.get("token_id") or t.get("tokenId") or "")
            return str(t) if t else ""
        up_token   = _extract_token(tokens[0]) if len(tokens) > 0 else ""
        down_token = _extract_token(tokens[1]) if len(tokens) > 1 else ""

        # FIX 5: validar token no vacío antes de continuar
        _token_to_use = up_token if agreed_side == "UP" else down_token
        if not _token_to_use:
            logger.warning(f"[Rush] Token vacío para {agreed_side} en {mkt.get('slug','?')} — intentando fallback")
            # Fallback: intentar otros campos del mercado
            alt_tokens = mkt.get("tokens", mkt.get("clobTokenIds", ""))
            if isinstance(alt_tokens, str):
                try:
                    alt_tokens = json.loads(alt_tokens)
                except Exception:
                    alt_tokens = []
            if isinstance(alt_tokens, list) and len(alt_tokens) >= 2:
                up_token   = alt_tokens[0] if not up_token   else up_token
                down_token = alt_tokens[1] if not down_token else down_token
                _token_to_use = up_token if agreed_side == "UP" else down_token
            if not _token_to_use:
                return [], f"Token vacío para lado {agreed_side} — mercado {mkt.get('slug','?')} puede no soportar CLOB"

        # ── Filtro de precio: Return Zone Filter ──────────────────────────
        # Zonas de retorno asimétrico: rechaza precios > 72¢ (retorno < 1.4x)
        # y precios < 20¢ (sin liquidez). Sweet spot: 20-42¢ (zona premium).
        # En lax mode / zone_filter_disabled: permitir rango 10-90¢
        raw_signal_price = up_price if agreed_side == "UP" else down_price
        _return_zone = RushEngine.classify_return_zone(raw_signal_price)
        if _return_zone == "rejected":
            if zone_filter_disabled or lax_mode:
                # En lax mode: permitir rango 0.10-0.90
                if raw_signal_price < 0.10 or raw_signal_price > 0.90:
                    return [], (
                        f"[LAX] Fuera de zona ampliada ({raw_signal_price*100:.1f}¢) — "
                        f"mínimo 10¢, máximo 90¢"
                    )
                _return_zone = "good"  # Tratar como zona normal en lax mode
            else:
                return [], (
                    f"Fuera de zona de retorno ({raw_signal_price*100:.1f}¢ — zona:{_return_zone}) — "
                    f"retorno {round(1/max(0.01,raw_signal_price),2)}x fuera del rango óptimo 20-78¢"
                )

        if agreed_side == "UP":
            signal_price  = up_price
            token         = up_token
            if poly_signal != "NEUTRAL":
                poly_strength = (up_price - _poly_high) / (1.0 - _poly_high)
            elif lax_mode:
                # En lax mode sin señal Poly: usar poly_strength reducida
                poly_strength = abs(up_price - 0.5) * 0.7
        else:
            signal_price  = down_price
            token         = down_token
            if poly_signal != "NEUTRAL":
                poly_strength = (_poly_low - up_price) / _poly_low
            elif lax_mode:
                poly_strength = abs(up_price - 0.5) * 0.7

        # FIX 6: Fuerza combinada ponderada — BTC momentum más inmediato que Poly (lag manual)
        btc_strength = min(1.0, abs(btc_change_pct) / 0.003)  # 0.3% = máximo
        if signal_is_strong:
            # Ambas señales coinciden: BTC pesa 60%, Poly 40% (BTC más inmediato y reactivo)
            combined_strength = round(btc_strength * 0.60 + poly_strength * 0.40, 3)
        elif btc_signal != "NEUTRAL":
            # Solo señal BTC (Poly neutral o lax): usar BTC directamente con leve descuento
            combined_strength = round(btc_strength * 0.80, 3)
        elif poly_signal != "NEUTRAL":
            # Solo señal Poly (BTC neutral): menos confiable — descuento mayor
            combined_strength = round(poly_strength * 0.60, 3)
        else:
            # Lax mode con movimiento sub-umbral: señal muy débil
            combined_strength = round(max(btc_strength, poly_strength) * 0.50, 3)

        # ── Volumen BTC — señal de confirmación adicional ────────────────
        vol_ratio = await self.get_btc_volume_ratio(window_ts)
        if vol_ratio >= 1.5:
            combined_strength = min(1.0, combined_strength * 1.1)
        elif vol_ratio < 0.7:
            combined_strength = combined_strength * 0.9

        # ── v2: Multi-period momentum (últimos 2 candles 1-min) ─────────────
        # Investigación: momentum sostenido tiene win rate ~72% vs ~63% single-period.
        _momentum_data = await self.get_btc_1min_momentum()
        _mt = _momentum_data["type"]
        _mm = _momentum_data["multiplier"]
        # Aplicar multiplicador al combined_strength (cap en 1.0)
        combined_strength = round(min(1.0, combined_strength * _mm), 3)
        # Si momentum es MIXED con señal, reducir pero no eliminar (puede haber otra fuente)
        # En lax mode: no cancelar por momentum mixto, solo reducir fuerza
        if _mt == "MIXED" and not signal_is_strong:
            if lax_mode:
                combined_strength = round(combined_strength * 0.5, 3)  # Penalizar en lax
            else:
                return [], (
                    f"Momentum mixto sin confirmación Poly ({_momentum_data['c1_pct']:+.3f}% / "
                    f"{_momentum_data['c2_pct']:+.3f}%) — señal cancelada"
                )

        # ── v2: Timing quality (Chainlink lag compensation) ─────────────────
        timing_q = RushEngine.classify_timing_quality(seconds_left)
        if timing_q == "critical":
            return [], f"Timing crítico ({seconds_left:.0f}s) — fill imposible en CLOB"

        # Clasificar calidad de señal
        _is_accelerating = _mt in ("ACCELERATING_UP", "ACCELERATING_DOWN")
        _is_sustained    = _mt in ("SUSTAINED_UP", "SUSTAINED_DOWN")
        if signal_is_strong and (_is_accelerating or (_is_sustained and vol_ratio >= 1.3)):
            signal_quality = "strong"
        elif signal_is_strong and abs(btc_change_pct) >= 0.002:
            signal_quality = "normal"
        elif signal_is_strong:
            signal_quality = "normal"
        else:
            signal_quality = "weak"

        cid      = mkt.get("conditionId", mkt.get("id", ""))
        question = mkt.get("question", slug)

        quality_label = {"strong": "FUERTE", "normal": "NORMAL", "weak": "DEBIL"}.get(signal_quality, "")

        # ── Kelly sizing asimétrico (v2: incorpora multiplicador de momentum) ──
        # BUG 1 fix: usar capital configurado por el usuario, no 3.0 hardcodeado.
        # _rush_max_capital es la variable global configurable via POST /api/rush/config.
        _rush_capital_avail = _rush_max_capital  # Usar capital configurado por el usuario

        # BUG 2 fix: actualizar correlaciones dinámicas máximo cada hora (throttle).
        import time as _time_corr
        if (_time_corr.time() - self._last_corr_update) > 3600:
            try:
                await self.update_dynamic_correlations()
                self._last_corr_update = _time_corr.time()
            except Exception as _ce:
                logger.debug(f"[Rush] Correlación update error: {_ce}")

        kelly_frac, kelly_usd, model_p = RushEngine.compute_rush_kelly(
            raw_signal_price, combined_strength, _rush_capital_avail
        )
        return_zone = RushEngine.classify_return_zone(raw_signal_price)
        zone_label  = {"premium": "PREMIUM", "good": "BUENA", "marginal": "MARGINAL"}.get(return_zone, "?")

        btc_opp = RushOpportunity(
            market_id=cid,
            question=question,
            slug=slug,
            window_ts=window_ts,
            up_price=round(up_price, 4),
            down_price=round(down_price, 4),
            side=agreed_side,
            signal_price=round(raw_signal_price, 4),
            signal_strength=combined_strength,
            seconds_remaining=round(seconds_left, 1),
            token=token,
            potential_return=round(1.0 / max(0.01, raw_signal_price), 3),
            btc_current=round(btc_current, 2),
            btc_open=round(btc_open, 2),
            btc_change_pct=round(btc_change_pct * 100, 4),
            poly_signal=poly_signal,
            btc_signal=btc_signal,
            btc_volume_ratio=vol_ratio,
            signal_quality=signal_quality,
            return_zone=return_zone,
            kelly_fraction=kelly_frac,
            kelly_bet_usd=kelly_usd,
            model_prob=model_p,
            momentum_type=_mt,
            momentum_multiplier=_mm,
            timing_quality=timing_q,
            asset="BTC",
        )
        opps = [btc_opp]

        # ── v2: ETH market (correlación BTC-ETH, dinámica si disponible) ──────
        # Si señal BTC es fuerte, buscar mercado ETH de la misma ventana.
        if RUSH_ETH_ENABLED and signal_is_strong and combined_strength >= 0.45:
            eth_corr = self._dynamic_correlations.get("ETH", RUSH_ETH_CORRELATION)
            eth_opp = await self._build_derived_opp(
                "ETH", self._fetch_eth_market, eth_corr, RUSH_ETH_MIN_CONFIDENCE,
                agreed_side, combined_strength, window_ts, seconds_left,
                btc_current, btc_open, btc_change_pct, poly_signal, btc_signal,
                vol_ratio, _mt, timing_q, _rush_capital_avail
            )
            if eth_opp:
                opps.append(eth_opp)

        # ── v3: SOL market (correlación BTC-SOL, dinámica si disponible) ──────
        # Solo si BTC fuerte y SOL habilitado (threshold ligeramente menor que ETH).
        if RUSH_SOL_ENABLED and signal_is_strong and combined_strength >= 0.46:
            sol_corr = self._dynamic_correlations.get("SOL", RUSH_SOL_CORRELATION)
            sol_opp = await self._build_derived_opp(
                "SOL", self._fetch_sol_market, sol_corr, RUSH_SOL_MIN_CONFIDENCE,
                agreed_side, combined_strength, window_ts, seconds_left,
                btc_current, btc_open, btc_change_pct, poly_signal, btc_signal,
                vol_ratio, _mt, timing_q, _rush_capital_avail
            )
            if sol_opp:
                opps.append(sol_opp)

        momentum_label = {"ACCELERATING_UP": "ACELERANDO↑", "ACCELERATING_DOWN": "ACELERANDO↓",
                          "SUSTAINED_UP": "SOSTENIDO↑", "SUSTAINED_DOWN": "SOSTENIDO↓",
                          "SINGLE_UP": "ÚNICO↑", "SINGLE_DOWN": "ÚNICO↓", "MIXED": "MIXTO"}.get(_mt, _mt)
        assets_note = "".join(f" +{o.asset}" for o in opps[1:])
        lax_tag = " [LAX]" if lax_mode else ""
        return opps, (f"✓ Señal{lax_tag} {quality_label} {momentum_label} zona:{zone_label} "
                      f"{raw_signal_price*100:.1f}¢→{round(1/max(0.01,raw_signal_price),2)}x "
                      f"Kelly=${kelly_usd:.2f} timing:{timing_q}{assets_note}")

    # ── Correlaciones dinámicas BTC-ETH / BTC-SOL ─────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # MODO CONTRARIO (MEAN REVERSION) — estrategia ddchack
    # ─────────────────────────────────────────────────────────────────────────────

    async def get_btc_rsi(self, period: int = 14, interval: str = "1m",
                          limit: int = 50) -> Optional[float]:
        """
        Calcula RSI de BTC usando klines de Binance (suavizado Wilder).
        RSI > 62 = sobrecomprado  → señal contraria DOWN
        RSI < 38 = sobrevendido   → señal contraria UP
        """
        try:
            resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
                timeout=5.0,
            )
            resp.raise_for_status()
            klines = resp.json()
            if len(klines) < period + 1:
                return None
            closes = [float(k[4]) for k in klines]
            deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            gains  = [max(0.0, d) for d in deltas]
            losses = [max(0.0, -d) for d in deltas]
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss < 1e-10:
                return 100.0
            rs = avg_gain / avg_loss
            return round(100.0 - 100.0 / (1.0 + rs), 2)
        except Exception as e:
            logger.debug(f"[Rush] RSI error: {e}")
            return None

    async def find_contrarian_opportunities(
        self,
        min_seconds: int = 60,
        max_seconds: int = 285,
        rsi_overbought: float = 55.0,   # v3: bajado de 58 → captura sobrecompra moderada
        rsi_oversold:   float = 45.0,   # v3: subido de 42 → captura sobrevendido moderado
        max_contrarian_price: float = 0.56,
        min_contrarian_price: float = 0.25,
    ) -> "tuple[list[RushOpportunity], str]":
        """
        Detecta oportunidades de mean-reversion en BTC 5-min.

        Estrategia: entra al INICIO del mercado (210-290s restantes = recién abierto).
        BTC se movió en una dirección y RSI confirma sobreextensión.
        Apuesta al lado PERDEDOR (precio 25-47¢, retorno 2.1x-4x) esperando
        que el movimiento revierta durante los ~3-4 minutos que quedan.
        Sale parcialmente cuando el precio sube ~17¢ (≈$1.50 por cada $1 invertido a 35¢).

        Ventana de entrada: 90-240s antes del cierre (tiempo suficiente para reversión).
        """
        now      = datetime.now(timezone.utc)
        slug, window_ts = self._current_window()

        mkt = await self._fetch_market_by_slug(slug)
        if not mkt:
            return [], f"Mercado no encontrado: {slug}"

        # ── Tiempo restante ─────────────────────────────────────────────────
        end_date_str = mkt.get("endDate", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                end_dt = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc)
        else:
            end_dt = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc)

        seconds_left = (end_dt - now).total_seconds()
        if seconds_left > max_seconds:
            return [], f"Muy pronto para contraria ({seconds_left:.0f}s, entra a {max_seconds}s)"
        if seconds_left < min_seconds:
            return [], f"Muy tarde para contraria ({seconds_left:.0f}s, mínimo {min_seconds}s)"

        # ── Precios Polymarket ───────────────────────────────────────────────
        prices_raw = mkt.get("outcomePrices", "")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        except (json.JSONDecodeError, TypeError):
            return [], "Error parseando precios Polymarket"
        if not prices or len(prices) < 2:
            return [], "Sin precios en el mercado"
        try:
            up_price   = max(0.01, min(0.99, float(prices[0])))
            down_price = max(0.01, min(0.99, float(prices[1])))
        except (ValueError, TypeError):
            return [], "Precios inválidos"

        # ── BTC momentum en la ventana actual ───────────────────────────────
        btc_current = await self.get_btc_price()
        btc_open    = await self.get_btc_window_open(window_ts)
        if not btc_current or not btc_open or btc_open <= 0:
            return [], "Sin datos BTC"
        btc_change_pct = (btc_current - btc_open) / btc_open  # fracción decimal

        # ── Filtro de tendencia multi-ventana ────────────────────────────────
        # Si la ventana ANTERIOR también fue en la misma dirección → tendencia sostenida.
        # Tendencia bajista = dos ventanas consecutivas DOWN: no apostar UP contrarian.
        # Tendencia alcista = dos ventanas consecutivas UP: no apostar DOWN contrarian.
        # Umbral: cada ventana debe moverse >0.04% para contar como "tendencia".
        TREND_THRESHOLD = 0.0004  # 0.04% por ventana = ~$30 en BTC a $73k
        prev_window_open = await self.get_btc_window_open(window_ts - 300)
        if prev_window_open and prev_window_open > 0:
            # prev_window_change: de apertura de ventana anterior → apertura de ventana actual
            prev_window_change = (btc_open - prev_window_open) / prev_window_open
            _trend_down = prev_window_change < -TREND_THRESHOLD and btc_change_pct < -TREND_THRESHOLD
            _trend_up   = prev_window_change > TREND_THRESHOLD  and btc_change_pct > TREND_THRESHOLD
            if _trend_down:
                # Dos ventanas bajistas consecutivas: crowd puede seguir bajando. No apostar UP.
                _trend_str = f"ventana anterior {prev_window_change*100:.2f}%, actual {btc_change_pct*100:.2f}%"
                # Solo bloquear si la señal contraria sería UP (apostar contra la bajada)
                # Si la señal RSI dice DOWN tampoco tendría sentido bloquearla
                # → Guardar estado; aplicar después de determinar contrarian_side
                _btc_trending_down = True
                _btc_trending_up   = False
                _trend_msg         = f"Tendencia bajista 2V: {_trend_str}"
            elif _trend_up:
                _trend_str = f"ventana anterior {prev_window_change*100:.2f}%, actual {btc_change_pct*100:.2f}%"
                _btc_trending_down = False
                _btc_trending_up   = True
                _trend_msg         = f"Tendencia alcista 2V: {_trend_str}"
            else:
                _btc_trending_down = False
                _btc_trending_up   = False
                _trend_msg         = ""
        else:
            _btc_trending_down = False
            _btc_trending_up   = False
            _trend_msg         = ""

        # ── RSI: confirmar sobreextensión ────────────────────────────────────
        rsi = await self.get_btc_rsi(period=14, interval="1m", limit=50)
        if rsi is None:
            return [], "RSI no disponible"

        # v3: dos caminos de señal —
        #   A) RSI-primario: BTC sobreextendido (RSI>55 ó <45) → señal directa
        #   B) Poly-puro: BTC lateral (<0.05%) pero crowd empujó ≥4¢ sin fundamento → reversion
        ALREADY_REVERSED  = 0.001   # BTC ya revirtió ≥0.10% → oportunidad pasó
        BTC_LATERAL_PCT   = 0.0005  # < 0.05% = BTC lateral (sin dirección clara)
        POLY_ONLY_BIAS    = 0.04    # crowd ≥4¢ (≥54¢ en un lado) para señal Poly-pura

        btc_is_lateral = abs(btc_change_pct) < BTC_LATERAL_PCT
        signal_mode    = "rsi"   # o "poly"

        if rsi >= rsi_overbought and btc_change_pct > -ALREADY_REVERSED:
            contrarian_side = "DOWN"   # RSI sobrecomprado + BTC no ha caído ya → apostar DOWN
            signal_price    = down_price
            rsi_tag         = f"sobrecomprado RSI:{rsi:.1f}"
        elif rsi <= rsi_oversold and btc_change_pct < ALREADY_REVERSED:
            contrarian_side = "UP"     # RSI sobrevendido + BTC no ha rebotado ya → apostar UP
            signal_price    = up_price
            rsi_tag         = f"sobrevendido RSI:{rsi:.1f}"
        elif btc_is_lateral and (0.50 - up_price) >= POLY_ONLY_BIAS:
            # Crowd apostó DOWN fuerte (DOWN≥54¢) pero BTC está plano → crowd sobrereaccionó → UP
            contrarian_side = "UP"
            signal_price    = up_price
            rsi_tag         = f"Poly-crowd DOWN:{down_price*100:.1f}¢, BTC lateral {btc_change_pct*100:+.3f}%"
            signal_mode     = "poly"
        elif btc_is_lateral and (0.50 - down_price) >= POLY_ONLY_BIAS:
            # Crowd apostó UP fuerte (UP≥54¢) pero BTC está plano → crowd sobrereaccionó → DOWN
            contrarian_side = "DOWN"
            signal_price    = down_price
            rsi_tag         = f"Poly-crowd UP:{up_price*100:.1f}¢, BTC lateral {btc_change_pct*100:+.3f}%"
            signal_mode     = "poly"
        else:
            reason = ""
            if rsi >= rsi_overbought:
                reason = f"RSI sobrecomprado ({rsi:.1f}) pero BTC ya cayó {btc_change_pct*100:+.3f}% (oportunidad DOWN pasó)"
            elif rsi <= rsi_oversold:
                reason = f"RSI sobrevendido ({rsi:.1f}) pero BTC ya rebotó {btc_change_pct*100:+.3f}% (oportunidad UP pasó)"
            elif btc_is_lateral:
                reason = f"RSI neutral ({rsi:.1f}), BTC lateral {btc_change_pct*100:+.3f}%, crowd {up_price*100:.1f}¢/{down_price*100:.1f}¢ insuficiente (necesita ≥54¢ en un lado)"
            else:
                reason = f"RSI neutral ({rsi:.1f}) — esperar RSI>{rsi_overbought} o RSI<{rsi_oversold}"
            return [], (
                f"Sin señal contraria — {reason} BTC:{btc_change_pct*100:+.3f}% "
                f"| Poly UP:{up_price*100:.1f}¢ DOWN:{down_price*100:.1f}¢ "
                f"({seconds_left:.0f}s restantes)"
            )

        # ── Bloquear si hay tendencia multi-ventana en dirección del bet ────────
        # Dos ventanas consecutivas DOWN → no apostar UP (tendencia, no reversión)
        # Dos ventanas consecutivas UP  → no apostar DOWN
        if contrarian_side == "UP" and _btc_trending_down:
            return [], f"⚠ {_trend_msg} → no apostar UP contra tendencia bajista confirmada"
        if contrarian_side == "DOWN" and _btc_trending_up:
            return [], f"⚠ {_trend_msg} → no apostar DOWN contra tendencia alcista confirmada"

        # ── Validar zona de precio del lado contrario ────────────────────────
        if not (min_contrarian_price <= signal_price <= max_contrarian_price):
            return [], (
                f"Precio contrario {signal_price * 100:.1f}¢ fuera de zona "
                f"({min_contrarian_price * 100:.0f}-{max_contrarian_price * 100:.0f}¢) "
                f"retorno:{round(1 / max(0.01, signal_price), 2)}x"
            )

        # ── Filtro de sesgo crowd ────────────────────────────────────────────
        # Señal RSI: mínimo 3¢ de sesgo (bajado de 5¢ → captura movimientos moderados).
        # Señal Poly-pura: ya tiene sesgo ≥4¢ incorporado en la condición de arriba.
        if signal_mode == "rsi":
            MIN_CROWD_BIAS_CENTS = 3  # v3: bajado de 5¢ a 3¢
            _crowd_bias = round(abs(signal_price - 0.50) * 100, 1)
            if signal_price > (0.50 - MIN_CROWD_BIAS_CENTS / 100):
                return [], (
                    f"Mercado sin sesgo ({up_price*100:.1f}¢/{down_price*100:.1f}¢) — "
                    f"lado {contrarian_side} a {signal_price*100:.1f}¢ solo {_crowd_bias:.1f}¢ bajo 50¢ "
                    f"(mínimo {MIN_CROWD_BIAS_CENTS}¢). RSI:{rsi:.1f} pero crowd neutral."
                )

        potential_return = round(1.0 / max(0.01, signal_price), 3)

        # ── Tokens ──────────────────────────────────────────────────────────
        tokens_raw = mkt.get("clobTokenIds", "")
        try:
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
        except (json.JSONDecodeError, TypeError):
            tokens = []

        def _ext(t) -> str:
            if isinstance(t, dict):
                return str(t.get("token_id") or t.get("tokenId") or "")
            return str(t) if t else ""

        up_token   = _ext(tokens[0]) if len(tokens) > 0 else ""
        down_token = _ext(tokens[1]) if len(tokens) > 1 else ""
        token = up_token if contrarian_side == "UP" else down_token
        if not token:
            return [], f"Token vacío para lado contrario {contrarian_side}"

        # ── Kelly conservador (contraria tiene menor confianza que señal directa) ─
        kelly_frac, kelly_usd, model_p = RushEngine.compute_rush_kelly(
            signal_price, 0.30, _rush_max_capital
        )

        cid = mkt.get("conditionId", mkt.get("id", ""))
        opp = RushOpportunity(
            market_id=cid,
            question=mkt.get("question", slug),
            slug=slug,
            window_ts=window_ts,
            up_price=round(up_price, 4),
            down_price=round(down_price, 4),
            side=contrarian_side,
            signal_price=round(signal_price, 4),
            signal_strength=0.30,
            seconds_remaining=round(seconds_left, 1),
            token=token,
            potential_return=potential_return,
            btc_current=round(btc_current, 2),
            btc_open=round(btc_open, 2),
            btc_change_pct=round(btc_change_pct * 100, 4),
            poly_signal="NEUTRAL",
            btc_signal="UP" if btc_change_pct > 0 else "DOWN",
            btc_volume_ratio=1.0,
            signal_quality="contrarian",
            return_zone=RushEngine.classify_return_zone(signal_price),
            kelly_fraction=kelly_frac,
            kelly_bet_usd=kelly_usd,
            model_prob=model_p,
            momentum_type="CONTRARIAN",
            momentum_multiplier=1.0,
            timing_quality=RushEngine.classify_timing_quality(seconds_left),
            asset="BTC",
        )

        return [opp], (
            f"🔄 Contraria {rsi_tag} BTC:{btc_change_pct * 100:+.3f}% → "
            f"BET {contrarian_side} @ {signal_price * 100:.1f}¢ "
            f"retorno:{potential_return:.2f}x {seconds_left:.0f}s"
        )

    async def update_dynamic_correlations(self) -> dict:
        """Calcula correlaciones BTC-ETH y BTC-SOL rolling 24h desde Binance."""
        try:
            btc_resp = await self._http.get(
                f"{BINANCE_API}/klines",
                params={"symbol": "BTCUSDT", "interval": "1h", "limit": 24},
                timeout=5.0
            )
            btc_resp.raise_for_status()
            btc_closes = [float(k[4]) for k in btc_resp.json()]
            if len(btc_closes) < 2:
                return {}
            btc_rets = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1] for i in range(1, len(btc_closes))]

            correlations = {}
            for asset, symbol in [("ETH", "ETHUSDT"), ("SOL", "SOLUSDT")]:
                try:
                    resp = await self._http.get(
                        f"{BINANCE_API}/klines",
                        params={"symbol": symbol, "interval": "1h", "limit": 24},
                        timeout=5.0
                    )
                    resp.raise_for_status()
                    closes = [float(k[4]) for k in resp.json()]
                    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                    n = min(len(btc_rets), len(rets))
                    if n >= 12:
                        x, y = btc_rets[-n:], rets[-n:]
                        mx, my = sum(x) / n, sum(y) / n
                        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
                        sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
                        sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
                        if sx > 1e-10 and sy > 1e-10:
                            corr = max(0.5, min(0.99, round(cov / (sx * sy), 4)))
                            correlations[asset] = corr
                            logger.info(f"[Rush] Correlación dinámica BTC-{asset}: {corr:.4f}")
                except Exception as e:
                    logger.debug(f"[Rush] Correlation {asset} error: {e}")

            if correlations:
                self._dynamic_correlations = correlations
            return correlations
        except Exception as e:
            logger.debug(f"[Rush] Dynamic correlation update error: {e}")
            return {}

    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Obtiene funding rate actual de Binance Futures. Positivo = longs pagan (sobrecomprado)."""
        try:
            resp = await self._http.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
                timeout=4.0
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return float(data[0]["fundingRate"])
        except Exception as e:
            logger.debug(f"[Rush] Funding rate {symbol} error: {e}")
        return None
