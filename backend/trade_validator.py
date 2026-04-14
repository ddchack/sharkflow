"""
══════════════════════════════════════════════════════════════
SharkFlow — TradeValidator
Pre-trade safety checks. All validations run BEFORE any order.
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent
BLOCKED_LOG = _DIR / "blocked_trades.log"

# ── Constantes ────────────────────────────────────────────────────────────────
CIRCUIT_BREAKER_THRESHOLD = 10  # bloqueos consecutivos → pausar bot
SPORTS_CACHE_TTL_SECONDS  = 300 # 5 minutos
POLYMARKET_MIN_BET        = 1.00
GAMMA_API_BASE            = "https://gamma-api.polymarket.com"

# Check 6: Liquidez mínima para evitar slippage destructivo
MIN_LIQUIDITY_USD         = 100.0   # Reducido para permitir mercados con menos liquidez
# Check 7: Volumen mínimo 24h — mercados sin actividad son más manipulables
MIN_VOLUME_24H_USD        = 10.0    # Reducido: mínimo volumen
# Check 8: Tiempo mínimo antes de resolución (evita entrar cuando el mercado ya casi cierra)
MIN_MINUTES_TO_RESOLUTION = 2.0    # Reducido: permite entrar hasta 2min antes
# Check 9: Anti-duplicados — ventana de exclusión por mercado
DEDUP_WINDOW_SECONDS      = 60     # 1 minuto en lugar de 5
# Check 10: Utilidad mínima — la ganancia potencial debe ser ≥20% del importe apostado
MIN_PROFIT_MARGIN         = 0.20   # 20% mínimo de retorno sobre el capital arriesgado
MAX_BET_PRICE             = round(1.0 / (1.0 + MIN_PROFIT_MARGIN), 4)  # 0.8333


class TradeValidator:
    """
    Validación pre-trade.
    Llama a validate_trade() ANTES de cualquier place_limit_order / place_market_order.
    """

    def __init__(self):
        self._lock = Lock()

        # Umbrales configurables
        self.MIN_PRICE: float   = 0.05   # 5¢
        self.MAX_PRICE: float   = 0.95   # 95¢
        self.MAX_SPREAD: float  = 0.90   # spread YES–NO
        self.MIN_KELLY_OVERRIDE = 0.10   # Reducido: el bot fuerza max(1.0, kelly) antes de llamar al validator

        # Umbrales — checks 6, 7, 8, 9
        self.MIN_LIQUIDITY_USD: float       = MIN_LIQUIDITY_USD
        self.MIN_VOLUME_24H_USD: float      = MIN_VOLUME_24H_USD
        self.MIN_MINUTES_TO_RESOLUTION: int = MIN_MINUTES_TO_RESOLUTION
        self.DEDUP_WINDOW_SECONDS: int      = DEDUP_WINDOW_SECONDS

        # Estado interno
        self._consecutive_blocked: int  = 0
        self._circuit_breaker_active: bool = False
        self._blocked_history: list[dict] = []  # últimas 500 entradas

        # Cache de resultados deportivos  {market_id: {"finished": bool, "ts": float}}
        self._sports_cache: dict[str, dict] = {}

        # Anti-duplicados: {market_id: timestamp_último_trade_aprobado}
        self._recent_trades: dict[str, float] = {}

        # Logger dedicado
        logging.basicConfig(level=logging.WARNING)
        self._log = logging.getLogger("trade_validator")

    # ══════════════════════════════════════════════════════════════════════
    # API PÚBLICA
    # ══════════════════════════════════════════════════════════════════════

    def validate_trade(self, market_id: str, outcome: str, amount: float,
                       kelly_amount: float, market_data: dict = None) -> tuple[bool, str]:
        """
        Validación completa pre-trade (síncrona).

        Parámetros:
            market_id    — ID del mercado (conditionId)
            outcome      — "YES" o "NO"
            amount       — tamaño de la orden en USD
            kelly_amount — tamaño Kelly calculado (puede diferir del amount si se forzó mínimo)
            market_data  — dict con datos del mercado del último scan

        Retorna:
            (True,  "OK")          si pasa todas las validaciones
            (False, "razón")       si debe bloquearse
        """
        mkt = market_data or {}
        outcome_upper = (outcome or "YES").upper()

        # ── 1. Validación de precio ────────────────────────────────────────
        yes_p  = float(mkt.get("yes_price")   or mkt.get("market_price") or 0.5)
        no_p   = float(mkt.get("no_price")    or (1.0 - yes_p))
        out_p  = yes_p if outcome_upper == "YES" else no_p

        if out_p < self.MIN_PRICE:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Precio indica resultado ya decidido (< 5¢): {out_p*100:.1f}¢")

        if out_p > self.MAX_PRICE:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Precio indica resultado ya decidido (> 95¢): {out_p*100:.1f}¢")

        spread = abs(yes_p - no_p)
        if spread > self.MAX_SPREAD:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Spread extremo indica mercado resuelto: {spread*100:.0f}¢")

        # ── 2. Estado del mercado ──────────────────────────────────────────
        if mkt.get("closed") is True:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               "Mercado cerrado")

        if mkt.get("active") is False:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               "Mercado inactivo")

        end_date_str = str(mkt.get("end_date") or mkt.get("end_date_iso") or "").strip()
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                if end_dt < datetime.now(timezone.utc):
                    return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                                       f"Evento ya finalizó (end_date: {end_date_str[:16]})")
            except (ValueError, TypeError):
                pass  # No bloquear si no se puede parsear

        # ── 3. Movimiento extremo reciente ────────────────────────────────
        # Señal de resultado ya conocido: precio en zona extrema (10-90%)
        # combina con la validación de precios; aquí cubrimos la zona 10–90%
        if out_p >= 0.90 or out_p <= 0.10:
            # En la zona 10–90% el bloque ya ocurrió arriba (5¢/95¢).
            # Si llegó aquí, el precio es entre 0.10–0.90: no bloquear.
            pass

        # ── 4. Kelly vs mínimo Polymarket ─────────────────────────────────
        # kelly_amount es el Kelly puro calculado.
        # amount puede ser max(1.0, kelly_amount) — el tamaño real de la orden.
        if 0 < kelly_amount < self.MIN_KELLY_OVERRIDE:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Kelly ${kelly_amount:.2f} es menos de la mitad del mínimo ($1.00). "
                               f"Riesgo desproporcionado.")

        # ── 10. Utilidad mínima — ganancia potencial ≥ 20% ───────────────
        # Si el precio del lado a apostar es > 0.8333, ganar solo nos da < 20%.
        # Ej: apostar YES a $0.90 → ganancia potencial 11% → inaceptable.
        if out_p > MAX_BET_PRICE:
            _max_return_pct = round((1.0 / out_p - 1.0) * 100, 1)
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Utilidad insuficiente: precio {out_p*100:.1f}¢ → "
                               f"retorno máximo {_max_return_pct}% "
                               f"(mínimo requerido: {int(MIN_PROFIT_MARGIN*100)}%)")

        # ── 5. Deportes/Esports — señales de resultado conocido ───────────
        category = str(mkt.get("category") or "").lower()
        is_sports = any(c in category for c in
                        ("sport", "esport", "nba", "nfl", "nhl", "mlb", "soccer",
                         "football", "basketball", "tennis", "cricket", "rugby"))
        if is_sports:
            # Precio ultra-extremo en mercados deportivos (98%/2%)
            if out_p > 0.98 or out_p < 0.02:
                return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                                   f"Mercado deportivo con precio extremo ({out_p*100:.1f}¢) "
                                   f"— posible resultado ya publicado")
            # Cache de resultados
            cached = self._sports_cache.get(market_id)
            if cached and (datetime.now(timezone.utc).timestamp() - cached["ts"]) < SPORTS_CACHE_TTL_SECONDS:
                if cached.get("finished"):
                    return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                                       f"Evento deportivo marcado como finalizado en cache: {cached.get('reason','')}")

        # ── 6. Liquidez mínima ────────────────────────────────────────────
        liquidity = float(mkt.get("liquidity") or mkt.get("liquidity_usd") or 0.0)
        if liquidity > 0 and liquidity < self.MIN_LIQUIDITY_USD:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Liquidez insuficiente (${liquidity:.0f} < ${self.MIN_LIQUIDITY_USD:.0f}): "
                               f"slippage destructivo")

        # ── 7. Volumen 24h mínimo ─────────────────────────────────────────
        volume_24h = float(mkt.get("volume_24h") or mkt.get("volume24hr") or 0.0)
        if volume_24h > 0 and volume_24h < self.MIN_VOLUME_24H_USD:
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Volumen 24h muy bajo (${volume_24h:.0f} < ${self.MIN_VOLUME_24H_USD:.0f}): "
                               f"mercado inactivo/manipulable")

        # ── 8. Tiempo mínimo a resolución ─────────────────────────────────
        _end_str = str(mkt.get("end_date") or mkt.get("end_date_iso") or "").strip()
        if _end_str:
            try:
                _end_dt = datetime.fromisoformat(_end_str.replace("Z", "+00:00"))
                if _end_dt.tzinfo is None:
                    _end_dt = _end_dt.replace(tzinfo=timezone.utc)
                _mins_left = (_end_dt - datetime.now(timezone.utc)).total_seconds() / 60.0
                if 0 < _mins_left < self.MIN_MINUTES_TO_RESOLUTION:
                    return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                                       f"Resolución inminente ({_mins_left:.1f} min) — demasiado tarde para entrar")
            except (ValueError, TypeError):
                pass

        # ── 9. Anti-duplicados — mismo mercado en ventana de 5 minutos ────
        _now_ts = datetime.now(timezone.utc).timestamp()
        with self._lock:
            _last_ts = self._recent_trades.get(market_id, 0.0)
        if _now_ts - _last_ts < self.DEDUP_WINDOW_SECONDS:
            _secs_ago = int(_now_ts - _last_ts)
            return self._block(market_id, outcome_upper, amount, kelly_amount, out_p,
                               f"Trade duplicado: mismo mercado hace {_secs_ago}s "
                               f"(ventana anti-dup: {self.DEDUP_WINDOW_SECONDS}s)")

        # ── Todas las validaciones pasadas ────────────────────────────────
        with self._lock:
            self._recent_trades[market_id] = _now_ts
            # Limpiar entradas antiguas (>ventana) para evitar memory leak
            _cutoff = _now_ts - self.DEDUP_WINDOW_SECONDS
            self._recent_trades = {k: v for k, v in self._recent_trades.items() if v > _cutoff}
        self._on_success()
        return True, "OK"

    def cancel_dedup(self, market_id: str) -> None:
        """
        Elimina market_id del registro anti-duplicados.
        Llamar cuando una orden FOK fue cancelada (no ejecutada) para permitir
        reintentos en el mismo mercado dentro de la misma ventana.
        """
        with self._lock:
            self._recent_trades.pop(market_id, None)

    async def validate_trade_full(self, market_id: str, outcome: str, amount: float,
                                  kelly_amount: float,
                                  market_data: dict = None) -> tuple[bool, str]:
        """
        Versión async: igual que validate_trade() pero además consulta Gamma API
        para confirmar que el mercado aún está activo. Usada en el endpoint /test.
        """
        # Validación síncrona primero
        ok, reason = self.validate_trade(market_id, outcome, amount, kelly_amount, market_data)
        if not ok:
            return ok, reason

        # Consulta fresca a Gamma API para verificar closed/active/resolution
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"{GAMMA_API_BASE}/markets/{market_id}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                    if resp.status == 200:
                        fresh = await resp.json()
                        # Verificar campos de estado
                        if fresh.get("closed") is True:
                            return self._block(market_id, outcome, amount, kelly_amount,
                                               float(market_data.get("yes_price", 0.5) if market_data else 0.5),
                                               "Mercado cerrado (verificado en Gamma API)")
                        if fresh.get("active") is False:
                            return self._block(market_id, outcome, amount, kelly_amount,
                                               float(market_data.get("yes_price", 0.5) if market_data else 0.5),
                                               "Mercado inactivo (verificado en Gamma API)")
                        # Precios frescos
                        try:
                            op = json.loads(fresh.get("outcomePrices", "[]") or "[]")
                            if op:
                                fresh_yes = float(op[0])
                                fresh_no  = float(op[1]) if len(op) > 1 else 1 - fresh_yes
                                fresh_out = fresh_yes if outcome.upper() == "YES" else fresh_no
                                if fresh_out < self.MIN_PRICE or fresh_out > self.MAX_PRICE:
                                    return self._block(
                                        market_id, outcome, amount, kelly_amount, fresh_out,
                                        f"Precio fresco de Gamma API indica resultado decidido: {fresh_out*100:.1f}¢")
                        except Exception:
                            pass
        except Exception as e:
            # Si la API no responde → no bloquear (fail-open en consulta externa)
            pass

        return True, "OK"

    def mark_sports_finished(self, market_id: str, reason: str = ""):
        """Marca un mercado deportivo como finalizado en cache."""
        with self._lock:
            self._sports_cache[market_id] = {
                "finished": True,
                "reason": reason,
                "ts": datetime.now(timezone.utc).timestamp(),
            }

    def check_circuit_breaker(self) -> bool:
        """True si el circuit breaker está activo (bot debe pausar)."""
        return self._circuit_breaker_active

    def reset_circuit_breaker(self):
        """Resetea el circuit breaker manualmente."""
        with self._lock:
            self._circuit_breaker_active = False
            self._consecutive_blocked = 0

    def get_status(self) -> dict:
        with self._lock:
            today = datetime.now(timezone.utc).date().isoformat()
            blocked_today = [b for b in self._blocked_history
                             if b["timestamp"][:10] == today]
            return {
                "active":                   True,
                "circuit_breaker_active":   self._circuit_breaker_active,
                "consecutive_blocked":      self._consecutive_blocked,
                "blocked_today":            len(blocked_today),
                "blocked_total_session":    len(self._blocked_history),
                "log_file":                 str(BLOCKED_LOG),
                "checks_active": 10,
                "thresholds": {
                    "1_min_price_cents":          int(self.MIN_PRICE * 100),
                    "2_max_price_cents":          int(self.MAX_PRICE * 100),
                    "3_max_spread_cents":         int(self.MAX_SPREAD * 100),
                    "4_market_state":             "closed+active+end_date",
                    "5_sports_extreme_price":     "98¢/2¢",
                    "6_min_liquidity_usd":        self.MIN_LIQUIDITY_USD,
                    "7_min_volume_24h_usd":       self.MIN_VOLUME_24H_USD,
                    "8_min_minutes_to_resolution": self.MIN_MINUTES_TO_RESOLUTION,
                    "9_dedup_window_seconds":     self.DEDUP_WINDOW_SECONDS,
                    "10_min_profit_margin_pct":   int(MIN_PROFIT_MARGIN * 100),
                    "10_max_bet_price_cents":     int(MAX_BET_PRICE * 100),
                    "min_kelly_for_override":     self.MIN_KELLY_OVERRIDE,
                    "polymarket_min_bet":         POLYMARKET_MIN_BET,
                    "circuit_breaker_threshold":  CIRCUIT_BREAKER_THRESHOLD,
                },
            }

    def get_blocked_history(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(reversed(self._blocked_history[-limit:]))

    # ══════════════════════════════════════════════════════════════════════
    # PRIVADO
    # ══════════════════════════════════════════════════════════════════════

    def _block(self, market_id: str, outcome: str, amount: float,
               kelly: float, price: float, reason: str) -> tuple[bool, str]:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "timestamp": ts,
            "market_id": market_id,
            "outcome":   outcome,
            "amount":    round(amount, 4),
            "kelly":     round(kelly, 4),
            "price":     round(price, 4),
            "reason":    reason,
        }
        with self._lock:
            self._consecutive_blocked += 1
            self._blocked_history.append(entry)
            if len(self._blocked_history) > 500:
                self._blocked_history = self._blocked_history[-500:]
            if self._consecutive_blocked >= CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_breaker_active = True

        # Log a fichero dedicado con rotación automática (max 5000 líneas ~250KB)
        try:
            _MAX_LOG_LINES = 5000
            log_line = (
                f"[{ts}] BLOCKED | Market: {market_id} | Razón: {reason} | "
                f"Precio: {price:.4f} | Kelly: ${kelly:.4f} | Outcome: {outcome} | "
                f"Amount: ${amount:.4f}\n"
            )
            # Rotar si el archivo es demasiado grande
            if BLOCKED_LOG.exists() and BLOCKED_LOG.stat().st_size > 200_000:  # >200KB
                try:
                    lines = BLOCKED_LOG.read_text(encoding="utf-8").splitlines(keepends=True)
                    if len(lines) > _MAX_LOG_LINES:
                        # Mantener solo las últimas 3000 líneas
                        BLOCKED_LOG.write_text("".join(lines[-3000:]), encoding="utf-8")
                except Exception:
                    pass  # Si falla la rotación, simplemente seguimos
            with open(BLOCKED_LOG, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

        self._log.warning("BLOCKED | %s | %s", market_id, reason)
        return False, reason

    def _on_success(self):
        with self._lock:
            self._consecutive_blocked = 0
