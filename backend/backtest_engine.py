"""
══════════════════════════════════════════════════════════════
Polymarket Bot - Backtesting Engine v2.0
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Simulates trading strategies against resolved Polymarket markets.

IMPORTANTE — Sin data leakage:
  - simulate_trade() NO conoce el resultado real al generar la probabilidad.
  - La probabilidad estimada se construye SOLO con datos disponibles en
    tiempo real: precio de mercado, volumen, liquidez, spread y señales
    de categoría derivadas de la pregunta.
  - El resultado resuelto se usa ÚNICAMENTE para calcular ganancia/pérdida
    DESPUÉS de que el motor ya tomó su decisión.
  - Corrección del 19/03/2026 — ddchack.
"""

import httpx
import asyncio
import json
import math
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from math_engine import MathEngine, MarketOdds
from bayesian_engine import BayesianProbabilityEngine
from category_models import CategoryEvaluator, classify_market, MarketCategory


@dataclass
class BacktestTrade:
    market_id: str
    question: str
    category: str
    side: str
    entry_price: float
    estimated_prob: float
    bet_usd: float
    edge_pct: float
    kelly: float
    resolved_outcome: str       # "YES" o "NO"  — solo para scoring
    won: bool
    pnl_usd: float
    confidence: float
    signals_detail: list = field(default_factory=list)


@dataclass
class BacktestResult:
    """Reporte completo de un backtest."""
    strategy_name: str
    period: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_edge: float
    avg_confidence: float
    max_drawdown_pct: float
    sharpe_estimate: float
    profit_factor: float
    brier_score: float = 0.0        # Calibración de probabilidades
    best_trade: Optional[dict] = None
    worst_trade: Optional[dict] = None
    capital_trajectory: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    calibration_curve: list = field(default_factory=list)
    category_metrics: dict = field(default_factory=dict)  # métricas win_rate/ROI por categoría


GAMMA_API = "https://gamma-api.polymarket.com"

# ── Caché de mercados resueltos (TTL: 5 minutos) ──────────────────────────
# Evita refetch repetitivo de Gamma API en runs consecutivos del backtest.
import time as _time
import asyncio as _asyncio
_resolved_cache: list = []
_resolved_cache_ts: float = 0.0
_RESOLVED_CACHE_TTL: int = 300  # 5 minutos
_resolved_cache_lock: "_asyncio.Lock | None" = None  # se inicializa al primer uso

def _get_cache_lock() -> "_asyncio.Lock":
    global _resolved_cache_lock
    if _resolved_cache_lock is None:
        _resolved_cache_lock = _asyncio.Lock()
    return _resolved_cache_lock


class BacktestEngine:
    """
    Backtesting sin data leakage sobre mercados resueltos de Polymarket.

    Flujo correcto:
        1. Obtener mercados resueltos (precio ya convergió a 0 o 1).
        2. Estimar probabilidad usando SOLO datos observables en tiempo real
           (precio pre-resolución + señales derivables sin conocer el outcome).
        3. Pasar esa probabilidad ciega al MathEngine.
        4. Registrar win/loss comparando la decisión del motor con el outcome real.

    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self, math_engine: MathEngine = None,
                 initial_capital: float = 100.0):
        self.math = math_engine or MathEngine(max_capital=initial_capital)
        self.initial_capital = initial_capital
        self.client = httpx.AsyncClient(timeout=30.0)
        self.bayesian = BayesianProbabilityEngine(prior_strength=10.0)
        self.cat_evaluator = CategoryEvaluator()

    async def close(self):
        await self.client.aclose()

    # ─────────────────────────────────────────────────────────
    # OBTENCIÓN DE DATOS HISTÓRICOS
    # ─────────────────────────────────────────────────────────

    async def fetch_resolved_markets(self, limit: int = 200,
                                      offset: int = 0) -> list[dict]:
        """
        Descarga mercados cerrados/resueltos de la Gamma API.
        Solo incluye mercados que tienen resolutionSource (resultado verificable).
        Usa caché de 5 minutos para evitar refetch en runs consecutivos.
        """
        global _resolved_cache, _resolved_cache_ts

        async with _get_cache_lock():
            # Verificar caché: si tenemos datos frescos y son suficientes, retornar directo
            now_ts = _time.time()
            if (_resolved_cache and
                    len(_resolved_cache) >= limit and
                    now_ts - _resolved_cache_ts < _RESOLVED_CACHE_TTL):
                print(f"[Backtest] Usando caché ({len(_resolved_cache)} mercados, "
                      f"TTL restante: {int(_RESOLVED_CACHE_TTL - (now_ts - _resolved_cache_ts))}s)")
                return _resolved_cache[:limit]

        params = {
            "limit": limit,
            "offset": offset,
            "closed": "true",
            "active": "false",
            "order": "endDate",
            "ascending": "false",
        }
        try:
            resp = await self.client.get(f"{GAMMA_API}/markets", params=params)
            resp.raise_for_status()
            markets = resp.json()
            filtered = [m for m in markets if m.get("resolutionSource")]
            # Actualizar caché si obtuvimos más de 10 mercados válidos
            if len(filtered) > 10:
                async with _get_cache_lock():
                    _resolved_cache = filtered
                    _resolved_cache_ts = _time.time()
                print(f"[Backtest] Caché actualizado: {len(filtered)} mercados")
            return filtered
        except Exception as e:
            print(f"[Backtest] Error fetching resolved markets: {e}")
            # Intentar usar caché aunque esté expirado como fallback
            if _resolved_cache:
                print(f"[Backtest] Usando caché expirado como fallback ({len(_resolved_cache)} mercados)")
                return _resolved_cache[:limit]
            return []

    def parse_resolved_outcome(self, market: dict) -> Optional[str]:
        """
        Determina el outcome real del mercado a partir de los precios finales.
        En mercados resueltos, uno de los precios converge a ~1.0 y el otro a ~0.0.
        """
        prices = market.get("outcomePrices", "")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                return None

        if not prices or len(prices) < 2:
            return None

        yes_price = float(prices[0])
        no_price = float(prices[1])

        if yes_price > 0.95:
            return "YES"
        elif no_price > 0.95:
            return "NO"
        return None

    def _infer_pre_resolution_price(self, market: dict) -> Optional[tuple[float, float]]:
        """
        Reconstruye el precio ANTES de la resolución.

        La Gamma API devuelve outcomePrices ya en estado resuelto (0 o 1).
        Para recuperar el precio pre-resolución usamos los campos alternativos:
          - lastTradePrice / bestBid / bestAsk  (cuando están disponibles)
          - Volume y liquidez como proxy de actividad del mercado antes del cierre

        Si el mercado solo tiene precio resuelto (0 o 1), lo descartamos
        porque no podemos saber qué precio tenía antes — mejor no simular
        que simular con datos incorrectos.
        """
        # Intentar extraer precio de trading activo
        best_bid = market.get("bestBid")
        best_ask = market.get("bestAsk")
        last_trade = market.get("lastTradePrice")

        # Prioridad: spread bid/ask → último precio → ignorar
        if best_bid and best_ask:
            try:
                bid = float(best_bid)
                ask = float(best_ask)
                mid = (bid + ask) / 2.0
                if 0.05 <= mid <= 0.95:
                    return mid, ask - bid  # (precio_mid, spread)
            except (ValueError, TypeError):
                pass

        if last_trade:
            try:
                p = float(last_trade)
                if 0.05 <= p <= 0.95:
                    return p, 0.04  # spread estimado
            except (ValueError, TypeError):
                pass

        # Si no hay precio activo, intentar el precio de resolución intermedio.
        # outcomePrices en algunos mercados puede tener el precio de cierre antes
        # de que el oracle lo fijara a 0/1.
        prices_str = market.get("outcomePrices", "")
        if isinstance(prices_str, str):
            try:
                prices = json.loads(prices_str) if prices_str else []
            except json.JSONDecodeError:
                prices = []
        else:
            prices = prices_str or []

        if len(prices) >= 2:
            try:
                yp = float(prices[0])
                np_ = float(prices[1])
                # Solo usar si NO está ya resuelto (evitar precio 0.00/1.00)
                if 0.05 <= yp <= 0.95 and 0.05 <= np_ <= 0.95:
                    spread = abs(yp + np_ - 1.0)
                    return yp, spread
            except (ValueError, TypeError):
                pass

        # No se puede reconstruir el precio pre-resolución → descartar
        return None

    # ─────────────────────────────────────────────────────────
    # ESTIMACIÓN DE PROBABILIDAD — SIN DATA LEAKAGE
    # ─────────────────────────────────────────────────────────

    def _estimate_probability_blind(self, market: dict,
                                     yes_price: float,
                                     spread: float) -> tuple[float, float, list]:
        """
        Estima probabilidad usando SOLO datos observables sin conocer el outcome.

        Fuentes de señal (todas disponibles en tiempo real):
          1. Precio de mercado → prior Bayesiano
          2. Volume 24h vs volumen total → momentum
          3. Liquidez → calidad del spread
          4. Categoría del mercado → modelo específico (CryptoModel, SportsModel…)
          5. Tendencia implícita en el spread (spread ancho = incertidumbre alta)

        NO se usa ni el resultado resuelto ni ninguna variante de él.
        """
        volume_24h = float(market.get("volume24hr", 0) or 0)
        volume_total = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)
        question = market.get("question", "")

        # ── Señal de momentum de volumen ──────────────────────────────────
        # Si volume_24h es alto relativo al total, hay actividad reciente
        avg_vol_proxy = volume_total / max(1, 30)  # volumen diario implícito
        vol_signal = self.bayesian.volume_momentum_signal(volume_24h, avg_vol_proxy)

        # ── Señal de calidad de liquidez ──────────────────────────────────
        liq_signal = self.bayesian.liquidity_quality_signal(spread, liquidity, volume_24h)

        # ── Señal de categoría (sin datos externos — solo estructura de pregunta) ──
        category = classify_market(question)
        cat_signal_val = None

        if category != MarketCategory.OTHER:
            cat_result = self.cat_evaluator.evaluate(question, yes_price, context={})
            # Solo usar si la confianza es mínimamente significativa
            if cat_result.confidence >= 0.15:
                from bayesian_engine import Signal
                cat_signal_val = Signal(
                    name="category_model",
                    value=cat_result.estimated_prob,
                    confidence=cat_result.confidence,
                    signal_type="probability"
                )

        # ── Construir lista de señales ────────────────────────────────────
        signals = [vol_signal, liq_signal]
        if cat_signal_val:
            signals.append(cat_signal_val)

        # ── Estimación Bayesiana ──────────────────────────────────────────
        estimate = self.bayesian.estimate_probability(yes_price, signals)

        return estimate.posterior, estimate.confidence, estimate.signals_used

    # ─────────────────────────────────────────────────────────
    # SIMULACIÓN DE TRADE — NÚCLEO
    # ─────────────────────────────────────────────────────────

    def simulate_trade(self, market: dict, capital: float) -> Optional[BacktestTrade]:
        """
        Simula la decisión del motor sobre un mercado histórico resuelto.

        GARANTÍA ANTI-LEAKAGE:
          - parse_resolved_outcome() se llama DESPUÉS de que el motor ya decidió.
          - _estimate_probability_blind() no recibe ni usa el outcome.
          - El outcome solo se compara con la decisión para calcular PnL.
        """
        # ── 1. Reconstruir precio pre-resolución ─────────────────────────
        price_data = self._infer_pre_resolution_price(market)
        if price_data is None:
            return None  # No hay precio usable → no simular

        yes_price, spread = price_data

        # ── 2. Construir MarketOdds con datos observables ─────────────────
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)
        odds = MarketOdds(
            yes_price=max(0.05, min(0.95, yes_price)),
            no_price=max(0.05, min(0.95, 1.0 - yes_price)),
            volume_24h=float(market.get("volume24hr", 0) or 0),
            liquidity=liquidity,
        )

        # ── 3. Estimar probabilidad SIN conocer el outcome ─────────────────
        estimated_prob, confidence, signals_detail = self._estimate_probability_blind(
            market, yes_price, spread
        )

        # ── 4. Pasar al MathEngine (Kelly, EV, edge) ──────────────────────
        rec = self.math.evaluate_market(
            market_id=market.get("conditionId", market.get("id", "")),
            question=market.get("question", ""),
            odds=odds,
            estimated_true_prob=estimated_prob,
            available_capital=capital,
        )

        if not rec or rec.recommended_bet_usd <= 0:
            return None

        # ── 5. Ahora sí: comparar con el resultado real ───────────────────
        resolved = self.parse_resolved_outcome(market)
        if not resolved:
            return None

        won = (rec.side == resolved)
        if won:
            pnl = rec.recommended_bet_usd * ((1.0 / rec.market_price) - 1.0)
        else:
            pnl = -rec.recommended_bet_usd

        # Registrar en el motor Bayesiano para calibración
        self.bayesian.record_prediction(
            market_id=market.get("conditionId", ""),
            predicted_prob=estimated_prob,
            actual_outcome=(resolved == "YES"),
        )

        return BacktestTrade(
            market_id=market.get("conditionId", market.get("id", "")),
            question=market.get("question", "")[:100],
            category=str(classify_market(market.get("question", ""))),
            side=rec.side,
            entry_price=rec.market_price,
            estimated_prob=estimated_prob,
            bet_usd=rec.recommended_bet_usd,
            edge_pct=rec.edge_percent,
            kelly=rec.kelly_fraction,
            resolved_outcome=resolved,
            won=won,
            pnl_usd=round(pnl, 4),
            confidence=confidence,
            signals_detail=signals_detail,
        )

    # ─────────────────────────────────────────────────────────
    # BACKTEST COMPLETO
    # ─────────────────────────────────────────────────────────

    async def run_backtest(self, strategy_name: str = "Kelly Quarter",
                           num_markets: int = 100) -> BacktestResult:
        """
        Ejecuta backtest completo sobre mercados resueltos.

        Args:
            strategy_name: Etiqueta para este run.
            num_markets: Cantidad de mercados resueltos a testear.
        """
        markets = await self.fetch_resolved_markets(limit=num_markets)
        if not markets:
            print("[Backtest] No se obtuvieron mercados resueltos.")

        capital = self.initial_capital
        trajectory = [capital]
        trades: list[BacktestTrade] = []
        peak_capital = capital

        total_fetched = len(markets)
        skipped = 0

        for mkt in markets:
            trade = self.simulate_trade(mkt, capital)
            if trade is None:
                skipped += 1
                continue

            if trade.bet_usd > 0:
                capital += trade.pnl_usd
                capital = max(0.0, capital)
                trajectory.append(round(capital, 4))
                trades.append(trade)

                if capital > peak_capital:
                    peak_capital = capital

                if capital < 1.0:
                    print("[Backtest] Capital agotado — deteniendo.")
                    break

        print(f"[Backtest] Mercados obtenidos: {total_fetched} | "
              f"Simulados: {len(trades)} | Descartados (sin precio): {skipped}")

        # ── Métricas ──────────────────────────────────────────────────────
        wins = sum(1 for t in trades if t.won)
        losses = len(trades) - wins
        gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))

        # Max drawdown
        max_dd = 0.0
        peak = self.initial_capital
        for val in trajectory:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Sharpe por trade
        if trades:
            returns = [t.pnl_usd / max(0.01, t.bet_usd) for t in trades]
            avg_ret = float(np.mean(returns))
            std_ret = float(np.std(returns)) if len(returns) > 1 else 1.0
            sharpe = avg_ret / std_ret if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        profit_factor = gross_profit / max(0.01, gross_loss)

        best = max(trades, key=lambda t: t.pnl_usd) if trades else None
        worst = min(trades, key=lambda t: t.pnl_usd) if trades else None

        # Brier Score y curva de calibración (calibración honesta)
        brier = self.bayesian.brier_score()
        cal_curve = self.bayesian.calibration_curve(bins=10)

        return BacktestResult(
            strategy_name=strategy_name,
            period=f"Últimos {num_markets} mercados resueltos "
                   f"({len(trades)} simulados, {skipped} sin precio)",
            initial_capital=self.initial_capital,
            final_capital=round(capital, 2),
            total_return_pct=round(
                ((capital - self.initial_capital) / self.initial_capital) * 100, 2
            ),
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=round(wins / max(1, len(trades)) * 100, 1),
            avg_edge=round(
                sum(t.edge_pct for t in trades) / max(1, len(trades)), 2
            ),
            avg_confidence=round(
                sum(t.confidence for t in trades) / max(1, len(trades)), 3
            ),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_estimate=round(sharpe, 3),
            profit_factor=round(profit_factor, 2),
            brier_score=brier,
            best_trade={
                "question": best.question,
                "pnl": best.pnl_usd,
                "side": best.side,
                "edge": best.edge_pct,
                "category": best.category,
            } if best else None,
            worst_trade={
                "question": worst.question,
                "pnl": worst.pnl_usd,
                "side": worst.side,
                "edge": worst.edge_pct,
                "category": worst.category,
            } if worst else None,
            capital_trajectory=trajectory,
            calibration_curve=cal_curve,
            trades=[{
                "question":    t.question[:60],
                "category":    t.category,
                "side":        t.side,
                "price":       t.entry_price,
                "est_prob":    t.estimated_prob,
                "resolved":    t.resolved_outcome,
                "bet":         t.bet_usd,
                "won":         t.won,
                "pnl":         t.pnl_usd,
                "edge":        t.edge_pct,
                "confidence":  t.confidence,
            } for t in trades],
            category_metrics=self._compute_category_metrics(trades),
        )

    def _compute_category_metrics(self, trades: list) -> dict:
        """
        Métricas de rendimiento por categoría.
        Útil para calibrar pesos del ensemble por tipo de mercado.
        """
        cats: dict[str, dict] = {}
        for t in trades:
            c = t.category or "other"
            if c not in cats:
                cats[c] = {"trades": 0, "wins": 0, "pnl": 0.0,
                            "total_bet": 0.0, "edge_sum": 0.0}
            cats[c]["trades"]    += 1
            cats[c]["wins"]      += int(t.won)
            cats[c]["pnl"]       += t.pnl_usd
            cats[c]["total_bet"] += t.bet_usd
            cats[c]["edge_sum"]  += t.edge_pct

        result = {}
        for c, d in cats.items():
            n = max(1, d["trades"])
            result[c] = {
                "trades":    d["trades"],
                "win_rate":  round(d["wins"] / n * 100, 1),
                "total_pnl": round(d["pnl"], 3),
                "roi_pct":   round(d["pnl"] / max(0.01, d["total_bet"]) * 100, 2),
                "avg_edge":  round(d["edge_sum"] / n, 2),
            }
        return result

    # ─────────────────────────────────────────────────────────
    # COMPARACIÓN DE ESTRATEGIAS
    # ─────────────────────────────────────────────────────────

    async def compare_strategies(self, strategies: list[dict],
                                  num_markets: int = 100) -> list[BacktestResult]:
        """
        Compara múltiples configuraciones de Kelly/edge sobre los mismos mercados.

        strategies = [
            {"name": "Quarter Kelly", "kelly_mult": 0.25, "min_edge": 0.03},
            {"name": "Half Kelly",    "kelly_mult": 0.50, "min_edge": 0.03},
            {"name": "Conservador",   "kelly_mult": 0.10, "min_edge": 0.05},
        ]
        """
        # Un solo fetch compartido para comparación justa
        markets = await self.fetch_resolved_markets(limit=num_markets)
        results = []

        for strat in strategies:
            engine = MathEngine(
                max_capital=self.initial_capital,
                kelly_multiplier=strat.get("kelly_mult", 0.25),
                min_edge=strat.get("min_edge", 0.03),
                min_ev=strat.get("min_ev", 0.05),
            )
            bt = BacktestEngine(
                math_engine=engine,
                initial_capital=self.initial_capital,
            )

            capital = self.initial_capital
            trajectory = [capital]
            trades: list[BacktestTrade] = []

            for mkt in markets:
                trade = bt.simulate_trade(mkt, capital)
                if trade and trade.bet_usd > 0:
                    capital += trade.pnl_usd
                    capital = max(0.0, capital)
                    trajectory.append(round(capital, 4))
                    trades.append(trade)
                    if capital < 1.0:
                        break

            wins = sum(1 for t in trades if t.won)
            gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
            gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))

            # Max drawdown
            max_dd = 0.0
            peak = self.initial_capital
            for val in trajectory:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            if trades:
                returns = [t.pnl_usd / max(0.01, t.bet_usd) for t in trades]
                avg_r = float(np.mean(returns))
                std_r = float(np.std(returns)) if len(returns) > 1 else 1.0
                sharpe = avg_r / std_r if std_r > 0 else 0.0
            else:
                sharpe = 0.0

            results.append(BacktestResult(
                strategy_name=strat["name"],
                period=f"Últimos {num_markets} resueltos",
                initial_capital=self.initial_capital,
                final_capital=round(capital, 2),
                total_return_pct=round(
                    ((capital - self.initial_capital) / self.initial_capital) * 100, 2
                ),
                total_trades=len(trades),
                wins=wins,
                losses=len(trades) - wins,
                win_rate=round(wins / max(1, len(trades)) * 100, 1),
                avg_edge=round(
                    sum(t.edge_pct for t in trades) / max(1, len(trades)), 2
                ) if trades else 0,
                avg_confidence=round(
                    sum(t.confidence for t in trades) / max(1, len(trades)), 3
                ) if trades else 0,
                max_drawdown_pct=round(max_dd * 100, 2),
                sharpe_estimate=round(sharpe, 3),
                profit_factor=round(
                    gross_profit / max(0.01, gross_loss), 2
                ),
                brier_score=bt.bayesian.brier_score(),
                capital_trajectory=trajectory,
            ))

            await bt.close()

        return results
