"""
══════════════════════════════════════════════════════════════
Polymarket Bot v5.0 - Full Algorithm Integration
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════
All algorithms from research integrated:
- KL Divergence opportunity ranking
- Multi-Kelly simultaneous optimization
- Category-specific models (ELO, Poisson, Oracle Lag, Polls, Weather)
- HMM Regime detection for position sizing
- Brier Score decomposition for self-calibration
- Monte Carlo portfolio simulation
- Gaussian Copula correlation modeling
- Platt scaling calibration
- LMSR fair price analysis
"""

import os, json, asyncio, math, statistics, logging
logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket as FWS, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
import pathlib

# v1 modules
from math_engine import MathEngine, MarketOdds
from market_scanner import MarketScanner
from news_sentiment import NewsSentimentAnalyzer
from trading_client import TradingClient
from backtest_engine import BacktestEngine
# v2 modules
from bayesian_engine import BayesianProbabilityEngine
from arbitrage_detector import ArbitrageDetector
from risk_manager import RiskManager, RiskLimits
from whale_tracker import WhaleTracker
from ws_client import RealtimeClient
# v3 modules
from advanced_math import (KLDivergence, MultiKelly, LMSR, BrierScoreAnalyzer,
                            MonteCarloSimulator, GaussianCopula, BinarySharpe, PlattCalibrator)
from category_models import (CategoryEvaluator, classify_market, MarketCategory, RegimeDetector)
from auto_allocator import AutoAllocator
from llm_engine import LLMEngine, ANTHROPIC_MODELS, OPENAI_MODELS
from swarm_engine import SwarmEngine, PERSONAS
from calibration_tracker import get_calibration_tracker
# v4 modules (research improvements)
from calibration_v2 import CalibrationSuite
from extremizer import Extremizer, DiversityTracker, AdaptiveAggregator
from mean_reversion import MeanReversionEngine
from microstructure import MicrostructureAnalyzer, OrderBookImbalance, VPIN
from trade_validator import TradeValidator, CIRCUIT_BREAKER_THRESHOLD
from escape_room import EscapeRoomEngine, EROpportunity, ER_KELLY_FRACTION, ER_MIN_ENTRY_PRICE, ER_MIN_LIQUIDITY
from rush_mode import (RushEngine, RushOpportunity, RUSH_MIN_SECONDS, RUSH_MAX_SECONDS,
                       RUSH_SCAN_INTERVAL, RUSH_MARKET_HIGH, RUSH_MARKET_LOW,
                       RUSH_BTC_MOMENTUM, RUSH_BTC_STRONG)
from dump_hedge import DumpHedgeEngine, DHOpportunity, DH_DEFAULT_CONFIG, taker_fee
from dump_hedge_ws import BinancePriceFeed, DumpDetector

# ═══════════ CONFIG ═══════════
MAX_CAPITAL = float(os.getenv("MAX_CAPITAL_USD", "100"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MIN_EDGE = float(os.getenv("MIN_EDGE_PERCENT", "3.0")) / 100
MIN_EV = float(os.getenv("MIN_EXPECTED_VALUE", "0.05"))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
SIG_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

# ═══════════ INSTANCES ═══════════
math_engine = MathEngine(max_capital=MAX_CAPITAL, kelly_multiplier=KELLY_FRACTION, min_edge=MIN_EDGE, min_ev=MIN_EV)
scanner = MarketScanner(math_engine=math_engine)
sentiment = NewsSentimentAnalyzer(newsapi_key=NEWSAPI_KEY)
trader = TradingClient(private_key=PRIVATE_KEY, funder=FUNDER, signature_type=SIG_TYPE, dry_run=True)
bt_engine = BacktestEngine(math_engine=math_engine, initial_capital=MAX_CAPITAL)
bayesian = BayesianProbabilityEngine(prior_strength=10.0)
arb = ArbitrageDetector(max_capital=MAX_CAPITAL)
risk = RiskManager(initial_capital=MAX_CAPITAL, limits=RiskLimits(
    min_liquidity_usd=1000.0,      # Subido a 1000 → evitar slippage en mercados de baja liquidez
    min_time_between_trades=1.0,   # El bot gestiona su propio intervalo
    max_spread_pct=12.0,           # Subido de 8% → mercados pequeños tienen spread alto
))
whales = WhaleTracker(min_trade_usd=500.0)
rt = RealtimeClient()
# v3
cat_eval = CategoryEvaluator()
calibrator = PlattCalibrator()  # Legacy compat
brier = BrierScoreAnalyzer()
regime = RegimeDetector()
allocator = AutoAllocator(kelly_fraction=KELLY_FRACTION)
llm = LLMEngine()
swarm_engine = SwarmEngine()
calibration_tracker = get_calibration_tracker()
# v4 (research improvements)
cal_suite = CalibrationSuite()  # Beta + Isotonic + Temp + Venn-ABERS
mr_engine = MeanReversionEngine()
micro = MicrostructureAnalyzer()
extremizer_d = 1.73  # Satopää optimal default
adaptive_agg = AdaptiveAggregator(n_models=4, learning_rate=0.5)
diversity = DiversityTracker()

# ── Pesos del ensemble por categoría (modificables via /api/tune/apply) ──────
# [bayesian_weight, cat_signal_weight_factor, market_price_weight]
_ENSEMBLE_WEIGHTS: dict = {
    "politics":    [0.35, 0.45, 0.20],
    # Sports: mercado tiene 70% de peso. Sin ELO/Poisson reales el modelo
    # no tiene ventaja sobre el crowd — el extremizer causaba edge falso en underdogs.
    "sports":      [0.08, 0.22, 0.70],
    "crypto":      [0.40, 0.30, 0.30],
    "economics":   [0.40, 0.35, 0.25],
    "geopolitics": [0.35, 0.40, 0.25],
    "tech":        [0.40, 0.35, 0.25],
    "default":     [0.45, 0.35, 0.20],
}
_ENSEMBLE_WEIGHTS_SOURCE: str = "default"  # "default" | "backtest_YYYY-MM-DD"
# ── TradeValidator ──
validator = TradeValidator()
trader.validator = validator   # safety net en place_limit/market_order
dashboard_ws: list[FWS] = []
# ── Dump & Hedge engine ──
_dh_engine = DumpHedgeEngine(trading_client=trader, scanner=scanner)
_dh_binance = BinancePriceFeed()  # feed de precios Binance en tiempo real

# Auto-backtest background task (runs every 6h to optimize ensemble weights)
_AUTO_BACKTEST_INTERVAL_H: int = 6   # horas entre backtests automáticos
_last_auto_backtest_ts: float = 0.0  # unix timestamp del último backtest
_backtest_task = None

# DDC Normalization bounds — actualizados por auto-backtest según distribución real de mercados
# KL norm: cuánto KL se considera "ideal" (máximo donde la señal es más útil)
# EV norm: cuánto EV se considera "ideal"
# Ambos se ajustan según los percentiles del backtest para que el score sea más discriminativo
_DDC_KL_NORM: float  = 0.10   # KL = 0.10 → score máximo (actualizable via backtest)
_DDC_EV_NORM: float  = 0.12   # EV = 0.12 → score máximo (actualizable via backtest)
_volume_history_cache: dict = {}  # BUG5: historial de volumen por market_id para detectar volume surge real

# DDC v3: normas por categoría — markets eficientes (politics) tienen menor norm
# porque en esos mercados cualquier edge ≥ norm es excepcional.
# Crypto tiene edges naturalmente más altos → norm más alta para ser discriminativo.
_DDC_CATEGORY_NORMS: dict = {
    "crypto":    {"kl": 0.14, "ev": 0.18},   # alta volatilidad, edges grandes
    "sports":    {"kl": 0.10, "ev": 0.12},   # base
    "politics":  {"kl": 0.07, "ev": 0.09},   # mercado eficiente, edges pequeños
    "economics": {"kl": 0.08, "ev": 0.10},   # entre politics y sports
    "default":   {"kl": 0.10, "ev": 0.12},   # fallback = global norms
}

# ── DDC v5 Helper Functions ──────────────────────────────────────────────────
# SharkFlow by Carlos David Donoso Cordero (ddchack)

def _ddc_bayesian_kl(kl: float, cat_norm: float, n_samples: int = 100) -> float:
    """
    Bayesian shrinkage of observed KL toward the category expected mean.
    When n_samples is small, trust the category prior more.
    When n_samples is large (≥200), use the observed KL directly.
    """
    alpha = min(n_samples / 200.0, 1.0)
    category_mean = cat_norm * 0.5  # expected mean KL = 50% of category norm
    return alpha * kl + (1 - alpha) * category_mean

def _ddc_kelly_consistency(composite: float, kelly_frac: float) -> float:
    """
    Reduce DDC score if kelly fraction is too small to be actionable.
    Tiny kelly → opportunity is statistically real but practically irrelevant.
    """
    if kelly_frac < 0.03:
        return composite * 0.80
    elif kelly_frac < 0.08:
        return composite * 0.90
    return composite

def _ddc_market_age_factor(hours_elapsed: float, total_duration_hours: float) -> float:
    """
    Penalize markets that are very new (unstable prices) or near resolution (sticky prices).
    Sweet spot: 15%-85% of market lifetime.
    """
    pct_elapsed = hours_elapsed / max(total_duration_hours, 1.0)
    if pct_elapsed < 0.05:
        return 0.70   # very new market, prices not yet stable
    elif pct_elapsed < 0.15:
        return 0.85   # recently opened
    elif pct_elapsed > 0.85:
        return 0.92   # near end, prices sticky (less opportunity)
    return 1.00       # normal market

def _ddc_consensus_bonus(diversity_score: float, model_count: int) -> float:
    """
    Bonus when multiple independent models strongly agree.
    High diversity_score + many models → robust signal.
    """
    if diversity_score >= 0.8 and model_count >= 6:
        return 0.06
    elif diversity_score >= 0.7 and model_count >= 5:
        return 0.03
    return 0.0

def _ddc_anti_overfit(composite: float, confidence: float, liquidity: float) -> float:
    """
    Guard against suspicious high-composite signals with low confidence and low liquidity.
    Pattern: high KL + low liquidity + low confidence = likely noise artifact.
    """
    if composite > 0.70 and confidence < 55 and liquidity < 2000:
        return min(composite, 0.65)
    return composite

# Track predictions for calibration — persisted to disk
_PRED_LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.json")
_pred_log_lock = asyncio.Lock()

def _load_prediction_log() -> list:
    """Carga prediction log con fallback a backup si está corrupto."""
    for path in [_PRED_LOG_PATH, _PRED_LOG_PATH + ".bak"]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except (json.JSONDecodeError, Exception) as _e:
            if path == _PRED_LOG_PATH:
                print(f"[PredLog] Principal corrupto, usando backup: {_e}")
    return []

def _save_prediction_log(log: list) -> None:
    """Guarda prediction log con atomic write para prevenir corrupción por concurrencia."""
    import tempfile, shutil
    try:
        dir_path = os.path.dirname(os.path.abspath(_PRED_LOG_PATH))
        os.makedirs(dir_path, exist_ok=True)
        # Backup antes de sobrescribir
        if os.path.exists(_PRED_LOG_PATH):
            try:
                shutil.copy2(_PRED_LOG_PATH, _PRED_LOG_PATH + ".bak")
            except Exception:
                pass
        # Atomic write: temp file → rename
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp", prefix="predlog_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(log[-500:], f, ensure_ascii=False)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        os.replace(tmp_path, _PRED_LOG_PATH)
    except Exception as _e:
        print(f"[PredLog] Error saving: {_e}")

_prediction_log: list = _load_prediction_log()
print(f"[PredLog] Loaded {len(_prediction_log)} predictions from disk")

# ── Bot scheduler state ──
_bot_running = False
_bot_mode = "demo"          # "demo" | "live"
_bot_interval = 30          # minutos
_bot_max_risk = 0.25        # kelly fraction
_bot_min_edge = 15.0        # porcentaje mínimo de edge (mín 15% para entrar)
_bot_time_horizon = "any"   # minutes | hours | days | weeks | months | any
_bot_min_bets = 1           # mínimo de apuestas por ciclo a intentar
_bot_max_bets = 5           # máximo de apuestas por ciclo
_bot_max_exposure = 80.0    # % máximo del capital total desplegado
_bot_max_bet_usd = 1.0      # apuesta máxima por trade en USD
_bot_task = None            # asyncio.Task
_sports_client = None       # SportsDataClient (lazy init)
_sports_intel_task = None   # background refresh task para SportsIntelligence

# ── Escape Room state ──
_er_running: bool = False
_er_mode: str = "demo"
_er_interval: int = 30
_er_max_bets: int = 3
_er_time_horizon: str = "any"
_er_task = None
_er_engine: Optional[EscapeRoomEngine] = None
_er_session_bets: int = 0       # apuestas colocadas en sesión actual
_er_session_capital: float = 0.0  # capital apostado en sesión actual
_er_session_scans: int = 0      # ciclos de scan ejecutados

def get_er_engine():
    global _er_engine
    if _er_engine is None:
        from escape_room import EscapeRoomEngine
        _er_engine = EscapeRoomEngine(llm_engine=llm, sports_client=get_sports_client())
    return _er_engine

# ── Rush Mode globals ──
_rush_running: bool = False
_rush_mode: str = "demo"       # "demo" o "live"
_rush_bet_usd: float = 1.0     # apuesta por operación
_rush_max_capital: float = 3.0 # capital máximo total
_rush_spent: float = 0.0       # capital gastado en sesión actual
_rush_window_min: int = 25     # segundos mínimos antes del cierre para entrar
_rush_window_max: int = 90     # segundos máximos antes del cierre para entrar (default ampliado a 90s)
_rush_optimize_timing: bool = True  # señal débil entra solo en últimos 35s (ventana ajustada)
_rush_wins: int = 0
_rush_losses: int = 0
_rush_pnl: float = 0.0
_rush_total_pnl: float = 0.0   # P&L acumulado de todas las sesiones
_rush_task: Optional[asyncio.Task] = None
_rush_engine: Optional[object] = None
_rush_history: list = []        # historial de apuestas (máx 50)
_rush_bet_counter: int = 0      # ID incremental de apuestas
# ── Modo contrario: posiciones abiertas para averaging-down / salida parcial ──
_contrarian_positions: list = []  # [{"id","market_id","side","token","entry_price","total_size","total_usd","window_ts","avg_down_done","partial_exit_done"}]
# ── Historial persistente multi-bot ──────────────────────────────────────────
_PERSISTENT_HISTORY_FILE = pathlib.Path(__file__).parent / "data" / "trade_history.json"
_persistent_history: list = []   # cargado al arrancar, máx 500 trades

def _load_persistent_history() -> list:
    """Carga historial de trades desde disco al iniciar."""
    try:
        _PERSISTENT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _PERSISTENT_HISTORY_FILE.exists():
            with open(_PERSISTENT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data[-500:] if len(data) > 500 else data
    except Exception as e:
        print(f"[History] No se pudo cargar historial: {e}")
    return []

def _save_persistent_history() -> None:
    """Persiste historial a disco."""
    try:
        _PERSISTENT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERSISTENT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_persistent_history[-500:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[History] Error guardando historial: {e}")

def _record_trade(bot: str, trade_data: dict) -> None:
    """Registra un trade en el historial persistente."""
    global _persistent_history
    from datetime import datetime as _dt
    import time as _time
    # ID único: bot_timestamp_ms para evitar colisiones entre sesiones
    _default_id = f"{bot}_{int(_time.time() * 1000)}"
    entry = {
        "id":        trade_data.get("id", _default_id),
        "bot":       bot,
        "ts":        trade_data.get("ts", _dt.now().isoformat()),
        "market":    trade_data.get("question", trade_data.get("market_question", "")),
        "side":      trade_data.get("side", ""),
        "bet_usd":   trade_data.get("bet_usd", trade_data.get("amount", 0.0)),
        "pnl":       trade_data.get("pnl", 0.0),
        "status":    trade_data.get("status", "pending"),
        "mode":      trade_data.get("mode", "demo"),
        "asset":     trade_data.get("asset", ""),
        "price":     trade_data.get("price", trade_data.get("signal_price", 0.0)),
    }
    _persistent_history.append(entry)
    if len(_persistent_history) > 500:
        _persistent_history = _persistent_history[-500:]
    _save_persistent_history()

_persistent_history = _load_persistent_history()
# ── Rush lax mode params ──
_rush_lax_mode: bool = False
_rush_btc_momentum_override: float = 0.0
_rush_poly_override: float = 0.0
_rush_zone_filter_disabled: bool = False
# ── Rush diagnostics ──
_rush_last_scan_msg: str = ""
_rush_last_scan_ts: str = ""

# ── Scan result cache (60s TTL) — evita re-escanear cuando el bot pide status cada 30s ──
_scan_cache: dict = {}          # key: (time_horizon, min_edge, category) → result
_scan_cache_ts: dict = {}       # key → unix timestamp del último scan
_SCAN_CACHE_TTL: int = 60       # segundos de validez del caché
# ── D&H lax mode params ──
_dh_lax_mode: bool = False
_dh_combined_override: float = 0.0
# ── D&H diagnostics ──
_dh_last_scan_msg: str = ""
_dh_last_scan_ts: str = ""
# ── ER config params ──
_er_require_dual_ai: bool = True
_er_score_min_override: float = 0.0
# ── ER diagnostics ──
_er_last_scan_msg: str = ""
_er_last_scan_ts: str = ""

def get_rush_engine():
    global _rush_engine
    if _rush_engine is None:
        _rush_engine = RushEngine(scanner)
    return _rush_engine

async def _auto_backtest_worker():
    """
    Backtest automático periódico: cada 6h analiza mercados resueltos,
    actualiza calibración y ajusta pesos del ensemble por categoría.
    SharkFlow by Carlos David Donoso Cordero (ddchack)
    """
    import time as _t
    global _ENSEMBLE_WEIGHTS, _ENSEMBLE_WEIGHTS_SOURCE, _last_auto_backtest_ts, _DDC_KL_NORM, _DDC_EV_NORM
    # Delay inicial de 8 minutos para no sobrecargar el arranque
    await asyncio.sleep(480)
    while True:
        try:
            _last_auto_backtest_ts = _t.time()
            print("[AutoBacktest] Iniciando backtest periódico (80 mercados)...")
            eng = MathEngine(max_capital=MAX_CAPITAL, kelly_multiplier=KELLY_FRACTION, min_edge=MIN_EDGE)
            bt = BacktestEngine(math_engine=eng, initial_capital=MAX_CAPITAL)
            result = await bt.run_backtest("auto_ddc_optimizer", num_markets=80)
            await bt.close()

            # Alimentar observaciones a la suite de calibración
            obs_fed = 0
            for t in result.trades:
                try:
                    _outcome = 1 if t.get("resolved_outcome") == "YES" else 0
                    _pred = t.get("estimated_prob", 0.5)
                    cal_suite.add_observation(float(_pred), int(_outcome))
                    obs_fed += 1
                except Exception:
                    pass
            if obs_fed >= 15:
                cal_suite.fit_all()
                print(f"[AutoBacktest] Calibración actualizada ({obs_fed} obs). Método: {cal_suite.get_status()['method']}")

            # Ajuste de pesos por categoría usando métricas del backtest
            cat_metrics = result.category_metrics or {}
            if result.total_trades >= 15:
                new_weights = {}
                for cat, m in cat_metrics.items():
                    wr = m.get("win_rate", 0.5)
                    n  = m.get("trades", 0)
                    if n < 3:
                        continue
                    # Si win_rate > 55% en una categoría → subir peso bayesiano (más confianza en el modelo)
                    # Si win_rate < 45% → bajar peso bayesiano, subir peso de mercado
                    base = _ENSEMBLE_WEIGHTS.get(cat, _ENSEMBLE_WEIGHTS["default"])
                    adj_bay = max(0.20, min(0.55, base[0] + (wr - 0.50) * 0.15))
                    adj_mkt = max(0.15, min(0.40, base[2] - (wr - 0.50) * 0.10))
                    adj_cat = round(1.0 - adj_bay - adj_mkt, 3)
                    adj_cat = max(0.20, min(0.55, adj_cat))
                    # Re-normalizar para que sumen ~1.0
                    total_w = adj_bay + adj_cat + adj_mkt
                    new_weights[cat] = [round(adj_bay/total_w, 3), round(adj_cat/total_w, 3), round(adj_mkt/total_w, 3)]

                if new_weights:
                    _ENSEMBLE_WEIGHTS.update(new_weights)
                    _ENSEMBLE_WEIGHTS_SOURCE = f"auto_backtest_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                    print(f"[AutoBacktest] Pesos DDC actualizados para {list(new_weights.keys())}. "
                          f"WR global: {result.win_rate:.1f}% | Trades: {result.total_trades} | "
                          f"Sharpe: {result.sharpe_estimate:.2f}")
            else:
                print(f"[AutoBacktest] Insuficientes trades ({result.total_trades}) — pesos no modificados.")

            # Actualizar bounds de normalización DDC con percentiles reales del backtest
            # Usar KL y edge_pct del 80th percentile de trades ganadores como referencia
            if result.trades:
                _winning_trades = [t for t in result.trades if t.get("resolved_outcome") == "YES"]
                if len(_winning_trades) >= 5:
                    _kl_vals = sorted([t.get("kl_divergence", 0) for t in _winning_trades])
                    _ev_vals = sorted([t.get("expected_value", 0) for t in _winning_trades])
                    _p80_idx = max(0, int(len(_kl_vals) * 0.80) - 1)
                    _new_kl_norm = round(_kl_vals[_p80_idx], 4) if _kl_vals else _DDC_KL_NORM
                    _new_ev_norm = round(_ev_vals[_p80_idx], 4) if _ev_vals else _DDC_EV_NORM
                    # Suavizar cambios para evitar oscilaciones (promedio con valor actual)
                    _DDC_KL_NORM = round(_DDC_KL_NORM * 0.6 + _new_kl_norm * 0.4, 4)
                    _DDC_EV_NORM = round(_DDC_EV_NORM * 0.6 + _new_ev_norm * 0.4, 4)
                    # Clamp en rangos razonables
                    _DDC_KL_NORM = max(0.04, min(0.25, _DDC_KL_NORM))
                    _DDC_EV_NORM = max(0.04, min(0.30, _DDC_EV_NORM))
                    print(f"[AutoBacktest] DDC bounds actualizados → KL_norm={_DDC_KL_NORM}, EV_norm={_DDC_EV_NORM}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[AutoBacktest] Error en ciclo: {e}")

        await asyncio.sleep(_AUTO_BACKTEST_INTERVAL_H * 3600)


# ── Worker de reconexión automática al CLOB de Polymarket ─────────────────
# Los creds generados por create_or_derive_api_creds() expiran en ~3-6h.
# Este worker refresca la conexión cada 2.5h para evitar errores de auth.
_reconnect_task = None
_CLOB_RECONNECT_INTERVAL_H: float = 2.5  # horas entre reconexiones preventivas
_clob_connection_status: dict = {"connected": False, "last_check": None, "consecutive_failures": 0}

async def _auto_reconnect_worker():
    """Refresca periódicamente los creds del CLOB con retry inteligente."""
    await asyncio.sleep(600)
    global _clob_connection_status
    while True:
        try:
            if trader.private_key and getattr(trader, 'funder', None):
                ok = trader.ensure_connected(max_age_hours=_CLOB_RECONNECT_INTERVAL_H, max_retries=5)
                _clob_connection_status["last_check"] = datetime.now(timezone.utc).isoformat()
                if ok:
                    prev = _clob_connection_status["consecutive_failures"]
                    _clob_connection_status["connected"] = True
                    _clob_connection_status["consecutive_failures"] = 0
                    if prev > 0:
                        print(f"[AutoReconnect] Conexion CLOB restaurada tras {prev} fallos.")
                    else:
                        print(f"[AutoReconnect] Conexion CLOB verificada/renovada OK.")
                else:
                    _clob_connection_status["connected"] = False
                    _clob_connection_status["consecutive_failures"] += 1
                    n = _clob_connection_status["consecutive_failures"]
                    print(f"[AutoReconnect] ALERTA: CLOB perdido ({n} fallos consecutivos).")
                    if n <= 5:
                        await asyncio.sleep(300)  # reintentar en 5 min
                        continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[AutoReconnect] Error: {e}")
        await asyncio.sleep(_CLOB_RECONNECT_INTERVAL_H * 3600)


async def _sports_intel_refresh_worker():
    """
    Refresca ELO y lesiones deportivas cada 24h.
    Espera 30min al inicio para no competir con el startup del servidor.
    """
    await asyncio.sleep(1800)  # 30 min antes del primer refresh
    while True:
        try:
            from sports_intel import get_sports_intel as _get_si
            _si = _get_si()
            if _si is not None:
                await _si.refresh()
                print("[SportsIntel] ELO + lesiones actualizados.")
        except asyncio.CancelledError:
            break
        except Exception as _e:
            print(f"[SportsIntel] Error en refresh periódico: {_e}")
        await asyncio.sleep(24 * 3600)  # cada 24h


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ██ SHARKFLOW v5.0 — Prediction Market Intelligence ██  ║")
    print("║  by Carlos David Donoso Cordero (ddchack)               ║")
    print("║  KL·MultiKelly·ELO·Poisson·Bayes·HMM·MonteCarlo·LMSR  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    # Auto-cargar API keys del .env al LLM engine (evita que estén vacías tras reinicio)
    try:
        import pathlib as _pl
        _env_path = _pl.Path(__file__).parent.parent / ".env"
        if not _env_path.exists():
            _env_path = _pl.Path(__file__).parent / ".env"
        if _env_path.exists():
            _env_kv = {}
            for _line in _env_path.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _env_kv[_k.strip()] = _v.strip().strip('"').strip("'")
            _ak = _env_kv.get("ANTHROPIC_API_KEY", "")
            _ok = _env_kv.get("OPENAI_API_KEY", "")
            _am = _env_kv.get("ANTHROPIC_MODEL", "")
            _om = _env_kv.get("OPENAI_MODEL", "")
            _md = _env_kv.get("LLM_MODE", "")
            if _ak or _ok:
                llm.configure(_ak, _ok, _am, _om, _md)
                swarm_engine.configure(
                    anthropic_key=_ak,
                    openai_key=_ok,
                    anthropic_model="claude-haiku-4-5-20251001",
                    openai_model="gpt-4o-mini",
                )
                calibration_tracker.apply_to_swarm_engine(swarm_engine)
                _keys_info = []
                if _ak: _keys_info.append("Anthropic")
                if _ok: _keys_info.append("OpenAI")
                print(f"[LLM] API keys cargadas desde .env: {', '.join(_keys_info)}")
                print(f"[SwarmEngine] Configurado con {len(PERSONAS)} personas")
    except Exception as _e:
        print(f"[LLM] No se pudieron cargar keys desde .env: {_e}")

    # Auto-cargar claves Polymarket del .env si no vienen de variables de entorno
    # (os.getenv() solo lee vars del shell; si el usuario usa .env pero no lo exporta, quedan vacías)
    try:
        _env_path2 = _pl.Path(__file__).parent.parent / ".env"
        if not _env_path2.exists():
            _env_path2 = _pl.Path(__file__).parent / ".env"
        if _env_path2.exists():
            _env_kv2 = {}
            for _line in _env_path2.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k2, _, _v2 = _line.partition("=")
                    _env_kv2[_k2.strip()] = _v2.strip().strip('"').strip("'")
            _pk2 = _env_kv2.get("POLYMARKET_PRIVATE_KEY", "")
            _fa2 = _env_kv2.get("POLYMARKET_FUNDER_ADDRESS", "")
            _st2 = int(_env_kv2.get("POLYMARKET_SIGNATURE_TYPE", "1"))
            _needs_update = False
            if _pk2 and not trader.private_key:
                trader.private_key = _pk2; _needs_update = True
            if _fa2 and not trader.funder:
                trader.funder = _fa2; _needs_update = True
            if _st2 and trader.signature_type != _st2:
                trader.signature_type = _st2
            if _needs_update:
                print(f"[Trader] Claves cargadas desde .env (no estaban en variables de entorno)")
    except Exception as _e2:
        print(f"[Trader] No se pudieron cargar claves desde .env: {_e2}")

    # Auto-inicializar trader si hay credenciales configuradas
    if trader.private_key and trader.funder:
        try:
            result = trader.initialize()
            if result.get("status") == "authenticated":
                print(f"[Trader] Autenticado en Polymarket CLOB — funder: {trader.funder[:10]}...")
            else:
                print(f"[Trader] Auth fallida: {result.get('message', 'desconocido')}")
        except Exception as e:
            print(f"[Trader] Error al inicializar: {e}")
    else:
        print("[Trader] Sin credenciales — modo DRY_RUN activo")

    # Iniciar worker de backtest automático periódico (cada 6h)
    global _backtest_task, _reconnect_task
    _backtest_task = asyncio.create_task(_auto_backtest_worker())
    print("[AutoBacktest] Worker iniciado — primer ciclo en 8 minutos")

    # Iniciar worker de reconexión automática al CLOB (cada 2.5h)
    _reconnect_task = asyncio.create_task(_auto_reconnect_worker())
    print("[AutoReconnect] Worker iniciado — reconexión preventiva cada 2.5h")

    # Iniciar SportsIntelligence — ELO histórico + lesiones ESPN
    global _sports_intel_task
    try:
        _newsapi_key_si = os.getenv("NEWSAPI_KEY", "")
        if not _newsapi_key_si and "_env_kv" in dir():
            _newsapi_key_si = _env_kv.get("NEWSAPI_KEY", "")
        from sports_intel import init_sports_intel
        _si = init_sports_intel(newsapi_key=_newsapi_key_si)
        asyncio.create_task(_si.async_init())   # fire-and-forget: no bloquea startup
        _sports_intel_task = asyncio.create_task(_sports_intel_refresh_worker())
        print("[SportsIntel] Inicializando ELO desde ESPN (background)…")
    except Exception as _sie:
        print(f"[SportsIntel] No se pudo iniciar: {_sie}")

    # Cargar pesos ensemble guardados en disco (persisten entre reinicios)
    global _ENSEMBLE_WEIGHTS, _ENSEMBLE_WEIGHTS_SOURCE
    _weights_path_load = os.path.join(os.path.dirname(__file__), "ensemble_weights.json")
    if os.path.exists(_weights_path_load):
        try:
            with open(_weights_path_load, "r", encoding="utf-8") as _wf:
                _saved = json.load(_wf)
                if isinstance(_saved.get("weights"), dict):
                    _ENSEMBLE_WEIGHTS.update(_saved["weights"])
                    _ENSEMBLE_WEIGHTS_SOURCE = _saved.get("source", "loaded_from_disk")
                    print(f"[Weights] Pesos ensemble cargados desde disco: {_ENSEMBLE_WEIGHTS_SOURCE}")
        except Exception as _we:
            print(f"[Weights] No se pudieron cargar pesos guardados: {_we}")

    yield

    # Cancelar workers al apagar
    for _task in (_backtest_task, _reconnect_task, _sports_intel_task):
        if _task and not _task.done():
            _task.cancel()
            try:
                await _task
            except asyncio.CancelledError:
                pass

    await scanner.close(); await sentiment.close(); await bt_engine.close()
    await arb.close(); await whales.close(); await rt.disconnect(); await llm.close()
    if _sports_client is not None:
        await _sports_client.close()
    try:
        from sports_intel import get_sports_intel as _get_si
        _si_inst = _get_si()
        if _si_inst:
            await _si_inst.close()
    except Exception:
        pass
    if _rush_engine is not None:
        await _rush_engine.close()

app = FastAPI(title="SharkFlow v6.0 — PolySwarm Edition", version="6.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ═══════════ MODELS ═══════════
class TradeRequest(BaseModel):
    token_id: str; side: str; amount_usd: float; order_type: str = "MARKET"
    price: Optional[float] = None; market_id: str = ""; question: str = ""
    confidence: float = 0; edge_pct: float = 0

class BacktestRequest(BaseModel):
    strategy_name: str = "Default"; num_markets: int = 50
    kelly_multiplier: float = 0.25; min_edge: float = 0.03

class MonteCarloRequest(BaseModel):
    n_simulations: int = 1000; kelly_fraction: float = 0.25

class AutoAllocateRequest(BaseModel):
    total_budget: float = 20.0
    time_horizon: str = "ALL"   # SHORT, MEDIUM, LONG, ALL
    min_bets: int = 5
    max_bets: int = 20

class WhaleAddRequest(BaseModel):
    address: str; alias: str = ""

class ConfigUpdate(BaseModel):
    max_capital: Optional[float] = None; kelly_fraction: Optional[float] = None
    min_edge: Optional[float] = None; dry_run: Optional[bool] = None

class KeysUpdate(BaseModel):
    polymarket_private_key: Optional[str] = None
    polymarket_funder_address: Optional[str] = None
    polymarket_signature_type: Optional[int] = None
    newsapi_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    backend_port: Optional[int] = None
    max_capital: Optional[float] = None
    kelly_fraction: Optional[float] = None
    min_edge: Optional[float] = None

# ═══════════ HELPERS ═══════════
async def broadcast(event: str, data: dict):
    msg = json.dumps({"event": event, "data": data, "ts": datetime.now(timezone.utc).isoformat()})
    dead = []
    for ws in dashboard_ws:
        try: await ws.send_text(msg)
        except: dead.append(ws)
    for ws in dead: dashboard_ws.remove(ws)

# ═══════════════════════════════════════════════════════════
# AI CACHE — evita re-llamar IAs para el mismo mercado en 5min
# SharkFlow by Carlos David Donoso Cordero (ddchack)
# ═══════════════════════════════════════════════════════════
import time as _time_module
_ai_cache: dict = {}      # market_id → (timestamp, result_dict)
AI_CACHE_TTL: int = 300   # 5 minutes

def _get_ai_cache(market_id: str) -> Optional[dict]:
    """Returns cached AI result if fresh, else None."""
    if market_id in _ai_cache:
        ts, result = _ai_cache[market_id]
        if _time_module.time() - ts < AI_CACHE_TTL:
            return result
        del _ai_cache[market_id]
    return None

def _set_ai_cache(market_id: str, result: dict) -> None:
    """Cache AI result for a market."""
    _ai_cache[market_id] = (_time_module.time(), result)

# ═══════════════════════════════════════════════════════════
# AI RECOMMENDATION VALUES — for weighted consensus
# ═══════════════════════════════════════════════════════════
_AI_REC_VALUES: dict = {
    "APOSTAR_FUERTE": 1.0,
    "APOSTAR":        0.7,
    "ESPERAR":        0.3,
    "NO_APOSTAR":     0.0,
    "PASAR":          0.0,
}

def _parse_structured_ai_response(text: str, market_idx: int) -> dict:
    """
    Parse structured JSON AI response for a single market in a batch.
    Falls back to legacy PROB/REC/CONF format, then to NO_APOSTAR on any error.
    """
    import re as _re_parse
    import json as _json_parse
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        d = _json_parse.loads(clean)
        raw_conf = float(d.get("confianza", 0.5))
        conf_f = raw_conf if raw_conf <= 1.0 else raw_conf / 100.0
        conf_num = max(0, min(100, int(conf_f * 100)))
        return {
            "ai_prob": max(0.02, min(0.98, float(d.get("probabilidad_estimada", 0.5)))),
            "ai_recommendation": str(d.get("recomendacion", "NO_APOSTAR")).upper(),
            "ai_confidence_num": conf_num,
            "ai_confidence": "ALTA" if conf_f >= 0.70 else ("MEDIA" if conf_f >= 0.50 else "BAJA"),
            "ai_edge_estimate": float(d.get("edge_estimado", 0.0)),
            "ai_risk_level": str(d.get("nivel_riesgo", "medium")).lower(),
            "ai_key_factors": d.get("factores_clave", []),
            "ai_reasoning": str(d.get("razonamiento", ""))[:200],
        }
    except Exception:
        # Try legacy format: "N. PROB=0.XX REC=APOSTAR CONF=ALTA"
        pat = rf"{market_idx}\.\s*PROB=(\d+\.\d+)\s+REC=(\w+)\s+CONF=(\w+)"
        m = _re_parse.search(pat, text, _re_parse.IGNORECASE)
        if m:
            conf_str = m.group(3).upper()
            conf_num = 80 if conf_str == "ALTA" else (60 if conf_str == "MEDIA" else 40)
            return {
                "ai_prob": float(m.group(1)),
                "ai_recommendation": m.group(2).upper(),
                "ai_confidence": conf_str,
                "ai_confidence_num": conf_num,
                "ai_edge_estimate": 0.0,
                "ai_risk_level": "medium",
                "ai_key_factors": [],
                "ai_reasoning": "",
            }
        return {
            "ai_prob": 0.5, "ai_recommendation": "NO_APOSTAR",
            "ai_confidence": "BAJA", "ai_confidence_num": 30,
            "ai_edge_estimate": 0.0, "ai_risk_level": "high",
            "ai_key_factors": [], "ai_reasoning": "Parse error",
        }

# ═══════════════════════════════════════════════════════════
# HELPER: AI ENSEMBLE BOOST — Integra LLM en pipeline de scan
# SharkFlow by Carlos David Donoso Cordero (ddchack)
# ═══════════════════════════════════════════════════════════

async def _ai_ensemble_boost(top_recs: list, llm_inst) -> dict:
    """
    Llama al LLM con un batch prompt para las top recomendaciones.
    Retorna dict {market_id: {"ai_prob": float, "ai_recommendation": str, ...}}.
    v5: structured JSON responses, AI cache (5min TTL), weighted consensus.
    El peso de la IA en el ensemble escala con composite_score:
      - composite < 0.30  → ai_weight = 0.05  (5% — presencia mínima)
      - composite 0.30-0.60 → ai_weight = 0.12 (12% — influencia moderada)
      - composite > 0.60  → ai_weight = 0.20  (20% — máxima influencia)
    """
    if not top_recs or (not llm_inst.anthropic_key and not llm_inst.openai_key):
        return {}

    # Check cache first — avoid re-calling AI for fresh results
    result: dict = {}
    uncached_recs = []
    for r in top_recs:
        mid = r.get("market_id", "")
        cached = _get_ai_cache(mid)
        if cached is not None:
            result[mid] = cached
        else:
            uncached_recs.append(r)

    if not uncached_recs:
        print(f"[AIBoost] All {len(result)} results served from cache.")
        return result

    # Construir prompt batch con formato estructurado mejorado
    # Obtener contexto BTC actual para mercados crypto
    _btc_ctx = ""
    _has_crypto = any(r.get("category", "").lower() in ("crypto", "cryptocurrency") for r in uncached_recs)
    if _has_crypto:
        try:
            import httpx as _hx
            async with _hx.AsyncClient(timeout=3.0) as _hc:
                _kr = await _hc.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT")
                if _kr.status_code == 200:
                    _kd = _kr.json()
                    _btc_price = float(_kd.get("lastPrice", 0))
                    _btc_chg24 = float(_kd.get("priceChangePercent", 0))
                    _btc_vol = float(_kd.get("volume", 0))
                    _btc_ctx = (
                        f"\nCONTEXTO CRYPTO ACTUAL: BTC=${_btc_price:,.0f} | "
                        f"24h: {_btc_chg24:+.2f}% | Vol24h: {_btc_vol:,.0f} BTC\n"
                    )
        except Exception:
            pass

    lines = []
    for i, r in enumerate(uncached_recs, 1):
        side_price = r["yes_price"] if r["side"] == "YES" else r["no_price"]
        kl_str = f" | KL={r['kl_divergence']:.4f}" if r.get('kl_divergence') else ""
        vol_str = f" | Volumen24h=${r.get('volume_24h', 0):,.0f}" if r.get('volume_24h', 0) > 0 else ""
        lines.append(
            f"{i}. [{r['category'].upper()}] {r['question']}\n"
            f"   Precio: YES={r['yes_price']:.3f} NO={r['no_price']:.3f} | "
            f"Lado sugerido: {r['side']} @ {side_price:.3f}\n"
            f"   Modelo: prob={r['ensemble_prob']:.3f} | "
            f"Edge={r.get('edge_pct',0)*100:.1f}% | EV={r.get('ev',0):.4f}{kl_str}\n"
            f"   Liquidez=${r.get('liquidity',0):,.0f}{vol_str} | DDC={int(r.get('composite_score',0)*100)}/100 | "
            f"Horas resolución: {r.get('hours_to_resolution','?')}"
        )

    prompt = (
        "Eres un trader cuantitativo especializado en mercados de predicción (Polymarket). "
        "Capital real limitado — sé conservador: prefiere NO_APOSTAR ante duda.\n"
        "Criterios de decisión: "
        "APOSTAR_FUERTE = edge >10% + alta confianza; "
        "APOSTAR = edge 5-10% o señal clara; "
        "ESPERAR = edge ambiguo 2-5%; "
        "NO_APOSTAR = edge <2% o precio justo.\n"
        + _btc_ctx +
        "\nFormato respuesta — una línea JSON por mercado (sin markdown):\n"
        '{"recomendacion":"APOSTAR_FUERTE|APOSTAR|ESPERAR|NO_APOSTAR",'
        '"probabilidad_estimada":0.XX,'
        '"confianza":0.XX,'
        '"edge_estimado":-0.20_to_0.40,'
        '"factores_clave":["f1","f2"],'
        '"nivel_riesgo":"low|medium|high",'
        '"razonamiento":"una frase concisa"}\n\n'
        "REGLAS: confianza en [0.0,1.0]. edge_estimado = tu_prob - precio_mercado. "
        "Si el modelo_prob es similar al precio de mercado (|diff|<0.03), responde NO_APOSTAR.\n\n"
        "MERCADOS:\n" + "\n".join(lines) + "\n\n"
        f"Responde exactamente {len(uncached_recs)} líneas JSON (en orden):"
    )

    try:
        import re as _re
        # Call both AIs in parallel when both keys available
        claude_text = None
        gpt_text = None

        async def _claude_call():
            if not llm_inst.anthropic_key:
                return None
            try:
                text, _ = await asyncio.wait_for(llm_inst._call_anthropic(prompt), timeout=20.0)
                return text
            except Exception as _ce:
                print(f"[AIBoost] Claude error: {_ce}")
                return None

        async def _gpt_call():
            if not llm_inst.openai_key:
                return None
            try:
                resp = await asyncio.wait_for(
                    llm_inst.client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {llm_inst.openai_key}", "Content-Type": "application/json"},
                        json={"model": llm_inst.openai_model, "max_tokens": 500,
                              "messages": [
                                  {"role": "system", "content": "Eres un analista cuantitativo de mercados de predicción. Responde ÚNICAMENTE con JSON válido por línea, sin markdown."},
                                  {"role": "user", "content": prompt}],
                              "temperature": 0.2}
                    ),
                    timeout=20.0
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as _ge:
                print(f"[AIBoost] GPT error: {_ge}")
                return None

        # Llamadas paralelas a ambas IAs (era secuencial, duplicaba latencia)
        claude_text, gpt_text = await asyncio.gather(_claude_call(), _gpt_call())

        # Parse responses and compute weighted consensus
        for i, r in enumerate(uncached_recs, 1):
            mid = r.get("market_id", "")
            # Extract line i as JSON object from each AI response
            claude_line = ""
            gpt_line = ""
            if claude_text:
                json_lines = [ln.strip() for ln in claude_text.split("\n") if ln.strip().startswith("{")]
                claude_line = json_lines[i - 1] if i - 1 < len(json_lines) else claude_text
            if gpt_text:
                json_lines_g = [ln.strip() for ln in gpt_text.split("\n") if ln.strip().startswith("{")]
                gpt_line = json_lines_g[i - 1] if i - 1 < len(json_lines_g) else gpt_text

            claude_parsed = _parse_structured_ai_response(claude_line, i) if claude_line else None
            gpt_parsed    = _parse_structured_ai_response(gpt_line, i)   if gpt_line    else None

            if claude_parsed and gpt_parsed:
                # Weighted consensus: confidence-weighted recommendation value
                c_conf = claude_parsed["ai_confidence_num"]
                g_conf = gpt_parsed["ai_confidence_num"]
                c_val  = _AI_REC_VALUES.get(claude_parsed["ai_recommendation"], 0.0)
                g_val  = _AI_REC_VALUES.get(gpt_parsed["ai_recommendation"], 0.0)
                total_conf = max(c_conf + g_conf, 1)
                consensus_val = (c_conf * c_val + g_conf * g_val) / total_conf
                if consensus_val >= 0.85:
                    consensus_rec = "APOSTAR_FUERTE"
                elif consensus_val >= 0.55:
                    consensus_rec = "APOSTAR"
                elif consensus_val >= 0.25:
                    consensus_rec = "ESPERAR"
                else:
                    consensus_rec = "NO_APOSTAR"
                consensus_prob = (c_conf * claude_parsed["ai_prob"] + g_conf * gpt_parsed["ai_prob"]) / total_conf
                entry = {
                    "ai_prob": round(consensus_prob, 4),
                    "ai_recommendation": consensus_rec,
                    "ai_confidence": "ALTA" if max(c_conf, g_conf) >= 70 else ("MEDIA" if max(c_conf, g_conf) >= 50 else "BAJA"),
                    "ai_confidence_num": round((c_conf + g_conf) / 2),
                    "ai_edge_estimate": round((claude_parsed["ai_edge_estimate"] + gpt_parsed["ai_edge_estimate"]) / 2, 4),
                    "ai_risk_level": claude_parsed["ai_risk_level"],
                    "ai_reasoning": f"Claude: conf={c_conf}% edge={claude_parsed['ai_edge_estimate']:.2f} -> {claude_parsed['ai_recommendation']} | GPT: conf={g_conf}% -> {gpt_parsed['ai_recommendation']}",
                    "ai_consensus_val": round(consensus_val, 3),
                }
            elif claude_parsed:
                entry = {**claude_parsed, "ai_consensus_val": _AI_REC_VALUES.get(claude_parsed["ai_recommendation"], 0)}
            elif gpt_parsed:
                entry = {**gpt_parsed, "ai_consensus_val": _AI_REC_VALUES.get(gpt_parsed["ai_recommendation"], 0)}
            else:
                entry = {"ai_prob": r["ensemble_prob"], "ai_recommendation": "NO_APOSTAR",
                         "ai_confidence": "BAJA", "ai_confidence_num": 30,
                         "ai_edge_estimate": 0.0, "ai_risk_level": "high",
                         "ai_consensus_val": 0.0, "ai_reasoning": "No AI response"}

            result[mid] = entry
            _set_ai_cache(mid, entry)

        return result
    except Exception as e:
        print(f"[AIBoost] Error en llamada LLM: {e}")
        return result  # return whatever was cached even if batch failed


# ═══════════════════════════════════════════════════════════
# CORE ENDPOINT: v3 SCANNER — Full Algorithm Pipeline
# ═══════════════════════════════════════════════════════════

@app.get("/api/scan")
async def full_scan(limit: int = Query(50), with_sentiment: bool = Query(False), with_whales: bool = Query(False), time_horizon: str = Query("any"), min_edge_pct: float = Query(-1.0), category: str = Query("")):
    """
    v3 Full Algorithm Pipeline:
    1. Fetch markets from Gamma API
    2. Classify each market by category
    3. Run category-specific model (ELO/Poisson/OracleLag/Polls/Weather)
    4. Run Bayesian engine with sentiment + whale signals
    5. Combine via weighted ensemble → estimated probability
    6. Rank by KL Divergence (proven profit metric)
    7. Apply Multi-Kelly for simultaneous position sizing
    8. Risk-validate each recommendation
    """
    # ── Caché de 60s: evita re-llamar Gamma API + re-correr modelos en ciclos rápidos ──
    _cache_key = (time_horizon, round(min_edge_pct, 2), category.lower(), with_sentiment, with_whales)
    import time as _scan_time
    _cached_ts = _scan_cache_ts.get(_cache_key, 0)
    if _scan_time.time() - _cached_ts < _SCAN_CACHE_TTL and _cache_key in _scan_cache:
        _cached = _scan_cache[_cache_key]
        # Recalcular capital disponible en tiempo real (puede cambiar)
        _cached["capital_available"] = round(risk.state.current_capital - risk.state.total_exposure_usd, 2)
        _cached["from_cache"] = True
        _cached["cache_age_s"] = round(_scan_time.time() - _cached_ts)
        return _cached

    # Fetch multi-estrategia en paralelo para diversificar más allá de deportes:
    # 1) Top volumen 24h (mercados activos más populares — suelen ser deportes)
    # 2) Por createdAt desc (mercados recientes — política, crypto, tecnología)
    # 3) Por liquidez desc (mercados con capital — más variado)
    effective_limit = max(limit, 150)
    _batches = await asyncio.gather(
        scanner.fetch_active_markets(limit=effective_limit, order="volume24hr",  ascending=False),
        scanner.fetch_active_markets(limit=effective_limit, order="createdAt",   ascending=False),
        scanner.fetch_active_markets(limit=effective_limit, order="liquidity",   ascending=False),
        return_exceptions=True,
    )
    _seen_ids: set = set()
    markets: list = []
    for batch in _batches:
        if isinstance(batch, Exception) or not isinstance(batch, list):
            continue
        for m in batch:
            cid = m.get("conditionId", m.get("id", ""))
            if cid and cid not in _seen_ids:
                _seen_ids.add(cid)
                markets.append(m)
    whale_moves = []
    if with_whales and whales.whales:
        whale_moves = await whales.scan_whale_moves()

    raw_recs = []

    for mkt in markets:
        odds = scanner.parse_market_odds(mkt)
        if not odds: continue
        q = mkt.get("question", ""); cid = mkt.get("conditionId", mkt.get("id", ""))
        yt, nt = scanner.extract_token_ids(mkt)
        mkt_category = classify_market(q)

        # ── Category-specific model ──
        # Para deportes: enriquecer contexto con ELO real de SportsIntelligence
        _cat_ctx: dict = {}
        try:
            from sports_intel import get_sports_intel as _gsi
            _si_inst = _gsi()
            if _si_inst and _si_inst._initialized and mkt_category.value == "sports":
                _cat_ctx = _si_inst.get_context(q)
        except Exception:
            pass
        cat_signal = cat_eval.evaluate(q, odds.yes_price, context=_cat_ctx)

        # ── Sentiment ──
        sent_score, n_art = 0.0, 0
        if with_sentiment:
            try:
                rpt = await sentiment.get_sentiment_for_market(q)
                sent_score, n_art = rpt.avg_sentiment, rpt.articles_found
            except: pass

        # ── Whale consensus ──
        wpos = whales.get_whale_consensus(cid) if with_whales else []

        # ── Calcular horas a resolución para time-decay Bayesian ──
        _hrs_to_res = None
        _end_date_str = mkt.get("endDate", "")
        if _end_date_str:
            try:
                _end_dt = datetime.fromisoformat(_end_date_str.replace("Z", "+00:00"))
                if _end_dt.tzinfo is None:
                    _end_dt = _end_dt.replace(tzinfo=timezone.utc)
                _hrs_to_res = (_end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # ── Bayesian posterior ──
        # BUG5: usar historial de volumen para detectar volume surge real
        _prev_vol_scan = _volume_history_cache.get(cid, None)
        _avg_vol_scan = _prev_vol_scan if (_prev_vol_scan and _prev_vol_scan > 0) else odds.volume_24h * 1.5
        _volume_history_cache[cid] = odds.volume_24h
        bay_est = bayesian.estimate_with_raw_data(
            market_price=odds.yes_price, sentiment_score=sent_score,
            num_articles=n_art, volume_24h=odds.volume_24h,
            avg_volume_7d=_avg_vol_scan, whale_positions=wpos,
            spread=odds.spread, liquidity=odds.liquidity,
            hours_to_resolution=_hrs_to_res)

        # ── v4 Ensemble: Extremized aggregation + Advanced calibration ──
        model_probs = [bay_est.posterior, cat_signal.estimated_prob, odds.yes_price]
        _cat_str = mkt_category.value if isinstance(mkt_category, MarketCategory) else str(mkt_category)
        # Usar pesos globales (ajustables via /api/tune/apply o /api/tune/weights)
        _base_w = _ENSEMBLE_WEIGHTS.get(_cat_str, _ENSEMBLE_WEIGHTS["default"])
        model_weights = [_base_w[0], cat_signal.confidence * _base_w[1], _base_w[2]]

        # Sports sin señal real (ELO/Poisson): saltarse el extremizer.
        # El extremizer Satopää (d=1.73) asume que los forecasters humanos son
        # sistemáticamente subconfiados — no aplica cuando solo tenemos precio de mercado.
        # Efecto con d=1.73 en YES=0.34: 0.34 → 0.24 → edge falso de 8-12% en NO.
        _is_sports_no_signal = (
            _cat_str == "sports" and cat_signal.confidence < 0.15
        )
        if _is_sports_no_signal:
            # Media ponderada simple — sin extremizar
            _total_w = sum(model_weights)
            ensemble_prob = sum(p * w for p, w in zip(model_probs, model_weights)) / max(_total_w, 1e-6)
        else:
            # Satopää extremization (d≈1.73, 10-25% Brier improvement)
            ensemble_prob = Extremizer.weighted_extremize(
                list(zip(model_probs, model_weights)), d=extremizer_d)
        ensemble_prob = max(0.03, min(0.97, ensemble_prob))

        # ── v4 Advanced Calibration (Beta > Isotonic > Temp > Platt) ──
        cal_pred = cal_suite.calibrate(ensemble_prob)
        ensemble_prob = cal_pred.calibrated
        venn_uncertainty = cal_pred.uncertainty  # For Kelly position sizing

        # ── Mean Reversion pre-calc (needed for ensemble adjustment) ──
        _mp_pre = max(0.02, min(0.98, odds.yes_price))
        _logit_dist_pre = abs(math.log(_mp_pre / (1 - _mp_pre)))
        _mr_proxy_pre = round(min(1.0, _logit_dist_pre / 2.5), 3)
        _mr_side_pre = "FADE_UP" if odds.yes_price > 0.72 else ("FADE_DOWN" if odds.yes_price < 0.28 else "NEUTRAL")

        # ── Integrate MR signal into ensemble when price is extreme and signal is strong ──
        # Research: markets at extremes (>72% or <28%) show mean-reversion tendency.
        # When MR signal is strong (proxy > 0.35), add it as a 4th model at 12% weight.
        if _mr_proxy_pre > 0.35 and _mr_side_pre != "NEUTRAL":
            _mr_ensemble_prob = min(0.75, 0.5 + _mr_proxy_pre * 0.25) if _mr_side_pre == "FADE_UP" else max(0.25, 0.5 - _mr_proxy_pre * 0.25)
            # Solo invertir para FADE_DOWN (el precio está muy bajo, esperamos regresión al alza)
            # Para FADE_UP no invertir (el precio está muy alto, esperamos regresión a la baja = probabilidad baja de YES)
            if _mr_side_pre == "FADE_DOWN":
                _mr_ensemble_prob = 1.0 - _mr_ensemble_prob
            _mr_weight = _mr_proxy_pre * 0.12  # Up to 12% weight, proportional to signal strength
            model_probs.append(round(_mr_ensemble_prob, 4))
            model_weights.append(_mr_weight)

        # ── Confidence Gate: adaptive — only pull to market when liquid (spread < 3%) ──
        _probs_std = statistics.stdev(model_probs) if len(model_probs) > 1 else 0
        _high_disagreement = _probs_std > 0.15
        _market_is_liquid = odds.spread < 0.03 and odds.liquidity > 2000
        if _high_disagreement:
            if _market_is_liquid:
                # Liquid market → trust market price more → strong pull
                ensemble_prob = ensemble_prob * 0.65 + odds.yes_price * 0.35
            else:
                # Illiquid market → market price may be wrong → weaker pull
                ensemble_prob = ensemble_prob * 0.80 + odds.yes_price * 0.20

        # ── KL Divergence (profit ranking metric) ──
        kl_div = KLDivergence.binary_kl(ensemble_prob, odds.yes_price)

        # ── Math engine evaluation ──
        edge_override = (min_edge_pct / 100.0) if min_edge_pct >= 0 else None
        rec = math_engine.evaluate_market(
            market_id=cid, question=q, odds=odds,
            estimated_true_prob=ensemble_prob, sentiment_score=sent_score,
            available_capital=risk.state.current_capital - risk.state.total_exposure_usd,
            min_edge_override=edge_override)

        # Price zone filter: sweet spot $0.15-$0.50, avoid comfort trap $0.62-$0.85
        _price = odds.yes_price if rec and rec.side == "YES" else odds.no_price
        _in_sweet_spot = 0.15 <= _price <= 0.50
        _in_comfort_trap = 0.62 <= _price <= 0.85

        # ── Filtro de utilidad mínima 20% ─────────────────────────────────
        # Precio > 0.8333 → retorno potencial < 20% → descartar
        if rec and _price > 0.8333:
            continue

        if rec and kl_div > 0.01:
            # Binary Sharpe
            sharpe = BinarySharpe.single_bet(ensemble_prob, odds.yes_price)

            # ── Mean Reversion Proxy (sin fetch extra — cálculo O(1)) ──
            # Research: 58% Polymarket markets show negative autocorrelation.
            # Proxy: logit distance from 0.5 indicates how "extreme" price is.
            _mp = max(0.02, min(0.98, odds.yes_price))
            _logit_dist = abs(math.log(_mp / (1 - _mp)))  # 0=neutral, high=extreme
            _mr_direction = "FADE_UP" if odds.yes_price > 0.72 else ("FADE_DOWN" if odds.yes_price < 0.28 else "NEUTRAL")
            # Mean reversion aligns with bet if ensemble says OPPOSITE to price extreme
            _mr_aligned = (rec.side == "NO" and odds.yes_price > 0.72) or (rec.side == "YES" and odds.yes_price < 0.28)
            _mr_proxy = round(min(1.0, _logit_dist / 2.5), 3)   # normalized 0-1

            raw_recs.append({
                "market_id": cid, "question": q, "slug": mkt.get("slug", ""),
                "category": mkt_category.value if isinstance(mkt_category, MarketCategory) else str(mkt_category),
                "side": rec.side, "market_price": rec.market_price,
                "ensemble_prob": round(ensemble_prob, 4),
                "bayesian_prob": bay_est.posterior, "prior": bay_est.prior,
                "category_prob": round(cat_signal.estimated_prob, 4),
                "category_model": cat_signal.model_name,
                "category_confidence": round(cat_signal.confidence, 3),
                "uncertainty": bay_est.uncertainty,
                "venn_uncertainty": venn_uncertainty,
                "calibration_method": cal_pred.method,
                "extremizer_d": extremizer_d,
                "kl_divergence": round(kl_div, 6),
                "binary_sharpe": round(sharpe, 3),
                "edge_pct": rec.edge_percent, "ev": rec.expected_value,
                "kelly_individual": rec.kelly_fraction,
                "confidence": rec.confidence_score, "risk": rec.risk_level,
                "reasoning": rec.reasoning + [cat_signal.reasoning],
                "signals": bay_est.signals_used,
                "time_sensitivity": cat_signal.time_sensitivity,
                "yes_token": yt, "no_token": nt,
                "yes_price": round(odds.yes_price, 4),
                "no_price": round(odds.no_price, 4),
                "volume_24h": odds.volume_24h, "liquidity": odds.liquidity,
                "spread": odds.spread, "image": mkt.get("image", ""),
                "end_date": mkt.get("endDate", ""),
                "price_zone": "sweet_spot" if _in_sweet_spot else ("comfort_trap" if _in_comfort_trap else "other"),
                "model_disagreement": round(_probs_std, 4),
                "mr_proxy": _mr_proxy,
                "mr_direction": _mr_direction,
                "mr_aligned": _mr_aligned,
                "hours_to_resolution": round(_hrs_to_res, 1) if _hrs_to_res is not None else None,
                # Breakdown de los 3 modelos del ensemble para auditoría
                "model_breakdown": {
                    "bayesian": round(model_probs[0], 4),
                    "category":  round(model_probs[1], 4),
                    "market":    round(model_probs[2], 4),
                    "weights":   [round(w, 3) for w in model_weights],
                },
            })

    # ── Composite DDC Score v5 ─────────────────────────────────────────────────
    # v5 adds on top of v3: Bayesian KL shrinkage, Kelly consistency penalty,
    #     Market age factor, Consensus diversity bonus, Anti-overfitting guard.
    # v3 base: KL(35%) + EV_liq(22%) + Conf(12%) + MR(10%) + Diversity(5%) +
    #          Zone(5%) + TimeUrgency(variable) + Volume + PriceEff + AI bonus.
    # Los bounds _DDC_KL_NORM y _DDC_EV_NORM se actualizan cada 6h via auto-backtest.
    for _r in raw_recs:
        # DDC v3: normalización por categoría — cada tipo de mercado tiene su propio "típico"
        _cat     = _r.get("category", "default").lower()
        _cat_nrm = _DDC_CATEGORY_NORMS.get(_cat, _DDC_CATEGORY_NORMS["default"])
        _kl_norm_use = max(0.04, _cat_nrm["kl"])
        _ev_norm_use = max(0.04, _cat_nrm["ev"])
        _liq    = _r.get("liquidity", 0.0)
        # DDC v5: Bayesian KL shrinkage — n_samples proportional to liquidity
        _n_samples_kl = min(int(_liq / 100), 500)
        _kl_shrunk = _ddc_bayesian_kl(_r["kl_divergence"], _cat_nrm["kl"], n_samples=_n_samples_kl)
        # KL normalizado (bound por categoría + adaptativo global mezclado 70/30) using shrunk KL
        _kl_n   = min(_kl_shrunk / (_kl_norm_use * 0.70 + _DDC_KL_NORM * 0.30), 1.0)
        # EV ajustado por liquidez — mercados ilíquidos tienen EV poco confiable
        _ev_raw = min(max(_r.get("ev", 0), 0.0) / (_ev_norm_use * 0.70 + _DDC_EV_NORM * 0.30), 1.0)
        _liq_f  = 0.20 if _liq < 500 else (0.50 if _liq < 2000 else (0.80 if _liq < 5000 else 1.00))
        _ev_n   = _ev_raw * _liq_f
        _conf_raw = _r.get("confidence", 50)
        _conf_n = min(_conf_raw / 100.0, 1.0)
        _mr_b   = 0.10 if _r.get("mr_aligned", False) and _r.get("mr_proxy", 0) > 0.3 else 0.0
        _div_n  = _r.get("diversity_score", 1.0)
        # Zone bonus/penalty: sweet_spot +5%, comfort_trap -8% (soft filter — incluye pero penaliza)
        _zone_b = (0.05 if _r.get("price_zone") == "sweet_spot"
                   else -0.08 if _r.get("price_zone") == "comfort_trap"
                   else 0.0)
        # Time urgency bonus: mercados que cierran pronto tienen edge más estable
        # v6: Category-adjusted urgency — crypto edges mueven rápido, política es estable
        _hrs    = _r.get("hours_to_resolution")
        _cat_urg_mult = {"crypto": 1.25, "sports": 1.20, "economics": 1.0,
                         "politics": 0.80, "default": 1.0}.get(_cat, 1.0)
        _urg_b_raw = (0.07 if _hrs is not None and _hrs <= 6
                      else 0.05 if _hrs is not None and _hrs <= 24
                      else 0.03 if _hrs is not None and _hrs <= 72
                      else 0.0)
        _urg_b = round(min(_urg_b_raw * _cat_urg_mult, 0.10), 4)
        # DDC v3: Close market premium — si cierra en ≤2h, bonus extra sobre urgency
        _close_prem = 0.03 if _hrs is not None and _hrs <= 2 else 0.0
        # DDC v3: Volume factor — mercados con volumen alto tienen precio más eficiente
        _vol_b   = 0.03 if _r.get("volume_24h", 0) > 10000 else 0.0
        # DDC v3: Price efficiency bonus — precios 0.45-0.55 indican mercado incierto = más oportunidad
        _yes_p   = _r.get("yes_price", 0.5)
        _price_eff = 0.04 if (0.45 <= _yes_p <= 0.55) else 0.0
        # AI bonus — v3: si AI recomienda APOSTAR con confianza alta → 0.03 (era 0.01)
        _ai_conf_high = _r.get("ai_confidence", "") in ("ALTA", "MEDIA")
        _ai_b   = (0.03 if _r.get("ai_recommendation") == "APOSTAR" and _ai_conf_high
                   else 0.01 if _r.get("ai_recommendation") == "APOSTAR" else 0.0)
        # DDC v5: Consensus diversity bonus
        _model_cnt = _r.get("model_count", 5)
        _v5_consensus_b = _ddc_consensus_bonus(_div_n, _model_cnt)
        # Raw composite before v5 correction passes
        _raw_composite = min(
            _kl_n * 0.35 + _ev_n * 0.22 + _conf_n * 0.12 + _mr_b + _div_n * 0.05
            + _zone_b + _urg_b + _close_prem + _vol_b + _price_eff + _ai_b + _v5_consensus_b,
            1.0
        )
        # DDC v5 Pass 1: Kelly consistency penalty
        _kelly_frac_v5 = _r.get("kelly_individual", 0.15) or 0.15
        _raw_composite = _ddc_kelly_consistency(_raw_composite, _kelly_frac_v5)
        # DDC v5 Pass 2: Anti-overfitting guard
        _raw_composite = _ddc_anti_overfit(_raw_composite, _conf_raw, _liq)
        # DDC v5 Pass 3: Market age factor (default market lifetime = 168h)
        _total_dur   = 168.0
        _hrs_elapsed = (_total_dur - _hrs) if _hrs is not None else (_total_dur * 0.5)
        _age_f       = _ddc_market_age_factor(_hrs_elapsed, _total_dur)
        _raw_composite = _raw_composite * _age_f
        _r["composite_score"] = round(min(_raw_composite, 1.0), 4)
        # Exponer los componentes para el dashboard
        _r["ddc_components"] = {
            "kl": round(_kl_n * 0.35, 4), "ev_liq": round(_ev_n * 0.22, 4),
            "conf": round(_conf_n * 0.12, 4), "mr_bonus": round(_mr_b, 4),
            "urgency": round(_urg_b, 4), "close_premium": round(_close_prem, 4),
            "volume_factor": round(_vol_b, 4), "price_efficiency": round(_price_eff, 4),
            "liq_factor": round(_liq_f, 2), "category_norm": _cat,
            "urg_cat_mult": round(_cat_urg_mult, 2),
            # DDC v5 debug components
            "bayesian_shrinkage": round(_kl_shrunk, 6),
            "market_age": round(_age_f, 3),
            "consensus": round(_v5_consensus_b, 4),
        }
    raw_recs.sort(key=lambda r: r["composite_score"], reverse=True)

    # ── AI Ensemble Boost: integra LLM en recomendaciones con score ≥ 0.05 ──
    # v3: red más amplia (era top 10 con score ≥ 0.10).
    # Para score 0.05-0.10: análisis ligero (solo APOSTAR/NO con 1 razón).
    # Para score > 0.10: análisis completo.
    # Falla silenciosamente → no bloquea el scan.
    _ai_boost_n = len(raw_recs)  # todos los elegibles
    _ai_boost_eligible = [r for r in raw_recs[:min(_ai_boost_n, 30)] if r.get("composite_score", 0) >= 0.05]
    if _ai_boost_eligible:
        try:
            _ai_results = await asyncio.wait_for(_ai_ensemble_boost(_ai_boost_eligible, llm), timeout=25.0)
        except asyncio.TimeoutError:
            print("[Scan] AI Ensemble Boost timeout (25s) — continuando sin ajuste IA")
            _ai_results = {}
        if _ai_results:
            for _r in raw_recs:
                _mid = _r.get("market_id", "")
                if _mid not in _ai_results:
                    continue
                _ai = _ai_results[_mid]
                _ai_prob = _ai.get("ai_prob", _r["ensemble_prob"])
                _ai_conf = _ai.get("ai_confidence", "BAJA")
                _ai_rec  = _ai.get("ai_recommendation", "PASAR")
                # Peso de IA: escala con composite_score → mayor beneficio = más peso IA
                _cs = _r.get("composite_score", 0)
                if _cs >= 0.60:
                    _ai_w = 0.20
                elif _cs >= 0.30:
                    _ai_w = 0.12
                else:
                    _ai_w = 0.05
                # Reducir peso si IA tiene baja confianza
                if _ai_conf == "BAJA":
                    _ai_w *= 0.5
                # Blend: final_prob = (1 - ai_w) * ensemble + ai_w * ai_prob
                _orig_ensemble = _r["ensemble_prob"]
                _new_ensemble  = round(_orig_ensemble * (1 - _ai_w) + _ai_prob * _ai_w, 4)
                _new_ensemble  = max(0.03, min(0.97, _new_ensemble))
                # Recompute KL, edge, EV con el nuevo ensemble
                _new_kl  = KLDivergence.binary_kl(_new_ensemble, _r["yes_price"])
                # edge = our_prob - market_price (positive = we have edge)
                # YES: P(YES) - yes_price  |  NO: P(NO) - no_price = (1-new_ensemble) - no_price
                _new_edge = (_new_ensemble - _r["yes_price"]) if _r["side"] == "YES" else ((1.0 - _new_ensemble) - _r["no_price"])
                # Actualizar campos — edge_pct en PORCENTAJE (como math_engine: edge_val * 100)
                _r["ensemble_prob"]        = _new_ensemble
                _r["kl_divergence"]        = round(_new_kl, 6)
                _r["edge_pct"]             = round(_new_edge * 100, 2)
                _r["ai_prob"]              = round(_ai_prob, 4)
                _r["ai_recommendation"]    = _ai_rec
                _r["ai_confidence"]        = _ai_conf
                _r["ai_weight_applied"]    = round(_ai_w, 3)
                _r["ai_ensemble_adjusted"] = True
                # Re-compute DDC score con nuevo KL y edge — normas por categoría (v3/v4)
                _cat2    = _r.get("category", "default").lower()
                _nrm2    = _DDC_CATEGORY_NORMS.get(_cat2, _DDC_CATEGORY_NORMS["default"])
                _kl_use2 = max(0.04, _nrm2["kl"] * 0.70 + _DDC_KL_NORM * 0.30)
                _ev_use2 = max(0.04, _nrm2["ev"] * 0.70 + _DDC_EV_NORM * 0.30)
                _kl_n2   = min(_r["kl_divergence"] / _kl_use2, 1.0)
                _liq2    = _r.get("liquidity", 0.0)
                _liq_f2  = 0.20 if _liq2 < 500 else (0.50 if _liq2 < 2000 else (0.80 if _liq2 < 5000 else 1.00))
                _ev_n2   = min(max(_r.get("ev", 0), 0.0) / _ev_use2, 1.0) * _liq_f2
                _conf_n2 = min(_r.get("confidence", 50) / 100.0, 1.0)
                _mr_b2   = 0.10 if _r.get("mr_aligned", False) and _r.get("mr_proxy", 0) > 0.3 else 0.0
                _div_n2  = _r.get("diversity_score", 1.0)
                _zone_b2 = 0.05 if _r.get("price_zone") == "sweet_spot" else 0.0
                _hrs2    = _r.get("hours_to_resolution")
                _urg_b2  = (0.07 if _hrs2 is not None and _hrs2 <= 6
                            else 0.05 if _hrs2 is not None and _hrs2 <= 24
                            else 0.03 if _hrs2 is not None and _hrs2 <= 72
                            else 0.0)
                _ai_bonus = (0.07 if _ai_rec in ("APOSTAR", "APOSTAR_FUERTE") and _ai_conf == "ALTA"
                             else 0.04 if _ai_rec == "APOSTAR" and _ai_conf == "MEDIA"
                             else 0.01 if _ai_rec == "APOSTAR"
                             else 0.0)
                _r["composite_score"] = round(min(
                    _kl_n2 * 0.35 + _ev_n2 * 0.22 + _conf_n2 * 0.12 + _mr_b2
                    + _div_n2 * 0.05 + _zone_b2 + _urg_b2 + _ai_bonus,
                    1.0
                ), 4)
            # Re-sort después del ajuste IA
            raw_recs.sort(key=lambda r: r["composite_score"], reverse=True)
            print(f"[AIBoost] Ajuste IA aplicado a {len(_ai_results)} recomendaciones.")

    # ── Diversity score: penaliza recomendaciones correlacionadas ──────────
    # Research: portafolios diversificados tienen menor drawdown sin sacrificar retorno.
    # Usamos ensemble_prob como proxy de correlación entre mercados.
    # Dos mercados con prob similar en misma categoría probablemente comparten drivers.
    _seen_cat_probs: list[tuple[str, float]] = []
    for _r in raw_recs:
        _cat = _r.get("category", "other")
        _ep  = _r.get("ensemble_prob", 0.5)
        # Contar cuántas recomendaciones previas en la misma categoría tienen prob similar (±5pp)
        _similar = sum(1 for _sc, _sp in _seen_cat_probs
                       if _sc == _cat and abs(_sp - _ep) < 0.05)
        # diversity_score: 1.0 = única, decrece con duplicados similares
        _div_score = round(1.0 / (1.0 + _similar * 0.5), 3)
        _r["diversity_score"] = _div_score
        _seen_cat_probs.append((_cat, _ep))

    # ── Siempre excluir mercados ya expirados (independiente del horizonte) ──
    _now_utc = datetime.now(timezone.utc)
    _valid_recs = []
    for _r in raw_recs:
        _ed = _r.get("end_date", "")
        if not _ed:
            _valid_recs.append(_r)
            continue
        try:
            _dt = datetime.fromisoformat(_ed.replace("Z", "+00:00"))
            if _dt.tzinfo is None:
                _dt = _dt.replace(tzinfo=timezone.utc)
            if _dt > _now_utc:
                _valid_recs.append(_r)
        except (ValueError, TypeError):
            _valid_recs.append(_r)
    raw_recs = _valid_recs

    # ── Diversificación de categorías ────────────────────────────────────
    # Con el multi-fetch ya traemos 3×80 = ~240 mercados únicos.
    # Subimos los caps para que se vean más oportunidades de cada categoría.
    _cat_cap_sports  = 4   # deportes (era 2 — subido para capturar más eventos)
    _cat_cap_crypto  = 5   # cripto
    _cat_cap_default = 5   # política, tecnología, economía, geopolítica, otros
    _cat_counts: dict = {}
    _diversified, _overflow = [], []
    for _r in raw_recs:
        _c = _r.get("category", "other")
        _cap = _cat_cap_crypto if _c == "crypto" else (_cat_cap_sports if _c == "sports" else _cat_cap_default)
        if _cat_counts.get(_c, 0) < _cap:
            _diversified.append(_r)
            _cat_counts[_c] = _cat_counts.get(_c, 0) + 1
        else:
            _overflow.append(_r)
    raw_recs = _diversified + _overflow

    # ── Multi-Kelly optimization for top recommendations ──
    # Calcular Kelly para todos los candidatos que el bot podría intentar ejecutar.
    # Antes era min(10, ...) → mercados en posición 11+ recibían bet_usd=0 → "Kelly $0".
    top_n = min(30, len(raw_recs))
    if top_n > 0:
        # Para apuestas NO, la probabilidad relevante para Kelly es 1 - prob_YES
        kelly_bets = [{"p": r["ensemble_prob"] if r["side"] == "YES" else 1.0 - r["ensemble_prob"],
                       "price": r["market_price"]} for r in raw_recs[:top_n]]
        optimal_fracs = MultiKelly.multi_kelly_optimize(kelly_bets, KELLY_FRACTION)
        available = max(0.0, risk.state.current_capital - risk.state.total_exposure_usd)
        for i in range(top_n):
            raw_recs[i]["multi_kelly_frac"] = round(optimal_fracs[i], 4)
            _unc = raw_recs[i].get("venn_uncertainty", 0.0)
            _conf_factor = max(0.25, 1.0 - _unc)  # Mínimo 25% del bet
            # Sports sin señal real → Kelly reducido al 50%: sin ELO/Poisson live no hay
            # ventaja informacional real sobre el mercado. Reduce el tamaño de apuesta.
            _is_sports_blind = (
                raw_recs[i].get("category") == "sports" and
                raw_recs[i].get("category_confidence", 1.0) < 0.15
            )
            _sports_mult = 0.50 if _is_sports_blind else 1.0
            _raw_bet = optimal_fracs[i] * _conf_factor * _sports_mult * available
            # Mínimo $1.00 (requisito Polymarket CLOB) — evita bets de centavos sin sentido
            raw_recs[i]["bet_usd"] = round(max(1.00, _raw_bet), 2)
            raw_recs[i]["confidence_factor"] = round(_conf_factor * _sports_mult, 3)
            if _is_sports_blind:
                raw_recs[i]["sports_kelly_penalty"] = True
    for r in raw_recs[top_n:]:
        r["multi_kelly_frac"] = 0
        r["bet_usd"] = 0

    # ── Aplicar filtro de horizonte si se especificó ──────────────────────
    if time_horizon and time_horizon != "any":
        raw_recs = _filter_recs_by_horizon(raw_recs, time_horizon)

    # ── Filtro por categoría (opcional) ──────────────────────────────────
    if category:
        _cat_lower = category.lower()
        raw_recs = [r for r in raw_recs
                    if _cat_lower in (r.get("category") or "").lower()]

    _scan_result = {
        "recommendations": raw_recs, "total": len(raw_recs),
        "time_horizon": time_horizon,
        "category_filter": category or None,
        "capital_available": round(risk.state.current_capital - risk.state.total_exposure_usd, 2),
        "algorithms": ["DDC_Composite_Score", "KL_Divergence", "Multi_Kelly", "Bayesian",
                        "Category_Models", f"Extremization_d={extremizer_d}",
                        f"Calibration_{cal_suite.get_status()['method']}",
                        "Mean_Reversion_Ensemble", "Adaptive_Confidence_Gate",
                        "AI_Ensemble_Boost" if (llm.anthropic_key or llm.openai_key) else "AI_Boost_Disabled"],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "from_cache": False,
    }
    # Guardar en caché para evitar re-escanear en los próximos 60s
    import time as _scan_time_save
    _scan_cache[_cache_key] = _scan_result
    _scan_cache_ts[_cache_key] = _scan_time_save.time()
    return _scan_result

# ═══════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Diagnóstico completo del sistema — módulos, calibración, configuración."""
    import sys
    _env_keys = {
        "polymarket_api_key": bool(os.getenv("POLYMARKET_API_KEY")),
        "anthropic_api_key":  bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai_api_key":     bool(os.getenv("OPENAI_API_KEY")),
        "private_key":        bool(os.getenv("PRIVATE_KEY")),
        "funder_address":     bool(os.getenv("FUNDER_ADDRESS")),
    }
    _cal_status = cal_suite.get_status()
    _risk_report = risk.get_risk_report()
    _validator_status = validator.get_status() if hasattr(validator, "get_status") else {}
    return {
        "status": "ok",
        "version": "5.0.0",
        "python": sys.version.split()[0],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env_keys_present": _env_keys,
        "all_keys_ok": all(_env_keys.values()),
        "calibration": {
            "method": _cal_status.get("method"),
            "observations": _cal_status.get("n_observations", 0),
            "beta_fitted": cal_suite.beta.fitted,
            "isotonic_fitted": cal_suite.isotonic.fitted,
            "temperature_fitted": cal_suite.temperature.fitted,
        },
        "extremizer_d": extremizer_d,
        "ensemble_weights": _ENSEMBLE_WEIGHTS,
        "risk": {
            "current_capital": _risk_report.get("current_capital"),
            "total_exposure_usd": _risk_report.get("total_exposure_usd"),
            "is_paused": risk.state.is_paused if hasattr(risk.state, "is_paused") else (risk.state.consecutive_losses >= 3),
            "consecutive_losses": risk.state.consecutive_losses if hasattr(risk.state, "consecutive_losses") else 0,
        },
        "validator": _validator_status,
        "dry_run": trader.dry_run,
        "prediction_log_size": len(_prediction_log),
    }


@app.get("/api/ai/health")
async def ai_health_check():
    """
    Comprueba conectividad real con Claude y OpenAI haciendo una llamada mínima.
    Usado por el indicador de salud IA en el header del dashboard.
    """
    results: dict = {}

    # Claude (Anthropic)
    if llm and getattr(llm, "anthropic_key", None):
        try:
            resp = await asyncio.wait_for(
                llm._call_anthropic(
                    "Responde exactamente con la palabra OK y nada más."
                ),
                timeout=8.0
            )
            text_r = resp[0] if isinstance(resp, tuple) else resp
            results["claude"] = "ok" if text_r and "OK" in text_r else f"respuesta_rara: {str(text_r)[:40]}"
        except asyncio.TimeoutError:
            results["claude"] = "timeout"
        except Exception as e:
            results["claude"] = f"error: {str(e)[:60]}"
    else:
        results["claude"] = "sin_api_key"

    # OpenAI
    if llm and getattr(llm, "openai_key", None):
        try:
            resp = await asyncio.wait_for(
                llm._call_openai(
                    "Responde exactamente con la palabra OK y nada más."
                ),
                timeout=8.0
            )
            text_r = resp[0] if isinstance(resp, tuple) else resp
            results["openai"] = "ok" if text_r and "OK" in text_r else f"respuesta_rara: {str(text_r)[:40]}"
        except asyncio.TimeoutError:
            results["openai"] = "timeout"
        except Exception as e:
            results["openai"] = f"error: {str(e)[:60]}"
    else:
        results["openai"] = "sin_api_key"

    # Gamma API (Polymarket)
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=5.0) as _c:
            _r = await _c.get("https://gamma-api.polymarket.com/markets?limit=1")
            results["gamma_api"] = "ok" if _r.status_code == 200 else f"http_{_r.status_code}"
    except Exception as e:
        results["gamma_api"] = f"error: {str(e)[:50]}"

    # Binance (para precios BTC)
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=5.0) as _c:
            _r = await _c.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
            results["binance"] = "ok" if _r.status_code == 200 else f"http_{_r.status_code}"
    except Exception as e:
        results["binance"] = f"error: {str(e)[:50]}"

    all_critical_ok = results.get("gamma_api") == "ok" and results.get("binance") == "ok"
    results["status"] = "ok" if all_critical_ok else "degraded"
    results["cache_size"]    = len(_ai_cache)
    results["cache_ttl_sec"] = AI_CACHE_TTL
    return results


@app.get("/api/system/test")
async def system_self_test():
    """
    Autotest rápido de los módulos principales del sistema.
    No hace trades. Verifica que los algoritmos producen resultados razonables.
    """
    import time as _time
    results = {}
    t0 = _time.time()

    # Test 1: KLDivergence
    try:
        from advanced_math import KLDivergence
        kl = KLDivergence.binary_kl(0.7, 0.5)
        results["kl_divergence"] = {"ok": kl > 0, "val": round(kl, 4)}
    except Exception as e:
        results["kl_divergence"] = {"ok": False, "error": str(e)[:80]}

    # Test 2: Extremizer
    try:
        from extremizer import Extremizer
        ep = Extremizer.weighted_extremize([(0.7, 1.0), (0.6, 0.8)], d=extremizer_d)
        results["extremizer"] = {"ok": 0 < ep < 1, "val": round(ep, 4), "d": extremizer_d}
    except Exception as e:
        results["extremizer"] = {"ok": False, "error": str(e)[:80]}

    # Test 3: Calibration
    try:
        cp = cal_suite.calibrate(0.65)
        results["calibration"] = {"ok": 0 < cp.calibrated < 1, "method": cp.method, "val": round(cp.calibrated, 4)}
    except Exception as e:
        results["calibration"] = {"ok": False, "error": str(e)[:80]}

    # Test 4: Bayesian engine
    try:
        bp = bayesian.estimate_with_raw_data(market_price=0.60)
        results["bayesian"] = {"ok": 0 < bp.posterior < 1, "posterior": bp.posterior}
    except Exception as e:
        results["bayesian"] = {"ok": False, "error": str(e)[:80]}

    # Test 5: Category classifier
    try:
        from category_models import classify_market, MarketCategory
        cat = classify_market("Will Bitcoin reach $100K?")
        results["classifier"] = {"ok": cat == MarketCategory.CRYPTO, "category": cat.value}
    except Exception as e:
        results["classifier"] = {"ok": False, "error": str(e)[:80]}

    # Test 6: TradeValidator (with dummy data)
    try:
        ok, reason = validator.validate_trade(
            market_id="test-123", outcome="YES", amount=2.0, kelly_amount=2.0,
            market_data={"yes_price": 0.6, "no_price": 0.4})
        results["validator"] = {"ok": isinstance(ok, bool), "passed": ok, "reason": reason[:60]}
    except Exception as e:
        results["validator"] = {"ok": False, "error": str(e)[:80]}

    elapsed_ms = round((_time.time() - t0) * 1000, 1)
    all_ok = all(v.get("ok", False) for v in results.values())
    return {
        "all_ok": all_ok,
        "elapsed_ms": elapsed_ms,
        "tests": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def _get_sports_intel_status() -> dict:
    """Helper para obtener status de SportsIntelligence sin crashear si no está inicializado."""
    try:
        from sports_intel import get_sports_intel
        si = get_sports_intel()
        return si.get_status() if si else {"initialized": False}
    except Exception:
        return {"initialized": False}

# ═══════════════════════════════════════════════════════════
# ADVANCED ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/api/status")
async def status():
    import time as _t_status
    _has_pk = bool(trader.private_key and not trader.private_key.startswith("0x_YOUR"))
    _age_h_status = round((_t_status.time() - trader._auth_timestamp) / 3600.0, 2) if trader._auth_timestamp > 0 else None
    return {"version": "5.0.0", "author": "Carlos David Donoso Cordero (ddchack)",
        "is_authenticated": trader.is_authenticated,
        "has_private_key": _has_pk,
        "creds_age_hours": _age_h_status,
        "funder": (trader.funder[:6] + "…" + trader.funder[-4:]) if trader.funder else None,
        "algorithms": {"kl_divergence": True, "multi_kelly": True, "category_models": True,
            "brier_decomposition": True, "monte_carlo": True, "copula": True,
            "beta_calibration": cal_suite.beta.fitted, "isotonic_calibration": cal_suite.isotonic.fitted,
            "temperature_scaling": cal_suite.temperature.fitted, "venn_abers": True,
            "extremization": True, "extremizer_d": extremizer_d,
            "mean_reversion": True, "obi_vpin": True,
            "adaptive_aggregation": True, "diversity_tracking": True,
            "regime_detection": True, "lmsr": True, "binary_sharpe": True},
        "config": {"max_capital": MAX_CAPITAL, "kelly_fraction": KELLY_FRACTION,
            "min_edge_pct": MIN_EDGE*100, "dry_run": trader.dry_run},
        "bots": {
            "auto_running": _bot_running, "auto_mode": _bot_mode,
            "rush_running": _rush_running, "rush_mode": _rush_mode,
            "er_running": _er_running,
            "dh_running": (_dh_engine._running if _dh_engine else False),
            "dh_mode": ("demo" if not _dh_engine or _dh_engine.config.get("dry_run", True) else "live"),
        },
        "risk": risk.get_risk_report(),
        "calibration": calibrator.get_params(),
        "predictions_logged": len(_prediction_log),
        "sports_intel": _get_sports_intel_status(),
        "clob_connection": _clob_connection_status}

@app.post("/api/config")
async def update_config(c: ConfigUpdate):
    global MAX_CAPITAL, KELLY_FRACTION, MIN_EDGE
    if c.max_capital is not None: MAX_CAPITAL = c.max_capital; math_engine.max_capital = c.max_capital
    if c.kelly_fraction is not None: KELLY_FRACTION = c.kelly_fraction; math_engine.kelly_multiplier = c.kelly_fraction
    if c.min_edge is not None: MIN_EDGE = c.min_edge; math_engine.min_edge = c.min_edge
    if c.dry_run is not None: trader.dry_run = c.dry_run
    return {"status": "updated"}

# ── LLM (IA) Configuration & Analysis ──
class LLMConfigRequest(BaseModel):
    anthropic_key: str = ""
    openai_key: str = ""
    anthropic_model: str = ""
    openai_model: str = ""
    mode: str = ""  # "anthropic", "openai", "ensemble", "none"

class LLMAnalyzeRequest(BaseModel):
    question: str
    category: str = "other"
    yes_price: float = 0.5
    no_price: float = 0.5
    volume: float = 0
    liquidity: float = 0
    end_date: str = ""

@app.get("/api/llm/config")
async def get_llm_config():
    return llm.get_config()

@app.post("/api/llm/config")
async def set_llm_config(req: LLMConfigRequest):
    llm.configure(req.anthropic_key, req.openai_key, req.anthropic_model, req.openai_model, req.mode)
    # Propagar keys al SwarmEngine
    swarm_engine.configure(
        anthropic_key=req.anthropic_key or llm.anthropic_key,
        openai_key=req.openai_key or llm.openai_key,
        anthropic_model="claude-haiku-4-5-20251001",
        openai_model="gpt-4o-mini",
    )
    return {"status": "configured", "config": llm.get_config()}

@app.post("/api/llm/analyze")
async def llm_analyze(req: LLMAnalyzeRequest):
    """Ejecuta TODOS los algoritmos primero, luego alimenta a la IA con los resultados."""
    if llm.mode == "none":
        return {"error": "IA no configurada. Ve a Ajustes → configura API keys."}

    # ── PASO 1: Ejecutar todos los algoritmos ──
    question = req.question
    category = classify_market(question)

    # Bayesian estimation
    bay_est = bayesian.estimate_with_raw_data(
        market_price=req.yes_price,
        sentiment_score=0, num_articles=0,
        volume_24h=req.volume, avg_volume_7d=req.volume * 1.5,  # BUG5: estimación conservadora
        spread=0.03, liquidity=req.liquidity)

    # Category model
    cat_signal = cat_eval.evaluate(question, req.yes_price)

    # Ensemble probability
    cat_w = cat_signal.confidence * 0.4
    ensemble_p = bay_est.posterior * (1 - cat_w) + cat_signal.estimated_prob * cat_w
    ensemble_p = max(0.03, min(0.97, ensemble_p))
    if calibrator.is_fitted:
        ensemble_p = calibrator.calibrate(ensemble_p)

    # KL Divergence
    kl = KLDivergence.binary_kl(ensemble_p, req.yes_price) if ensemble_p != req.yes_price else 0

    # Kelly
    kelly_f = MultiKelly.single_kelly(ensemble_p, req.yes_price)
    kelly_q = kelly_f * KELLY_FRACTION

    # Sharpe
    sharpe = BinarySharpe.single_bet(ensemble_p, req.yes_price)

    # EV
    ev = math_engine.expected_value(ensemble_p, req.yes_price)

    # Edge
    edge = (ensemble_p - req.yes_price) * 100

    # Sentiment (quick)
    sent_score, sent_articles, sent_label = 0, 0, "NEUTRAL"
    try:
        rpt = await sentiment.get_sentiment_for_market(question)
        sent_score, sent_articles, sent_label = rpt.avg_sentiment, rpt.articles_found, rpt.sentiment_label
    except: pass

    # Regime
    regime_data = {"regime": "NORMAL", "vol_multiplier": 1.0}

    # ── PASO 2: Empaquetar TODO para la IA ──
    algo_data = {
        "category": category.value if hasattr(category, 'value') else str(category),
        "yes_price": req.yes_price,
        "no_price": req.no_price,
        "volume_24h": req.volume,
        "liquidity": req.liquidity,
        "end_date": req.end_date,
        "spread": 0.03,
        "bayesian_prob": bay_est.posterior,
        "prior": bay_est.prior,
        "uncertainty": bay_est.uncertainty,
        "signals": bay_est.signals_used,
        "kl_divergence": kl,
        "edge_pct": edge,
        "ev": ev,
        "kelly_full": kelly_f,
        "kelly_quarter": kelly_q,
        "kelly_bet_usd": round(kelly_q * MAX_CAPITAL, 2),
        "binary_sharpe": sharpe,
        "category_model": cat_signal.model_name,
        "category_prob": cat_signal.estimated_prob,
        "category_confidence": cat_signal.confidence,
        "category_reasoning": cat_signal.reasoning,
        "time_sensitivity": cat_signal.time_sensitivity,
        "sentiment_score": sent_score,
        "sentiment_articles": sent_articles,
        "sentiment_label": sent_label,
        "whale_positions": [],
        "regime": regime_data["regime"],
        "vol_multiplier": regime_data["vol_multiplier"],
        "brier_score": calibrator.get_params().get("n_observations", 0) > 10 and "calculado" or "N/A",
        "n_predictions": len(_prediction_log),
        "platt_status": "Calibrado" if calibrator.is_fitted else "Pendiente (necesita 30+ resultados)",
    }

    # ── PASO 3: Enviar a la IA con TODOS los datos ──
    result = await llm.analyze_with_algorithms(question, algo_data)

    # Añadir resumen de lo que se alimentó
    result["algoritmos_ejecutados"] = {
        "bayesian": f"{bay_est.posterior*100:.1f}% (±{bay_est.uncertainty*100:.1f}%)",
        "kl_divergence": f"{kl:.4f}" + (" ✓ RENTABLE" if kl > 0.05 else " △ marginal" if kl > 0.01 else " ✗ bajo"),
        "kelly_recomendado": f"${round(kelly_q * MAX_CAPITAL, 2)}",
        "sharpe": f"{sharpe:.3f}",
        "ev": f"{'+' if ev > 0 else ''}{ev:.4f} por $1",
        "modelo_categoria": f"{cat_signal.model_name} → {cat_signal.estimated_prob*100:.1f}%",
        "sentimiento": f"{sent_label} ({sent_score:+.2f}, {sent_articles} artículos)",
        "edge": f"+{edge:.1f}% vs mercado",
    }

    return result

@app.get("/api/llm/models")
async def get_available_models():
    return {"anthropic": ANTHROPIC_MODELS, "openai": OPENAI_MODELS}

# ── KL Divergence Calculator ──
@app.get("/api/kl-divergence")
async def calc_kl(p_model: float = Query(...), p_market: float = Query(...)):
    kl = KLDivergence.binary_kl(p_model, p_market)
    actionable = KLDivergence.is_actionable(p_model, p_market)
    return {"kl_divergence": round(kl, 6), "actionable": actionable,
            "interpretation": "HIGH_EDGE" if kl > 0.1 else "MODERATE" if kl > 0.05 else "LOW"}

# ── Multi-Kelly Optimizer ──
@app.post("/api/multi-kelly")
async def calc_multi_kelly(bets: list[dict]):
    fracs = MultiKelly.multi_kelly_optimize(bets, KELLY_FRACTION)
    return {"optimal_fractions": [round(f, 4) for f in fracs], "kelly_fraction": KELLY_FRACTION,
            "total_allocation": round(sum(fracs), 4)}

# ── Monte Carlo Simulation ──
@app.post("/api/monte-carlo")
async def run_monte_carlo(req: MonteCarloRequest):
    scan = await full_scan(limit=30, with_sentiment=False, with_whales=False)
    recs = scan["recommendations"][:15]
    if not recs:
        return {"error": "No recommendations to simulate"}
    bets = [{"p": r["ensemble_prob"], "price": r["market_price"],
             "category": r["category"]} for r in recs]
    result = MonteCarloSimulator.simulate_portfolio(
        bets, MAX_CAPITAL, req.n_simulations, req.kelly_fraction)
    return result

# ── Auto-Allocation (SHARKFLOW core feature) ──
@app.post("/api/auto-allocate")
async def auto_allocate(req: AutoAllocateRequest):
    """
    SHARKFLOW Smart Allocation: distribute budget across 5-20 bets.
    Higher confidence = proportionally more capital.
    Respects category diversification and time horizon.
    """
    scan = await full_scan(limit=60, with_sentiment=True, with_whales=False)
    recs = scan["recommendations"]
    if not recs:
        return {"error": "No markets found"}
    plan = allocator.allocate(recs, req.total_budget, req.time_horizon, req.min_bets, req.max_bets)
    return {
        "plan_id": plan.plan_id,
        "total_budget": plan.total_budget,
        "total_allocated": plan.total_allocated,
        "reserve": plan.reserve_usd,
        "n_bets": len(plan.slots),
        "time_horizon": plan.time_horizon,
        "expected_ev": plan.expected_ev,
        "portfolio_sharpe": plan.portfolio_sharpe,
        "effective_exposure": plan.effective_exposure,
        "category_breakdown": plan.category_breakdown,
        "slots": plan.slots,
        "created_at": plan.created_at,
    }

@app.post("/api/auto-allocate/execute")
async def execute_allocation(req: AutoAllocateRequest):
    """Execute the auto-allocation plan: scan → allocate → place all trades."""
    if not risk.state.is_active or risk.state.is_paused:
        return {"error": risk.state.pause_reason}
    scan = await full_scan(limit=60, with_sentiment=True, with_whales=True)
    plan = allocator.allocate(scan["recommendations"], req.total_budget, req.time_horizon, req.min_bets, req.max_bets)
    executed = []
    for slot in plan.slots:
        if slot["allocated_usd"] < 1:
            continue
        tid = slot["yes_token"] if slot["side"] == "YES" else slot["no_token"]
        if not tid:
            continue
        v = risk.validate_trade(slot["allocated_usd"], slot["confidence"], 10000, 0.03, slot["market_id"])
        if not v["approved"]:
            continue
        r = trader.place_market_order(tid, v["adjusted_size"], "BUY", slot["market_id"], slot["question"], slot["confidence"], slot["edge_pct"])
        risk.record_trade(slot["market_id"], v["adjusted_size"])
        async with _pred_log_lock:
            _prediction_log.append({"market_id": slot["market_id"], "predicted": slot["ensemble_prob"], "side": slot["side"], "ts": datetime.now(timezone.utc).isoformat()})
            _save_prediction_log(_prediction_log)
        executed.append({"question": slot["question"][:60], "side": slot["side"], "usd": v["adjusted_size"], "category": slot["category"], "kl": slot["kl_divergence"], "status": r.status})
    await broadcast("allocation_executed", {"n_bets": len(executed), "total": sum(e["usd"] for e in executed)})
    return {"executed": executed, "total_executed": len(executed), "total_spent": round(sum(e["usd"] for e in executed), 2), "risk": risk.get_risk_report()["capital"]}

# ── Brier Score & Calibration ──
@app.get("/api/calibration")
async def get_calibration():
    if len(_prediction_log) < 10:
        return {"status": "insufficient_data", "predictions": len(_prediction_log),
                "message": "Need at least 10 resolved predictions"}
    preds = [p["predicted"] for p in _prediction_log if "actual" in p]
    actuals = [p["actual"] for p in _prediction_log if "actual" in p]
    if len(preds) < 10:
        return {"status": "insufficient_resolved", "total": len(_prediction_log)}
    decomp = brier.decompose(preds, actuals)
    return {"brier": decomp, "platt": calibrator.get_params(), "n_predictions": len(preds)}

@app.post("/api/calibration/record")
async def record_outcome(market_id: str, outcome: int):
    """Record actual outcome for a prediction (for calibration tracking)."""
    n_matched = 0
    for p in _prediction_log:
        if p["market_id"] == market_id and "actual" not in p:
            p["actual"] = outcome
            n_matched += 1
            # ── v1 calibrator (Platt) ──
            calibrator.add_observation(p["predicted"], outcome)
            # ── v2 CalibrationSuite (Beta/Isotonic/Temperature/Venn) ──
            cal_suite.add_observation(p["predicted"], outcome)

    # Re-fit both calibrators when enough observations
    resolved = [p for p in _prediction_log if "actual" in p]
    if len(resolved) >= 30:
        calibrator.fit()
    if len(resolved) >= 30:
        cal_suite.fit_all()

    # Persistir cambios al disco
    async with _pred_log_lock:
        _save_prediction_log(_prediction_log)

    return {
        "status": "recorded",
        "market_id": market_id,
        "outcome": outcome,
        "n_matched_predictions": n_matched,
        "total_resolved": len(resolved),
        "calibrator_fitted": calibrator.is_fitted,
        "cal_suite_status": cal_suite.get_status(),
    }

@app.get("/api/calibration/predictions")
async def get_prediction_log(resolved_only: bool = Query(False), limit: int = Query(50)):
    """Lista el prediction log — útil para auditar qué predicciones se han registrado."""
    preds = _prediction_log
    if resolved_only:
        preds = [p for p in preds if "actual" in p]
    preds = list(reversed(preds))[:limit]  # Most recent first
    # Compute per-pred hit/miss when available
    enriched = []
    for p in preds:
        row = dict(p)
        if "actual" in p:
            # Correcto si el lado apostado gana (predicted > 0.5 → YES expected)
            row["correct"] = (p["predicted"] > 0.5) == (p["actual"] == 1)
        enriched.append(row)

    resolved_count = sum(1 for p in _prediction_log if "actual" in p)
    correct_count  = sum(1 for p in _prediction_log if "actual" in p and (p["predicted"] > 0.5) == (p["actual"] == 1))
    return {
        "total_predictions": len(_prediction_log),
        "resolved": resolved_count,
        "win_rate": round(correct_count / max(1, resolved_count) * 100, 1),
        "predictions": enriched,
    }

@app.get("/api/portfolio/analysis")
async def portfolio_analysis():
    """
    Análisis completo del portfolio actual: concentración, correlación estimada,
    Sharpe ponderado, y alertas de rebalanceo.
    """
    rr = risk.get_risk_report()
    open_positions = rr.get("open_positions", [])

    if not open_positions:
        return {
            "status": "empty",
            "message": "No hay posiciones abiertas en el portfolio.",
            "total_exposure_usd": 0,
            "alerts": [],
        }

    total_exp = sum(p.get("size_usd", 0) for p in open_positions)
    if total_exp <= 0:
        return {"status": "empty", "total_exposure_usd": 0, "alerts": []}

    # ── Concentración por categoría ──
    cat_map = {}
    for pos in open_positions:
        cat = pos.get("category", "unknown")
        cat_map[cat] = cat_map.get(cat, 0) + pos.get("size_usd", 0)
    cat_pct = {k: round(v / total_exp * 100, 1) for k, v in cat_map.items()}

    # ── Herfindahl-Hirschman Index (concentración) ──
    # HHI = 0 → perfectamente diversificado, HHI = 10000 → todo en una categoría
    hhi = sum((pct / 100) ** 2 for pct in cat_pct.values()) * 10000

    # ── Sharpe ponderado del portfolio ──
    from advanced_math import BinarySharpe
    sharpe_items = []
    for pos in open_positions:
        ep = pos.get("ensemble_prob", 0.5)
        mp = pos.get("market_price", pos.get("entry_price", 0.5))
        w  = pos.get("size_usd", 0) / max(1, total_exp)
        sharpe_items.append({"p": ep, "m": mp, "w": w})
    port_sharpe = BinarySharpe.portfolio(sharpe_items) if sharpe_items else 0

    # ── Alertas ──
    alerts = []
    for cat, pct in cat_pct.items():
        if pct > 50:
            alerts.append({"type": "CONCENTRATION", "severity": "HIGH",
                           "message": f"Concentración excesiva en '{cat}': {pct}% del portfolio"})
        elif pct > 35:
            alerts.append({"type": "CONCENTRATION", "severity": "MEDIUM",
                           "message": f"Alta concentración en '{cat}': {pct}% del portfolio"})
    if hhi > 5000:
        alerts.append({"type": "HHI", "severity": "HIGH",
                       "message": f"Portfolio muy concentrado (HHI={hhi:.0f}, máx recomendado: 2500)"})
    elif hhi > 2500:
        alerts.append({"type": "HHI", "severity": "MEDIUM",
                       "message": f"Portfolio moderadamente concentrado (HHI={hhi:.0f})"})
    if len(open_positions) < 3:
        alerts.append({"type": "DIVERSIFICATION", "severity": "HIGH",
                       "message": f"Solo {len(open_positions)} posición(es) — insuficiente diversificación"})
    if port_sharpe < 0.5 and len(open_positions) >= 3:
        alerts.append({"type": "SHARPE", "severity": "MEDIUM",
                       "message": f"Sharpe ponderado bajo ({port_sharpe:.3f}) — considerar rebalanceo"})

    return {
        "status": "ok",
        "n_positions": len(open_positions),
        "total_exposure_usd": round(total_exp, 2),
        "portfolio_sharpe": round(port_sharpe, 3),
        "hhi": round(hhi, 1),
        "hhi_label": "ALTO" if hhi > 5000 else ("MEDIO" if hhi > 2500 else "BAJO"),
        "category_breakdown_pct": cat_pct,
        "alerts": alerts,
        "alert_count": len(alerts),
    }


# ── LMSR Fair Price ──
@app.get("/api/lmsr")
async def lmsr_price(yes_qty: float = Query(100), no_qty: float = Query(100), b: float = Query(50)):
    yp, np_ = LMSR.fair_price_binary(yes_qty, no_qty, b)
    return {"yes_price": round(yp, 4), "no_price": round(np_, 4),
            "max_mm_loss": round(LMSR.max_market_maker_loss(b, 2), 2)}

# ── Regime Detection ──
@app.get("/api/regime")
async def detect_regime(token_id: str = Query("")):
    """Detect current market regime from price history."""
    if token_id:
        history = await scanner.fetch_price_history(token_id, "1d")
        if history:
            prices = [float(h.get("p", h.get("price", 0))) for h in history if h.get("p") or h.get("price")]
            if len(prices) > 5:
                returns = list(np.diff(np.log(np.array(prices) + 1e-10)))
                return regime.detect_regime(returns)
    return {"regime": "UNKNOWN", "vol_multiplier": 0.75, "confidence": 0}

# ── Category Classification ──
@app.get("/api/classify")
async def classify(question: str):
    cat = classify_market(question)
    return {"category": cat.value, "allocation": CategoryEvaluator.get_category_allocation().get(cat.value, {})}

# ── Copula Correlation ──
@app.get("/api/correlation")
async def calc_correlation(p1: float = Query(...), p2: float = Query(...), rho: float = Query(0.5)):
    joint = GaussianCopula.joint_probability(p1, p2, rho)
    return {"joint_probabilities": joint, "correlation": rho}

# ── Effective Exposure ──
@app.get("/api/effective-exposure")
async def calc_exposure():
    positions = [{"size": abs(v), "category": "other", "market_id": k}
                 for k, v in risk._position_tracker.items()]
    if not positions:
        return {"effective_exposure": 0, "raw_exposure": 0, "n_positions": 0}
    eff = GaussianCopula.effective_exposure(positions)
    raw = sum(p["size"] for p in positions)
    return {"effective_exposure": eff, "raw_exposure": round(raw, 2),
            "correlation_multiplier": round(eff / max(1, raw), 2), "n_positions": len(positions)}

# ═══════════ EXISTING ENDPOINTS (updated) ═══════════

@app.get("/api/markets")
async def get_markets(limit: int = Query(50), offset: int = Query(0)):
    markets = await scanner.fetch_active_markets(limit=limit, offset=offset)
    enriched = []
    for m in markets:
        odds = scanner.parse_market_odds(m)
        yt, nt = scanner.extract_token_ids(m)
        cat = classify_market(m.get("question", ""))
        enriched.append({"id": m.get("conditionId", m.get("id","")), "question": m.get("question",""),
            "category": cat.value, "yes_price": odds.yes_price if odds else 0,
            "no_price": odds.no_price if odds else 0, "volume_24h": float(m.get("volume24hr",0) or 0),
            "liquidity": float(m.get("liquidity",0) or 0), "yes_token": yt, "no_token": nt,
            "image": m.get("image",""), "end_date": m.get("endDate","")})
    return {"markets": enriched, "total": len(enriched)}

@app.get("/api/scan/ai-analysis")
async def ai_analysis_top_recs():
    """
    Analiza las top 3 recomendaciones con LLM (Claude/GPT).
    Devuelve un resumen narrativo en español para cada oportunidad.
    SharkFlow by Carlos David Donoso Cordero (ddchack)
    """
    try:
        # Obtener recomendaciones actuales (top 3 por composite_score o KL)
        raw = await scanner.scan_markets()
        from math_engine import MathEngine as _ME
        _meng = _ME()
        recs = []
        for m in raw[:30]:
            try:
                ep = m.yes_price
                r = _meng.full_analysis(ep, m.liquidity, m.spread, m.category, m.question)
                if r.get("edge_pct", 0) > 0.03:
                    recs.append({**r, "question": m.question, "category": m.category,
                                 "yes_price": m.yes_price, "market_id": m.market_id})
            except Exception:
                pass
        recs.sort(key=lambda x: x.get("composite_score", x.get("kl_divergence", 0)), reverse=True)
        top3 = recs[:3]
        if not top3:
            return {"analysis": "No hay recomendaciones activas con suficiente edge para analizar.", "analyzed": 0}

        # Construir prompt
        lines = []
        for i, rec in enumerate(top3, 1):
            side = "YES" if rec.get("edge_pct", 0) > 0 else "NO"
            lines.append(
                f"{i}. [{rec['category'].upper()}] {rec['question']}\n"
                f"   Precio actual YES: {rec['yes_price']:.2f} | Modelo estima: {rec.get('ensemble_prob',0.5):.2f}\n"
                f"   Edge: {rec.get('edge_pct',0)*100:.1f}% | EV: {rec.get('ev',0):.4f} | "
                f"KL: {rec.get('kl_divergence',0):.4f} | DDC Score: {int(rec.get('composite_score',0)*100)}/100\n"
                f"   Kelly recomendado: ${rec.get('bet_usd',0):.2f} | Lado: {side}"
            )
        prompt = (
            "Eres un analista cuantitativo de mercados de predicción. "
            "Analiza brevemente estas 3 oportunidades detectadas por el sistema SharkFlow. "
            "Para cada una: (1) explica en 1-2 oraciones qué hace al precio atractivo, "
            "(2) menciona el riesgo principal, (3) da una recomendación concisa (APOSTAR / PASAR / ESPERAR). "
            "Sé directo y profesional. Responde en español.\n\n"
            + "\n".join(lines)
        )
        # Llamar directamente al proveedor disponible
        if llm.anthropic_key:
            response, _ms = await llm._call_anthropic(prompt)
        elif llm.openai_key:
            import httpx as _hx
            _oai_resp = await llm.client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {llm.openai_key}", "Content-Type": "application/json"},
                json={"model": llm.openai_model, "max_tokens": 600,
                      "messages": [
                          {"role": "system", "content": "Eres un analista de mercados de predicción. Responde en español de forma concisa y profesional."},
                          {"role": "user", "content": prompt}],
                      "temperature": 0.4}
            )
            _oai_resp.raise_for_status()
            _oai_data = _oai_resp.json()
            response = _oai_data["choices"][0]["message"]["content"] if _oai_data.get("choices") else ""
        else:
            return {"analysis": "No hay API keys configuradas. Ve a Configuración → Claves API para agregar Claude o GPT.", "analyzed": 0}
        return {"analysis": response, "analyzed": len(top3), "opportunities": [
            {"question": r["question"], "ddc_score": int(r.get("composite_score",0)*100)} for r in top3
        ]}
    except Exception as e:
        return {"analysis": f"Error al consultar IA: {str(e)}", "analyzed": 0}

@app.get("/api/arbitrage")
async def scan_arb():
    result = await arb.full_scan()
    if result["total_opportunities"] > 0:
        await broadcast("arbitrage", {"count": result["total_opportunities"]})
    return result

@app.get("/api/risk")
async def get_risk(): return risk.get_risk_report()
@app.get("/api/risk/status")
async def get_risk_status():
    """Estado de pausa del bot — muestra motivo y cómo reactivar manualmente."""
    s = risk.state
    return {
        "is_active": s.is_active,
        "is_paused": s.is_paused,
        "pause_reason": s.pause_reason if s.is_paused else None,
        "consecutive_losses": s.consecutive_losses,
        "max_consecutive_losses": risk.limits.max_consecutive_losses,
        "daily_pnl": round(s.daily_pnl, 2),
        "current_drawdown_pct": round(s.current_drawdown_pct, 2),
        "resume_endpoint": "POST /api/risk/resume  (reactivar manualmente)",
    }
@app.post("/api/risk/resume")
async def risk_resume(): risk.resume(); return {"ok": True}
@app.post("/api/risk/pause")
async def risk_pause(): risk._trigger_circuit_breaker("Manual"); return {"ok": True}
@app.post("/api/risk/kill")
async def risk_kill(): risk._kill_switch("Manual"); return {"ok": True}
@app.post("/api/risk/activate")
async def risk_activate(): risk.activate(); return {"ok": True}

@app.get("/api/whales")
async def get_whales(): return {"whales": whales.get_tracked_whales(), "total": len(whales.whales)}
@app.post("/api/whales/add")
async def add_whale(r: WhaleAddRequest): whales.add_whale(r.address, r.alias); return {"ok": True}
@app.get("/api/whales/scan")
async def scan_whales(days: int = Query(7, ge=1, le=90)):
    m, stats = await whales.scan_whale_moves(max_age_days=days)
    return {
        "moves":      whales.get_recent_moves_report(),
        "total":      len(m),
        "wallets":    len(whales.whales),
        "days":       days,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "stats":      stats,
    }

@app.get("/api/calculate")
async def calc(true_prob: float = Query(...), market_price: float = Query(...), capital: float = Query(100)):
    ev = math_engine.expected_value(true_prob, market_price)
    kl = KLDivergence.binary_kl(true_prob, market_price)
    sr = BinarySharpe.single_bet(true_prob, market_price)
    return {"ev": round(ev,4), "edge_pct": round((true_prob-market_price)*100,2),
        "kelly_full": round(MultiKelly.single_kelly(true_prob, market_price),4),
        "kelly_quarter": round(MultiKelly.single_kelly(true_prob, market_price)*0.25,4),
        "bet_usd": round(MultiKelly.single_kelly(true_prob, market_price)*0.25*capital,2),
        "kl_divergence": round(kl,6), "binary_sharpe": round(sr,3),
        "profitable": ev > 0, "actionable": kl > 0.05}

@app.get("/api/sentiment/{query}")
async def get_sentiment(query: str):
    rpt = await sentiment.get_sentiment_for_market(query)
    return {"sentiment": rpt.avg_sentiment, "label": rpt.sentiment_label,
            "articles": rpt.articles_found, "confidence": rpt.confidence}

@app.post("/api/trade")
async def place_trade(req: TradeRequest):
    v = risk.validate_trade(req.amount_usd, req.confidence, 10000, 0.03, req.market_id)
    if not v["approved"]: return {"status": "REJECTED", "reasons": v["reasons"]}
    sz = v["adjusted_size"]
    if req.order_type == "MARKET":
        r = trader.place_market_order(req.token_id, sz, req.side, req.market_id, req.question, req.confidence, req.edge_pct)
    else:
        if not req.price: raise HTTPException(400, "Price required")
        r = trader.place_limit_order(req.token_id, req.price, sz/req.price, req.side, req.market_id, req.question, req.confidence, req.edge_pct)
    risk.record_trade(req.market_id, sz)
    # Log prediction for calibration
    async with _pred_log_lock:
        _prediction_log.append({"market_id": req.market_id, "predicted": req.confidence/100, "side": req.side, "ts": datetime.now(timezone.utc).isoformat()})
        _save_prediction_log(_prediction_log)
    await broadcast("trade", {"question": req.question[:60], "side": req.side, "amount": sz, "status": r.status})
    return {"status": r.status, "order_id": r.order_id, "size": r.size_usd, "risk_adjusted": sz}

@app.post("/api/trade/auto")
async def auto_trade(max_trades: int = Query(3)):
    if not risk.state.is_active or risk.state.is_paused:
        return {"error": risk.state.pause_reason, "executed": []}
    scan = await full_scan(limit=30, with_sentiment=True, with_whales=True)
    executed = []
    for rec in scan["recommendations"][:max_trades*2]:
        if len(executed) >= max_trades: break
        if rec["bet_usd"] < 1 or rec["confidence"] < 40: continue
        if rec["risk"] == "HIGH" and rec["confidence"] < 65: continue
        tid = rec["yes_token"] if rec["side"] == "YES" else rec["no_token"]
        if not tid: continue
        v = risk.validate_trade(rec["bet_usd"], rec["confidence"], rec["liquidity"], rec["spread"], rec["market_id"])
        if not v["approved"]: continue
        r = trader.place_market_order(tid, v["adjusted_size"], "BUY", rec["market_id"], rec["question"], rec["confidence"], rec["edge_pct"])
        risk.record_trade(rec["market_id"], v["adjusted_size"])
        async with _pred_log_lock:
            _prediction_log.append({"market_id": rec["market_id"], "predicted": rec["ensemble_prob"], "side": rec["side"], "ts": datetime.now(timezone.utc).isoformat()})
            _save_prediction_log(_prediction_log)
        executed.append({"question": rec["question"][:80], "side": rec["side"], "bet_usd": v["adjusted_size"],
            "ensemble_prob": rec["ensemble_prob"], "kl_div": rec["kl_divergence"],
            "category": rec["category"], "status": r.status})
    return {"executed": executed, "total_scanned": scan["total"], "total_executed": len(executed),
            "risk": risk.get_risk_report()["capital"]}

@app.get("/api/trade/history")
async def trade_history(): return trader.get_trade_summary()

@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    eng = MathEngine(max_capital=MAX_CAPITAL, kelly_multiplier=req.kelly_multiplier, min_edge=req.min_edge)
    bt = BacktestEngine(math_engine=eng, initial_capital=MAX_CAPITAL)
    try:
        r = await bt.run_backtest(req.strategy_name, req.num_markets); await bt.close()
        # ── Feed backtest observations → calibration suite (mejora automática) ──
        _obs_fed = 0
        for t in r.trades:
            try:
                _outcome = 1 if t.get("resolved_outcome") == "YES" else 0
                _pred = t.get("estimated_prob", 0.5)
                cal_suite.add_observation(float(_pred), int(_outcome))
                _obs_fed += 1
            except Exception:
                pass
        if _obs_fed >= 20:
            cal_suite.fit_all()
        return {"strategy": r.strategy_name, "final_capital": r.final_capital, "return_pct": r.total_return_pct,
            "trades": r.total_trades, "wins": r.wins, "losses": r.losses, "win_rate": r.win_rate,
            "sharpe": r.sharpe_estimate, "max_dd": r.max_drawdown_pct, "profit_factor": r.profit_factor,
            "brier_score": r.brier_score, "calibration_curve": r.calibration_curve,
            "category_metrics": r.category_metrics,
            "calibration_obs_fed": _obs_fed,
            "trajectory": r.capital_trajectory, "trade_log": r.trades[:30]}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/auth/connect")
async def connect(): return trader.initialize()

@app.post("/api/auth/reconnect")
async def reconnect():
    """Fuerza renovación de creds CLOB (útil cuando la conexión expira)."""
    import time as _t
    old_ts = trader._auth_timestamp
    trader.is_authenticated = False   # forzar re-auth
    result = trader.initialize()
    new_ts = trader._auth_timestamp
    result["age_before_s"] = round(_t.time() - old_ts) if old_ts > 0 else None
    result["reconnected"]  = new_ts > old_ts
    return result

@app.get("/api/auth/status")
async def auth_status():
    """Estado de la conexión con Polymarket CLOB."""
    import time as _t
    has_pk = bool(trader.private_key and not trader.private_key.startswith("0x_YOUR"))
    has_funder = bool(trader.funder and not trader.funder.startswith("0x_YOUR"))
    age_h = round((_t.time() - trader._auth_timestamp) / 3600, 1) if trader._auth_timestamp > 0 else None
    return {
        "has_private_key": has_pk,
        "has_funder": has_funder,
        "is_authenticated": trader.is_authenticated,
        "creds_age_hours": age_h,
        "funder": trader.funder[:10] + "..." + trader.funder[-4:] if has_funder and len(trader.funder) > 14 else trader.funder,
        "dry_run": trader.dry_run,
    }

@app.get("/api/balance")
async def get_balance():
    """
    Consulta el balance real de USDC del funder address en Polygon.
    Si la consulta tiene éxito, sincroniza risk.state.current_capital
    para que los cálculos de Kelly usen el capital real.
    """
    result = trader.get_wallet_balance()
    usdc = result.get("usdc", 0.0)

    # Sincronizar con RiskManager y math_engine solo si obtuvimos un balance válido
    if usdc > 0 and result.get("source") not in ("not_configured", "rpc_failed"):
        risk.state.current_capital = usdc
        math_engine.max_capital = usdc
        if usdc > risk.state.peak_capital:
            risk.state.peak_capital = usdc

    return {
        "usdc":              usdc,
        "source":            result.get("source"),
        "address":           result.get("address", ""),
        "error":             result.get("error"),
        "needs_private_key": result.get("needs_private_key", False),
        "synced_to_risk":    usdc > 0 and result.get("source") not in ("not_configured", "rpc_failed"),
    }

@app.get("/api/balance/debug")
async def balance_debug():
    """Debug: prueba todas las variantes de get_balance_allowance."""
    result = {
        "is_authenticated": trader.is_authenticated,
        "funder": trader.funder,
        "signature_type": trader.signature_type,
        "has_creds": False,
        "signer_address": None,
        "variants": {},
    }

    if not (trader.is_authenticated and trader.clob_client):
        return result

    try:
        creds  = trader.clob_client.creds
        signer = trader.clob_client.signer
        result["has_creds"] = creds is not None
        if signer:
            result["signer_address"] = signer.address()
        if creds:
            result["api_key_prefix"] = creds.api_key[:8] + "..."

        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        for label, params in [
            ("A_no_asset_type",  BalanceAllowanceParams()),
            ("B_collateral",     BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)),
            ("C_conditional",    BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)),
        ]:
            try:
                result["variants"][label] = trader.clob_client.get_balance_allowance(params)
            except Exception as e:
                result["variants"][label] = {"error": f"{type(e).__name__}: {e}"}

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result

@app.post("/api/setup-allowances")
async def setup_allowances():
    """
    Configura los permisos (ERC-20 approve + setApprovalForAll) necesarios para
    operar en el CLOB de Polymarket. Requiere MATIC en el wallet para gas.
    """
    if not trader.is_authenticated:
        return {"ok": False, "error": "Trader no autenticado. Configura las API keys primero."}
    try:
        status = trader._setup_allowances()
        return {"ok": True, "status": status}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN DE MERCADOS — verifica si trades abiertos ya resolvieron
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/resolve-check")
async def resolve_check(ids: str = Query("")):
    """
    Verifica si una lista de mercados se resolvieron en Polymarket.
    Consulta la Gamma API por cada ID (conditionId o slug).
    Devuelve {resolved: {market_id: {outcome, resolved_at, question}}}.
    Desarrollado por Carlos David Donoso Cordero (ddchack).
    """
    import urllib.request as _req
    import json as _j

    market_ids = [i.strip() for i in ids.split(",") if i.strip()][:20]
    resolved: dict = {}

    for market_id in market_ids:
        try:
            url = f"https://gamma-api.polymarket.com/markets/{market_id}"
            req = _req.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "SharkFlow/4.0"},
            )
            with _req.urlopen(req, timeout=6) as resp:
                data = _j.loads(resp.read().decode("utf-8"))

            # El mercado debe estar cerrado/resuelto
            if not (data.get("closed") or data.get("resolved")):
                continue

            # outcomePrices: lista de probabilidades finales
            # YES resuelve → prices[0] ≈ 1.0, prices[1] ≈ 0.0
            # NO  resuelve → prices[0] ≈ 0.0, prices[1] ≈ 1.0
            outcome_prices = data.get("outcomePrices", [])
            if not outcome_prices:
                continue

            try:
                prices = [float(p) for p in outcome_prices]
            except (TypeError, ValueError):
                continue

            if len(prices) < 2:
                continue

            if prices[0] > 0.95:
                outcome = "YES"
            elif prices[1] > 0.95:
                outcome = "NO"
            else:
                continue  # No hay resolución clara (ej: dispute en curso)

            resolved[market_id] = {
                "outcome":     outcome,
                "resolved_at": (data.get("endDate") or
                                data.get("resolutionTime") or
                                datetime.now(timezone.utc).isoformat()),
                "question":    data.get("question", ""),
            }

        except Exception:
            continue

    return {
        "resolved": resolved,
        "checked":  len(market_ids),
        "found":    len(resolved),
    }

# ═══════════ v4 RESEARCH ENDPOINTS ═══════════

@app.get("/api/mean-reversion/{token_id}")
async def check_mean_reversion(token_id: str):
    """Detect overreaction and mean reversion opportunity."""
    history = await scanner.fetch_price_history(token_id, "1h")
    if not history or len(history) < 10:
        return {"error": "Historial insuficiente"}
    prices = [float(h.get("p", h.get("price", 0.5))) for h in history if h.get("p") or h.get("price")]
    if len(prices) < 10:
        return {"error": "Datos de precio insuficientes"}
    fade = mr_engine.fade_score(prices)
    return fade

@app.get("/api/microstructure/{token_id}")
async def analyze_microstructure(token_id: str):
    """OBI + VPIN analysis from orderbook."""
    try:
        ob = await scanner.fetch_orderbook(token_id)
        bids = ob.get("bids", [])[:10]
        asks = ob.get("asks", [])[:10]
        return micro.analyze(bids, asks)
    except:
        return {"error": "No se pudo obtener orderbook", "obi": {}, "vpin": {}}

@app.get("/api/calibration/v2")
async def get_calibration_v2():
    """Advanced calibration status: Beta + Isotonic + Temp + Venn-ABERS."""
    status = cal_suite.get_status()
    status["extremizer_d"] = extremizer_d
    return status

@app.post("/api/calibration/v2/record")
async def record_v2(market_id: str, predicted: float, outcome: int):
    """Record outcome for v2 calibration suite."""
    cal_suite.add_observation(predicted, outcome)
    calibrator.add_observation(predicted, outcome)  # Legacy too
    fit_result = cal_suite.fit_all()
    return {"status": "recorded", "fit": fit_result}

@app.get("/api/extremize")
async def extremize_probs(probs: str = "", d: float = Query(1.73)):
    """Extremize a comma-separated list of probabilities."""
    try:
        p_list = [float(x.strip()) for x in probs.split(",") if x.strip()]
        if not p_list:
            return {"error": "Envía probabilidades separadas por coma"}
        result = Extremizer.extremize(p_list, d)
        simple_avg = sum(p_list) / len(p_list)
        return {"extremized": round(result, 4), "simple_average": round(simple_avg, 4),
                "d": d, "n_inputs": len(p_list),
                "shift": round((result - simple_avg) * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: FWS):
    await websocket.accept(); dashboard_ws.append(websocket)
    try:
        while True:
            await websocket.send_text(json.dumps({"event": "heartbeat",
                "data": {"risk": risk.get_risk_report(), "calibration": calibrator.get_params()},
                "ts": datetime.now(timezone.utc).isoformat()}))
            await asyncio.sleep(5)
    except: pass
    finally:
        if websocket in dashboard_ws: dashboard_ws.remove(websocket)

def _serve_dashboard():
    resp = FileResponse(pathlib.Path(__file__).parent / "dashboard.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/app")
async def dashboard_app():
    return _serve_dashboard()

@app.get("/")
async def dashboard_root():
    return _serve_dashboard()

# ═══════════ KEYS & CONFIG MANAGEMENT ═══════════

import re as _re

def mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "••••••••" if key else ""
    return key[:4] + "•" * (len(key) - 8) + key[-4:]

def _get_env_path():
    # Buscar .env en la raíz del proyecto (un nivel arriba de backend/)
    backend_dir = pathlib.Path(__file__).parent
    env_candidates = [backend_dir.parent / ".env", backend_dir / ".env", pathlib.Path(".env")]
    for p in env_candidates:
        if p.exists():
            return p
    return backend_dir.parent / ".env"  # default

def _read_env_file(path) -> dict:
    result = {}
    try:
        content = path.read_text(encoding='utf-8')
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                result[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return result

def _write_env_file(path, updates: dict):
    # Lee el .env actual, actualiza solo las claves especificadas, mantiene comentarios
    try:
        # Asegurar que el directorio padre existe
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            content = path.read_text(encoding='utf-8')
        else:
            content = ""
        lines = content.splitlines()
        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                k, _, _ = stripped.partition('=')
                k = k.strip()
                if k in updates:
                    new_lines.append(f"{k}={updates[k]}")
                    updated_keys.add(k)
                    continue
            new_lines.append(line)
        # Agregar claves nuevas que no existían
        for k, v in updates.items():
            if k not in updated_keys:
                new_lines.append(f"{k}={v}")
        path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    except Exception as e:
        print(f"[Keys] _write_env_file error: {type(e).__name__}: {e}")
        raise HTTPException(500, f"Error escribiendo .env: {e}")

@app.get("/api/keys")
async def get_keys():
    """Retorna las API keys actuales (enmascaradas para seguridad)."""
    env_path = _get_env_path()
    env_data = _read_env_file(env_path)
    return {
        "polymarket_private_key":    mask_key(env_data.get("POLYMARKET_PRIVATE_KEY",    os.getenv("POLYMARKET_PRIVATE_KEY", ""))),
        "polymarket_funder_address": mask_key(env_data.get("POLYMARKET_FUNDER_ADDRESS", os.getenv("POLYMARKET_FUNDER_ADDRESS", ""))),
        "polymarket_signature_type": env_data.get("POLYMARKET_SIGNATURE_TYPE", os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")),
        "newsapi_key":    mask_key(env_data.get("NEWSAPI_KEY",       os.getenv("NEWSAPI_KEY", ""))),
        "anthropic_key":  mask_key(env_data.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))),
        "openai_key":     mask_key(env_data.get("OPENAI_API_KEY",    os.getenv("OPENAI_API_KEY", ""))),
        "backend_port":   env_data.get("BACKEND_PORT", os.getenv("BACKEND_PORT", "8888")),
        "max_capital":    env_data.get("MAX_CAPITAL_USD", str(MAX_CAPITAL)),
        "kelly_fraction": env_data.get("KELLY_FRACTION", str(KELLY_FRACTION)),
        "min_edge":       env_data.get("MIN_EDGE_PERCENT", str(MIN_EDGE * 100)),
        "env_file":  str(env_path),
        "env_exists": env_path.exists(),
    }

@app.post("/api/keys")
async def save_keys(req: KeysUpdate):
    """Guarda las API keys en el archivo .env del proyecto."""
    env_path = _get_env_path()
    print(f"[Keys] POST /api/keys | path={env_path}")

    # Mapeo directo campo -> variable de entorno (solo valores reales, sin enmascarar)
    MAP = [
        (req.polymarket_private_key,   "POLYMARKET_PRIVATE_KEY"),
        (req.polymarket_funder_address, "POLYMARKET_FUNDER_ADDRESS"),
        (req.newsapi_key,              "NEWSAPI_KEY"),
        (req.anthropic_key,            "ANTHROPIC_API_KEY"),
        (req.openai_key,               "OPENAI_API_KEY"),
    ]
    updates = {}
    for val, key in MAP:
        if val and val.strip() and "\u2022" not in val:  # \u2022 = bullet •
            updates[key] = val.strip()
    if req.polymarket_signature_type is not None:
        updates["POLYMARKET_SIGNATURE_TYPE"] = str(req.polymarket_signature_type)
    if req.backend_port is not None:
        updates["BACKEND_PORT"] = str(req.backend_port)
    if req.max_capital is not None:
        updates["MAX_CAPITAL_USD"] = str(req.max_capital)
    if req.kelly_fraction is not None:
        updates["KELLY_FRACTION"] = str(req.kelly_fraction)
    if req.min_edge is not None:
        updates["MIN_EDGE_PERCENT"] = str(req.min_edge)

    real = {k: v for k, v in updates.items() if k != "POLYMARKET_SIGNATURE_TYPE"}
    if not real:
        print(f"[Keys] Sin claves reales para guardar")
        return {"status": "no_changes", "message": "Ingresa al menos una API key"}

    print(f"[Keys] Guardando: {list(updates.keys())} en {env_path}")

    # Leer .env actual (si existe) y hacer merge
    existing: dict = {}
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
    except Exception as e:
        print(f"[Keys] WARN leyendo .env: {e}")

    existing.update(updates)

    # Escribir directamente
    try:
        content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
        env_path.write_text(content, encoding="utf-8")
        print(f"[Keys] OK - {env_path} escrito con {len(existing)} variables")
    except Exception as e:
        print(f"[Keys] ERROR escribiendo {env_path}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al escribir .env: {e}")

    for k, v in updates.items():
        os.environ[k] = v

    # Actualizar el objeto trader en memoria (sin reiniciar el servidor)
    if "POLYMARKET_PRIVATE_KEY"    in updates: trader.private_key = updates["POLYMARKET_PRIVATE_KEY"]
    if "POLYMARKET_FUNDER_ADDRESS" in updates: trader.funder      = updates["POLYMARKET_FUNDER_ADDRESS"]
    if any(k in updates for k in ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS")):
        # Resetear estado de auth para forzar re-init con nuevas credenciales
        trader.is_authenticated = False
        trader.clob_client = None

    return {
        "status": "saved",
        "updated_keys": list(updates.keys()),
        "env_file": str(env_path),
        "message": f"{len(updates)} clave(s) guardada(s)"
    }

@app.get("/api/config/full")
async def get_full_config():
    """Configuración completa del sistema para el dashboard."""
    return {
        "trading": {
            "max_capital": MAX_CAPITAL,
            "kelly_fraction": KELLY_FRACTION,
            "min_edge_pct": MIN_EDGE * 100,
            "dry_run": trader.dry_run,
        },
        "risk": risk.get_risk_report(),
        "algorithms": {
            "bayesian": True, "kl_divergence": True, "multi_kelly": True,
            "category_models": True, "extremization": True, "mean_reversion": True,
            "calibration": cal_suite.get_status()['method'],
        },
        "integrations": {
            "newsapi": bool(NEWSAPI_KEY),
            "polymarket": bool(PRIVATE_KEY and FUNDER),
            "llm": llm.mode if hasattr(llm, 'mode') else "none",
        },
        "version": "5.0.0",
    }

# ═══════════ WHALES LEADERBOARD ═══════════

@app.get("/api/whales/leaderboard")
async def get_whale_leaderboard():
    """Obtiene top traders de Polymarket para seguir (múltiples endpoints de fallback)."""
    import httpx as _httpx
    leaders = []
    last_error = ""

    # Construir set de direcciones ya rastreadas
    tracked_addrs = set()
    for w in whales.whales:
        if isinstance(w, str):
            tracked_addrs.add(w.lower())
        elif hasattr(w, 'address'):
            tracked_addrs.add(w.address.lower())

    def _parse_items(items):
        result = []
        for i, item in enumerate(items[:20]):
            addr = (item.get("proxyWallet") or item.get("address") or
                    item.get("user") or item.get("userId") or "")
            if not addr:
                continue
            # v1 API usa 'pnl' y 'vol' directamente
            profit = 0.0
            try:
                profit = float(item.get("pnl") or item.get("profit") or item.get("cashPnl") or
                               item.get("totalProfit") or 0)
            except (ValueError, TypeError):
                pass
            volume = 0.0
            try:
                volume = float(item.get("vol") or item.get("volume") or 0)
            except (ValueError, TypeError):
                pass
            roi = 0.0
            try:
                roi_raw = float(item.get("profitRoi") or item.get("roi") or
                                item.get("pnlRoi") or item.get("returnOnInvestment") or 0)
                # Normalizar ROI: si está en decimal (e.g. 0.45 = 45%), convertir
                if abs(roi_raw) < 10 and abs(roi_raw) > 0:
                    roi = roi_raw * 100
                else:
                    roi = roi_raw
            except (ValueError, TypeError):
                pass
            # Si no hay ROI pero hay profit y volume, calcularlo
            if roi == 0 and profit != 0 and volume > 0:
                roi = (profit / volume) * 100
            trades = 0
            try:
                trades = int(item.get("tradesCount") or item.get("trades") or
                             item.get("totalTrades") or item.get("positionsCount") or 0)
            except (ValueError, TypeError):
                pass
            rank = item.get("rank", i + 1)
            result.append({
                "address": addr,
                "alias": item.get("userName") or item.get("name") or item.get("username") or f"Top Trader #{rank}",
                "roi": round(roi, 2),
                "profit": round(profit, 2),
                "volume": round(volume, 2),
                "trades": trades,
                "rank": rank,
                "already_tracked": addr.lower() in tracked_addrs,
            })
        return result

    endpoints = [
        # v1 API oficial de Polymarket (docs.polymarket.com)
        ("https://data-api.polymarket.com/v1/leaderboard",
         {"timePeriod": "ALL", "orderBy": "PNL", "limit": 20}),
        ("https://data-api.polymarket.com/v1/leaderboard",
         {"timePeriod": "MONTH", "orderBy": "PNL", "limit": 20}),
        ("https://data-api.polymarket.com/v1/leaderboard",
         {"timePeriod": "ALL", "orderBy": "VOL", "limit": 20}),
        # Fallback: endpoint legacy sin versionar
        ("https://data-api.polymarket.com/leaderboard",
         {"timePeriod": "ALL", "orderBy": "PNL", "limit": 20}),
    ]

    async with _httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for url, params in endpoints:
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    items = (data if isinstance(data, list)
                             else data.get("data", data.get("results",
                                  data.get("traders", data.get("profiles", [])))))
                    if items and isinstance(items, list) and len(items) > 0:
                        leaders = _parse_items(items)
                        if leaders:
                            break
                else:
                    last_error = f"HTTP {resp.status_code} en {url}"
            except Exception as e:
                last_error = str(e)[:80]
                print(f"[Whales] {url}: {e}")
                continue

    if not leaders:
        return {
            "leaders": [],
            "source": "unavailable",
            "error": last_error or "API de Polymarket no retornó datos",
            "tip": "Busca top traders en polymarket.com/leaderboard y agrega sus direcciones manualmente"
        }

    return {"leaders": leaders, "source": "polymarket", "count": len(leaders)}


@app.post("/api/whales/auto-discover")
async def auto_discover_whales(
    min_profit: float = Query(500.0, description="Beneficio mínimo en USDC"),
    min_trades: int   = Query(5,     description="Trades mínimos realizados"),
    limit:      int   = Query(20,    ge=1, le=50, description="Máximo de wallets a agregar"),
):
    """Obtiene el leaderboard de Polymarket y añade automáticamente los top traders."""
    # Reutilizar la lógica del endpoint leaderboard
    lb_resp = await get_whale_leaderboard()
    leaders = lb_resp.get("leaders", [])

    if not leaders:
        return {
            "ok": False,
            "added": 0,
            "skipped": 0,
            "error": lb_resp.get("error", "No se pudo obtener el leaderboard"),
        }

    added, skipped_already, skipped_filter = 0, 0, 0
    added_names = []

    for leader in leaders[:limit]:
        addr = leader.get("address", "")
        if not addr:
            continue

        # Filtrar por criterios mínimos
        profit = leader.get("profit", 0) or 0
        trades = leader.get("trades", 0) or 0
        if profit < min_profit or trades < min_trades:
            skipped_filter += 1
            continue

        # No agregar duplicados
        if leader.get("already_tracked"):
            skipped_already += 1
            continue

        alias = leader.get("alias") or f"Top#{leader.get('rank', added + 1)}"
        whales.add_whale(addr, alias)
        added += 1
        added_names.append(alias)

    return {
        "ok": True,
        "added": added,
        "skipped_duplicates": skipped_already,
        "skipped_filter": skipped_filter,
        "added_names": added_names[:10],
        "source": lb_resp.get("source", "unknown"),
    }


# ═══════════ BOT SCHEDULER ═══════════

def _filter_recs_by_horizon(recs: list, horizon: str) -> list:
    """
    Filtra recomendaciones según horizonte temporal.
    Semántica: "máximo X tiempo hasta resolución".
    No usa mínimo — si el mercado cierra antes, igual es aceptable.
    Excluye mercados ya expirados.
    Deportes reciben +4h buffer (resuelven DESPUÉS del partido).
    """
    if not horizon or horizon == "any":
        return recs
    import re as _re
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    # Máximo de horas hasta cierre del mercado por horizonte
    _max_hours = {
        "minutes": 6,     # cierra en las próximas 6h
        "hours":   24,    # cierra hoy
        "days":    168,   # cierra en los próximos 7 días
        "weeks":   720,   # cierra en los próximos 30 días
        "months":  4320,  # cierra en los próximos 180 días
    }
    max_h = _max_hours.get(horizon)
    if max_h is None:
        return recs
    cutoff = now + timedelta(hours=max_h)

    def _is_sports_q(q: str) -> bool:
        ql = q.lower()
        return bool(
            _re.search(r'\bvs\.?\s', ql) or " beat " in ql or " beats " in ql or
            "win on" in ql or "championship" in ql or "playoff" in ql or "match result" in ql
        )

    def _is_crypto_q(q: str) -> bool:
        ql = q.lower()
        return any(tok in ql for tok in ["btc", "eth", "sol", "bitcoin", "ethereum",
                                          "solana", "crypto", "doge", "xrp", "bnb",
                                          "avax", "ada", "matic", "link", "usdc",
                                          "stablecoin", "defi", "nft", "blockchain",
                                          "coinbase", "binance", "kraken"])

    filtered = []
    for r in recs:
        ed = r.get("end_date", "")
        if not ed:
            continue
        try:
            dt = datetime.fromisoformat(ed.replace("Z", "+00:00")) if "T" in ed else \
                 datetime.fromisoformat(ed + "T23:59:59+00:00")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            # Deportes: +4h buffer — resuelven DESPUÉS del partido
            cat = r.get("category", "").lower()
            is_sports = (cat == "sports") or _is_sports_q(r.get("question", ""))
            dt_effective = dt + timedelta(hours=4) if is_sports else dt

            # Excluir expirados
            if dt_effective < now:
                continue

            # Solo incluir si cierra antes del cutoff
            if dt_effective > cutoff:
                continue

            filtered.append(r)
        except (ValueError, TypeError):
            pass

    # Para horizontes cortos, priorizar cripto (mayor volatilidad intradía)
    if horizon in ("minutes", "hours"):
        crypto = [r for r in filtered if r.get("category","")=="crypto" or _is_crypto_q(r.get("question",""))]
        others = [r for r in filtered if r not in crypto]
        filtered = crypto + others

    return filtered


async def _bot_worker():
    """
    Bot worker v2 — directo, con logging completo.
    Mantiene algoritmos matemáticos (Kelly·KL·Bayesian·Ensemble).
    Elimina bloqueos silenciosos y circuit breaker del loop principal.
    SharkFlow by Carlos David Donoso Cordero (ddchack)
    """
    global _bot_running, _already_invested_today
    from datetime import datetime as _dt
    import pathlib as _pl
    ts = lambda: _dt.now().strftime("%H:%M:%S")

    # Log también a archivo para debug cuando WS no está disponible
    _bot_log_file = _pl.Path(__file__).parent / "bot_activity.log"
    def _flog(line: str):
        try:
            with open(_bot_log_file, "a", encoding="utf-8") as _f:
                _f.write(f"{_dt.now().strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except Exception:
            pass

    async def _log(line: str):
        _flog(line)
        await broadcast("bot_log", {"line": line})

    _flog(f"═══ BOT {_bot_mode.upper()} INICIADO — intervalo:{_bot_interval}min edge≥{_bot_min_edge}% ═══")
    await broadcast("bot_log", {"line": f"[{ts()}] ═══ BOT {_bot_mode.upper()} — intervalo:{_bot_interval}min edge≥{_bot_min_edge}% ═══"})

    async def _cycle():
        global _bot_running
        try:
            # ── 1. Autenticación ──────────────────────────────────────────────
            if _bot_mode == "live":
                if not trader.is_authenticated:
                    await _log(f"[{ts()}] Conectando wallet Polymarket...")
                    result = trader.initialize()
                    st = result.get("status","")
                    if st != "authenticated":
                        await _log(f"[{ts()}] ❌ Auth fallida: {result.get('message','error')} — ciclo abortado")
                        return
                    await _log(f"[{ts()}] ✓ Wallet autenticada — funder:{(trader.funder or '')[:14]}...")
                else:
                    await _log(f"[{ts()}] ✓ Wallet ya autenticada (is_authenticated=True)")

            # ── 2. Balance real ──────────────────────────────────────────────
            bal = trader.get_wallet_balance()
            usdc = bal.get("usdc", 0.0)
            src  = bal.get("source","?")
            await _log(f"[{ts()}] Balance: ${usdc:.2f} USDC (fuente:{src})")
            if usdc >= 0.01:
                risk.state.current_capital = usdc
                math_engine.max_capital    = usdc
            capital = risk.state.current_capital
            if capital < 1.0:
                await _log(f"[{ts()}] ❌ Capital ${capital:.2f} < $1 — ciclo abortado")
                return

            # ── 3. Posiciones abiertas hoy (anti-duplicados) ─────────────────
            try:
                filled_today = trader.get_filled_positions_today()
                _already_invested_today = set(filled_today.keys())
            except Exception:
                _already_invested_today = set()
            if _already_invested_today:
                await _log(f"[{ts()}] Posiciones ya abiertas hoy: {len(_already_invested_today)}")

            # ── 4. Scan con algoritmos matemáticos ──────────────────────────
            await _log(f"[{ts()}] Escaneando (Kelly:{_bot_max_risk*100:.0f}% | Edge≥{_bot_min_edge}% | {_bot_time_horizon})...")
            scan = await full_scan(
                limit=50, with_sentiment=False, with_whales=False,
                time_horizon=_bot_time_horizon,
                min_edge_pct=float(_bot_min_edge),
                category=""
            )
            recs  = scan.get("recommendations", [])
            total = scan.get("total", len(recs))
            await _log(f"[{ts()}] {total} mercados → {len(recs)} con edge≥{_bot_min_edge}%")

            if not recs:
                await _log(f"[{ts()}] Sin oportunidades — próximo ciclo en {_bot_interval}min")
                return

            # Broadcast tabla para el dashboard
            top_n = min(_bot_max_bets, len(recs))
            algo_signals = {
                "kl_divergence": round(sum(r.get("kl_divergence",0)  for r in recs[:top_n])/max(top_n,1),4),
                "bayesian":      round(sum(abs(r.get("bayesian_prob",0.5)-r.get("market_price",0.5)) for r in recs[:top_n])/max(top_n,1),4),
                "extremization": round(sum(r.get("extremizer_d",0)   for r in recs[:top_n])/max(top_n,1),4),
                "category_conf": round(sum(r.get("category_confidence",0) for r in recs[:top_n])/max(top_n,1),4),
                "sharpe":        round(sum(r.get("binary_sharpe",0)  for r in recs[:top_n])/max(top_n,1),4),
                "edge_avg":      round(sum(r.get("edge_pct",0)       for r in recs[:top_n])/max(top_n,1),2),
            }
            await broadcast("bot_demo_scan", {
                "recommendations": recs[:top_n], "total": len(recs),
                "algo_signals": algo_signals, "min_bets": _bot_min_bets, "max_bets": _bot_max_bets,
            })

            # ── DEMO: simulación ─────────────────────────────────────────────
            if _bot_mode != "live":
                await _log(f"[{ts()}] ── SIMULACIÓN DEMO (top {top_n}) ──")
                _demo_used_tokens: set = set()
                _demo_count = 0
                for rec in recs[:top_n]:
                    _tokens_d = _event_tokens(rec.get("question",""))
                    if _tokens_d & _demo_used_tokens:
                        await _log(f"[{ts()}] ↷ [DEMO] Correlacionado — omitido: {rec.get('question','')[:40]}...")
                        continue
                    _demo_count += 1
                    _demo_bet = max(1.00, min(rec.get("bet_usd", 1.0), _bot_max_bet_usd))
                    await _log(
                        f"[{ts()}] #{_demo_count} {rec.get('side','?')} ${_demo_bet:.2f}"
                        f" | Edge:{rec.get('edge_pct',0):+.1f}%"
                        f" | KL:{rec.get('kl_divergence',0):.3f}"
                        f" | {rec.get('question','')[:40]}..."
                    )
                    _demo_used_tokens.update(_tokens_d)
                await _log(f"[{ts()}] [DEMO] {_demo_count} trades simulados (sin correlaciones) — no se usó dinero real")
                return

            # ── LIVE: ejecutar trades reales ─────────────────────────────────
            pool = min(_bot_max_bets * 4, len(recs))
            await _log(f"[{ts()}] ── LIVE: capital=${capital:.2f} | pool={pool} candidatos ──")
            executed = 0

            # Anti-correlación: tokens de eventos ya apostados en este ciclo
            # Palabras irrelevantes que no identifican un evento
            _STOP_WORDS = {
                'will','the','and','for','not','are','was','has','have','its','that','this',
                'with','from','over','under','more','less','than','win','lose','wins','loses',
                'game','match','series','point','points','goal','goals','score','cover','spread',
                'total','round','period','half','quarter','season','week','day','year','month',
                'who','what','when','which','does','make','play','team','home','away','vs',
                'between','during','before','after','first','last','next','per',
                'march','april','may','june','july','august','september','october','november','december',
                'jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec',
            }
            def _event_tokens(question: str) -> frozenset:
                """Palabras clave que identifican el evento (equipos, jugadores, torneos)."""
                words = question.lower().replace('-',' ').replace('.',' ').replace(',',' ').split()
                return frozenset(w for w in words if len(w) >= 4 and w not in _STOP_WORDS and not w.isdigit())

            _used_event_tokens: set = set()  # tokens de eventos ya apostados este ciclo

            for rec in recs[:pool]:
                if not _bot_running or executed >= _bot_max_bets:
                    break

                q    = rec.get("question","?")[:45]
                side = rec.get("side","YES")
                edge = rec.get("edge_pct", 0)
                kl   = rec.get("kl_divergence", 0)
                k_usd = rec.get("bet_usd", 0)

                # a) Token ID requerido por el CLOB
                tid = rec.get("yes_token") if side == "YES" else rec.get("no_token")
                if not tid:
                    await _log(f"[{ts()}] ⚠ Sin token_id ({side}): {q}...")
                    continue

                # b) No apostar en mercado ya abierto hoy
                mid = rec.get("market_id","")
                if mid in _already_invested_today:
                    await _log(f"[{ts()}] ↷ Ya apostado hoy: {q}...")
                    continue

                # c) Precio extremo = mercado ya decidido
                side_p = float(rec.get("yes_price" if side=="YES" else "no_price") or 0.5)
                if side_p < 0.05 or side_p > 0.95:
                    await _log(f"[{ts()}] ⚠ Precio extremo {side_p:.2f} — mercado ya decidido: {q}...")
                    continue

                # d) Anti-correlación: un solo bet por evento en el ciclo
                _full_q = rec.get("question", "")
                _tokens = _event_tokens(_full_q)
                _overlap = _tokens & _used_event_tokens
                if _overlap:
                    _overlap_str = ", ".join(sorted(_overlap)[:3])
                    await _log(f"[{ts()}] ↷ Correlacionado ({_overlap_str}): {q}...")
                    continue

                # e) Tamaño apuesta: Kelly fraccionado, hard cap según configuración
                bet = min(k_usd, _bot_max_bet_usd)  # nunca más del máximo configurado
                bet = max(bet, 1.00)                 # mínimo $1.00 (requisito Polymarket CLOB)
                bet = round(bet, 2)

                await _log(
                    f"[{ts()}] → {side} ${bet:.2f} | Edge:{edge:.1f}% KL:{kl:.3f} | {q}..."
                )

                # f) Ejecutar orden
                r = trader.place_market_order(
                    tid, bet, "BUY", mid, rec.get("question",""),
                    rec.get("confidence", 50), edge, market_data=rec
                )

                if r.status == "FILLED":
                    _already_invested_today.add(mid)
                    _used_event_tokens.update(_tokens)  # marcar evento como usado
                    risk.record_trade(mid, bet)
                    async with _pred_log_lock:
                        _prediction_log.append({
                            "market_id": mid, "predicted": rec.get("ensemble_prob", 0.5),
                            "side": side, "ts": datetime.now(timezone.utc).isoformat()
                        })
                        _save_prediction_log(_prediction_log)
                    await _log(f"[{ts()}] ✅ TRADE REAL: {side} ${bet:.2f} FILLED — {q}...")
                    await broadcast("trade", {
                        "question": rec.get("question","")[:60], "side": side,
                        "amount": bet, "status": "FILLED", "market_id": mid
                    })
                    executed += 1
                else:
                    err = (r.error_msg or r.status or "sin detalle")[:120]
                    await _log(f"[{ts()}] ✗ FALLÓ ({r.status}): {err}")

                await asyncio.sleep(1)

            if executed > 0:
                await _log(f"[{ts()}] ✅ {executed} trades ejecutados este ciclo")
                # Refrescar balance tras trades
                await asyncio.sleep(3)
                bal2 = trader.get_wallet_balance()
                usdc2 = bal2.get("usdc", 0.0)
                if usdc2 > 0:
                    risk.state.current_capital = usdc2
                    math_engine.max_capital    = usdc2
                    await _log(f"[{ts()}] Balance actualizado: ${usdc2:.2f} USDC")
            else:
                await _log(f"[{ts()}] ⚠ 0 trades ejecutados — ver detalles arriba")

        except Exception as e:
            import traceback as _tb
            tb_short = _tb.format_exc()[-600:]
            _flog(f"❌ ERROR en ciclo: {str(e)}\n{tb_short}")
            await broadcast("bot_log", {"line": f"[{ts()}] ❌ ERROR en ciclo: {str(e)[:100]}"})
            print(f"[Bot] ERROR completo:\n{tb_short}")

    # ── Loop principal ────────────────────────────────────────────────────────
    try:
        await asyncio.sleep(1)
        await _cycle()

        while _bot_running:
            wait_secs = _bot_interval * 60
            for i in range(wait_secs):
                if not _bot_running:
                    return
                await asyncio.sleep(1)
                if i == wait_secs - 60:
                    await _log(f"[{ts()}] Próximo escaneo en 1 min...")
            if _bot_running:
                await _cycle()

    except asyncio.CancelledError:
        pass
    finally:
        _flog("─── Bot detenido ───")
        await broadcast("bot_log", {"line": f"[{ts()}] ─── Bot detenido ───"})

# ═══════════ ESCAPE ROOM SCANNER DIRECTO ═══════════

async def _er_scan_direct(max_price: float = 0.33, fetch_limit: int = 300) -> list:
    """
    Scanner directo para Escape Room: busca mercados con precio ≤ max_price
    sin depender de full_scan(). Consulta ~4×fetch_limit mercados crudos
    y filtra por precio, sin requerir edge algorítmico previo.
    La validación de edge real queda en manos del análisis dual-IA.
    """
    _batches = await asyncio.gather(
        scanner.fetch_active_markets(limit=fetch_limit, order="volume24hr",  ascending=False),
        scanner.fetch_active_markets(limit=fetch_limit, order="createdAt",   ascending=False),
        scanner.fetch_active_markets(limit=fetch_limit, order="liquidity",   ascending=False),
        scanner.fetch_active_markets(limit=fetch_limit, order="volume24hr",  ascending=True),  # niche low-volume
        return_exceptions=True,
    )

    _seen_ids: set = set()
    markets: list = []
    for batch in _batches:
        if isinstance(batch, Exception) or not isinstance(batch, list):
            continue
        for m in batch:
            cid = m.get("conditionId", m.get("id", ""))
            if cid and cid not in _seen_ids:
                _seen_ids.add(cid)
                markets.append(m)

    _now_utc = datetime.now(timezone.utc)
    opportunities: list = []

    for mkt in markets:
        odds = scanner.parse_market_odds(mkt)
        if not odds:
            continue

        yes_qualifies = ER_MIN_ENTRY_PRICE <= odds.yes_price <= max_price
        no_qualifies  = ER_MIN_ENTRY_PRICE <= odds.no_price  <= max_price
        if not (yes_qualifies or no_qualifies):
            continue

        if odds.liquidity < ER_MIN_LIQUIDITY:
            continue

        # Excluir expirados
        end_date = mkt.get("endDate", "")
        if end_date:
            try:
                _dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if _dt.tzinfo is None:
                    _dt = _dt.replace(tzinfo=timezone.utc)
                if _dt <= _now_utc:
                    continue
            except (ValueError, TypeError):
                pass

        # Scoring con ensemble model
        q    = mkt.get("question", "")
        cid  = mkt.get("conditionId", mkt.get("id", ""))
        yt, nt = scanner.extract_token_ids(mkt)
        category = classify_market(q)
        _cat_str = category.value if isinstance(category, MarketCategory) else str(category)

        cat_signal = cat_eval.evaluate(q, odds.yes_price)
        # BUG5: usar historial de volumen para detectar volume surge real
        _prev_vol_er = _volume_history_cache.get(cid, None)
        _avg_vol_er = _prev_vol_er if (_prev_vol_er and _prev_vol_er > 0) else odds.volume_24h * 1.5
        _volume_history_cache[cid] = odds.volume_24h
        bay_est = bayesian.estimate_with_raw_data(
            market_price=odds.yes_price, sentiment_score=0.0,
            num_articles=0, volume_24h=odds.volume_24h,
            avg_volume_7d=_avg_vol_er, whale_positions=[],
            spread=odds.spread, liquidity=odds.liquidity)

        model_probs = [bay_est.posterior, cat_signal.estimated_prob, odds.yes_price]
        _base_w2 = _ENSEMBLE_WEIGHTS.get(_cat_str, _ENSEMBLE_WEIGHTS["default"])
        model_weights = [_base_w2[0], cat_signal.confidence * _base_w2[1], _base_w2[2]]
        ensemble_prob = Extremizer.weighted_extremize(
            list(zip(model_probs, model_weights)), d=extremizer_d)
        ensemble_prob = max(0.03, min(0.97, ensemble_prob))
        cal_pred = cal_suite.calibrate(ensemble_prob)
        ensemble_prob = cal_pred.calibrated

        kl_div = KLDivergence.binary_kl(ensemble_prob, odds.yes_price)

        # Determinar mejor lado (YES o NO)
        if yes_qualifies and no_qualifies:
            yes_edge = (ensemble_prob - odds.yes_price) / max(0.01, odds.yes_price) * 100
            no_edge  = ((1 - ensemble_prob) - odds.no_price) / max(0.01, odds.no_price) * 100
            if yes_edge >= no_edge:
                side, entry_price, edge_pct = "YES", odds.yes_price, yes_edge
            else:
                side, entry_price, edge_pct = "NO", odds.no_price, no_edge
        elif yes_qualifies:
            side = "YES"
            entry_price = odds.yes_price
            edge_pct = (ensemble_prob - odds.yes_price) / max(0.01, odds.yes_price) * 100
        else:
            side = "NO"
            entry_price = odds.no_price
            edge_pct = ((1 - ensemble_prob) - odds.no_price) / max(0.01, odds.no_price) * 100

        opp = EROpportunity(
            market_id=cid, question=q, slug=mkt.get("slug", ""),
            category=_cat_str, side=side,
            entry_price=round(entry_price, 4),
            multiplier=round(1.0 / max(0.01, entry_price), 2),
            ensemble_prob=round(ensemble_prob, 4),
            edge_pct=round(edge_pct, 2),
            kl_divergence=round(kl_div, 6),
            yes_token=yt, no_token=nt,
        )
        opportunities.append(opp)

    # Primero edge positivo, luego por KL divergence (la IA decide sobre los negativos)
    opportunities.sort(key=lambda o: (o.edge_pct > 0, o.kl_divergence), reverse=True)
    return opportunities[:50]


# ═══════════ ESCAPE ROOM BOT ═══════════

async def _er_worker():
    """Bot del Escape Room: busca mercados ≤ $0.33 con análisis dual-IA."""
    global _er_running, _er_last_scan_msg, _er_last_scan_ts
    from datetime import datetime as _dt
    ts = lambda: _dt.now().strftime("%H:%M:%S")

    lax_tag = " [SIN_DUAL_IA]" if not _er_require_dual_ai else ""
    await broadcast("er_log", {"line": f"[{ts()}] Escape Room iniciado{lax_tag} — modo {_er_mode.upper()} — buscando oportunidades >=3x..."})

    async def _run_er_cycle():
        global _er_running, _er_session_bets, _er_session_capital, _er_session_scans
        global _er_last_scan_msg, _er_last_scan_ts
        _er_session_scans += 1
        try:
            engine = get_er_engine()

            # 1. Scanner directo: busca mercados con precio ≤ 0.33 sin filtros de full_scan
            await broadcast("er_log", {"line": f"[{ts()}] Escaneando mercados (modo directo ~1200 mercados)..."})
            opportunities = await _er_scan_direct(max_price=0.33, fetch_limit=300)

            if not opportunities:
                _er_last_scan_msg = "Sin mercados con precio ≤ 0.33 encontrados"
                _er_last_scan_ts  = ts()
                await broadcast("er_log", {"line": f"[{ts()}] Sin oportunidades >=3x (precio ≤ $0.33) en este ciclo."})
                return

            scan_summary = (f"[ER] Escaneadas {len(opportunities)} oportunidades >=3x — "
                            f"analizando top {min(_er_max_bets * 2, 6)}")
            _er_last_scan_msg = scan_summary
            _er_last_scan_ts  = ts()
            await broadcast("er_log", {"line": f"[{ts()}] {scan_summary}"})

            # 3. Analizar top oportunidades con dual-IA
            top = opportunities[:min(_er_max_bets * 2, 6)]
            await broadcast("er_log", {"line": f"[{ts()}] Analizando {len(top)} oportunidades con Claude + GPT..."})

            enriched = []
            for opp in top:
                if not _er_running:
                    break
                sports_ctx = await engine.enrich_with_sports_context(opp)
                opp.sports_context = sports_ctx
                if llm.anthropic_key or llm.openai_key:
                    opp = await engine.analyze_with_dual_ai(opp, sports_ctx)
                else:
                    opp.er_score = opp.edge_pct * 10
                # Log individual IA result
                await broadcast("er_log", {
                    "line": (f"[{ts()}] IA: Claude={opp.claude_recommendation} "
                             f"GPT={opp.gpt_recommendation} "
                             f"score={opp.er_score:.0f} "
                             f"@ {opp.entry_price:.2f} ({opp.multiplier:.1f}x) — "
                             f"{opp.question[:50]}")
                })
                enriched.append(opp)

            # Broadcast resultados al dashboard
            er_data = [
                {
                    "market_id": o.market_id, "question": o.question,
                    "category": o.category, "side": o.side,
                    "entry_price": o.entry_price, "multiplier": o.multiplier,
                    "ensemble_prob": o.ensemble_prob, "edge_pct": o.edge_pct,
                    "claude_prob": o.claude_prob, "claude_confidence": o.claude_confidence,
                    "claude_reasoning": o.claude_reasoning,
                    "claude_recommendation": o.claude_recommendation,
                    "gpt_prob": o.gpt_prob, "gpt_confidence": o.gpt_confidence,
                    "gpt_reasoning": o.gpt_reasoning,
                    "gpt_recommendation": o.gpt_recommendation,
                    "ai_consensus": o.ai_consensus,
                    "ai_combined_prob": o.ai_combined_prob,
                    "sports_context": o.sports_context,
                    "er_score": o.er_score,
                    "bet_usd": o.bet_usd,
                    "yes_token": o.yes_token, "no_token": o.no_token,
                }
                for o in enriched
            ]
            await broadcast("er_opportunities", {"opportunities": er_data})

            # 4. Ejecutar trades si modo LIVE
            if _er_mode == "live":
                available = max(0.0, risk.state.current_capital - risk.state.total_exposure_usd)
                executed = 0
                for opp in enriched:
                    if not _er_running or executed >= _er_max_bets:
                        break
                    bet_size = engine.calculate_bet_size(
                        opp, available,
                        require_dual_ai=_er_require_dual_ai,
                        score_min_override=_er_score_min_override,
                    )
                    if bet_size <= 0:
                        continue
                    opp.bet_usd = bet_size

                    # Validar con risk manager
                    val = risk.validate_trade(
                        bet_usd=bet_size, confidence=opp.er_score,
                        market_liquidity=500.0, spread=0.05,
                        market_id=opp.market_id
                    )
                    if not val["approved"]:
                        await broadcast("er_log", {"line": f"[{ts()}] Bloqueado: {opp.question[:40]}... -> {val['reasons']}"})
                        continue

                    # Ejecutar
                    tid = opp.yes_token if opp.side == "YES" else opp.no_token
                    try:
                        result = trader.place_market_order(
                            tid, bet_size, "BUY",
                            market_id=opp.market_id,
                            question=opp.question,
                        )
                        if result.status == "FILLED":
                            await broadcast("er_log", {"line": f"[{ts()}] TRADE: {opp.side} ${bet_size:.2f} @ {opp.entry_price} ({opp.multiplier}x) — {opp.question[:40]}..."})
                            executed += 1
                            available -= bet_size
                            _er_session_bets += 1
                            _er_session_capital += bet_size
                            _record_trade("er", {
                                "id": f"er_{_er_session_bets}",
                                "question": opp.question,
                                "side": opp.side,
                                "bet_usd": bet_size,
                                "status": "pending",
                                "mode": "live",
                                "price": opp.entry_price,
                                "asset": opp.category,
                            })
                        else:
                            await broadcast("er_log", {"line": f"[{ts()}] Error: {result.status} — {result.error_msg or 'unknown'}"})
                    except Exception as e:
                        await broadcast("er_log", {"line": f"[{ts()}] Excepcion: {str(e)[:80]}"})
            else:
                # DEMO: simular
                _min_demo_score = (_er_score_min_override if _er_score_min_override > 0
                                   else (35.0 if not _er_require_dual_ai else 40.0))
                for opp in enriched[:_er_max_bets]:
                    demo_bet = engine.calculate_bet_size(
                        opp, 100.0,
                        require_dual_ai=_er_require_dual_ai,
                        score_min_override=_er_score_min_override,
                    )
                    if opp.er_score >= _min_demo_score and demo_bet > 0:
                        demo_size = opp.bet_usd or 1.0
                        await broadcast("er_log", {"line": f"[{ts()}] [DEMO] {opp.side} ${demo_size:.2f} @ {opp.entry_price} ({opp.multiplier}x) — ER Score: {opp.er_score} — {opp.question[:50]}..."})
                        _er_session_bets += 1
                        _er_session_capital += demo_size
                        _record_trade("er", {
                            "id": f"er_demo_{_er_session_bets}",
                            "question": opp.question,
                            "side": opp.side,
                            "bet_usd": demo_size,
                            "status": "pending",
                            "mode": "demo",
                            "price": opp.entry_price,
                            "asset": opp.category,
                        })

        except Exception as e:
            await broadcast("er_log", {"line": f"[{ts()}] Error en ciclo: {e}"})
            logger.exception("ER cycle error")

    try:
        await asyncio.sleep(1)
        await _run_er_cycle()

        while _er_running:
            wait_total = _er_interval * 60
            for i in range(wait_total):
                if not _er_running:
                    return
                await asyncio.sleep(1)
                if i == wait_total - 60:
                    await broadcast("er_log", {"line": f"[{ts()}] Proximo ciclo Escape Room en 1 minuto..."})
            if not _er_running:
                return
            await _run_er_cycle()

    except asyncio.CancelledError:
        pass
    finally:
        await broadcast("er_log", {"line": f"[{ts()}] Escape Room detenido."})

# ═══════════ BOT CONTROL ═══════════

@app.post("/api/bot/start")
async def start_bot(request: Request):
    """Inicia el bot con la configuración especificada."""
    global _bot_running, _bot_mode, _bot_interval, _bot_max_risk, _bot_min_edge, _bot_time_horizon, _bot_min_bets, _bot_max_bets, _bot_max_exposure, _bot_max_bet_usd, _bot_task
    try:
        req = await request.json()
    except Exception:
        req = {}
    _bot_mode = req.get("mode", "demo")
    _bot_interval = int(req.get("interval_minutes", 30))
    _bot_max_risk = float(req.get("max_risk_pct", 0.25))
    _bot_min_edge = float(req.get("min_edge_pct", 15.0))
    _bot_time_horizon = req.get("time_horizon", "any")
    _bot_min_bets = max(1, int(req.get("min_bets", 1)))
    _bot_max_bets = max(_bot_min_bets, int(req.get("max_bets", 5)))
    _bot_max_exposure = max(10.0, min(100.0, float(req.get("max_exposure_pct", 80.0))))
    _bot_max_bet_usd  = max(1.00, float(req.get("max_bet", 1.0)))  # Polymarket mínimo $1
    risk.limits.max_total_exposure_pct = _bot_max_exposure

    # dry_run: DEMO=simulación, LIVE=operaciones reales
    trader.dry_run = (_bot_mode == "demo")
    # Limpiar circuit breaker al arrancar — bloqueos de sesiones anteriores no deben detener el bot nuevo
    validator.reset_circuit_breaker()
    # Si es LIVE y aún no autenticado, intentar inicializar ahora
    if _bot_mode == "live" and not trader.is_authenticated:
        try:
            result = trader.initialize()
            if result.get("status") != "authenticated":
                return {"ok": False, "error": f"No se pudo autenticar con Polymarket: {result.get('message', '')}"}
        except Exception as e:
            return {"ok": False, "error": f"Error de autenticación: {str(e)}"}

    # Cancelar task anterior si existe
    if _bot_task and not _bot_task.done():
        _bot_running = False
        _bot_task.cancel()
        try:
            await _bot_task
        except asyncio.CancelledError:
            pass

    _bot_running = True
    _bot_task = asyncio.create_task(_bot_worker())
    return {
        "ok": True,
        "mode": _bot_mode,
        "interval": _bot_interval,
        "max_risk": _bot_max_risk,
        "min_edge": _bot_min_edge,
        "message": f"Bot iniciado en modo {_bot_mode.upper()} — escaneo cada {_bot_interval} min"
    }

@app.post("/api/bot/stop")
async def stop_bot():
    """Detiene el bot."""
    global _bot_running, _bot_task
    _bot_running = False
    if _bot_task and not _bot_task.done():
        _bot_task.cancel()
        try:
            await _bot_task
        except asyncio.CancelledError:
            pass
    return {"ok": True, "message": "Bot detenido"}

@app.get("/api/bot/log")
async def get_bot_log(lines: int = Query(100)):
    """Devuelve las últimas N líneas del bot_activity.log para diagnóstico."""
    import pathlib as _pl
    log_path = _pl.Path(__file__).parent / "bot_activity.log"
    if not log_path.exists():
        return {"lines": [], "note": "bot_activity.log no existe aún — inicia el bot primero"}
    try:
        all_lines = log_path.read_text(encoding="utf-8").splitlines()
        return {"lines": all_lines[-lines:], "total": len(all_lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)}

@app.get("/api/bot/status")
async def get_bot_status():
    """Retorna el estado actual del bot."""
    return {
        "running": _bot_running,
        "mode": _bot_mode,
        "interval_minutes": _bot_interval,
        "max_risk_pct": _bot_max_risk,
        "min_edge_pct": _bot_min_edge,
        "time_horizon": _bot_time_horizon,
        "task_alive": _bot_task is not None and not _bot_task.done(),
    }

# ═══════════ ESCAPE ROOM ENDPOINTS ═══════════

@app.get("/api/escape-room/scan")
async def er_scan(limit: int = Query(50, ge=10, le=200),
                  min_multiplier: float = Query(3.0, ge=2.0, le=20.0)):
    """Escanea y devuelve oportunidades Escape Room (precio <= 1/min_multiplier).
    Usa scanner directo (~1200 mercados crudos) en vez de full_scan para encontrar
    mercados de baja probabilidad que full_scan excluiría por sus filtros de edge.
    """
    try:
        max_price = round(1.0 / min_multiplier, 4)
        fetch_limit = max(limit * 3, 300)
        opps = await _er_scan_direct(max_price=max_price, fetch_limit=fetch_limit)

        return {
            "opportunities": [
                {
                    "market_id": o.market_id, "question": o.question,
                    "category": o.category, "side": o.side,
                    "entry_price": o.entry_price, "multiplier": o.multiplier,
                    "ensemble_prob": o.ensemble_prob, "edge_pct": o.edge_pct,
                    "kl_divergence": o.kl_divergence,
                    "er_score": o.er_score,
                    "yes_token": o.yes_token, "no_token": o.no_token,
                }
                for o in opps
            ],
            "total": len(opps),
            "max_entry_price": max_price,
            "min_multiplier": min_multiplier,
        }
    except Exception as e:
        raise HTTPException(500, f"ER scan error: {e}")


@app.post("/api/escape-room/analyze/{market_id}")
async def er_analyze(market_id: str):
    """Análisis dual-IA profundo de una oportunidad específica."""
    try:
        # Usar scanner directo para encontrar el mercado aunque tenga precio bajo
        all_opps = await _er_scan_direct(max_price=0.50, fetch_limit=300)
        opp = next((o for o in all_opps if o.market_id == market_id), None)
        if not opp:
            raise HTTPException(404, "Mercado no encontrado o no califica para Escape Room")

        engine = get_er_engine()
        if opp.entry_price > 0.33:
            raise HTTPException(400, "Este mercado no califica para Escape Room (precio > 0.33)")

        sports_ctx = await engine.enrich_with_sports_context(opp)
        opp.sports_context = sports_ctx
        opp = await engine.analyze_with_dual_ai(opp, sports_ctx)

        available = max(0.0, risk.state.current_capital - risk.state.total_exposure_usd)
        opp.bet_usd = engine.calculate_bet_size(opp, available)

        # Para el lado NO, la probabilidad relevante a mostrar es 1 - ensemble_prob
        _model_prob_side = opp.ensemble_prob if opp.side == "YES" else (1.0 - opp.ensemble_prob)
        return {
            "market_id": opp.market_id, "question": opp.question,
            "pregunta_es": opp.pregunta_es or opp.question,
            "category": opp.category, "side": opp.side,
            "entry_price": opp.entry_price, "multiplier": opp.multiplier,
            "ensemble_prob": round(_model_prob_side, 4), "edge_pct": opp.edge_pct,
            "claude_prob": opp.claude_prob, "claude_confidence": opp.claude_confidence,
            "claude_reasoning": opp.claude_reasoning,
            "claude_recommendation": opp.claude_recommendation,
            "gpt_prob": opp.gpt_prob, "gpt_confidence": opp.gpt_confidence,
            "gpt_reasoning": opp.gpt_reasoning,
            "gpt_recommendation": opp.gpt_recommendation,
            "ai_consensus": opp.ai_consensus, "ai_combined_prob": opp.ai_combined_prob,
            "sports_context": opp.sports_context,
            "er_score": opp.er_score, "bet_usd": opp.bet_usd,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"ER analyze error: {e}")


@app.post("/api/escape-room/start")
async def er_start(request: Request):
    """Inicia el bot Escape Room."""
    global _er_running, _er_mode, _er_interval, _er_max_bets, _er_time_horizon, _er_task
    global _er_session_bets, _er_session_capital, _er_session_scans
    global _er_require_dual_ai, _er_score_min_override
    try:
        req = await request.json()
    except Exception:
        req = {}
    _er_mode = req.get("mode", "demo")
    _er_interval = int(req.get("interval_minutes", 30))
    _er_max_bets = max(1, int(req.get("max_bets", 3)))
    _er_time_horizon = req.get("time_horizon", "any")
    _er_require_dual_ai = bool(req.get("require_dual_ai", True))
    _er_score_min_override = float(req.get("score_min", 0.0))
    # Resetear stats de sesión al iniciar
    _er_session_bets = 0
    _er_session_capital = 0.0
    _er_session_scans = 0

    trader.dry_run = (_er_mode == "demo")
    if _er_mode == "live" and not trader.is_authenticated:
        try:
            result = trader.initialize()
            if result.get("status") != "authenticated":
                return {"ok": False, "error": f"No autenticado: {result.get('message', '')}"}
        except Exception as e:
            return {"ok": False, "error": f"Error auth: {str(e)}"}

    if _er_task and not _er_task.done():
        _er_running = False
        _er_task.cancel()
        try:
            await _er_task
        except asyncio.CancelledError:
            pass

    _er_running = True
    _er_task = asyncio.create_task(_er_worker())
    return {"ok": True, "mode": _er_mode, "interval": _er_interval,
            "require_dual_ai": _er_require_dual_ai,
            "score_min": _er_score_min_override,
            "message": f"Escape Room iniciado en modo {_er_mode.upper()}"}


@app.post("/api/escape-room/stop")
async def er_stop():
    """Detiene el bot Escape Room."""
    global _er_running, _er_task
    _er_running = False
    if _er_task and not _er_task.done():
        _er_task.cancel()
        try:
            await _er_task
        except asyncio.CancelledError:
            pass
    return {"ok": True, "message": "Escape Room detenido"}


@app.get("/api/escape-room/status")
async def er_status():
    """Estado del bot Escape Room."""
    return {
        "running": _er_running,
        "mode": _er_mode,
        "interval_minutes": _er_interval,
        "max_bets": _er_max_bets,
        "time_horizon": _er_time_horizon,
        "task_alive": _er_task is not None and not _er_task.done(),
        "session_bets": _er_session_bets,
        "session_capital": round(_er_session_capital, 2),
        "session_scans": _er_session_scans,
    }

# ═══════════ SWARM INTEL v6.0 ENDPOINTS ═══════════

# Auto-fetch Polymarket — Carlos David Donoso Cordero (ddchack)
async def fetch_polymarket_options(slug: str) -> tuple[list, dict, str]:
    """
    Retorna (options, market_prices, question_title)
    Consulta Gamma API de Polymarket para obtener opciones de un mercado automáticamente.
    """
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=10) as client:
        # Intentar events primero (torneos multi-opción)
        try:
            r = await client.get(f"https://gamma-api.polymarket.com/events?slug={slug}")
            if r.status_code == 200:
                data = r.json()
                if data and len(data) > 0:
                    event = data[0]
                    markets = event.get("markets", [])
                    title = event.get("title", slug)
                    if len(markets) > 1:
                        options = [m.get("question", m.get("groupItemTitle", "")) for m in markets]
                        prices = {}
                        for m in markets:
                            op = m.get("outcomePrices", ["0.5"])
                            # outcomePrices puede llegar como string JSON — Carlos David Donoso Cordero (ddchack)
                            if isinstance(op, str):
                                import json as _jt
                                try:
                                    op = _jt.loads(op)
                                except Exception:
                                    op = ["0.5"]
                            key = m.get("question", m.get("groupItemTitle", ""))
                            try:
                                prices[key] = float(op[0]) if op else 0.5
                            except Exception:
                                prices[key] = 0.5
                        return options, prices, title
                    elif len(markets) == 1:
                        # Extraer precio binario (YES) del único mercado — Carlos David Donoso Cordero (ddchack)
                        _m1 = markets[0]
                        _op = _m1.get("outcomePrices", ["0.5", "0.5"])
                        if isinstance(_op, str):
                            import json as _j2
                            try:
                                _op = _j2.loads(_op)
                            except Exception:
                                _op = ["0.5"]
                        try:
                            _binary_price = float(_op[0]) if _op else 0.5
                        except Exception:
                            _binary_price = 0.5
                        return [], {"__binary__": _binary_price}, _m1.get("question", title)
        except Exception as e:
            logger.warning(f"Gamma events fetch failed for slug={slug}: {e}")

        # Fallback: markets endpoint
        try:
            r = await client.get(f"https://gamma-api.polymarket.com/markets?slug={slug}")
            if r.status_code == 200:
                data = r.json()
                if data:
                    m = data[0] if isinstance(data, list) else data
                    return [], {}, m.get("question", slug)
        except Exception as e:
            logger.warning(f"Gamma markets fetch failed for slug={slug}: {e}")

    return [], {}, slug


@app.get("/api/swarm/personas")
async def get_swarm_personas():
    """Lista completa de las 50 personas del swarm con su estado actual."""
    calibration_tracker.apply_to_swarm_engine(swarm_engine)
    personas_status = swarm_engine.get_all_personas_status()
    return {"personas": personas_status, "total": len(personas_status)}


@app.post("/api/swarm/analyze")
async def swarm_analyze(request: Request):
    """
    Análisis full-swarm de un mercado.
    Body: {"question": str, "market_id": str (optional), "algo_data": dict (optional), "options": list[str] (optional, torneo)}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    question = body.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="question requerido")

    market_id = body.get("market_id", "")
    algo_data = body.get("algo_data", {})
    # Opciones manuales como fallback — Carlos David Donoso Cordero (ddchack)
    manual_options = [str(o).strip() for o in body.get("options", []) if str(o).strip()]

    # Auto-fetch desde Polymarket Gamma API — Carlos David Donoso Cordero (ddchack)
    auto_options: list = []
    market_prices: dict = {}
    question_title: str = question
    slug = market_id or ""
    # Extraer slug de URL si viene la URL completa en market_id o question
    import re as _re
    for _src in [market_id, question]:
        _m = _re.search(r'polymarket\.com/event/([^/?#\s]+)', _src or '')
        if _m:
            slug = _m.group(1)
            break

    if slug:
        try:
            auto_options, market_prices, question_title = await fetch_polymarket_options(slug)
            logger.info(f"Gamma auto-fetch: slug={slug} → {len(auto_options)} opciones, título='{question_title}'")
        except Exception as e:
            logger.warning(f"fetch_polymarket_options error: {e}")

    # Prioridad: opciones de Gamma > manuales
    options = auto_options if auto_options else manual_options

    # Inyectar precio binario de Gamma en algo_data para que el swarm lo use — Carlos David Donoso Cordero (ddchack)
    if "__binary__" in market_prices:
        algo_data.setdefault("yes_price", market_prices["__binary__"])

    # Propagar keys del llm al swarm_engine si no están configuradas
    if not swarm_engine.anthropic_key and llm.anthropic_key:
        swarm_engine.configure(
            anthropic_key=llm.anthropic_key,
            openai_key=llm.openai_key,
            anthropic_model="claude-haiku-4-5-20251001",
        )

    calibration_tracker.apply_to_swarm_engine(swarm_engine)
    result = await swarm_engine.analyze(question, algo_data, market_id=market_id, timeout_seconds=20.0, options=options)
    result_dict = swarm_engine.swarm_result_to_dict(result)
    # Debug torneo — Carlos David Donoso Cordero (ddchack)
    logger.info(f"SWARM DEBUG: options={options}, market_type en result={result_dict.get('market_type')}, option_probs keys={list(result_dict.get('option_probs',{}).keys())}")
    # Añadir metadatos de Polymarket al resultado
    result_dict["question_title"] = question_title
    result_dict["market_prices"] = market_prices
    # Sobrescribir market_price con precio real de Gamma si está disponible — Carlos David Donoso Cordero (ddchack)
    if "__binary__" in market_prices:
        result_dict["market_price"] = market_prices["__binary__"]
    return result_dict


@app.get("/api/swarm/brier-scores")
async def get_swarm_brier_scores():
    """Tabla de Brier scores por persona y resumen por arquetipo."""
    return {
        "scores": calibration_tracker.get_all_scores(),
        "archetype_summary": calibration_tracker.get_archetype_summary(),
    }


@app.post("/api/swarm/record-resolution")
async def record_swarm_resolution(request: Request):
    """
    Registra la resolución de un mercado para actualizar Brier scores.
    Body: {"resolutions": [{"name": str, "archetype": str, "predicted_prob": float, "outcome": 0|1}]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    resolutions = body.get("resolutions", [])
    updated = []
    for r in resolutions:
        name = r.get("name", "")
        arch = r.get("archetype", "")
        prob = float(r.get("predicted_prob", 0.5))
        outcome = int(r.get("outcome", 0))
        if name:
            new_brier = calibration_tracker.record_resolution(name, arch, prob, outcome)
            updated.append({"name": name, "new_brier_ema": round(new_brier, 5)})
    calibration_tracker.apply_to_swarm_engine(swarm_engine)
    return {"updated": len(updated), "details": updated}


@app.post("/api/bot/horizon")
async def update_bot_horizon(request: Request):
    """Actualiza el horizonte temporal del bot en caliente (sin reiniciar)."""
    global _bot_time_horizon
    try:
        req = await request.json()
    except Exception:
        req = {}
    h = req.get("time_horizon", _bot_time_horizon)
    valid = {"minutes", "hours", "days", "weeks", "months", "any"}
    if h in valid:
        _bot_time_horizon = h
    return {"ok": True, "time_horizon": _bot_time_horizon}

# ═══════════ TRADE VALIDATOR ENDPOINTS ═══════════

@app.get("/api/blocked-trades")
async def get_blocked_trades(limit: int = Query(100)):
    """Historial de trades bloqueadas por el TradeValidator."""
    history = validator.get_blocked_history(limit=limit)
    return {
        "blocked_trades": history,
        "total": len(history),
        "log_file": str(validator._blocked_history and "blocked_trades.log"),
    }

@app.get("/api/validator/status")
async def get_validator_status():
    """Estado actual del TradeValidator."""
    return validator.get_status()

@app.post("/api/validator/reset")
async def reset_validator():
    """Resetea el circuit breaker manualmente."""
    was_active = validator.check_circuit_breaker()
    validator.reset_circuit_breaker()
    return {"ok": True, "circuit_breaker_was_active": was_active,
            "message": "Circuit breaker reseteado. El bot puede reanudar."}

class ValidatorConfigRequest(BaseModel):
    min_price_cents: Optional[float] = None          # default 5
    max_price_cents: Optional[float] = None          # default 95
    min_liquidity_usd: Optional[float] = None        # default 500
    min_volume_24h_usd: Optional[float] = None       # default 100
    min_minutes_to_resolution: Optional[int] = None  # default 10
    dedup_window_seconds: Optional[int] = None       # default 300
    min_kelly_override: Optional[float] = None       # default 0.50

@app.post("/api/validator/config")
async def config_validator(req: ValidatorConfigRequest):
    """Actualiza umbrales del TradeValidator en runtime (sin reiniciar)."""
    changed = {}
    if req.min_price_cents is not None:
        validator.MIN_PRICE = req.min_price_cents / 100.0
        changed["min_price"] = validator.MIN_PRICE
    if req.max_price_cents is not None:
        validator.MAX_PRICE = req.max_price_cents / 100.0
        changed["max_price"] = validator.MAX_PRICE
    if req.min_liquidity_usd is not None:
        validator.MIN_LIQUIDITY_USD = req.min_liquidity_usd
        changed["min_liquidity_usd"] = req.min_liquidity_usd
    if req.min_volume_24h_usd is not None:
        validator.MIN_VOLUME_24H_USD = req.min_volume_24h_usd
        changed["min_volume_24h_usd"] = req.min_volume_24h_usd
    if req.min_minutes_to_resolution is not None:
        validator.MIN_MINUTES_TO_RESOLUTION = req.min_minutes_to_resolution
        changed["min_minutes_to_resolution"] = req.min_minutes_to_resolution
    if req.dedup_window_seconds is not None:
        validator.DEDUP_WINDOW_SECONDS = req.dedup_window_seconds
        changed["dedup_window_seconds"] = req.dedup_window_seconds
    if req.min_kelly_override is not None:
        validator.MIN_KELLY_OVERRIDE = req.min_kelly_override
        changed["min_kelly_override"] = req.min_kelly_override
    return {"ok": True, "changed": changed, "current": validator.get_status()["thresholds"]}


class EnsembleWeightsRequest(BaseModel):
    weights: dict   # {"politics": [0.35, 0.45, 0.20], ...}
    source: str = "manual"

@app.post("/api/tune/apply")
async def apply_ensemble_weights(req: EnsembleWeightsRequest):
    """
    Aplica pesos del ensemble en runtime. Los pesos afectan inmediatamente
    al /api/scan sin reiniciar el servidor.

    Formato: {"politics": [bay_w, cat_w_factor, mkt_w], "sports": [...], ...}
    """
    global _ENSEMBLE_WEIGHTS, _ENSEMBLE_WEIGHTS_SOURCE
    applied = {}
    for cat, w in req.weights.items():
        if not isinstance(w, list) or len(w) != 3:
            continue
        if all(isinstance(x, (int, float)) for x in w):
            _ENSEMBLE_WEIGHTS[cat] = [float(x) for x in w]
            applied[cat] = _ENSEMBLE_WEIGHTS[cat]
    _ENSEMBLE_WEIGHTS_SOURCE = req.source
    # Persistir pesos a disco para sobrevivir reinicios
    try:
        _weights_path = os.path.join(os.path.dirname(__file__), "ensemble_weights.json")
        with open(_weights_path, "w", encoding="utf-8") as _wf:
            json.dump({
                "weights": _ENSEMBLE_WEIGHTS,
                "source": _ENSEMBLE_WEIGHTS_SOURCE,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }, _wf, ensure_ascii=False, indent=2)
        print(f"[Weights] Pesos ensemble guardados en {_weights_path}")
    except Exception as _we:
        print(f"[Weights] Error guardando pesos: {_we}")
    return {
        "ok": True, "applied": applied,
        "current_weights": _ENSEMBLE_WEIGHTS,
        "source": _ENSEMBLE_WEIGHTS_SOURCE,
    }

@app.get("/api/tune/weights/current")
async def get_current_weights():
    """Retorna los pesos del ensemble actualmente en uso."""
    return {
        "weights": _ENSEMBLE_WEIGHTS,
        "source": _ENSEMBLE_WEIGHTS_SOURCE,
        "extremizer_d": extremizer_d,
    }

@app.post("/api/tune/extremizer")
async def tune_extremizer(num_markets: int = Query(50, ge=10, le=200)):
    """
    Optimiza el parámetro d del extremizer de Satopää usando backtest histórico.
    Minimiza Brier Score sobre mercados resueltos → actualiza extremizer_d global.
    """
    global extremizer_d
    bt = BacktestEngine()
    try:
        result = await bt.run_backtest(num_markets=min(num_markets, 100))
    except Exception as e:
        return {"error": f"Backtest falló: {str(e)}"}

    # Extraer (estimated_prob, outcome) pairs desde los trades del backtest
    trades = result.trades
    if len(trades) < 10:
        return {"error": "Insuficientes trades resueltos para optimizar d", "trades": len(trades)}

    # Convertimos la prob estimada en lista de listas (1 "modelo" por trade)
    # find_optimal_d acepta predictions_list: [[p_model1, p_model2...], ...]
    predictions_list = [[t["estimated_prob"]] for t in trades]
    outcomes = [1 if t["resolved_outcome"] == "YES" else 0 for t in trades]

    # d óptimo minimizando Brier Score
    old_d = extremizer_d
    new_d = Extremizer.find_optimal_d(predictions_list, outcomes)
    # Clamp a rango razonable (0.8 - 3.5)
    new_d = max(0.8, min(3.5, new_d))
    extremizer_d = new_d

    # Brier score antes vs después (aproximación con los datos disponibles)
    def brier_at(d_val):
        total = 0.0
        for t, outcome in zip(trades, outcomes):
            p = t["estimated_prob"]
            # Re-extremizar desde logit medio (solo 1 prob aquí)
            from extremizer import Extremizer as _E
            ext = _E.extremize([p], d=d_val)
            total += (ext - outcome) ** 2
        return total / len(trades)

    bs_old = round(brier_at(old_d), 4)
    bs_new = round(brier_at(new_d), 4)
    improvement_pct = round((bs_old - bs_new) / max(0.0001, bs_old) * 100, 1)

    return {
        "ok": True,
        "old_d": old_d,
        "new_d": new_d,
        "brier_old": bs_old,
        "brier_new": bs_new,
        "improvement_pct": improvement_pct,
        "trades_analyzed": len(trades),
        "note": f"extremizer_d actualizado: {old_d} → {new_d} ({improvement_pct:+.1f}% Brier)",
    }

@app.get("/api/validator/test/{market_id}")
async def test_validator(market_id: str, outcome: str = Query("YES"),
                         amount: float = Query(1.0)):
    """
    Testa si un mercado pasaría las validaciones sin ejecutar ninguna orden.
    Consulta Gamma API para obtener datos frescos del mercado.
    """
    # Fetch datos frescos del mercado
    market_data: dict = {}
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = f"https://gamma-api.polymarket.com/markets/{market_id}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    import json as _json
                    op = _json.loads(raw.get("outcomePrices", "[]") or "[]")
                    market_data = {
                        "yes_price":  float(op[0]) if op else 0.5,
                        "no_price":   float(op[1]) if len(op) > 1 else 0.5,
                        "end_date":   raw.get("endDate", ""),
                        "closed":     raw.get("closed", False),
                        "active":     raw.get("active", True),
                        "category":   raw.get("category", ""),
                        "volume_24h": float(raw.get("volume24hr", 0) or 0),
                    }
    except Exception as e:
        return {"error": f"No se pudo obtener datos del mercado: {str(e)}",
                "market_id": market_id}

    # Ejecutar validación completa (incluye Gamma API refresh)
    is_valid, reason = await validator.validate_trade_full(
        market_id=market_id,
        outcome=outcome,
        amount=amount,
        kelly_amount=amount,
        market_data=market_data,
    )
    return {
        "market_id": market_id,
        "outcome":   outcome,
        "amount":    amount,
        "valid":     is_valid,
        "reason":    reason,
        "market_data_used": market_data,
    }


def _generate_monitoring_analysis(positions: list, total_invested: float, total_pnl: float) -> list:
    """Genera análisis IA basado en posiciones activas."""
    from datetime import datetime, timezone
    if not positions:
        return ["Sin posiciones activas todavía. Usa Automatización IA para ejecutar apuestas."]
    now = datetime.now(timezone.utc)
    insights = []
    roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    if total_pnl > 0.01:
        insights.append(f"📈 Portfolio en positivo: +${total_pnl:.2f} ({roi:+.1f}% ROI). El edge estadístico está funcionando.")
    elif total_pnl < -0.01:
        insights.append(f"📉 Portfolio en negativo: ${total_pnl:.2f} ({roi:.1f}% ROI). Normal en mercados volátiles — el edge se materializa con el tiempo.")
    else:
        insights.append("➡ Portfolio en equilibrio. Esperando resolución de mercados.")
    winning = [p for p in positions if p["pnl"] > 0.005]
    losing  = [p for p in positions if p["pnl"] < -0.005]
    if winning:
        best = max(winning, key=lambda p: p["pnl"])
        insights.append(f"🏆 Mejor posición: {best['question'][:55]}… ({best['side']}) +${best['pnl']:.2f}")
    if losing:
        worst = min(losing, key=lambda p: p["pnl"])
        insights.append(f"⚠ Bajo presión: {worst['question'][:55]}… (${worst['pnl']:.2f}). Verifica si el edge original sigue vigente.")
    for p in positions:
        if not p.get("end_date") or not p.get("is_active"):
            continue
        try:
            ed = datetime.fromisoformat(p["end_date"].replace("Z", "+00:00"))
            h = (ed - now).total_seconds() / 3600
            if 0 < h < 3:
                insights.append(f"⏰ Cierra en {h:.1f}h: {p['question'][:60]}…")
            elif 0 < h < 12:
                insights.append(f"⌛ Cierra en {h:.0f}h: {p['question'][:60]}…")
        except Exception:
            pass
    if total_invested > 0 and len(positions) >= 2:
        max_pos = max(positions, key=lambda p: p["size_usd"])
        conc = max_pos["size_usd"] / total_invested * 100
        if conc > 60:
            insights.append(f"🔍 Concentración alta: {conc:.0f}% del capital en una sola posición. Considera diversificar.")
    no_data = sum(1 for p in positions if not p.get("has_market_data"))
    if no_data > 0:
        insights.append(f"ℹ {no_data} posición(es) sin precio en tiempo real (mercado cerrado o fuera del feed activo).")
    if len(insights) <= 1:
        insights.append("✅ Sin alertas activas. Portfolio equilibrado.")
    return insights


@app.get("/api/monitoring/positions")
async def get_monitoring_positions():
    """Posiciones activas enriquecidas con precios actuales para el panel de Monitoreo."""
    try:
        import json as _json
        from datetime import datetime, timezone
        summary = trader.get_trade_summary()
        history = summary.get("history", [])
        filled  = [t for t in history if t.get("status") == "FILLED"]
        if not filled:
            return {
                "positions": [], "summary": {"total_invested": 0, "n_positions": 0, "total_pnl": 0, "n_active": 0},
                "ai_analysis": ["Sin posiciones registradas. Usa Automatización IA para ejecutar apuestas."]
            }
        market_map: dict = {}
        try:
            batches = await asyncio.gather(
                scanner.fetch_active_markets(limit=150, order="volume24hr", ascending=False),
                scanner.fetch_active_markets(limit=150, order="createdAt", ascending=False),
                return_exceptions=True,
            )
            for batch in batches:
                if isinstance(batch, list):
                    for m in batch:
                        cid = m.get("conditionId") or m.get("id", "")
                        if cid:
                            market_map[cid] = m
        except Exception:
            pass
        now_utc = datetime.now(timezone.utc)
        positions = []
        for t in filled:
            mid    = t.get("market_id", "")
            mkt    = market_map.get(mid, {})
            side   = t.get("side", "YES")
            entry_p = float(t.get("price", 0.5) or 0.5)
            size   = float(t.get("size_usd", 0) or 0)
            current_p = entry_p
            if mkt:
                try:
                    op = mkt.get("outcomePrices", "[]")
                    if isinstance(op, str):
                        op = _json.loads(op)
                    idx = 0 if side == "YES" else 1
                    if len(op) > idx and op[idx]:
                        current_p = float(op[idx])
                except Exception:
                    pass
            shares        = size / entry_p if entry_p > 0 else 0
            current_value = shares * current_p
            pnl           = current_value - size
            end_date = mkt.get("endDate") or mkt.get("end_date_iso") or t.get("end_date", "")
            is_active = False
            if end_date:
                try:
                    ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    is_active = ed > now_utc
                except Exception:
                    is_active = True
            positions.append({
                "market_id":       mid,
                "question":        t.get("question", "")[:90],
                "side":            side,
                "entry_price":     round(entry_p, 4),
                "current_price":   round(current_p, 4),
                "price_change":    round(current_p - entry_p, 4),
                "size_usd":        round(size, 2),
                "shares":          round(shares, 4),
                "current_value":   round(current_value, 2),
                "pnl":             round(pnl, 2),
                "end_date":        end_date,
                "timestamp":       t.get("timestamp", ""),
                "confidence":      t.get("confidence", 0),
                "is_active":       is_active,
                "has_market_data": bool(mkt),
            })
        total_invested = sum(p["size_usd"] for p in positions)
        total_pnl      = sum(p["pnl"] for p in positions)
        return {
            "positions": positions,
            "summary": {
                "n_positions":        len(positions),
                "n_active":           sum(1 for p in positions if p["is_active"]),
                "total_invested":     round(total_invested, 2),
                "total_pnl":          round(total_pnl, 2),
                "win_count":          sum(1 for p in positions if p["pnl"] > 0),
                "total_current_value": round(sum(p["current_value"] for p in positions), 2),
            },
            "ai_analysis": _generate_monitoring_analysis(positions, total_invested, total_pnl),
        }
    except Exception as e:
        return {"positions": [], "summary": {}, "error": str(e),
                "ai_analysis": [f"Error al cargar datos: {str(e)[:100]}"]}


# ══════════════════════════════════════════════════════════════
# SPORTS ENDPOINTS (v5 - Free APIs only)
# ══════════════════════════════════════════════════════════════

def get_sports_client():
    global _sports_client
    if _sports_client is None:
        from sports_data import SportsDataClient
        _sports_client = SportsDataClient()
    return _sports_client


@app.get("/api/sports/markets")
async def get_sports_markets(
    limit: int = Query(50, ge=1, le=200),
    min_volume: float = Query(0.0, ge=0),
):
    """Mercados deportivos activos en Polymarket."""
    try:
        client = get_sports_client()
        markets = await client.get_polymarket_sports_markets(limit=limit)
        if min_volume > 0:
            markets = [m for m in markets if m["volume"] >= min_volume]
        return {
            "markets": markets,
            "total": len(markets),
            "source": "polymarket_gamma_api",
        }
    except Exception as e:
        raise HTTPException(500, f"Sports markets error: {e}")


_VALID_LEAGUES = {"nba", "nfl", "mlb", "nhl", "soccer_epl", "soccer_laliga", "soccer_champions", "ncaab", "ufc"}

@app.get("/api/sports/live-scores")
async def get_live_scores(league: Optional[str] = Query(None)):
    """Scores en vivo de ligas principales via ESPN API."""
    if league and league.lower() not in _VALID_LEAGUES:
        raise HTTPException(400, f"Liga no soportada. Valores válidos: {sorted(_VALID_LEAGUES)}")
    try:
        client = get_sports_client()
        if league:
            data = await client.get_espn_scoreboard(league.lower())
        else:
            data = await client.get_live_scores_all()
        return data
    except Exception as e:
        raise HTTPException(500, f"Live scores error: {e}")


@app.get("/api/sports/edge-opportunities")
async def get_sports_edge_opportunities(min_edge_pct: float = Query(3.0, ge=0)):
    """
    Detecta oportunidades de edge en mercados deportivos de Polymarket.
    Compara precios vs liquidez para detectar posible mispricing.
    NOTA: Edges reales sostenibles vs Pinnacle son 1-3%; cifras de 6-12%
    citadas en guías incluyen sesgo de selección y no son reproducibles.
    """
    try:
        client = get_sports_client()
        markets = await client.get_polymarket_sports_markets(limit=200)

        opportunities = []
        for m in markets:
            yes_price = m.get("yes_price", 0.5)
            no_price = m.get("no_price", 0.5)
            volume = m.get("volume", 0)
            liquidity = m.get("liquidity", 0)

            if liquidity < 100:
                continue
            if yes_price <= 0.02 or yes_price >= 0.98:
                continue

            in_sweet_spot = 0.15 <= yes_price <= 0.50 or 0.15 <= no_price <= 0.50
            in_comfort_trap = 0.62 <= yes_price <= 0.85

            if in_sweet_spot and volume > 1000:
                liquidity_ratio = liquidity / max(volume, 1)
                estimated_edge = max(0, (0.3 - liquidity_ratio) * 20)

                if estimated_edge >= min_edge_pct:
                    opportunities.append({
                        "market_id": m.get("id"),
                        "question": m.get("question"),
                        "slug": m.get("slug"),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume": volume,
                        "liquidity": liquidity,
                        "liquidity_ratio": round(liquidity_ratio, 4),
                        "estimated_edge_pct": round(estimated_edge, 2),
                        "price_zone": "sweet_spot" if in_sweet_spot else ("comfort_trap" if in_comfort_trap else "neutral"),
                        "tags": m.get("tags", []),
                        "end_date": m.get("end_date", ""),
                    })

        opportunities.sort(key=lambda x: x["estimated_edge_pct"], reverse=True)
        return {
            "opportunities": opportunities[:50],
            "total": len(opportunities),
            "min_edge_filter": min_edge_pct,
            "note": "Edge estimado heurístico — edges reales vs Pinnacle suelen ser 1-3%; las guías citan 6-12% que incluye sesgo de selección",
        }
    except Exception as e:
        raise HTTPException(500, f"Sports edge error: {e}")


# ═══════════ RUSH MODE ═══════════

def _rush_win_rate() -> float:
    """Porcentaje de acierto sobre apuestas resueltas."""
    total = _rush_wins + _rush_losses
    return round(_rush_wins / total * 100, 1) if total > 0 else 0.0


async def _rush_monitor_contrarian(engine):
    """
    Monitorea posiciones contrarias abiertas.
    - Averaging down: si precio cayó ≥10¢ desde entrada, RSI sigue válido y quedan ≥120s → compra adicional.
    - Salida parcial: ganancia ≥20% del precio de entrada → vende la mitad (~$1.20 por $1 invertido).
    Solo ejecuta si hay capital disponible y la posición no fue ya procesada.
    """
    global _rush_spent, _contrarian_positions
    from datetime import datetime as _dt, timezone as _tz
    _ts = lambda: _dt.now().strftime("%H:%M:%S")
    if not _contrarian_positions:
        return

    now_ts = _dt.now(_tz.utc).timestamp()

    for pos in list(_contrarian_positions):
        if pos.get("closed"):
            continue

        # ── Tiempo restante en la ventana ──────────────────────────────────
        seconds_left = (pos["window_ts"] + 300) - now_ts
        if seconds_left < 15:
            pos["closed"] = True  # ventana expirada
            continue

        # ── Precio actual del token en Polymarket ──────────────────────────
        try:
            mkt = await engine._fetch_market_by_slug(pos["slug"])
            if not mkt:
                continue
            prices_raw = mkt.get("outcomePrices", "")
            import json as _jmod
            prices = _jmod.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if not prices or len(prices) < 2:
                continue
            current_up   = max(0.01, min(0.99, float(prices[0])))
            current_down = max(0.01, min(0.99, float(prices[1])))
            current_price = current_up if pos["side"] == "UP" else current_down
        except Exception:
            continue

        entry_price = pos["entry_price"]
        price_delta = current_price - entry_price  # positivo = precio subió (bueno)

        # ── SALIDA PARCIAL: ganancia ≥20% del precio de entrada ────────────
        # Ej: entrada 35¢ → salir a 42¢ (+7¢ = $1.20 por $1); entrada 50¢ → salir a 60¢ (+10¢ = $1.20)
        exit_threshold = entry_price * 0.20
        if price_delta >= exit_threshold and not pos.get("partial_exit_done"):
            sell_size = round(pos["total_size"] * 0.50, 4)  # vender 50%
            if sell_size >= 1.0 and _rush_mode in ("live", "contrarian_live"):
                try:
                    result = trader.place_limit_order(
                        pos["token"],
                        round(current_price - 0.02, 4),  # limit ligeramente bajo el mercado
                        sell_size,
                        "SELL",
                        market_id=pos["market_id"],
                        question=pos.get("question", ""),
                    )
                    if result.status in ("FILLED", "DRY_RUN"):
                        sell_usd = round(sell_size * current_price, 2)
                        recovered = round(sell_usd - pos["total_usd"] * 0.5, 2)
                        pos["partial_exit_done"] = True
                        pos["total_size"] = round(pos["total_size"] - sell_size, 4)
                        await broadcast("rush_log", {
                            "line": (
                                f"[{_ts()}] 💰 SALIDA PARCIAL contraria — "
                                f"vendido {sell_size:.1f} tokens @ {current_price*100:.1f}¢ "
                                f"(+${recovered:.2f}) — resto jugando libre"
                            )
                        })
                except Exception as _e:
                    await broadcast("rush_log", {"line": f"[{_ts()}] ⚠ Error salida parcial: {_e}"})
            elif _rush_mode in ("demo", "contrarian"):
                pos["partial_exit_done"] = True
                recovered = round(pos["total_usd"] * 0.5 * (price_delta / entry_price), 2)
                await broadcast("rush_log", {
                    "line": (
                        f"[{_ts()}] [DEMO] 💰 SALIDA PARCIAL contraria "
                        f"@ {current_price*100:.1f}¢ (+${recovered:.2f} simulado)"
                    )
                })

        # ── AVERAGING DOWN: precio bajó ≥10¢ y quedan ≥120s ───────────────
        elif price_delta <= -0.10 and not pos.get("avg_down_done") and seconds_left >= 120:
            rsi = await engine.get_btc_rsi(period=14, interval="1m", limit=30)
            remaining_capital = _rush_max_capital - _rush_spent
            add_bet = min(remaining_capital, _rush_bet_usd, 2.0)

            # Confirmar que RSI sigue en zona de sobreextensión
            rsi_still_valid = (
                (pos["side"] == "DOWN" and rsi is not None and rsi >= 58) or
                (pos["side"] == "UP"   and rsi is not None and rsi <= 42)
            )

            if rsi_still_valid and remaining_capital >= 1.0 and add_bet >= 1.0:
                # Precio redondeado a 2 decimales (CLOB requirement)
                avg_limit = round(min(0.97, current_price + 0.02), 2)
                # Tokens ENTEROS para garantizar makerAmount exacto a 2 decimales
                import math as _math2
                _avg_cents  = round(avg_limit * 100)
                _add_cents  = round(add_bet * 100)
                add_tokens  = float(_math2.ceil(_add_cents / _avg_cents))
                if _rush_mode in ("live", "contrarian_live"):
                    try:
                        result = trader.place_limit_order(
                            pos["token"],
                            avg_limit,
                            add_tokens,
                            "BUY",
                            market_id=pos["market_id"],
                            question=pos.get("question", ""),
                        )
                        if result.status in ("FILLED", "DRY_RUN"):
                            # Recalcular precio promedio de entrada
                            prev_total_usd = pos["total_usd"]
                            pos["total_usd"]   += add_bet
                            pos["total_size"]  += add_tokens
                            pos["entry_price"]  = round(pos["total_usd"] / max(0.01, pos["total_size"]), 4)
                            pos["avg_down_done"] = True
                            _rush_spent += add_bet
                            await broadcast("rush_log", {
                                "line": (
                                    f"[{_ts()}] 📉 AVERAGING DOWN contraria — "
                                    f"+{add_tokens:.1f} tokens @ {avg_limit*100:.1f}¢ "
                                    f"(RSI:{rsi:.1f}) entrada prom:{pos['entry_price']*100:.1f}¢ "
                                    f"retorno estimado:{round(1/pos['entry_price'],2)}x"
                                )
                            })
                    except Exception as _e:
                        await broadcast("rush_log", {"line": f"[{_ts()}] ⚠ Error averaging down: {_e}"})
                elif _rush_mode in ("demo", "contrarian"):
                    prev_usd = pos["total_usd"]
                    pos["total_usd"]   += add_bet
                    pos["total_size"]  += add_tokens
                    pos["entry_price"]  = round(pos["total_usd"] / max(0.01, pos["total_size"]), 4)
                    pos["avg_down_done"] = True
                    _rush_spent += add_bet
                    await broadcast("rush_log", {
                        "line": (
                            f"[{_ts()}] [DEMO] 📉 AVERAGING DOWN contraria "
                            f"+${add_bet:.2f} @ {current_price*100:.1f}¢ "
                            f"(RSI:{rsi:.1f}) entrada prom:{pos['entry_price']*100:.1f}¢"
                        )
                    })


async def _rush_check_pending(engine):
    """Verifica apuestas pendientes y actualiza su resultado usando precio de cierre BTC."""
    global _rush_wins, _rush_losses, _rush_pnl, _rush_total_pnl, _rush_spent
    from datetime import datetime as _dt
    _ts = lambda: _dt.now().strftime("%H:%M:%S")
    for trade in _rush_history:
        if trade["status"] != "pending":
            continue
        result = await engine.check_btc_resolution(
            trade["window_ts"], trade["side"], trade["btc_open"]
        )
        if result is None:
            continue
        trade["status"] = result
        bet = trade["bet_usd"]
        if result == "won":
            _rush_wins += 1
            trade["pnl"] = round(bet * (trade["potential_return"] - 1), 3)
            # Reciclar capital: el bet apostado volvió + ganancia
            _rush_spent = max(0.0, _rush_spent - bet)
            _oid = trade.get("order_id", "")
            _oid_info = f" | order_id: {_oid[:18]}..." if _oid else ""
            await broadcast("rush_log", {
                "line": (
                    f"[{_ts()}] ✅ GANADO ${bet:.2f} → retorno ${bet * trade['potential_return']:.2f}"
                    f"{_oid_info} | Capital reciclado — ${_rush_spent:.2f} gastados de ${_rush_max_capital:.2f}. "
                    f"{'⚠ RECLAMA en polymarket.com → Portfolio' if _rush_mode in ('live', 'contrarian_live') else ''}"
                )
            })
        else:
            _rush_losses += 1
            trade["pnl"] = round(-bet, 3)
            await broadcast("rush_log", {"line": f"[{_ts()}] ❌ PERDIDO ${bet:.2f}"})
        _rush_pnl += trade["pnl"]
        _rush_total_pnl += trade["pnl"]
        # Persistir en historial multi-bot
        trade["mode"] = _rush_mode
        _record_trade("rush", trade)
        await broadcast("rush_result", {
            "id": trade["id"],
            "status": result,
            "pnl": trade["pnl"],
            "side": trade["side"],
            "wins": _rush_wins,
            "losses": _rush_losses,
            "session_pnl": round(_rush_pnl, 3),
            "total_pnl": round(_rush_total_pnl, 3),
            "win_rate": _rush_win_rate(),
        })


async def _rush_worker():
    """Worker del modo Rush: busca y apuesta en mercados BTC 5-min."""
    global _rush_running, _rush_spent, _rush_wins, _rush_losses, _rush_pnl
    global _rush_history, _rush_bet_counter
    global _rush_last_scan_msg, _rush_last_scan_ts
    from datetime import datetime as _dt
    ts = lambda: _dt.now().strftime("%H:%M:%S")

    engine = get_rush_engine()
    _last_bet_window_ts: int = 0  # evita apostar más de una vez por ventana
    _last_logged_msg: str = ""    # dedup: evita spam de mensajes repetidos
    _last_log_time: float = 0.0   # timestamp del último log

    def _fmt_s(s): return f"{s//60}:{s%60:02d}" if s >= 60 else f"{s}s"
    await broadcast("rush_log", {
        "line": f"[{ts()}] ⚡ Rush iniciado — modo {_rush_mode.upper()} — ${_rush_bet_usd}/op (máx ${_rush_max_capital}) — ventana {_fmt_s(_rush_window_min)}-{_fmt_s(_rush_window_max)}"
    })

    try:
        while _rush_running:
            if _rush_spent >= _rush_max_capital:
                await broadcast("rush_log", {"line": f"[{ts()}] 🛑 Capital máximo (${_rush_max_capital}) alcanzado. Deteniendo Rush."})
                break

            # Verificar apuestas pendientes de ventanas anteriores
            await _rush_check_pending(engine)

            # Monitorear posiciones contrarias abiertas (averaging-down / salida parcial)
            if _rush_mode in ("contrarian", "contrarian_live"):
                await _rush_monitor_contrarian(engine)

            # Buscar oportunidad en ventana actual
            if _rush_mode in ("contrarian", "contrarian_live"):
                opps, debug_msg = await engine.find_contrarian_opportunities(
                    min_seconds=_rush_window_min,
                    max_seconds=_rush_window_max,
                )
            else:
                opps, debug_msg = await engine.find_rush_opportunities(
                    min_seconds=_rush_window_min,
                    max_seconds=_rush_window_max,
                    optimize_timing=_rush_optimize_timing,
                    lax_mode=_rush_lax_mode,
                    btc_momentum_override=_rush_btc_momentum_override,
                    poly_high_override=_rush_poly_override,
                    poly_low_override=(1.0 - _rush_poly_override) if _rush_poly_override > 0 else 0.0,
                    zone_filter_disabled=_rush_zone_filter_disabled,
                )
            _rush_last_scan_msg = debug_msg
            _rush_last_scan_ts  = ts()
            btc_price = opps[0].btc_current if opps else await engine.get_btc_price()
            btc_str = f"${btc_price:,.0f}" if btc_price else "N/A"

            total_resolved = _rush_wins + _rush_losses
            wr_str = f" | Acierto: {_rush_win_rate():.0f}%" if total_resolved > 0 else ""

            await broadcast("rush_status", {
                "running": _rush_running,
                "mode": _rush_mode,
                "btc_price": btc_price,
                "spent": round(_rush_spent, 2),
                "max_capital": _rush_max_capital,
                "wins": _rush_wins,
                "losses": _rush_losses,
                "pnl": round(_rush_pnl, 3),
                "total_pnl": round(_rush_total_pnl, 3),
                "win_rate": _rush_win_rate(),
            })

            if not opps:
                import time as _time_mod
                _rush_status_msg = debug_msg or "Sin oportunidades en este ciclo"
                _now_float = _time_mod.time()
                # Dedup: comparar solo primeras 15 chars (captura tipo de msg sin incluir segundos/RSI)
                _msg_key = _rush_status_msg[:15]
                _msg_changed = _msg_key != _last_logged_msg
                _timeout_expired = (_now_float - _last_log_time) >= 30.0
                if _msg_changed or _timeout_expired:
                    await broadcast("rush_log", {
                        "line": f"[{ts()}] BTC {btc_str}{wr_str} — 🔍 {_rush_status_msg}"
                    })
                    _last_logged_msg = _msg_key
                    _last_log_time = _now_float
                # Si está esperando la ventana óptima ("confirmando"), reducir sleep a 1s
                # para no saltar por encima del momento de entrada (bug: 28s→sleep5s→23s→cerrado)
                _near_window = "confirmando" in debug_msg or "Señal débil" in debug_msg
                if _near_window:
                    if not _rush_running:
                        break
                    await asyncio.sleep(1)
                    continue
            else:
                best = opps[0]
                chg = f"+{best.btc_change_pct:.3f}%" if best.btc_change_pct >= 0 else f"{best.btc_change_pct:.3f}%"
                _is_contrarian_mode = _rush_mode in ("contrarian", "contrarian_live")
                if _is_contrarian_mode:
                    _signal_prefix = "🔄"
                    _signal_label  = f"CONTRARIO — UP:{best.up_price*100:.1f}¢ DOWN:{best.down_price*100:.1f}¢"
                else:
                    _signal_prefix = "⚡"
                    _signal_label  = (
                        f"Poly {best.up_price*100:.0f}% {'▲' if best.poly_signal=='UP' else '▼'}"
                        if best.poly_signal != "NEUTRAL"
                        else f"Poly neutral {best.up_price*100:.0f}% — MOMENTUM PURO"
                    )
                await broadcast("rush_log", {
                    "line": (
                        f"[{ts()}] {_signal_prefix} {best.seconds_remaining:.0f}s — "
                        f"BTC {chg} {'▲' if best.btc_signal=='UP' else '▼'} | "
                        f"{_signal_label} — "
                        f"SEÑAL: {best.side} @ {best.signal_price*100:.1f}% (fuerza {best.signal_strength:.2f})"
                    )
                })
                await broadcast("rush_opportunity", {
                    "market_id": best.market_id,
                    "question": best.question,
                    "side": best.side,
                    "signal_price": best.signal_price,
                    "signal_strength": best.signal_strength,
                    "seconds_remaining": best.seconds_remaining,
                    "potential_return": best.potential_return,
                    "btc_price": btc_price,
                    "btc_change_pct": best.btc_change_pct,
                    "poly_signal": best.poly_signal,
                    "btc_signal": best.btc_signal,
                    "btc_volume_ratio": getattr(best, "btc_volume_ratio", 1.0),
                    "signal_quality": getattr(best, "signal_quality", "normal"),
                })

                # ── UNA sola apuesta por ventana de 5 min ──────────
                if best.window_ts == _last_bet_window_ts:
                    await broadcast("rush_log", {
                        "line": f"[{ts()}] Ya aposté en esta ventana ({best.slug[-12:]}). Esperando siguiente."
                    })
                    for _ in range(RUSH_SCAN_INTERVAL):
                        if not _rush_running:
                            break
                        await asyncio.sleep(1)
                    continue

                # Registrar apuesta (live o demo)
                _rush_bet_counter += 1
                bet_record = {
                    "id": _rush_bet_counter,
                    "time": ts(),
                    "ts": _dt.now().isoformat(),
                    "side": best.side,
                    "signal_price": best.signal_price,
                    "potential_return": best.potential_return,
                    "seconds_at_bet": best.seconds_remaining,
                    "btc_open": best.btc_open,
                    "btc_entry": best.btc_current,
                    "btc_change_pct": best.btc_change_pct,
                    "window_ts": best.window_ts,
                    "slug": best.slug,
                    "mode": _rush_mode,
                    "status": "pending",
                    "pnl": 0.0,
                    "bet_usd": 0.0,
                    "question": best.question,
                    "asset": getattr(best, "asset", "BTC"),
                }

                if _rush_mode in ("live", "contrarian_live"):
                    remaining_capital = _rush_max_capital - _rush_spent
                    CLOB_MIN_USD = 1.0
                    _is_contra = _rush_mode == "contrarian_live"

                    if _is_contra:
                        # Contrarian — CLOB precision constraint:
                        # makerAmount = size_tokens × limit_price MUST be exactly 2 decimal places.
                        # Con decimales fraccionarios (ej: 1.9608 × 0.51 = 1.000008) el CLOB rechaza.
                        # Solución: usar tokens ENTEROS. N × 0.51 = exactamente N.NN siempre.
                        # Costo real puede diferir levemente del target (ej: 2 × 0.51 = $1.02).
                        import math as _math
                        limit_price     = round(min(0.97, best.signal_price + 0.01), 2)
                        _price_cents    = round(limit_price * 100)   # ej: 0.51 → 51
                        _target_cents   = round(min(_rush_bet_usd, remaining_capital) * 100)  # ej: 100
                        # Mínimo tokens enteros para cubrir el target (bet ≥ target)
                        size_tokens     = float(_math.ceil(_target_cents / _price_cents))  # ej: ceil(100/51)=2
                        bet             = round(size_tokens * limit_price, 2)              # ej: 2×0.51=1.02
                    else:
                        # Rush momentum: lógica original con Kelly dinámico y mínimo 5 tokens.
                        CLOB_MIN_TOKENS = 5.0
                        limit_price = round(min(0.97, best.signal_price + 0.03), 2)  # max 2 dec CLOB
                        _kelly_dynamic = round(best.kelly_fraction * remaining_capital, 2) if best.kelly_fraction > 0 else _rush_bet_usd
                        _dynamic_bet   = max(0.25, min(_kelly_dynamic, _rush_bet_usd, remaining_capital))
                        size_tokens = round(max(CLOB_MIN_TOKENS, _dynamic_bet / limit_price), 4)
                        bet         = round(size_tokens * limit_price, 2)  # 2 dec = maker amount USDC
                    if not _is_contra and bet > remaining_capital:
                        # Fallback momentum: intentar con mínimo en USD ($1).
                        _fb_tokens = round(max(1.0, CLOB_MIN_USD / limit_price), 4)
                        _fb_bet    = round(_fb_tokens * limit_price, 2)
                        if _fb_bet <= remaining_capital and _fb_bet >= CLOB_MIN_USD:
                            size_tokens = _fb_tokens
                            bet         = _fb_bet
                            await broadcast("rush_log", {"line": (
                                f"[{ts()}] ℹ Mínimo CLOB ajustado a ${_fb_bet:.2f} "
                                f"({_fb_tokens:.2f} tokens × {limit_price*100:.1f}¢)"
                            )})
                        else:
                            _last_bet_window_ts = best.window_ts
                            _precio_min_necesario = round(CLOB_MIN_TOKENS * limit_price, 2)
                            await broadcast("rush_log", {"line": (
                                f"[{ts()}] ⚠ Señal bloqueada: mínimo CLOB ${_precio_min_necesario:.2f} "
                                f"({CLOB_MIN_TOKENS:.0f} tokens × {limit_price*100:.1f}¢) > capital ${remaining_capital:.2f}. "
                                f"Aumenta max_capital a ≥${_precio_min_necesario:.2f} para operar en esta zona de precio."
                            )})
                    if bet <= remaining_capital:
                        try:
                            result = trader.place_limit_order(
                                best.token, limit_price, size_tokens, "BUY",
                                market_id=best.market_id,
                                question=best.question,
                                market_data={
                                    "yes_price": best.up_price,
                                    "no_price": best.down_price,
                                    "liquidity": getattr(best, 'liquidity_usd', 1000.0),
                                    "volume_24h": getattr(best, 'volume_24h', 500.0),
                                    "active": True,
                                    "closed": False,
                                    "question": getattr(best, 'question', ''),
                                    "market_id": getattr(best, 'market_id', ''),
                                },
                                fok=True,   # FOK: si no hay contraparte inmediata, se cancela
                                            # GTC causaba órdenes fantasma: accepted=True pero sin fill real
                            )
                            if result.status in ("FILLED", "DRY_RUN"):
                                _rush_spent += bet
                                _last_bet_window_ts = best.window_ts
                                bet_record["bet_usd"] = bet
                                bet_record["order_id"] = result.order_id  # para verificar en Polymarket
                                _rush_history.append(bet_record)
                                if len(_rush_history) > 50:
                                    _rush_history.pop(0)
                                # Registrar posición contraria para monitoring
                                if _rush_mode in ("contrarian", "contrarian_live"):
                                    _contrarian_positions.append({
                                        "id": bet_record["id"],
                                        "market_id": best.market_id,
                                        "slug": best.slug,
                                        "question": best.question,
                                        "side": best.side,
                                        "token": best.token,
                                        "entry_price": limit_price,
                                        "total_size": size_tokens,
                                        "total_usd": bet,
                                        "window_ts": best.window_ts,
                                        "avg_down_done": False,
                                        "partial_exit_done": False,
                                        "closed": False,
                                    })
                                await broadcast("rush_result", {
                                    "id": bet_record["id"], "status": "pending",
                                    "side": best.side, "pnl": 0,
                                    "wins": _rush_wins, "losses": _rush_losses,
                                    "session_pnl": round(_rush_pnl, 3),
                                    "total_pnl": round(_rush_total_pnl, 3),
                                    "win_rate": _rush_win_rate(),
                                    "history": _rush_history[-20:],
                                })
                                dry_tag = " [DRY RUN]" if result.status == "DRY_RUN" else ""
                                oid_tag = f" | order:{result.order_id[:10]}..." if result.order_id and not dry_tag else ""
                                await broadcast("rush_log", {
                                    "line": f"[{ts()}] ✅ COMPRA: {best.side} ${bet:.2f} @ {limit_price*100:.1f}¢ ({size_tokens:.1f} tokens) — retorno {round(1/limit_price,3)}x{dry_tag}{oid_tag}"
                                })
                            elif result.status == "FOK_CANCELLED":
                                # FOK sin fill = sin liquidez inmediata — reintentar en siguiente scan
                                # (no marcamos _last_bet_window_ts para permitir reintento si aparece liquidez)
                                await broadcast("rush_log", {"line": f"[{ts()}] ⚠ Sin liquidez: FOK cancelado a {limit_price*100:.1f}¢ — reintentando en {RUSH_SCAN_INTERVAL}s"})
                            else:
                                # Error de validación CLOB — no reintentar esta ventana
                                _last_bet_window_ts = best.window_ts
                                await broadcast("rush_log", {"line": f"[{ts()}] ❌ Error CLOB: {result.status} — {result.error_msg or 'unknown'}"})
                        except Exception as e:
                            # Excepción inesperada — bloquear ventana para no repetir el error
                            _last_bet_window_ts = best.window_ts
                            await broadcast("rush_log", {"line": f"[{ts()}] ❌ Excepción: {str(e)[:120]}"})
                else:
                    # DEMO: registrar simulado
                    _demo_remaining = _rush_max_capital - _rush_spent
                    if _rush_mode == "contrarian":
                        # Contraria demo: usar monto exacto configurado (igual que en live)
                        _demo_bet = min(_rush_bet_usd, _demo_remaining)
                    else:
                        # Rush demo: Kelly dinámico
                        _demo_kelly = round(best.kelly_fraction * _demo_remaining, 2) if best.kelly_fraction > 0 else _rush_bet_usd
                        _demo_bet   = max(0.25, min(_demo_kelly, _rush_bet_usd, _demo_remaining))
                    bet_record["bet_usd"] = _demo_bet
                    _rush_spent += _demo_bet
                    _last_bet_window_ts = best.window_ts  # no repetir en esta ventana
                    _rush_history.append(bet_record)
                    # Registrar posición contraria para monitoring (modo demo)
                    if _rush_mode in ("contrarian", "contrarian_live"):
                        _contrarian_positions.append({
                            "id": bet_record["id"],
                            "market_id": best.market_id,
                            "slug": best.slug,
                            "question": best.question,
                            "side": best.side,
                            "token": best.token,
                            "entry_price": best.signal_price,
                            "total_size": round(_demo_bet / max(0.01, best.signal_price), 4),
                            "total_usd": _demo_bet,
                            "window_ts": best.window_ts,
                            "avg_down_done": False,
                            "partial_exit_done": False,
                            "closed": False,
                        })
                    if len(_rush_history) > 50:
                        _rush_history.pop(0)
                    await broadcast("rush_result", {
                        "id": bet_record["id"], "status": "pending",
                        "side": best.side, "pnl": 0,
                        "wins": _rush_wins, "losses": _rush_losses,
                        "session_pnl": round(_rush_pnl, 3),
                        "total_pnl": round(_rush_total_pnl, 3),
                        "win_rate": _rush_win_rate(),
                        "history": _rush_history[-20:],
                    })
                    _demo_tag = "[CONTRARIO-DEMO]" if _rush_mode in ("contrarian", "contrarian_live") else "[DEMO]"
                    await broadcast("rush_log", {
                        "line": f"[{ts()}] {_demo_tag} COMPRA {best.side} [{best.asset}] ${_demo_bet:.2f} (Kelly {best.kelly_fraction*100:.0f}%) @ {best.signal_price*100:.1f}% — retorno {best.potential_return}x"
                    })

            # Esperar RUSH_SCAN_INTERVAL segundos antes del próximo scan
            for _ in range(RUSH_SCAN_INTERVAL):
                if not _rush_running:
                    break
                await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        await broadcast("rush_log", {"line": f"[{ts()}] Error en Rush: {e}"})
        logger.exception("Rush worker error")
    finally:
        _rush_running = False
        await broadcast("rush_log", {"line": f"[{ts()}] ⚡ Rush detenido."})
        await broadcast("rush_status", {
            "running": False, "mode": _rush_mode,
            "spent": round(_rush_spent, 2), "pnl": round(_rush_pnl, 3),
            "total_pnl": round(_rush_total_pnl, 3),
            "wins": _rush_wins, "losses": _rush_losses,
            "win_rate": _rush_win_rate(),
        })


@app.post("/api/rush/start")
async def rush_start(request: Request):
    global _rush_running, _rush_mode, _rush_bet_usd, _rush_max_capital
    global _rush_spent, _rush_wins, _rush_losses, _rush_pnl, _rush_task
    global _rush_window_min, _rush_window_max, _rush_optimize_timing
    global _rush_lax_mode, _rush_btc_momentum_override, _rush_poly_override, _rush_zone_filter_disabled
    if _rush_running:
        return {"status": "already_running"}
    body = await request.json()
    _rush_mode = body.get("mode", "demo")
    _requested_bet = float(body.get("bet_usd", 1.0))
    _rush_max_capital = float(body.get("max_capital", 3.0))
    _rush_window_min = int(body.get("min_window", _rush_window_min))
    _rush_window_max = int(body.get("max_window", _rush_window_max))
    _rush_optimize_timing = bool(body.get("optimize_risk", True))
    # Lax mode params
    _rush_lax_mode = bool(body.get("lax_mode", False))
    # Slider BTC muestra "(value/100)%" — value=6 → "0.06%" → decimal=0.0006
    # Por eso dividimos por 10000 (= /100 del slider, /100 de % a decimal)
    _rush_btc_momentum_override = float(body.get("btc_momentum_pct", 0.0)) / 10000.0
    # Slider Poly muestra "value%" directamente — value=52 → "52%" → decimal=0.52
    _rush_poly_override = float(body.get("poly_signal_pct", 0.0)) / 100.0            # pct → decimal
    _rush_zone_filter_disabled = bool(body.get("zone_filter_disabled", False))

    # Sin guard — el usuario ya configuró el bet_usd conscientemente
    if _requested_bet > 10.0:
        print(f"[Rush] Advertencia: bet_usd=${_requested_bet:.2f} es alto. Continuando según configuración del usuario.")
    _rush_bet_usd = _requested_bet

    # Modo contrario: entra al INICIO del mercado (recién abierto, 210-290s restantes)
    # Acepta mode="contrarian" (demo) o mode="contrarian_live" (real)
    if _rush_mode in ("contrarian", "contrarian_live"):
        _rush_window_min = int(body.get("min_window", 60))
        _rush_window_max = int(body.get("max_window", 285))
        _contrarian_positions.clear()

    # dry_run: DEMO/CONTRARIAN=simulación, LIVE=operaciones reales
    _is_live = _rush_mode in ("live", "contrarian_live")
    trader.dry_run = not _is_live
    if _is_live and not trader.is_authenticated:
        return {"status": "error", "message": "Modo Live requiere Private Key configurada. Ve a la sección Operar → API Keys."}
    # Reset session stats (limpiar historial previo para evitar que apuestas
    # pending de sesiones anteriores contaminen el P&L de la nueva sesión)
    _rush_spent = 0.0
    _rush_wins = 0
    _rush_losses = 0
    _rush_pnl = 0.0
    _rush_history.clear()
    _rush_running = True
    # Sincronizar capital con el módulo rush_mode
    try:
        import backend.rush_mode as _rm_module
        _rm_module._rush_max_capital = float(_rush_max_capital)
    except Exception:
        try:
            import rush_mode as _rm_module
            _rm_module._rush_max_capital = float(_rush_max_capital)
        except Exception as _rme:
            print(f"[Rush] Advertencia: no se pudo sincronizar _rush_max_capital con rush_mode: {_rme}")
    _rush_task = asyncio.create_task(_rush_worker())
    return {"status": "started", "mode": _rush_mode, "bet_usd": _rush_bet_usd,
            "max_capital": _rush_max_capital,
            "window": {"min": _rush_window_min, "max": _rush_window_max},
            "lax_mode": _rush_lax_mode,
            "btc_momentum_override": _rush_btc_momentum_override,
            "poly_override": _rush_poly_override,
            "zone_filter_disabled": _rush_zone_filter_disabled}


@app.post("/api/rush/window")
async def rush_set_window(request: Request):
    """Actualiza la ventana de entrada sin reiniciar el bot."""
    global _rush_window_min, _rush_window_max, _rush_optimize_timing
    body = await request.json()
    new_min = int(body.get("min_window", _rush_window_min))
    new_max = int(body.get("max_window", _rush_window_max))
    if new_min < 5 or new_max > 300 or new_min >= new_max:
        return {"ok": False, "error": "Valores inválidos (min≥5, max≤300, min<max)"}
    _rush_window_min = new_min
    _rush_window_max = new_max
    if "optimize_risk" in body:
        _rush_optimize_timing = bool(body["optimize_risk"])
    return {"ok": True, "min_window": _rush_window_min, "max_window": _rush_window_max,
            "optimize_timing": _rush_optimize_timing}


@app.post("/api/rush/stop")
async def rush_stop():
    global _rush_running, _rush_task
    _rush_running = False
    if _rush_task and not _rush_task.done():
        _rush_task.cancel()
    return {"status": "stopped"}


@app.get("/api/rush/status")
async def rush_status():
    engine = get_rush_engine()
    btc_price = await engine.get_btc_price()
    return {
        "running": _rush_running,
        "mode": _rush_mode,
        "bet_usd": _rush_bet_usd,
        "max_capital": _rush_max_capital,
        "spent": round(_rush_spent, 2),
        "remaining": round(max(0, _rush_max_capital - _rush_spent), 2),
        "wins": _rush_wins,
        "losses": _rush_losses,
        "pnl": round(_rush_pnl, 3),
        "total_pnl": round(_rush_total_pnl, 3),
        "win_rate": _rush_win_rate(),
        "btc_price": btc_price,
        "thresholds": {"high": RUSH_MARKET_HIGH, "low": RUSH_MARKET_LOW, "btc_momentum": RUSH_BTC_MOMENTUM, "btc_strong_bypass": RUSH_BTC_STRONG},
        "window_seconds": {"min": _rush_window_min, "max": _rush_window_max},
        "optimize_risk": _rush_optimize_timing,
    }


@app.get("/api/rush/history")
async def rush_history_endpoint():
    return {"history": _rush_history[-20:]}


@app.get("/api/history")
async def get_full_history(
    bot:    str   = Query("all"),
    status: str   = Query("all"),
    mode:   str   = Query("all"),
    limit:  int   = Query(100),
    offset: int   = Query(0),
):
    """
    Historial persistente multi-bot: Rush + D&H + ER.
    Filtros: bot (rush|dh|er|all), status (won|lost|pending|all), mode (live|demo|all).
    El historial persiste entre sesiones en data/trade_history.json.
    """
    trades = _persistent_history[:]

    # merge session rush_history trades that aren't yet persisted
    # Usar (bot, id) como clave compuesta para evitar colisiones de IDs int entre sesiones
    persisted_keys = {(t.get("bot",""), t["id"]) for t in trades}
    for t in _rush_history:
        if t.get("id") and ("rush", t["id"]) not in persisted_keys:
            from datetime import datetime as _dt
            trades.append({
                "id":     t.get("id", ""),
                "bot":    "rush",
                "ts":     t.get("ts", ""),
                "market": t.get("question", ""),
                "side":   t.get("side", ""),
                "bet_usd": t.get("bet_usd", 0.0),
                "pnl":    t.get("pnl", 0.0),
                "status": t.get("status", "pending"),
                "mode":   t.get("mode", "demo"),
                "asset":  t.get("asset", "BTC"),
                "price":  t.get("signal_price", 0.0),
            })

    if bot != "all":
        trades = [t for t in trades if t.get("bot") == bot]
    if status != "all":
        trades = [t for t in trades if t.get("status") == status]
    if mode != "all":
        trades = [t for t in trades if t.get("mode") == mode]

    trades_sorted = sorted(trades, key=lambda x: x.get("ts", ""), reverse=True)
    total = len(trades_sorted)
    page  = trades_sorted[offset: offset + limit]

    won   = sum(1 for t in trades_sorted if t.get("status") == "won")
    lost  = sum(1 for t in trades_sorted if t.get("status") == "lost")
    total_pnl = round(sum(t.get("pnl", 0) for t in trades_sorted), 3)

    return {
        "trades":    page,
        "total":     total,
        "won":       won,
        "lost":      lost,
        "win_rate":  round(won / max(won + lost, 1) * 100, 1),
        "total_pnl": total_pnl,
        "offset":    offset,
        "limit":     limit,
    }


@app.delete("/api/history")
async def clear_history(bot: str = Query("all")):
    """Limpia el historial persistente (bot=all|rush|dh|er)."""
    global _persistent_history
    if bot == "all":
        _persistent_history = []
    else:
        _persistent_history = [t for t in _persistent_history if t.get("bot") != bot]
    _save_persistent_history()
    return {"cleared": True, "bot": bot}


@app.get("/api/diagnostics/scan")
async def get_scan_diagnostics():
    """Retorna los últimos resultados de scan de todos los bots para diagnóstico."""
    return {
        "rush": {
            "last_msg": _rush_last_scan_msg,
            "last_ts":  _rush_last_scan_ts,
            "lax_mode": _rush_lax_mode,
            "btc_momentum_override": _rush_btc_momentum_override,
            "poly_override": _rush_poly_override,
        },
        "dh": {
            "last_msg": _dh_last_scan_msg,
            "last_ts":  _dh_last_scan_ts,
            "lax_mode": _dh_lax_mode,
            "combined_override": _dh_combined_override,
        },
        "er": {
            "last_msg": _er_last_scan_msg,
            "last_ts":  _er_last_scan_ts,
            "require_dual_ai": _er_require_dual_ai,
            "score_min_override": _er_score_min_override,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# DUMP & HEDGE ENDPOINTS
# Motor: dump_hedge.py | WS feed: dump_hedge_ws.py
# Desarrollado por: Carlos David Donoso Cordero (ddchack)
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/dump-hedge/status")
async def dh_status():
    """Estado actual del motor D&H + precios Binance (WS si activo, REST si no)."""
    status = _dh_engine.get_status()
    ws_prices = _dh_binance.prices
    if ws_prices:
        status["binance_prices"] = ws_prices
    else:
        try:
            rest_prices = await asyncio.wait_for(
                _dh_engine.get_binance_prices_rest(), timeout=4.0
            )
            status["binance_prices"] = rest_prices
        except Exception:
            status["binance_prices"] = {}
    return status


@app.get("/api/dump-hedge/prices")
async def dh_prices():
    """
    Precios completos para el monitor del dashboard:
    - binance: precio actual de BTC/ETH/SOL/XRP (WS o REST)
    - poly: precio UP/DOWN de cada mercado 5m (outcomePrices de Gamma, sin CLOB)
    """
    import json as _json

    # ── Binance ──────────────────────────────────────────────────────
    ws_prices = _dh_binance.prices
    if ws_prices:
        binance = ws_prices
    else:
        try:
            binance = await asyncio.wait_for(
                _dh_engine.get_binance_prices_rest(), timeout=4.0
            )
        except Exception:
            binance = {}

    # ── Polymarket UP/DOWN — directo de outcomePrices (rápido, sin CLOB) ──
    poly_prices: dict = {}
    try:
        markets = await asyncio.wait_for(_dh_engine._fetch_markets(), timeout=10.0)
        for m in markets:
            asset = m.get("_asset", "")
            tf    = m.get("_tf", "")
            if not asset or tf != "5m":
                continue
            try:
                raw = m.get("outcomePrices", "[]")
                prices = _json.loads(raw) if isinstance(raw, str) else (raw or [])
                if len(prices) >= 2:
                    up_p = float(prices[0])
                    dn_p = float(prices[1])
                    poly_prices[asset] = {
                        "up":       round(up_p, 4),
                        "down":     round(dn_p, 4),
                        "combined": round(up_p + dn_p, 4),
                        "question": (m.get("question") or "")[:70],
                    }
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"[D&H prices] poly fetch error: {e}")

    return {
        "binance":  binance,
        "poly":     poly_prices,
        "ts":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/dump-hedge/opportunities")
async def dh_opportunities():
    """Oportunidades D&H detectadas en el último escaneo (caché)."""
    cached = _dh_engine.get_opportunities()
    # Si no hay caché, hacer un escaneo rápido
    if not cached:
        try:
            opps = await asyncio.wait_for(_dh_engine.scan_opportunities(), timeout=12.0)
            cached = [_dh_engine._opp_to_dict(o) for o in opps]
        except asyncio.TimeoutError:
            pass
    return {
        "opportunities": cached,
        "count": len(cached),
        "config": {
            "dump_threshold": _dh_engine.config["dump_threshold"],
            "max_combined_price": _dh_engine.config["max_combined_price"],
        },
    }


@app.post("/api/dump-hedge/start")
async def dh_start(request: Request):
    """Inicia el bot D&H en modo automático."""
    global _dh_lax_mode, _dh_combined_override
    body = await request.json()
    mode = body.get("mode", "demo")
    _dh_engine.config["dry_run"] = (mode != "live")
    if mode == "live" and not trader.is_authenticated:
        raise HTTPException(status_code=400, detail="API Keys no configuradas para modo live")

    # Actualizar config con parámetros opcionales del body
    for key in ("dump_threshold", "max_combined_price", "bet_per_leg",
                "max_pyramid_layers", "leg2_wait_timeout", "use_maker_orders",
                "max_pyramid_layers"):
        if key in body:
            _dh_engine.config[key] = body[key]

    # Lax mode params
    _dh_lax_mode = bool(body.get("lax_mode", False))
    _dh_combined_override = float(body.get("lax_max_combined", 0.0))
    if _dh_lax_mode and _dh_combined_override <= 0:
        _dh_combined_override = 0.97  # default sensible para lax mode

    # Iniciar feed Binance si no está corriendo
    if not _dh_binance._running:
        _dh_binance.start()

    # Wrapper de broadcast que intercepta dh_hedge_complete para historial persistente
    async def _dh_broadcast_wrapper(event: str, data: dict):
        if event == "dh_hedge_complete":
            hedge = data.get("hedge", {})
            _record_trade("dh", {
                "id": hedge.get("id", f"dh_{len(_persistent_history)+1}"),
                "question": hedge.get("market", {}).get("question", "D&H Hedge"),
                "side": "HEDGE",
                "bet_usd": float(hedge.get("total_invested", hedge.get("leg1_amount", 0))),
                "pnl": float(data.get("session_pnl", 0)),
                "status": "won" if float(data.get("session_pnl", 0)) > 0 else "lost",
                "mode": "demo" if _dh_engine.config.get("dry_run", True) else "live",
                "price": float(hedge.get("entry_combined_price", 0)),
                "asset": hedge.get("asset", "BTC"),
            })
        await broadcast(event, data)

    ok = _dh_engine.start(broadcast_fn=_dh_broadcast_wrapper)
    if not ok:
        return {"status": "already_running", "dry_run": _dh_engine.config["dry_run"]}
    return {"status": "started", "mode": mode, "dry_run": _dh_engine.config["dry_run"],
            "config": _dh_engine.config,
            "lax_mode": _dh_lax_mode,
            "lax_max_combined": _dh_combined_override}


@app.post("/api/dump-hedge/stop")
async def dh_stop():
    """Detiene el bot D&H."""
    _dh_engine.stop()
    return {"status": "stopped"}


@app.post("/api/dump-hedge/config")
async def dh_config(request: Request):
    """Actualiza la configuración del motor D&H en caliente."""
    body = await request.json()
    allowed = {
        "dump_threshold", "max_combined_price", "bet_per_leg",
        "max_legs_per_window", "max_pyramid_layers", "leg2_wait_timeout",
        "stop_loss_on_orphan", "use_maker_orders", "scan_interval",
        "assets", "timeframes",
    }
    updated = {}
    for k, v in body.items():
        if k in allowed:
            _dh_engine.config[k] = v
            updated[k] = v
    # Limpiar caché de mercados si cambiaron activos o timeframes
    if "assets" in updated or "timeframes" in updated:
        _dh_engine._markets_cache = []
        _dh_engine._markets_cache_ts = 0.0
    return {"ok": True, "updated": updated, "config": _dh_engine.config}


@app.get("/api/dump-hedge/history")
async def dh_history(limit: int = Query(50)):
    """Historial de hedges completados (hasta 50 más recientes)."""
    return {"history": _dh_engine.get_history(limit), "total": len(_dh_engine._completed_hedges)}


@app.get("/api/dump-hedge/stats")
async def dh_stats():
    """Estadísticas de la sesión D&H."""
    stats = _dh_engine.get_stats()
    stats["binance_prices"] = _dh_binance.prices
    return stats


@app.post("/api/dump-hedge/execute-manual")
async def dh_execute_manual(request: Request):
    """
    Ejecuta una pierna manualmente (debug / operación puntual).
    Body: {leg: "1"|"2", asset, side, price, token_id, amount_usd, leg1_id?}
    """
    body = await request.json()
    leg = str(body.get("leg", "1"))

    if leg == "1":
        # Buscar la oportunidad activa del activo
        asset = body.get("asset", "btc")
        opps  = [o for o in _dh_engine._opportunities if o.asset == asset]
        if not opps:
            # Construir oportunidad sintética desde los datos del body
            from dump_hedge import DHOpportunity
            opp = DHOpportunity(
                asset=asset,
                timeframe=body.get("timeframe", "5m"),
                market_id=body.get("market_id", ""),
                question=body.get("question", "Manual"),
                up_token=body.get("up_token", ""),
                down_token=body.get("down_token", ""),
                up_price=float(body.get("up_price", 0.1)),
                down_price=float(body.get("down_price", 0.1)),
                combined_price=float(body.get("up_price", 0.1)) + float(body.get("down_price", 0.1)),
                dump_side=body.get("side", "UP"),
                dump_price=float(body.get("price", 0.1)),
                potential_profit_usd=0,
                potential_roi_pct=0,
                net_profit_after_fees=0,
            )
        else:
            opp = opps[0]
        result_leg = await _dh_engine.execute_leg1(opp)
        if result_leg:
            return {"ok": True, "leg": _dh_engine._leg_to_dict(result_leg)}
        raise HTTPException(status_code=500, detail="No se pudo ejecutar Leg1")

    elif leg == "2":
        leg1_id  = body.get("leg1_id")
        leg1_obj = next((l for l in _dh_engine._active_legs if l.leg_id == leg1_id), None)
        if not leg1_obj:
            raise HTTPException(status_code=404, detail="Leg1 no encontrada en piernas activas")
        price2  = float(body.get("price", 0.1))
        token2  = body.get("token_id", "")
        result_leg2 = await _dh_engine.execute_leg2(leg1_obj, price2, token2)
        if result_leg2:
            hedge = _dh_engine._completed_hedges[-1] if _dh_engine._completed_hedges else None
            return {
                "ok": True,
                "leg2": _dh_engine._leg_to_dict(result_leg2),
                "hedge": _dh_engine._hedge_to_dict(hedge) if hedge else None,
            }
        raise HTTPException(status_code=500, detail="No se pudo ejecutar Leg2")

    raise HTTPException(status_code=400, detail="leg debe ser '1' o '2'")


@app.post("/api/dump-hedge/calculate")
async def dh_calculate(request: Request):
    """
    Calculadora de ganancia garantizada.
    Body: {leg1_price, leg1_amount, leg2_price, leg2_amount?, use_maker?}
    """
    body = await request.json()
    p1  = float(body.get("leg1_price", 0.1))
    a1  = float(body.get("leg1_amount", 1.0))
    p2  = float(body.get("leg2_price", 0.1))
    # Si no viene leg2_amount, usar dutching óptimo
    combined = p1 + p2
    total    = a1 * 2  # asumiendo bet_per_leg = a1/2 * 2 = a1
    a2 = float(body.get("leg2_amount", total * p2 / combined if combined > 0 else a1))
    orig_maker = _dh_engine.config.get("use_maker_orders", True)
    if "use_maker" in body:
        _dh_engine.config["use_maker_orders"] = bool(body["use_maker"])
    result = _dh_engine.calculate_guaranteed_profit(p1, a1, p2, a2)
    _dh_engine.config["use_maker_orders"] = orig_maker
    return result


@app.post("/api/dump-hedge/resume")
async def dh_resume():
    """Reanuda el bot si está pausado por circuit breaker."""
    _dh_engine.resume()
    return {"ok": True, "is_paused": _dh_engine._is_paused}


@app.get("/api/dump-hedge/scan")
async def dh_scan():
    """Fuerza un nuevo escaneo de mercados y retorna oportunidades."""
    _dh_engine._markets_cache = []  # Invalidar caché
    try:
        opps = await asyncio.wait_for(_dh_engine.scan_opportunities(), timeout=15.0)
        await broadcast("dh_opportunities", [_dh_engine._opp_to_dict(o) for o in opps[:8]])
        return {
            "opportunities": [_dh_engine._opp_to_dict(o) for o in opps],
            "count": len(opps),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
    except asyncio.TimeoutError:
        return {"opportunities": [], "count": 0, "error": "timeout"}


@app.post("/api/tune/weights")
async def tune_ensemble_weights(num_markets: int = Query(50, ge=20, le=200),
                                 auto_apply: bool = Query(False)):
    """
    Ejecuta un mini-backtest (N mercados resueltos) y sugiere pesos de ensemble.
    Si auto_apply=true, aplica los pesos sugeridos directamente.
    óptimos por categoría basados en el rendimiento real de cada señal.

    Retorna los pesos actuales y los sugeridos. NO aplica los cambios automáticamente.
    """
    eng = MathEngine(max_capital=MAX_CAPITAL, kelly_multiplier=KELLY_FRACTION)
    bt  = BacktestEngine(math_engine=eng, initial_capital=MAX_CAPITAL)
    try:
        r = await asyncio.wait_for(bt.run_backtest("tune", num_markets), timeout=30.0)
        await bt.close()
    except asyncio.TimeoutError:
        return {"error": "timeout en backtest (Gamma API lenta)"}
    except Exception as e:
        return {"error": str(e)}

    cat_m = r.category_metrics

    # ── Alimentar calibración con resultados del backtest ──
    _cal_obs = 0
    for t in r.trades:
        try:
            _outcome = 1 if t.get("resolved_outcome") == "YES" else 0
            _pred = t.get("estimated_prob", 0.5)
            cal_suite.add_observation(float(_pred), int(_outcome))
            _cal_obs += 1
        except Exception:
            pass
    if _cal_obs >= 20:
        cal_suite.fit_all()

    # Pesos actuales hardcodeados en api_server
    _current_weights = {
        "politics":  [0.35, "cat*0.45", 0.20],
        "sports":    [0.20, "cat*0.60", 0.20],
        "crypto":    [0.40, "cat*0.30", 0.30],
        "economics": [0.40, "cat*0.35", 0.25],
        "default":   [0.45, "cat*0.35", 0.20],
    }

    # Sugerir ajustes: categorías con win_rate > 60% → subir peso cat_signal
    # categorías con win_rate < 45% → bajar peso cat_signal, subir bayesian
    suggestions = {}
    for cat, m in cat_m.items():
        wr   = m.get("win_rate", 50)
        roi  = m.get("roi_pct", 0)
        n    = m.get("trades", 0)
        note = ""
        if n < 5:
            note = f"pocos trades ({n}) — no ajustar"
            suggestions[cat] = {"note": note, **m}
            continue
        if wr >= 60 and roi > 0:
            note = f"BUENA categoría (WR={wr:.0f}%, ROI={roi:.1f}%) — subir peso cat_signal +0.05"
        elif wr <= 45 or roi < -3:
            note = f"MAL rendimiento (WR={wr:.0f}%, ROI={roi:.1f}%) — reducir peso cat_signal -0.05"
        else:
            note = f"rendimiento normal (WR={wr:.0f}%, ROI={roi:.1f}%) — mantener pesos"
        suggestions[cat] = {"note": note, **m}

    # ── Auto-aplicar si se solicitó ──────────────────────────────
    auto_applied = {}
    if auto_apply:
        from datetime import date as _date
        _today = _date.today().isoformat()
        for cat, m in cat_m.items():
            wr  = m.get("win_rate", 50)
            roi = m.get("roi_pct", 0)
            n   = m.get("trades", 0)
            if n < 5:
                continue
            base = _ENSEMBLE_WEIGHTS.get(cat, _ENSEMBLE_WEIGHTS["default"])[:]
            if wr >= 60 and roi > 0:
                base[1] = round(min(0.90, base[1] + 0.05), 3)  # subir cat_signal
            elif wr <= 45 or roi < -3:
                base[1] = round(max(0.10, base[1] - 0.05), 3)  # bajar cat_signal
                base[0] = round(min(0.65, base[0] + 0.03), 3)  # subir bayesian
            _ENSEMBLE_WEIGHTS[cat] = base
            auto_applied[cat] = base
        if auto_applied:
            _ENSEMBLE_WEIGHTS_SOURCE = f"backtest_{_today}"
            # Persistir pesos actualizados por backtest
            try:
                _weights_path = os.path.join(os.path.dirname(__file__), "ensemble_weights.json")
                with open(_weights_path, "w", encoding="utf-8") as _wf:
                    json.dump({
                        "weights": _ENSEMBLE_WEIGHTS,
                        "source": _ENSEMBLE_WEIGHTS_SOURCE,
                        "saved_at": datetime.now(timezone.utc).isoformat()
                    }, _wf, ensure_ascii=False, indent=2)
                print(f"[Weights] Pesos backtest guardados en {_weights_path}")
            except Exception as _we:
                print(f"[Weights] Error guardando pesos backtest: {_we}")

    return {
        "backtest_markets": r.total_trades,
        "overall_win_rate": r.win_rate,
        "overall_return_pct": r.total_return_pct,
        "category_metrics": cat_m,
        "suggestions": suggestions,
        "current_weights": _ENSEMBLE_WEIGHTS,
        "auto_applied": auto_applied,
        "weights_source": _ENSEMBLE_WEIGHTS_SOURCE,
        "note": "Usa auto_apply=true para aplicar cambios en runtime, o POST /api/tune/apply con pesos manuales.",
        "brier_score": r.brier_score,
    }


# ai_health duplicado eliminado — usar GET /api/ai/health definido arriba (línea ~1244)


if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8888")))
