"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Motor de IA Alimentado por Algoritmos
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Las IAs NO analizan en vacío. Antes de emitir criterio reciben:
- Probabilidad Bayesiana posterior (con señales descompuestas)
- KL Divergence (métrica de rentabilidad esperada)
- Kelly Criterion (tamaño óptimo de apuesta)
- Sentimiento de noticias (score + artículos)
- Posiciones de whales (consenso de traders grandes)
- Modelo de categoría (ELO/Poisson/Oracle Lag/Polls/Forecast)
- Detección de régimen (volatilidad del mercado)
- Datos de calibración (Brier Score histórico)
- Arbitraje detectado (si YES+NO < $1)
- Sharpe Ratio binario

La IA evalúa estos datos matemáticos y LUEGO emite su propia estimación.
"""

import asyncio
import httpx
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from swarm_engine import SwarmEngine, SwarmResult

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysis:
    provider: str
    model: str
    estimated_prob: float
    confidence: float
    reasoning: str
    agrees_with_algorithms: bool
    algorithm_assessment: str
    factors_yes: list
    factors_no: list
    recommendation: str
    suggested_bet_pct: float
    risk_notes: str
    latency_ms: int
    cost_estimate: float


# Fix Anthropic 400 — Carlos David Donoso Cordero (ddchack)
# Modelos válidos a 2025-04: claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-opus-4-6
ANTHROPIC_MODELS = {
    "claude-sonnet-4-6": {"name": "Claude Sonnet 4.6", "cost_per_1k": 0.003, "speed": "rápido"},
    "claude-haiku-4-5-20251001": {"name": "Claude Haiku 4.5", "cost_per_1k": 0.001, "speed": "ultra-rápido"},
    "claude-opus-4-6": {"name": "Claude Opus 4.6", "cost_per_1k": 0.015, "speed": "profundo"},
}

OPENAI_MODELS = {
    "gpt-4o": {"name": "GPT-4o", "cost_per_1k": 0.005, "speed": "rápido"},
    "gpt-4o-mini": {"name": "GPT-4o Mini", "cost_per_1k": 0.0002, "speed": "ultra-rápido"},
    "gpt-4-turbo": {"name": "GPT-4 Turbo", "cost_per_1k": 0.01, "speed": "profundo"},
    "o3-mini": {"name": "o3-mini", "cost_per_1k": 0.001, "speed": "razonamiento"},
}


# ═══════════════════════════════════════════════════════════
# PROMPT CON DATOS ALGORÍTMICOS COMPLETOS
# ═══════════════════════════════════════════════════════════

ALGORITHM_ENRICHED_PROMPT = """Eres un analista experto en mercados de predicción (Polymarket). Tu trabajo es evaluar una oportunidad de apuesta usando tanto tu conocimiento del mundo como los DATOS MATEMÁTICOS Y ESTADÍSTICOS que nuestros algoritmos ya calcularon.

IMPORTANTE — ANTI-ANCHORING: No ancles tu estimación al precio de mercado. El precio de mercado puede estar sesgado por liquidez baja, manipulación o narrativas populares. Forma tu propia estimación independiente primero, luego compara.

═══ MERCADO ═══
Pregunta: {question}
Categoría: {category}
Fecha de resolución: {end_date}

═══ DATOS DE MERCADO ═══
Volumen 24h: ${volume}
Liquidez: ${liquidity}
Spread: {spread}%
[NOTA METODOLÓGICA: Los precios de mercado se omiten del análisis primario para
prevenir anchoring bias. Tu estimación debe basarse en los datos algorítmicos
y tu conocimiento del mundo, NO en el consenso del mercado. Los precios de
mercado se revelan en el paso de challenge/validación posterior.]

═══ ANÁLISIS BAYESIANO (nuestro modelo) ═══
Probabilidad posterior Bayesiana: {bayesian_prob}%
Prior (antes de señales): {prior}%
Incertidumbre: ±{uncertainty}%
Señales procesadas:
{signals_detail}

═══ MÉTRICAS MATEMÁTICAS ═══
KL Divergence: {kl_divergence} (>0.05 = oportunidad rentable, >0.10 = alta)
Edge vs mercado: {edge_pct}%
Valor Esperado (EV): {ev} por cada $1 apostado
Kelly Criterion completo: {kelly_full}% del bankroll
Kelly conservador (25%): {kelly_quarter}% → ${kelly_bet_usd} recomendado
Sharpe Ratio binario: {binary_sharpe} (>0.2 = bueno, >0.5 = excelente)

═══ MODELO DE CATEGORÍA ({category}) ═══
Modelo usado: {category_model}
Probabilidad del modelo: {category_prob}%
Confianza del modelo: {category_confidence}%
Sensibilidad temporal: {time_sensitivity}
{category_reasoning}

═══ SENTIMIENTO DE NOTICIAS ═══
Score de sentimiento: {sentiment_score} (-1=negativo, +1=positivo)
Artículos analizados: {sentiment_articles}
Tendencia: {sentiment_label}

═══ CONSENSO DE WHALES (traders grandes) ═══
{whale_detail}

═══ DETECCIÓN DE RÉGIMEN ═══
Régimen actual: {regime}
Multiplicador de volatilidad: {vol_multiplier}x (1.0=normal, <1=reducir posiciones)

═══ CALIBRACIÓN DEL SISTEMA ═══
Brier Score: {brier_score} (0=perfecto, 0.25=aleatorio)
Predicciones registradas: {n_predictions}
Calibración Platt: {platt_status}

═══ ARBITRAJE ═══
YES + NO = {arb_combined}¢ (si <100¢ hay ganancia garantizada)
Ganancia por arbitraje: {arb_profit}¢

═══ TU TAREA ═══
1. REVISA todos los datos algorítmicos de arriba
2. EVALÚA si los algoritmos están acertados o si hay factores que no capturaron
3. USA tu conocimiento del mundo real para complementar los datos
4. Si los datos muestran edge positivo Y tú estás de acuerdo, refuerza la señal
5. Si ves algo que los algoritmos no capturaron (cambio reciente, noticia, contexto), AJUSTA
6. Sé específico sobre POR QUÉ ajustas o confirmas la estimación algorítmica

Responde ÚNICAMENTE en este formato JSON exacto (sin markdown, sin backticks):
{{
  "probabilidad_estimada": 0.XX,
  "confianza": 0.XX,
  "coincide_con_algoritmos": true/false,
  "evaluacion_algoritmos": "Los algoritmos [aciertan/fallan] porque...",
  "factores_si": ["factor1", "factor2", "factor3"],
  "factores_no": ["factor1", "factor2"],
  "razonamiento": "Mi análisis completo considerando los datos algorítmicos y mi conocimiento...",
  "recomendacion": "COMPRAR_SI" o "COMPRAR_NO" o "NO_APOSTAR",
  "porcentaje_apuesta_sugerido": X.X,
  "notas_riesgo": "Riesgos principales a considerar..."
}}"""


class LLMEngine:
    """
    Motor de IA v6.0 — SwarmEngine (50 personas) como método principal.
    Ensemble de 3 modelos como fallback si swarm tarda >8s.
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    def __init__(self):
        self.anthropic_key: str = ""
        self.openai_key: str = ""
        self.anthropic_model: str = "claude-sonnet-4-6"  # Fix Anthropic 400 — Carlos David Donoso Cordero (ddchack)
        self.openai_model: str = "gpt-4o-mini"
        self.mode: str = "none"
        self.client = httpx.AsyncClient(timeout=90.0)
        self.swarm: SwarmEngine = SwarmEngine()
        self._swarm_timeout: float = 8.0  # segundos; si excede → fallback ensemble

    def configure(self, anthropic_key="", openai_key="", anthropic_model="",
                   openai_model="", mode=""):
        if anthropic_key: self.anthropic_key = anthropic_key
        if openai_key: self.openai_key = openai_key
        if anthropic_model and anthropic_model in ANTHROPIC_MODELS:
            self.anthropic_model = anthropic_model
        if openai_model and openai_model in OPENAI_MODELS:
            self.openai_model = openai_model
        if mode: self.mode = mode
        # Propagar keys al swarm
        self.swarm.configure(
            anthropic_key=self.anthropic_key,
            openai_key=self.openai_key,
            anthropic_model="claude-haiku-4-5-20251001",  # Haiku para velocidad en swarm
            openai_model="gpt-4o-mini",
        )

    def get_config(self) -> dict:
        return {
            "anthropic_configured": bool(self.anthropic_key),
            "openai_configured": bool(self.openai_key),
            "anthropic_model": self.anthropic_model,
            "openai_model": self.openai_model,
            "mode": self.mode,
            "anthropic_models": ANTHROPIC_MODELS,
            "openai_models": OPENAI_MODELS,
        }

    # ─────────────────────────────────────────────────────
    # BUILD PROMPT WITH ALL ALGORITHM DATA
    # ─────────────────────────────────────────────────────

    def _build_enriched_prompt(self, question: str, algo_data: dict) -> str:
        """
        Construye el prompt inyectando TODOS los datos algorítmicos.
        algo_data viene del pipeline completo del scanner v3.
        """
        # Señales Bayesianas detalladas
        signals = algo_data.get("signals", [])
        signals_text = ""
        if signals:
            for s in signals:
                signals_text += f"  - {s.get('name','?')}: valor={s.get('value',0)}, confianza={s.get('confidence',0)}\n"
        else:
            signals_text = "  (sin señales externas disponibles)\n"

        # Whale detail
        whale_data = algo_data.get("whale_positions", [])
        if whale_data:
            yes_total = sum(w.get("size_usd", 0) for w in whale_data if w.get("side") == "YES")
            no_total = sum(w.get("size_usd", 0) for w in whale_data if w.get("side") == "NO")
            whale_text = f"Whales apostando SÍ: ${yes_total:,.0f} | Whales apostando NO: ${no_total:,.0f}\n"
            whale_text += f"Total whales monitoreados: {len(whale_data)}"
        else:
            whale_text = "Sin datos de whales disponibles"

        # Category model reasoning
        cat_reasoning = algo_data.get("category_reasoning", "")
        if not cat_reasoning:
            model_name = algo_data.get("category_model", "default")
            cat_reasoning = f"Modelo {model_name} usado para esta categoría"

        # Arbitrage check
        yes_p = algo_data.get("yes_price", 0.5)
        no_p = algo_data.get("no_price", 0.5)
        arb_combined = round((yes_p + no_p) * 100, 1)
        arb_profit = round(max(0, 100 - arb_combined), 1)

        base_prompt = ALGORITHM_ENRICHED_PROMPT.format(
            question=question,
            category=algo_data.get("category", "otro"),
            end_date=algo_data.get("end_date", "No especificada"),
            volume=f"{algo_data.get('volume_24h', 0):,.0f}",
            liquidity=f"{algo_data.get('liquidity', 0):,.0f}",
            spread=round(algo_data.get("spread", 0) * 100, 2),
            bayesian_prob=round(algo_data.get("bayesian_prob", 0.5) * 100, 1),
            prior=round(algo_data.get("prior", 0.5) * 100, 1),
            uncertainty=round(algo_data.get("uncertainty", 0) * 100, 1),
            signals_detail=signals_text,
            kl_divergence=round(algo_data.get("kl_divergence", 0), 4),
            edge_pct=round(algo_data.get("edge_pct", 0), 1),
            ev=round(algo_data.get("ev", 0), 4),
            kelly_full=round(algo_data.get("kelly_full", 0) * 100, 1),
            kelly_quarter=round(algo_data.get("kelly_quarter", 0) * 100, 1),
            kelly_bet_usd=round(algo_data.get("kelly_bet_usd", 0), 2),
            binary_sharpe=round(algo_data.get("binary_sharpe", 0), 3),
            category_model=algo_data.get("category_model", "default"),
            category_prob=round(algo_data.get("category_prob", 0.5) * 100, 1),
            category_confidence=round(algo_data.get("category_confidence", 0) * 100, 1),
            time_sensitivity=algo_data.get("time_sensitivity", "MEDIUM"),
            category_reasoning=cat_reasoning,
            sentiment_score=round(algo_data.get("sentiment_score", 0), 3),
            sentiment_articles=algo_data.get("sentiment_articles", 0),
            sentiment_label=algo_data.get("sentiment_label", "NEUTRAL"),
            whale_detail=whale_text,
            regime=algo_data.get("regime", "DESCONOCIDO"),
            vol_multiplier=algo_data.get("vol_multiplier", 1.0),
            brier_score=algo_data.get("brier_score", "N/A"),
            n_predictions=algo_data.get("n_predictions", 0),
            platt_status=algo_data.get("platt_status", "No calibrado aún"),
            arb_combined=arb_combined,
            arb_profit=arb_profit,
        )
        return base_prompt

    def _build_second_pass_challenge(self, question: str, market_price: float,
                                      llm_prob: float) -> str:
        """
        Second-pass challenge: solo se activa si la divergencia LLM vs mercado > 15%.
        Muestra el precio real al LLM y le pide reconsiderar.
        """
        divergence = abs(llm_prob - market_price)
        if divergence <= 0.15:
            return ""
        direction = "mayor" if llm_prob > market_price else "menor"
        return (
            f"\n\n═══ CHALLENGE — SEGUNDA PASADA ═══\n"
            f"Tu estimación ({llm_prob*100:.1f}%) diverge {divergence*100:.1f}pp del precio de mercado "
            f"({market_price*100:.1f}%). Tu estimación es {direction} que el mercado.\n"
            f"Pregunta: {question}\n"
            f"¿Hay alguna razón sólida para esta divergencia? Si no la hay, ajusta tu estimación "
            f"hacia el precio de mercado. Responde con el mismo formato JSON, actualizando "
            f"'probabilidad_estimada' si reconsideras."
        )

    def _build_challenge_prompt(self, question: str, ensemble_prob: float,
                                 market_price: float, category: str) -> str:
        """
        Second-pass challenge prompt: reveal market price only when ensemble
        diverges significantly (> 15pp) from market. Anti-central-tendency bias.
        """
        divergence = abs(ensemble_prob - market_price)
        if divergence < 0.15:
            return ""  # No challenge needed, ensemble close to market

        direction = "SUPERIOR" if ensemble_prob > market_price else "INFERIOR"
        return f"""VALIDACIÓN SECUNDARIA — {question}

Nuestros algoritmos estiman una probabilidad de {ensemble_prob*100:.1f}% para YES.
El mercado (Polymarket) actualmente cotiza a {market_price*100:.1f}¢.
La divergencia es de {divergence*100:.1f}pp — nuestros modelos están {divergence*100:.1f}% {direction} al mercado.

Esta divergencia grande ({divergence*100:.1f}pp) requiere validación.
Categoría: {category}

¿Esta divergencia tiene sentido? ¿Los modelos capturan algo que el mercado no refleja,
o el mercado tiene información que nuestros algoritmos no tienen?

Responde SOLO en JSON:
{{"validacion": "CONFIRMAR" o "REDUCIR" o "RECHAZAR",
  "razon": "...",
  "prob_ajustada": 0.XX}}"""

    # ─────────────────────────────────────────────────────
    # API CALLS
    # ─────────────────────────────────────────────────────

    async def _call_anthropic(self, prompt: str) -> tuple[str, int]:
        start = time.time()
        resp = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.anthropic_key, "anthropic-version": "2023-06-01",
                      "content-type": "application/json"},
            json={"model": self.anthropic_model, "max_tokens": 1500,
                  "messages": [{"role": "user", "content": prompt}]})
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"] if data.get("content") else ""
        return text, int((time.time() - start) * 1000)

    async def _call_openai(self, prompt: str) -> tuple[str, int]:
        start = time.time()
        resp = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_key}",
                      "Content-Type": "application/json"},
            json={"model": self.openai_model, "max_tokens": 1500,
                  "messages": [
                      {"role": "system", "content": "Eres un analista experto en mercados de predicción. Recibirás datos matemáticos y estadísticos de nuestros algoritmos. SIEMPRE considera estos datos antes de emitir tu criterio. Responde solo en JSON válido."},
                      {"role": "user", "content": prompt}],
                  "temperature": 0.3})
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
        return text, int((time.time() - start) * 1000)

    # ─────────────────────────────────────────────────────
    # PARSE RESPONSE
    # ─────────────────────────────────────────────────────

    def _parse(self, text: str, provider: str, model: str, ms: int) -> LLMAnalysis:
        try:
            clean = text.strip()
            # Extraer JSON de bloques markdown (```json ... ``` o ``` ... ```)
            _md = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if _md:
                clean = _md.group(1).strip()
            # Buscar el primer objeto JSON si hay texto alrededor
            if not clean.startswith("{"):
                _obj = re.search(r"\{[\s\S]*\}", clean)
                if _obj:
                    clean = _obj.group(0)
            d = json.loads(clean)
            cost = ANTHROPIC_MODELS.get(model, {}).get("cost_per_1k", 0.003) if provider == "anthropic" else OPENAI_MODELS.get(model, {}).get("cost_per_1k", 0.003)
            return LLMAnalysis(
                provider=provider, model=model,
                estimated_prob=max(0.02, min(0.98, float(d.get("probabilidad_estimada", 0.5)))),
                confidence=max(0, min(1, float(d.get("confianza", 0.5)))),
                reasoning=d.get("razonamiento", ""),
                agrees_with_algorithms=d.get("coincide_con_algoritmos", True),
                algorithm_assessment=d.get("evaluacion_algoritmos", ""),
                factors_yes=d.get("factores_si", []),
                factors_no=d.get("factores_no", []),
                recommendation=d.get("recomendacion", "NO_APOSTAR"),
                suggested_bet_pct=float(d.get("porcentaje_apuesta_sugerido", 0)),
                risk_notes=d.get("notas_riesgo", ""),
                latency_ms=ms, cost_estimate=round(cost * 3, 4))
        except Exception as e:
            return LLMAnalysis(
                provider=provider, model=model,
                estimated_prob=0.5, confidence=0.1,
                reasoning=f"Error: {str(e)[:100]}",
                agrees_with_algorithms=False,
                algorithm_assessment="No se pudo parsear la respuesta",
                factors_yes=[], factors_no=[],
                recommendation="NO_APOSTAR",
                suggested_bet_pct=0, risk_notes="Error de parseo",
                latency_ms=ms, cost_estimate=0)

    # ─────────────────────────────────────────────────────
    # MAIN: ANALYZE WITH FULL ALGORITHM CONTEXT
    # ─────────────────────────────────────────────────────

    async def analyze_with_algorithms(self, question: str, algo_data: dict) -> dict:
        """
        Análisis alimentado por algoritmos.
        algo_data contiene TODOS los outputs previos del pipeline v3.
        """
        prompt = self._build_enriched_prompt(question, algo_data)
        results = {}

        if self.mode in ("anthropic", "ensemble") and self.anthropic_key:
            try:
                text, ms = await self._call_anthropic(prompt)
                results["anthropic"] = self._parse(text, "anthropic", self.anthropic_model, ms)
            except Exception as e:
                results["anthropic_error"] = str(e)[:200]

        if self.mode in ("openai", "ensemble") and self.openai_key:
            try:
                text, ms = await self._call_openai(prompt)
                results["openai"] = self._parse(text, "openai", self.openai_model, ms)
            except Exception as e:
                results["openai_error"] = str(e)[:200]

        # Ensemble
        ensemble = None
        if "anthropic" in results and "openai" in results:
            a, o = results["anthropic"], results["openai"]
            w_a = a.confidence / max(0.01, a.confidence + o.confidence)
            ens_prob = a.estimated_prob * w_a + o.estimated_prob * (1 - w_a)
            ensemble = {
                "probabilidad_ensemble": round(ens_prob, 4),
                "confianza_combinada": round((a.confidence + o.confidence) / 2, 3),
                "prob_claude": a.estimated_prob,
                "prob_gpt": o.estimated_prob,
                "acuerdo": round(1.0 - abs(a.estimated_prob - o.estimated_prob), 3),
                "ambas_coinciden_algoritmos": a.agrees_with_algorithms and o.agrees_with_algorithms,
            }

        return {
            "question": question,
            "algorithm_data_fed": {
                "bayesian_prob": algo_data.get("bayesian_prob"),
                "kl_divergence": algo_data.get("kl_divergence"),
                "edge_pct": algo_data.get("edge_pct"),
                "category_model": algo_data.get("category_model"),
                "sentiment_score": algo_data.get("sentiment_score"),
                "n_signals": len(algo_data.get("signals", [])),
            },
            "anthropic": self._to_dict(results.get("anthropic")),
            "openai": self._to_dict(results.get("openai")),
            "ensemble": ensemble,
            "errors": {k: v for k, v in results.items() if k.endswith("_error")},
            "mode": self.mode,
        }

    # Backward compatible simple version
    async def analyze_market(self, question: str, category: str = "other",
                              yes_price: float = 0.5, no_price: float = 0.5,
                              volume: float = 0, liquidity: float = 0,
                              end_date: str = "") -> dict:
        """Simple analysis (sin datos algorítmicos previos)."""
        basic_data = {
            "category": category, "yes_price": yes_price, "no_price": no_price,
            "volume_24h": volume, "liquidity": liquidity, "end_date": end_date,
            "bayesian_prob": yes_price, "prior": yes_price, "uncertainty": 0.1,
            "kl_divergence": 0, "edge_pct": 0, "ev": 0,
            "kelly_full": 0, "kelly_quarter": 0, "kelly_bet_usd": 0,
            "binary_sharpe": 0, "category_model": "ninguno",
            "category_prob": yes_price, "category_confidence": 0,
            "time_sensitivity": "MEDIUM", "sentiment_score": 0,
            "sentiment_articles": 0, "sentiment_label": "NEUTRAL",
            "regime": "DESCONOCIDO", "vol_multiplier": 1.0,
            "brier_score": "N/A", "n_predictions": 0,
            "platt_status": "No calibrado", "signals": [], "spread": 0,
        }
        return await self.analyze_with_algorithms(question, basic_data)

    @staticmethod
    def _to_dict(a) -> dict:
        if not a: return None
        return {
            "provider": a.provider, "model": a.model,
            "probabilidad": a.estimated_prob,
            "confianza": a.confidence,
            "razonamiento": a.reasoning,
            "coincide_con_algoritmos": a.agrees_with_algorithms,
            "evaluacion_algoritmos": a.algorithm_assessment,
            "factores_si": a.factors_yes[:5],
            "factores_no": a.factors_no[:5],
            "recomendacion": a.recommendation,
            "apuesta_sugerida_pct": a.suggested_bet_pct,
            "notas_riesgo": a.risk_notes,
            "latencia_ms": a.latency_ms,
            "costo_usd": a.cost_estimate,
        }

    # ─────────────────────────────────────────────────────
    # v6.0 — SWARM-FIRST ANALYSIS
    # ─────────────────────────────────────────────────────

    async def analyze_with_swarm(self, question: str, algo_data: dict) -> dict:
        """
        Método principal v6.0: intenta el swarm de 50 personas.
        Si tarda más de _swarm_timeout segundos → cae al ensemble clásico de 3 modelos.
        """
        swarm_result = None
        used_swarm = False
        swarm_error = None

        # Solo intentar swarm si hay API key configurada
        if self.anthropic_key or self.openai_key:
            try:
                swarm_result = await asyncio.wait_for(
                    self.swarm.analyze(question, algo_data, timeout_seconds=self._swarm_timeout),
                    timeout=self._swarm_timeout + 1.0,
                )
                used_swarm = True
            except asyncio.TimeoutError:
                swarm_error = "SwarmEngine timeout — falling back to ensemble"
                logger.warning(swarm_error)
            except Exception as e:
                swarm_error = f"SwarmEngine error: {str(e)[:150]}"
                logger.error(swarm_error)

        # Siempre correr el ensemble clásico (puede ser en paralelo, pero para no doblar costos,
        # solo se corre si el swarm falló o como segunda opinión si dissent_ratio > 0.6)
        ensemble_result = None
        run_fallback = (not used_swarm) or (
            swarm_result is not None and swarm_result.dissent_ratio > 0.6
        )
        if run_fallback:
            try:
                ensemble_result = await self.analyze_with_algorithms(question, algo_data)
            except Exception as e:
                logger.error(f"Fallback ensemble error: {e}")

        # Construir respuesta combinada
        response: dict = {
            "question": question,
            "version": "v6.0-swarm",
            "used_swarm": used_swarm,
            "swarm_error": swarm_error,
        }

        if swarm_result is not None:
            response["swarm"] = self.swarm.swarm_result_to_dict(swarm_result)
            # Usar la probabilidad del swarm como estimación principal
            response["probabilidad_final"] = swarm_result.probability
            response["kelly_fraction"] = swarm_result.kelly_fraction
            response["dissent_ratio"] = swarm_result.dissent_ratio
            response["n_personas"] = swarm_result.n_personas_used

        if ensemble_result is not None:
            response["ensemble_fallback"] = ensemble_result

        # Si no hay swarm ni ensemble, devolver basic
        if swarm_result is None and ensemble_result is None:
            response["probabilidad_final"] = float(algo_data.get("yes_price", 0.5))
            response["swarm"] = None

        return response

    async def close(self):
        await self.client.aclose()
        await self.swarm.close()
