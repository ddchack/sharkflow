"""
SharkFlow Escape Room Engine — Alto Riesgo, Alto Retorno
=========================================================
Filtra mercados con precio ≤ $0.33 (retorno potencial 3x o más).
Usa análisis dual-IA (Claude + OpenAI) para detectar edge en
mercados asimétricos con baja probabilidad de mercado implícita.

ZONA DE RIESGO: estos mercados tienen mayor probabilidad de perder
pero el payoff asimétrico justifica posiciones pequeñas cuando
ambas IAs coinciden en un edge real.

Retorno potencial = 1 / precio_entrada
- precio 0.33 → 3x
- precio 0.20 → 5x
- precio 0.10 → 10x
- precio 0.05 → 20x
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Zona de precio objetivo ──
ER_MAX_ENTRY_PRICE = 0.33   # 3x mínimo
ER_MIN_ENTRY_PRICE = 0.05   # 20x máximo (muy especulativo, requiere confianza alta)
ER_MIN_LIQUIDITY   = 500.0  # Liquidez mínima (menor que el bot normal)
ER_KELLY_FRACTION  = 0.10   # Kelly conservador para alto riesgo
ER_MAX_BET_PCT     = 0.05   # Máximo 5% del capital por trade

# ── Multiplicador mínimo por categoría ───────────────────────────────────────
# Crypto/sports son más predecibles → aceptamos retornos más bajos (3x).
# Geopolitics es menos predecible → requerimos mayor retorno para justificar riesgo.
ER_MIN_MULTIPLIER_BY_CATEGORY: dict[str, float] = {
    "crypto":      3.0,   # Predecible por correlación con precio spot
    "sports":      3.5,   # Moderadamente predecible con datos ESPN
    "economics":   4.0,   # Depende de datos macro; más incierto
    "tech":        4.0,   # Noticias tech tienen alta varianza
    "politics":    5.0,   # Muy difícil de predecir con precisión
    "geopolitics": 6.0,   # El menos predecible — solo vale si el edge es muy claro
    "other":       4.0,   # Default conservador
}


@dataclass
class EROpportunity:
    """Una oportunidad de Escape Room analizada."""
    market_id: str
    question: str
    slug: str
    category: str
    side: str                # "YES" o "NO"
    entry_price: float       # precio de entrada (≤ 0.33)
    multiplier: float        # 1/entry_price = retorno potencial
    ensemble_prob: float     # probabilidad modelo algorítmico
    edge_pct: float          # edge calculado
    kl_divergence: float
    yes_token: str
    no_token: str
    # Dual AI
    claude_prob: float = 0.0
    claude_confidence: float = 0.0
    claude_reasoning: str = ""
    claude_recommendation: str = "NO_APOSTAR"
    gpt_prob: float = 0.0
    gpt_confidence: float = 0.0
    gpt_reasoning: str = ""
    gpt_recommendation: str = "NO_APOSTAR"
    ai_consensus: float = 0.0      # 0-1, qué tan de acuerdo están las IAs
    ai_combined_prob: float = 0.0  # probabilidad combinada de ambas IAs
    # Sports context
    sports_context: str = ""       # texto con datos del partido si es deportivo
    # Traducción
    pregunta_es: str = ""          # pregunta traducida al español
    # Scoring final
    er_score: float = 0.0          # puntuación final Escape Room (0-100)
    bet_usd: float = 0.0
    cat_min_multiplier: float = 3.0  # Multiplicador mínimo exigido para esta categoría


class EscapeRoomEngine:
    """
    Motor del módulo Escape Room.

    Filtra mercados de full_scan() para quedarse solo con los que
    tienen precio de entrada ≤ $0.33, y aplica análisis dual-IA.
    """

    def __init__(self, llm_engine, sports_client=None):
        self.llm = llm_engine
        self.sports = sports_client

    def filter_opportunities(self, scan_results: list, max_price: Optional[float] = None) -> list[EROpportunity]:
        """
        Filtra recomendaciones del scan principal para el Escape Room.
        Solo pasan mercados con precio de entrada ≤ max_price (default: ER_MAX_ENTRY_PRICE=0.33).
        Pasar max_price evita mutar el módulo global (thread-safe).
        """
        effective_max = max_price if max_price is not None else ER_MAX_ENTRY_PRICE
        opportunities = []
        for r in scan_results:
            side = r.get("side", "YES")
            entry_price = r.get("yes_price", 0.5) if side == "YES" else r.get("no_price", 0.5)

            # Filtro principal: precio ≤ effective_max → retorno ≥ 3x
            if not (ER_MIN_ENTRY_PRICE <= entry_price <= effective_max):
                continue

            # Liquidez mínima
            if r.get("liquidity", 0) < ER_MIN_LIQUIDITY:
                continue

            # Edge positivo requerido
            if r.get("edge_pct", 0) <= 0:
                continue

            multiplier = round(1.0 / max(0.01, entry_price), 2)

            # Multiplicador mínimo por categoría
            cat = r.get("category", "other")
            cat_min_mult = ER_MIN_MULTIPLIER_BY_CATEGORY.get(cat, ER_MIN_MULTIPLIER_BY_CATEGORY["other"])
            if multiplier < cat_min_mult:
                continue

            opp = EROpportunity(
                market_id=r.get("market_id", ""),
                question=r.get("question", ""),
                slug=r.get("slug", ""),
                category=cat,
                side=side,
                entry_price=round(entry_price, 4),
                multiplier=multiplier,
                ensemble_prob=r.get("ensemble_prob", 0.5),
                edge_pct=r.get("edge_pct", 0),
                kl_divergence=r.get("kl_divergence", 0),
                yes_token=r.get("yes_token", ""),
                no_token=r.get("no_token", ""),
                cat_min_multiplier=cat_min_mult,
            )
            opportunities.append(opp)

        # Ordenar por KL divergence (mejor edge primero)
        opportunities.sort(key=lambda x: x.kl_divergence, reverse=True)
        return opportunities[:20]  # Top 20

    async def enrich_with_sports_context(self, opp: EROpportunity) -> str:
        """
        Si el mercado es deportivo, obtiene contexto en vivo via ESPN API.
        Retorna string con el contexto o vacío si no es deportivo.
        """
        if opp.category not in ("sports", "esports") or not self.sports:
            return ""

        context_parts = []
        question_lower = opp.question.lower()

        # Detectar liga por keywords en la pregunta
        league_map = {
            "nba": ["nba", "lakers", "celtics", "warriors", "bulls", "heat", "knicks"],
            "nfl": ["nfl", "super bowl", "touchdown", "patriots", "chiefs", "cowboys"],
            "mlb": ["mlb", "baseball", "yankees", "dodgers", "world series"],
            "nhl": ["nhl", "hockey", "stanley cup"],
            "soccer_epl": ["premier league", "epl", "arsenal", "chelsea", "manchester"],
            "soccer_laliga": ["la liga", "real madrid", "barcelona", "atletico"],
        }

        detected_league = None
        for league, keywords in league_map.items():
            if any(kw in question_lower for kw in keywords):
                detected_league = league
                break

        if detected_league:
            try:
                scores = await self.sports.get_espn_scoreboard(detected_league)
                games = scores.get("games", [])
                if games:
                    live_games = [g for g in games if g.get("is_live")]
                    recent_final = [g for g in games if g.get("is_final")]

                    if live_games:
                        g = live_games[0]
                        teams = g.get("teams", [])
                        if len(teams) >= 2:
                            context_parts.append(
                                f"EN VIVO ({detected_league.upper()}): "
                                f"{teams[0]['name']} {teams[0]['score']} - "
                                f"{teams[1]['score']} {teams[1]['name']} "
                                f"({g.get('status_short', '')})"
                            )
                    elif recent_final:
                        g = recent_final[0]
                        teams = g.get("teams", [])
                        if len(teams) >= 2:
                            winner = next((t for t in teams if t.get("winner")), None)
                            context_parts.append(
                                f"ÚLTIMO RESULTADO ({detected_league.upper()}): "
                                f"{teams[0]['name']} {teams[0]['score']} - "
                                f"{teams[1]['score']} {teams[1]['name']}"
                                + (f" | Ganador: {winner['name']}" if winner else "")
                            )
            except Exception as e:
                logger.warning(f"Sports context error for {opp.question[:30]}: {e}")

        return " | ".join(context_parts)

    async def analyze_with_dual_ai(self, opp: EROpportunity,
                                    sports_context: str = "") -> EROpportunity:
        """
        Analiza una oportunidad con AMBAS IAs (Claude + OpenAI) en paralelo.
        Usa un prompt especializado para mercados de alto riesgo/retorno.
        """
        # Construir prompt especializado para Escape Room
        prompt = self._build_er_prompt(opp, sports_context)

        claude_result = None
        gpt_result = None

        # Llamar ambas IAs en paralelo
        tasks = []
        if self.llm.anthropic_key:
            tasks.append(("claude", self.llm._call_anthropic(prompt)))
        if self.llm.openai_key:
            tasks.append(("gpt", self.llm._call_openai(prompt)))

        if not tasks:
            opp.claude_reasoning = "Sin API keys — configúralas en Control"
            opp.gpt_reasoning = "Sin API keys — configúralas en Control"
            opp.claude_recommendation = "NO_APOSTAR"
            opp.gpt_recommendation = "NO_APOSTAR"
            # Fallback: score mínimo basado en edge algorítmico (sin IA, máx 30)
            opp.er_score = round(min(30, max(0, opp.edge_pct * 0.5)), 1)
            return opp

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        _pregunta_es_found = False
        for (provider, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"ER {provider} error: {result}")
                continue
            text, ms = result

            # Extraer traducción al español del JSON crudo (solo la primera que llegue)
            if not _pregunta_es_found:
                try:
                    _clean = text.strip()
                    if _clean.startswith("```"):
                        _clean = _clean.split("\n", 1)[1].rsplit("```", 1)[0]
                    _tr = json.loads(_clean).get("pregunta_en_espanol", "")
                    if _tr:
                        opp.pregunta_es = _tr
                        _pregunta_es_found = True
                except Exception:
                    pass

            parsed = self.llm._parse(text, provider,
                                      self.llm.anthropic_model if provider == "claude" else self.llm.openai_model,
                                      ms)
            if provider == "claude":
                claude_result = parsed
                opp.claude_prob = parsed.estimated_prob
                opp.claude_confidence = parsed.confidence
                opp.claude_reasoning = parsed.reasoning[:300] if parsed.reasoning else ""
                opp.claude_recommendation = parsed.recommendation
            else:
                gpt_result = parsed
                opp.gpt_prob = parsed.estimated_prob
                opp.gpt_confidence = parsed.confidence
                opp.gpt_reasoning = parsed.reasoning[:300] if parsed.reasoning else ""
                opp.gpt_recommendation = parsed.recommendation

        # Calcular consenso y probabilidad combinada
        if claude_result and gpt_result:
            # BUG 4 fix: consenso basado en acuerdo DIRECCIONAL, no solo distancia numérica.
            # Antes: Claude=0.3 y GPT=0.7 producían consensus=0.6 aunque estén en desacuerdo.
            # Ahora: si ambas IAs recomiendan lo mismo → consenso alto; si divergen → bajo.
            _claude_rec = opp.claude_recommendation or ""
            _gpt_rec    = opp.gpt_recommendation or ""
            _positive_words = {"APOSTAR", "BET", "YES", "STRONG", "APOSTAR_FUERTE"}
            claude_positive = any(w in _claude_rec.upper() for w in _positive_words)
            gpt_positive    = any(w in _gpt_rec.upper() for w in _positive_words)
            _prob_dist = min(1.0, abs(claude_result.estimated_prob - gpt_result.estimated_prob))
            if claude_positive == gpt_positive:
                # Mismo sentido: consenso alto, modulado por cercanía de probabilidades
                direction_consensus = 0.8 + 0.2 * (1.0 - _prob_dist)
            else:
                # Sentidos opuestos: consenso bajo, penalizado por distancia
                direction_consensus = max(0.0, 0.2 * (1.0 - _prob_dist))
            opp.ai_consensus = round(direction_consensus, 3)
            w_c = claude_result.confidence / max(0.01, claude_result.confidence + gpt_result.confidence)
            opp.ai_combined_prob = round(
                claude_result.estimated_prob * w_c + gpt_result.estimated_prob * (1 - w_c), 4)
        elif claude_result:
            opp.ai_combined_prob = claude_result.estimated_prob
            opp.ai_consensus = 0.5
        elif gpt_result:
            opp.ai_combined_prob = gpt_result.estimated_prob
            opp.ai_consensus = 0.5

        # ER Score (0-100): combina edge, consenso IA, y potencial de retorno
        edge_score   = max(0, min(40, opp.edge_pct * 4))   # Hasta 40pts por edge (clamped 0-40)
        consensus_sc = min(30, opp.ai_consensus * 30)      # Hasta 30pts por acuerdo IA (clamped)
        return_score = min(20, max(0, (opp.multiplier - 3) * 2))  # Hasta 20pts por retorno
        ai_edge_sc   = min(10, max(0, (opp.ai_combined_prob - opp.entry_price) * 100))  # AI edge (clamped 0-10)

        # v5 improvement: +5pts if AI edge_estimate > 0.05 (extract from reasoning text)
        ai_edge_bonus = 0.0
        _ai_edge_str = ""
        if claude_result:
            _ai_edge_str = claude_result.algorithm_assessment or ""
        _ai_edge_val = 0.0
        try:
            import re as _re_er
            _em = _re_er.search(r"edge[_\s]*estimate[:\s]+([0-9.-]+)", _ai_edge_str, _re_er.IGNORECASE)
            if _em:
                _ai_edge_val = float(_em.group(1))
        except Exception:
            pass
        if _ai_edge_val > 0.05:
            ai_edge_bonus = 5.0

        opp.er_score = round(min(100, max(0, edge_score + consensus_sc + return_score + ai_edge_sc + ai_edge_bonus)), 1)

        # Log AI reasoning clearly
        if claude_result:
            logger.info(
                f"[ER] Claude: conf={int(claude_result.confidence*100)}% -> {claude_result.recommendation} | "
                f"prob={claude_result.estimated_prob:.3f} | {opp.question[:50]}"
            )
        if gpt_result:
            logger.info(
                f"[ER] GPT:    conf={int(gpt_result.confidence*100)}% -> {gpt_result.recommendation} | "
                f"prob={gpt_result.estimated_prob:.3f} | {opp.question[:50]}"
            )

        return opp

    def _build_er_prompt(self, opp: EROpportunity, sports_context: str = "") -> str:
        """Prompt especializado para análisis de mercados de alto riesgo/retorno."""
        sports_block = f"\nCONTEXTO DEPORTIVO EN TIEMPO REAL:\n{sports_context}\n" if sports_context else ""

        # ¡CRÍTICO! La probabilidad del modelo debe ser para el LADO que se apuesta,
        # no siempre para YES. ensemble_prob es siempre P(YES), así que para NO hay que invertir.
        model_prob_side = opp.ensemble_prob if opp.side == "YES" else (1.0 - opp.ensemble_prob)
        ratio_str = f"{model_prob_side / max(0.01, opp.entry_price):.2f}"

        # Explicación clara del lado apostado
        if opp.side == "YES":
            side_bet_desc = f"Apostarás a que SÍ ocurre lo preguntado."
        else:
            side_bet_desc = f"Apostarás a que NO ocurre lo preguntado (el evento descrito NO se cumplirá)."

        return f"""Eres un analista experto en apuestas asimétricas de alto riesgo/retorno (Polymarket).
Tu única tarea: estimar la probabilidad de que el LADO APOSTADO gane y elegir la recomendación.

═══════════════════════════════════════
PREGUNTA DEL MERCADO: {opp.question}
Categoría: {opp.category}

LADO A APOSTAR: {opp.side} — {side_bet_desc}
Precio actual del lado {opp.side}: {opp.entry_price:.3f} ({opp.entry_price:.1%} de probabilidad implícita)
Si aciertas el retorno es: {opp.multiplier:.1f}x tu inversión

Modelo algorítmico estima P({opp.side} gana): {model_prob_side:.1%}
(El mercado dice {opp.entry_price:.1%} — ratio modelo/mercado: {ratio_str}x)
Edge algorítmico calculado: {opp.edge_pct:.1f}%
{sports_block}═══════════════════════════════════════

ATENCIÓN: "probabilidad_estimada" en tu respuesta debe ser P({opp.side} GANE),
es decir la probabilidad de que {"ocurra" if opp.side == "YES" else "NO ocurra"} lo que pregunta el mercado.

PASO 1 — ESTIMA P({opp.side} GANE):
- Para deportes: lesiones confirmadas, forma reciente, H2H, sede
- Para política: encuestas actuales, historial electoral, momentum
- Para crypto: narrativa dominante, catalizadores próximos
- Considera que el mercado implica {opp.entry_price:.1%} para {opp.side} — ¿es correcto ese precio?

PASO 2 — REGLA MECÁNICA (usa tu_prob_estimada ÷ {opp.entry_price:.4f}):
ratio ≥ 2.0  →  "APOSTAR_FUERTE"
ratio ≥ 1.3  →  "APOSTAR"
ratio ≥ 0.9  →  "ESPERAR"
ratio < 0.9  →  "NO_APOSTAR"  ← usa esto si crees que el precio {opp.entry_price:.1%} es justo o alto

RESPONDE SOLO CON JSON VÁLIDO (sin markdown, sin texto extra):
{{
  "pregunta_en_espanol": "traducción exacta al español de: {opp.question}",
  "probabilidad_estimada": 0.XX,
  "confianza": 0.XX,
  "razonamiento": "2-3 frases sobre por qué el lado {opp.side} tiene o no tiene edge",
  "factores_si": ["factor a favor del lado {opp.side}"],
  "factores_no": ["factor en contra del lado {opp.side}"],
  "recomendacion": "APOSTAR_FUERTE|APOSTAR|ESPERAR|NO_APOSTAR",
  "porcentaje_apuesta_sugerido": X.X,
  "coincide_con_algoritmos": true,
  "evaluacion_algoritmos": "opinión sobre el edge algorítmico de {opp.edge_pct:.1f}%",
  "notas_riesgo": "principal riesgo de apostar {opp.side} aquí"
}}"""

    def calculate_bet_size(
        self,
        opp: EROpportunity,
        available_capital: float,
        require_dual_ai: bool = True,
        score_min_override: float = 0.0,
        broadcast_fn=None,
    ) -> float:
        """
        Calcula el tamaño de apuesta para Escape Room.
        Usa Kelly conservador (0.10) ajustado por ER score y consenso IA.

        require_dual_ai: si False, permite entrada con solo una IA (score mínimo 35).
        score_min_override: si > 0, sobreescribe el score mínimo (40 default).
        broadcast_fn: función sync para loguear rechazos (recibe str).
        """
        def _log(msg: str):
            logger.debug(msg)
            if broadcast_fn:
                try:
                    broadcast_fn(msg)
                except Exception:
                    pass

        if opp.entry_price <= 0 or opp.entry_price >= 1:
            return 0.0

        # Kelly fraction ajustado por ER score
        score_factor = opp.er_score / 100.0  # 0.0 - 1.0

        # Umbral de score mínimo (configurable)
        if require_dual_ai:
            effective_min_score = score_min_override if score_min_override > 0 else 40.0
        else:
            effective_min_score = score_min_override if score_min_override > 0 else 35.0

        if opp.er_score < effective_min_score:
            _log(f"[ER] Score {opp.er_score:.1f} < mínimo {effective_min_score:.0f} — rechazado "
                 f"({opp.question[:50]})")
            return 0.0

        # v5: risk_level guard — if high risk + low confidence, refuse even if score passes
        _claude_risk = getattr(opp, "claude_risk_level", "medium") or "medium"
        _gpt_risk    = getattr(opp, "gpt_risk_level", "medium") or "medium"
        _claude_conf_pct = opp.claude_confidence * 100.0  # confidence is 0-1 in EROpportunity
        _gpt_conf_pct    = opp.gpt_confidence * 100.0
        if _claude_risk == "high" and _claude_conf_pct < 70:
            _log(f"[ER] Claude risk=high conf={_claude_conf_pct:.0f}% < 70 — rechazado por riesgo alto ({opp.question[:40]})")
            return 0.0
        if _gpt_risk == "high" and _gpt_conf_pct < 70:
            _log(f"[ER] GPT risk=high conf={_gpt_conf_pct:.0f}% < 70 — rechazado por riesgo alto ({opp.question[:40]})")
            return 0.0

        # Verificar consenso IA
        claude_ok = opp.claude_recommendation in ("APOSTAR", "APOSTAR_FUERTE")
        gpt_ok    = opp.gpt_recommendation in ("APOSTAR", "APOSTAR_FUERTE")

        if require_dual_ai:
            both_bullish = claude_ok and gpt_ok
            if not both_bullish:
                _log(f"[ER] IA: Claude={opp.claude_recommendation} (conf={_claude_conf_pct:.0f}%) "
                     f"GPT={opp.gpt_recommendation} (conf={_gpt_conf_pct:.0f}%) "
                     f"— consenso dual insuficiente ({opp.question[:40]})")
                return 0.0
        else:
            # Solo una IA necesaria
            one_bullish = claude_ok or gpt_ok
            if not one_bullish:
                _log(f"[ER] IA: Claude={opp.claude_recommendation} GPT={opp.gpt_recommendation} "
                     f"— ninguna IA recomienda ({opp.question[:40]})")
                return 0.0

        # ── Umbral de consenso extra para mercados ultra-baratos ──────────
        # Precio < 10¢ (10x+) → la IA puede alucinar; requerir consenso alto
        if opp.entry_price < 0.10:
            if opp.ai_consensus < 0.70:
                _log(f"[ER] Consenso IA {opp.ai_consensus:.2f} < 0.70 para precio {opp.entry_price:.2f} (<10¢)")
                return 0.0
            if opp.er_score < 55:
                _log(f"[ER] Score {opp.er_score:.1f} < 55 para ultra-especulativo (<10¢)")
                return 0.0

        # ── Umbral adicional si solo hay una IA disponible ────────────────
        has_both_ias = (opp.claude_prob > 0 and opp.gpt_prob > 0)
        if not has_both_ias and opp.entry_price < 0.20 and require_dual_ai:
            _log(f"[ER] Requiere dual-IA para precio {opp.entry_price:.2f} (<20¢) — solo una IA disponible")
            return 0.0

        b = (1.0 - opp.entry_price) / opp.entry_price  # odds netas
        p = opp.ai_combined_prob if opp.ai_combined_prob > 0 else opp.ensemble_prob
        q = 1.0 - p

        full_kelly = (p * b - q) / b if b > 0 else 0
        if full_kelly <= 0:
            _log(f"[ER] Kelly negativo (p={p:.3f}, b={b:.3f}) — sin edge matemático")
            return 0.0

        # Quarter-Kelly ajustado por score y consenso
        adjusted_kelly = full_kelly * ER_KELLY_FRACTION * score_factor * max(0.3, opp.ai_consensus)
        max_bet = available_capital * ER_MAX_BET_PCT
        bet = min(adjusted_kelly * available_capital, max_bet)
        return round(max(1.0, bet), 2) if bet >= 1.0 else 0.0

    def filter_opportunities_verbose(
        self,
        scan_results: list,
        max_price: float = None,
        broadcast_fn=None,
    ) -> tuple[list["EROpportunity"], str]:
        """
        Versión verbose de filter_opportunities.
        Retorna (oportunidades, resumen_mensaje).
        broadcast_fn: función sync(msg: str) para loguear rechazos.
        """
        def _log(msg: str):
            logger.debug(msg)
            if broadcast_fn:
                try:
                    broadcast_fn(msg)
                except Exception:
                    pass

        effective_max = max_price if max_price is not None else ER_MAX_ENTRY_PRICE
        n_total = len(scan_results)
        n_price_fail = 0
        n_liq_fail   = 0
        n_edge_fail  = 0
        n_mult_fail  = 0
        opportunities = []

        for r in scan_results:
            side = r.get("side", "YES")
            entry_price = r.get("yes_price", 0.5) if side == "YES" else r.get("no_price", 0.5)

            if not (ER_MIN_ENTRY_PRICE <= entry_price <= effective_max):
                n_price_fail += 1
                continue

            if r.get("liquidity", 0) < ER_MIN_LIQUIDITY:
                n_liq_fail += 1
                continue

            if r.get("edge_pct", 0) <= 0:
                n_edge_fail += 1
                continue

            multiplier = round(1.0 / max(0.01, entry_price), 2)
            cat = r.get("category", "other")
            cat_min_mult = ER_MIN_MULTIPLIER_BY_CATEGORY.get(cat, ER_MIN_MULTIPLIER_BY_CATEGORY["other"])
            if multiplier < cat_min_mult:
                n_mult_fail += 1
                continue

            opp = EROpportunity(
                market_id=r.get("market_id", ""),
                question=r.get("question", ""),
                slug=r.get("slug", ""),
                category=cat,
                side=side,
                entry_price=round(entry_price, 4),
                multiplier=multiplier,
                ensemble_prob=r.get("ensemble_prob", 0.5),
                edge_pct=r.get("edge_pct", 0),
                kl_divergence=r.get("kl_divergence", 0),
                yes_token=r.get("yes_token", ""),
                no_token=r.get("no_token", ""),
                cat_min_multiplier=cat_min_mult,
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda x: x.kl_divergence, reverse=True)
        result = opportunities[:20]

        summary = (
            f"[ER] Escaneadas {n_total} oportunidades — {len(result)} pasaron filtros "
            f"(rechazadas: precio={n_price_fail}, liquidez={n_liq_fail}, "
            f"edge={n_edge_fail}, multiplicador={n_mult_fail})"
        )
        _log(summary)
        return result, summary
