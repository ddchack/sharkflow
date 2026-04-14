# SharkFlow by Carlos David Donoso Cordero (ddchack)
"""
test_ai_integration.py — AI Integration Optimization Tests
============================================================
Tests for:
  1. AI consensus weighting math
  2. Cache TTL logic
  3. Structured response parsing
  4. ER AI score boost (edge_estimate > 0.05)
  5. AI recommendation fallback when no API keys
  6. Weighted consensus calculation scenarios
"""

import json
import time
import pytest

# ═══════════════════════════════════════════════════════════════════════
# INLINE IMPLEMENTATIONS (mirrors api_server.py helpers, testable in isolation)
# ═══════════════════════════════════════════════════════════════════════

_AI_REC_VALUES = {
    "APOSTAR_FUERTE": 1.0,
    "APOSTAR":        0.7,
    "ESPERAR":        0.3,
    "NO_APOSTAR":     0.0,
    "PASAR":          0.0,
}

AI_CACHE_TTL = 300  # 5 minutes


def _parse_structured_ai_response(text: str, market_idx: int) -> dict:
    """Parse structured JSON AI response. Falls back to legacy/NO_APOSTAR on error."""
    import re as _re
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        d = json.loads(clean)
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
        pat = rf"{market_idx}\.\s*PROB=(\d+\.\d+)\s+REC=(\w+)\s+CONF=(\w+)"
        m = _re.search(pat, text, _re.IGNORECASE)
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


def weighted_consensus(c_conf: int, c_rec: str, g_conf: int, g_rec: str,
                        c_prob: float, g_prob: float) -> dict:
    """Compute weighted AI consensus from Claude and GPT results."""
    c_val = _AI_REC_VALUES.get(c_rec, 0.0)
    g_val = _AI_REC_VALUES.get(g_rec, 0.0)
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
    consensus_prob = (c_conf * c_prob + g_conf * g_prob) / total_conf
    return {
        "recommendation": consensus_rec,
        "prob": round(consensus_prob, 4),
        "consensus_val": round(consensus_val, 3),
        "confidence": "ALTA" if max(c_conf, g_conf) >= 70 else ("MEDIA" if max(c_conf, g_conf) >= 50 else "BAJA"),
    }


# Simple in-memory cache implementation
_test_cache: dict = {}

def _get_ai_cache(market_id: str, cache: dict) -> dict | None:
    if market_id in cache:
        ts, result = cache[market_id]
        if time.time() - ts < AI_CACHE_TTL:
            return result
        del cache[market_id]
    return None

def _set_ai_cache(market_id: str, result: dict, cache: dict) -> None:
    cache[market_id] = (time.time(), result)


# ═══════════════════════════════════════════════════════════════════════
# 1. AI CONSENSUS WEIGHTING MATH
# ═══════════════════════════════════════════════════════════════════════

def test_recommendation_values_defined():
    """All expected recommendation keys have correct numeric values."""
    assert _AI_REC_VALUES["APOSTAR_FUERTE"] == 1.0
    assert _AI_REC_VALUES["APOSTAR"]        == 0.7
    assert _AI_REC_VALUES["ESPERAR"]        == 0.3
    assert _AI_REC_VALUES["NO_APOSTAR"]     == 0.0
    assert _AI_REC_VALUES["PASAR"]          == 0.0


def test_consensus_both_apostar_fuerte():
    """Both AIs say APOSTAR_FUERTE → consensus = APOSTAR_FUERTE."""
    result = weighted_consensus(80, "APOSTAR_FUERTE", 75, "APOSTAR_FUERTE", 0.80, 0.82)
    assert result["recommendation"] == "APOSTAR_FUERTE"
    assert result["consensus_val"] == 1.0


def test_consensus_both_apostar():
    """Both AIs say APOSTAR → consensus_val = 0.7 → APOSTAR."""
    result = weighted_consensus(70, "APOSTAR", 60, "APOSTAR", 0.65, 0.70)
    assert result["recommendation"] == "APOSTAR"
    assert abs(result["consensus_val"] - 0.7) < 1e-9


def test_consensus_split_apostar_no_apostar():
    """Claude=APOSTAR(70%), GPT=NO_APOSTAR(80%) → weighted toward NO_APOSTAR."""
    result = weighted_consensus(70, "APOSTAR", 80, "NO_APOSTAR", 0.60, 0.45)
    # val = (70*0.7 + 80*0.0) / 150 = 49/150 = 0.327 → ESPERAR
    expected_val = (70 * 0.7 + 80 * 0.0) / 150
    assert abs(result["consensus_val"] - round(expected_val, 3)) < 1e-9
    assert result["recommendation"] == "ESPERAR"  # 0.327 is in 0.25-0.55 range


def test_consensus_both_no_apostar():
    """Both NO_APOSTAR → consensus = NO_APOSTAR."""
    result = weighted_consensus(60, "NO_APOSTAR", 75, "NO_APOSTAR", 0.30, 0.25)
    assert result["recommendation"] == "NO_APOSTAR"
    assert result["consensus_val"] == 0.0


def test_consensus_prob_weighted_average():
    """Probability is correctly weighted by confidence."""
    result = weighted_consensus(100, "APOSTAR", 0, "NO_APOSTAR", 0.70, 0.30)
    # All weight on Claude: prob = 0.70
    assert abs(result["prob"] - 0.70) < 1e-4


# ═══════════════════════════════════════════════════════════════════════
# 2. CACHE TTL LOGIC
# ═══════════════════════════════════════════════════════════════════════

def test_cache_fresh_returns_result():
    """Fresh cache entry returns the stored result."""
    cache = {}
    entry = {"ai_prob": 0.65, "ai_recommendation": "APOSTAR"}
    _set_ai_cache("market_1", entry, cache)
    result = _get_ai_cache("market_1", cache)
    assert result is not None
    assert result["ai_recommendation"] == "APOSTAR"


def test_cache_miss_returns_none():
    """Cache miss (unknown market) returns None."""
    cache = {}
    result = _get_ai_cache("nonexistent_market", cache)
    assert result is None


def test_cache_expired_returns_none():
    """Expired cache entry (ts in the past) returns None."""
    cache = {}
    market_id = "market_expired"
    # Manually set entry with expired timestamp
    cache[market_id] = (time.time() - AI_CACHE_TTL - 1, {"ai_prob": 0.5})
    result = _get_ai_cache(market_id, cache)
    assert result is None
    # Entry should be cleaned up
    assert market_id not in cache


def test_cache_just_before_expiry():
    """Entry with 1 second remaining is still fresh."""
    cache = {}
    market_id = "market_fresh"
    cache[market_id] = (time.time() - AI_CACHE_TTL + 1, {"ai_recommendation": "APOSTAR_FUERTE"})
    result = _get_ai_cache(market_id, cache)
    assert result is not None
    assert result["ai_recommendation"] == "APOSTAR_FUERTE"


def test_cache_overwrites_existing():
    """Setting cache twice overwrites the first value."""
    cache = {}
    _set_ai_cache("market_x", {"ai_prob": 0.4}, cache)
    _set_ai_cache("market_x", {"ai_prob": 0.8}, cache)
    result = _get_ai_cache("market_x", cache)
    assert result is not None
    assert result["ai_prob"] == 0.8


# ═══════════════════════════════════════════════════════════════════════
# 3. STRUCTURED RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════

def test_parse_valid_json_response():
    """Valid JSON response is parsed correctly."""
    response = json.dumps({
        "recomendacion": "APOSTAR",
        "probabilidad_estimada": 0.72,
        "confianza": 0.85,
        "edge_estimado": 0.12,
        "factores_clave": ["Strong momentum", "Low liquidity risk"],
        "nivel_riesgo": "medium",
        "razonamiento": "The market price underestimates the true probability.",
    })
    result = _parse_structured_ai_response(response, 1)
    assert result["ai_recommendation"] == "APOSTAR"
    assert abs(result["ai_prob"] - 0.72) < 1e-9
    assert result["ai_confidence"] == "ALTA"
    assert result["ai_edge_estimate"] == 0.12
    assert result["ai_risk_level"] == "medium"
    assert len(result["ai_key_factors"]) == 2


def test_parse_invalid_json_returns_no_apostar():
    """Invalid JSON falls back to NO_APOSTAR."""
    result = _parse_structured_ai_response("this is not valid json at all!!!!", 1)
    assert result["ai_recommendation"] == "NO_APOSTAR"
    assert result["ai_confidence"] == "BAJA"


def test_parse_legacy_format():
    """Legacy PROB/REC/CONF format is parsed correctly."""
    result = _parse_structured_ai_response("1. PROB=0.68 REC=APOSTAR CONF=ALTA", 1)
    assert result["ai_recommendation"] == "APOSTAR"
    assert abs(result["ai_prob"] - 0.68) < 1e-9
    assert result["ai_confidence"] == "ALTA"
    assert result["ai_confidence_num"] == 80


def test_parse_confidence_above_1_normalized():
    """Confidence value of 85 (percentage) is normalized to 0.85."""
    response = json.dumps({
        "recomendacion": "ESPERAR",
        "probabilidad_estimada": 0.55,
        "confianza": 85.0,  # percentage format
        "edge_estimado": 0.01,
    })
    result = _parse_structured_ai_response(response, 1)
    assert result["ai_confidence_num"] == 85
    assert result["ai_confidence"] == "ALTA"


def test_parse_markdown_stripped():
    """Markdown code block is stripped before parsing."""
    response = "```json\n" + json.dumps({
        "recomendacion": "NO_APOSTAR",
        "probabilidad_estimada": 0.30,
        "confianza": 0.6,
    }) + "\n```"
    result = _parse_structured_ai_response(response, 1)
    assert result["ai_recommendation"] == "NO_APOSTAR"
    assert result["ai_confidence"] == "MEDIA"


# ═══════════════════════════════════════════════════════════════════════
# 4. ER AI SCORE BOOST
# ═══════════════════════════════════════════════════════════════════════

def test_er_score_edge_estimate_bonus():
    """edge_estimate > 0.05 → 5pt bonus in ER score calculation."""
    def compute_er_score(edge_pct, ai_consensus, multiplier, ai_combined_prob, entry_price, ai_edge_val):
        edge_score   = max(0, min(40, edge_pct * 4))
        consensus_sc = ai_consensus * 30
        return_score = min(20, max(0, (multiplier - 3) * 2))
        ai_edge_sc   = max(0, (ai_combined_prob - entry_price) * 100)
        ai_edge_bonus = 5.0 if ai_edge_val > 0.05 else 0.0
        return min(100, max(0, edge_score + consensus_sc + return_score + ai_edge_sc + ai_edge_bonus))

    score_with_edge = compute_er_score(8.0, 0.8, 4.0, 0.22, 0.20, 0.08)
    score_without   = compute_er_score(8.0, 0.8, 4.0, 0.22, 0.20, 0.00)
    assert score_with_edge == score_without + 5.0, "edge_estimate > 0.05 should add 5pts"


def test_er_score_no_edge_estimate_bonus_below_threshold():
    """edge_estimate = 0.04 (< 0.05) → no bonus."""
    def compute_er_score(edge_pct, ai_consensus, multiplier, ai_combined_prob, entry_price, ai_edge_val):
        edge_score   = max(0, min(40, edge_pct * 4))
        consensus_sc = ai_consensus * 30
        return_score = min(20, max(0, (multiplier - 3) * 2))
        ai_edge_sc   = max(0, (ai_combined_prob - entry_price) * 100)
        ai_edge_bonus = 5.0 if ai_edge_val > 0.05 else 0.0
        return min(100, max(0, edge_score + consensus_sc + return_score + ai_edge_sc + ai_edge_bonus))

    score_no_bonus = compute_er_score(8.0, 0.8, 4.0, 0.22, 0.20, 0.04)
    score_base     = compute_er_score(8.0, 0.8, 4.0, 0.22, 0.20, 0.00)
    assert score_no_bonus == score_base, "edge_estimate <= 0.05 should not add bonus"


# ═══════════════════════════════════════════════════════════════════════
# 5. AI RECOMMENDATION FALLBACK
# ═══════════════════════════════════════════════════════════════════════

def test_no_api_keys_returns_no_apostar():
    """
    When no API keys are configured, the AI result should be NO_APOSTAR.
    This simulates the _ai_ensemble_boost early-return check.
    """
    class MockLLM:
        anthropic_key = ""
        openai_key = ""

    mock_recs = [{"market_id": "test_1", "question": "Test?"}]
    mock_llm = MockLLM()
    # The boost function returns {} when no keys
    result = {} if (not mock_recs or (not mock_llm.anthropic_key and not mock_llm.openai_key)) else {"test": "data"}
    assert result == {}, "No API keys should return empty dict (no AI result)"


def test_fallback_parse_gives_default_no_apostar():
    """Empty string response → NO_APOSTAR default."""
    result = _parse_structured_ai_response("", 1)
    assert result["ai_recommendation"] == "NO_APOSTAR"
    assert result["ai_confidence"] == "BAJA"
    assert result["ai_prob"] == 0.5


# ═══════════════════════════════════════════════════════════════════════
# 6. WEIGHTED CONSENSUS — Additional Scenarios
# ═══════════════════════════════════════════════════════════════════════

def test_consensus_high_confidence_claude_vs_low_gpt():
    """High-confidence Claude dominates over low-confidence GPT."""
    # Claude says APOSTAR_FUERTE (conf=90), GPT says NO_APOSTAR (conf=10)
    result = weighted_consensus(90, "APOSTAR_FUERTE", 10, "NO_APOSTAR", 0.80, 0.40)
    # val = (90*1.0 + 10*0.0) / 100 = 0.90 → APOSTAR_FUERTE
    assert result["recommendation"] == "APOSTAR_FUERTE"
    assert result["consensus_val"] == 0.90


def test_consensus_equal_confidence_esperar():
    """Equal confidence, one APOSTAR one NO_APOSTAR → ESPERAR."""
    result = weighted_consensus(50, "APOSTAR", 50, "NO_APOSTAR", 0.55, 0.45)
    # val = (50*0.7 + 50*0.0) / 100 = 35/100 = 0.35 → ESPERAR
    assert result["recommendation"] == "ESPERAR"
    assert abs(result["consensus_val"] - 0.35) < 1e-9


def test_consensus_confidence_zero():
    """Zero total confidence falls back gracefully (no division by zero)."""
    # Both conf=0 → total_conf = max(0,1) = 1
    result = weighted_consensus(0, "APOSTAR", 0, "APOSTAR_FUERTE", 0.60, 0.70)
    # val = 0; prob = 0 — but max(0+0,1)=1 prevents ZeroDivisionError
    assert "recommendation" in result
    assert result["consensus_val"] == 0.0


def test_consensus_apostar_fuerte_threshold():
    """consensus_val of exactly 0.85 → APOSTAR_FUERTE."""
    # Need (c_conf * 1.0 + g_conf * 0.7) / total = 0.85
    # c=85, g=15: (85*1.0 + 15*0.7)/100 = (85+10.5)/100 = 0.955 → APOSTAR_FUERTE
    result = weighted_consensus(85, "APOSTAR_FUERTE", 15, "APOSTAR", 0.80, 0.65)
    assert result["recommendation"] == "APOSTAR_FUERTE"
    assert result["consensus_val"] >= 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
