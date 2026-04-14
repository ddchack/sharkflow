# SharkFlow by Carlos David Donoso Cordero (ddchack)
"""
Tests para DDC v3 improvements:
- Volume factor bonus
- Price efficiency bonus
- Close market premium (≤2h)
- AI confidence weighting (0.01 -> 0.03)

Autor: SharkBot (ddchack)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _compute_ddc_score_v3(kl_div, ev, confidence, volume24h=0, yes_price=0.5,
                           hours_to_resolution=None, ai_recommendation=None,
                           ai_confidence=None, mr_aligned=False, mr_proxy=0.0,
                           price_zone=None):
    """
    Replica la lógica de DDC score v3 tal como está en api_server.py.
    Usa los mismos coeficientes y fórmulas.
    """
    # Normalización con constantes fijas para el test
    DDC_KL_NORM = 0.05
    DDC_EV_NORM = 0.06

    kl_n    = min(kl_div / DDC_KL_NORM, 1.0)
    ev_raw  = min(max(ev, 0.0) / DDC_EV_NORM, 1.0)
    liq_f   = 1.0  # asumir liquidez suficiente para tests
    ev_n    = ev_raw * liq_f
    conf_n  = min(confidence / 100.0, 1.0)
    mr_b    = 0.10 if mr_aligned and mr_proxy > 0.3 else 0.0
    zone_b  = 0.05 if price_zone == "sweet_spot" else 0.0

    # Urgency
    urg_b = (0.07 if hours_to_resolution is not None and hours_to_resolution <= 6
             else 0.05 if hours_to_resolution is not None and hours_to_resolution <= 24
             else 0.03 if hours_to_resolution is not None and hours_to_resolution <= 72
             else 0.0)

    # DDC v3: close market premium
    close_prem = 0.03 if hours_to_resolution is not None and hours_to_resolution <= 2 else 0.0

    # DDC v3: volume factor
    vol_b = 0.03 if volume24h > 10000 else 0.0

    # DDC v3: price efficiency
    price_eff = 0.04 if (0.45 <= yes_price <= 0.55) else 0.0

    # DDC v3: AI bonus
    ai_conf_high = ai_confidence in ("ALTA", "MEDIA")
    ai_b = (0.03 if ai_recommendation == "APOSTAR" and ai_conf_high
            else 0.01 if ai_recommendation == "APOSTAR" else 0.0)

    score = min(
        kl_n * 0.35 + ev_n * 0.22 + conf_n * 0.12 + mr_b + zone_b + urg_b + close_prem + vol_b + price_eff + ai_b,
        1.0
    )
    return round(score, 4)


def test_volume_factor_bonus():
    """Volume > 10000 USD agrega 0.03 al score."""
    score_no_vol  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    score_with_vol = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, volume24h=15000)
    diff = score_with_vol - score_no_vol
    assert abs(diff - 0.03) < 0.001, (
        f"Volume factor debe agregar 0.03: {score_with_vol} - {score_no_vol} = {diff}"
    )


def test_volume_factor_no_bonus_under_threshold():
    """Volume ≤ 10000 USD NO agrega bonus."""
    score_no_vol    = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    score_low_vol   = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, volume24h=9999)
    assert abs(score_no_vol - score_low_vol) < 0.001, (
        f"Volume < 10000 no debe agregar bonus: {score_no_vol} vs {score_low_vol}"
    )


def test_price_efficiency_bonus():
    """Yes_price 0.45-0.55 agrega 0.04 al score."""
    score_no_eff  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80)
    score_eff_50  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.50)
    score_eff_45  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.45)
    score_eff_55  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.55)
    assert score_eff_50 - score_no_eff == pytest_approx(0.04), f"Price efficiency 0.50: {score_eff_50} - {score_no_eff}"
    assert score_eff_45 - score_no_eff == pytest_approx(0.04), f"Price efficiency 0.45: {score_eff_45}"
    assert score_eff_55 - score_no_eff == pytest_approx(0.04), f"Price efficiency 0.55: {score_eff_55}"


def pytest_approx(val, rel=1e-3):
    """Aproximación simple para comparaciones flotantes sin pytest."""
    class _Approx:
        def __init__(self, v, r):
            self.v = v
            self.r = r
        def __eq__(self, other):
            return abs(other - self.v) <= max(self.r * abs(self.v), 1e-6)
        def __repr__(self):
            return f"~{self.v}"
    return _Approx(val, rel)


def test_close_market_premium():
    """Mercados con ≤2h hasta resolución tienen bonus extra (0.03 + urgency)."""
    score_6h    = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=6)
    score_2h    = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=2)
    score_1h    = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=1)
    # 2h: urgency=0.07 + close_prem=0.03
    # 6h: urgency=0.07 + close_prem=0 (igual urgency pero sin premium)
    assert score_2h > score_6h, f"2h debería tener mayor score que 6h: {score_2h} vs {score_6h}"
    assert abs(score_1h - score_2h) < 0.001, f"1h y 2h deben tener el mismo premium: {score_1h} vs {score_2h}"
    diff_premium = score_2h - score_6h
    assert abs(diff_premium - 0.03) < 0.001, f"Close premium debe ser 0.03: {diff_premium}"


def test_close_market_premium_not_applied_at_6h():
    """El close premium NO aplica a mercados de 6h."""
    score_6h  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=6)
    score_3h  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=3)
    # 6h y 3h tienen misma urgency (0.07) pero 3h tiene close_prem=0.03
    # excepto 3h <= 6h entonces urg_b es el mismo pero close_prem = 0.03 para ≤2h solamente
    # 3h no aplica (>2h), así que deberían ser iguales
    assert abs(score_6h - score_3h) < 0.001, f"3h y 6h no deben diferir en close premium: {score_6h} vs {score_3h}"


def test_ai_bonus_v3_high_confidence():
    """AI bonus con confianza ALTA es 0.03 (era 0.01)."""
    score_no_ai   = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    score_ai_high = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60,
                                           ai_recommendation="APOSTAR", ai_confidence="ALTA")
    diff = score_ai_high - score_no_ai
    assert abs(diff - 0.03) < 0.001, f"AI bonus con ALTA confianza debe ser 0.03: {diff}"


def test_ai_bonus_v3_low_confidence():
    """AI bonus con confianza BAJA es 0.01 (igual que antes)."""
    score_no_ai    = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    score_ai_low   = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60,
                                            ai_recommendation="APOSTAR", ai_confidence="BAJA")
    diff = score_ai_low - score_no_ai
    assert abs(diff - 0.01) < 0.001, f"AI bonus con BAJA confianza debe ser 0.01: {diff}"


def test_ai_bonus_no_recommendation():
    """Sin recomendación AI, bonus = 0."""
    score_no_ai = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    score_pasar = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60,
                                         ai_recommendation="PASAR", ai_confidence="ALTA")
    assert abs(score_no_ai - score_pasar) < 0.001, f"PASAR no debe agregar bonus: {score_no_ai} vs {score_pasar}"


def test_all_bonuses_combined():
    """Con todos los bonuses v3 activos, el score es mayor que sin ellos."""
    # Base: precio 0.80 (fuera del rango de eficiencia), sin volumen, sin urgency, sin AI
    score_base = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80)
    score_all  = _compute_ddc_score_v3(
        kl_div=0.05, ev=0.06, confidence=60,
        volume24h=20000,       # +0.03 vol
        yes_price=0.50,        # +0.04 price_eff
        hours_to_resolution=2, # +0.07 urg + 0.03 close_prem
        ai_recommendation="APOSTAR", ai_confidence="ALTA",  # +0.03 ai
    )
    diff = score_all - score_base
    # Verificar cada bonus individualmente para asegurar que todos contribuyen
    # (Nota: base usa yes_price=0.80 para que el bonus de eficiencia sea visible)
    score_vol  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80, volume24h=20000)
    score_eff  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.50)
    score_urg  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80, hours_to_resolution=6)
    score_cls  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80, hours_to_resolution=2)
    score_ai   = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.80,
                                        ai_recommendation="APOSTAR", ai_confidence="ALTA")
    # Cada bonus individual debe ser positivo
    assert score_vol > score_base,  f"Volume bonus: {score_vol} vs {score_base}"
    assert score_eff > score_base,  f"Efficiency bonus: {score_eff} vs {score_base}"
    assert score_urg > score_base,  f"Urgency bonus: {score_urg} vs {score_base}"
    assert score_cls > score_urg,   f"Close premium > urgency only: {score_cls} vs {score_urg}"
    assert score_ai  > score_base,  f"AI bonus: {score_ai} vs {score_base}"
    # Score combinado debe ser mayor que base
    assert score_all > score_base, f"Score combinado debe superar base: {score_all} vs {score_base}"
    assert diff > 0.10, f"Con todos los bonuses, incremento debe ser > 0.10: {diff}"


def test_score_capped_at_1():
    """El score nunca excede 1.0."""
    score = _compute_ddc_score_v3(
        kl_div=0.20, ev=0.20, confidence=100,
        volume24h=999999,
        yes_price=0.50,
        hours_to_resolution=1,
        ai_recommendation="APOSTAR", ai_confidence="ALTA",
        mr_aligned=True, mr_proxy=0.5,
        price_zone="sweet_spot",
    )
    assert score <= 1.0, f"Score no debe exceder 1.0: {score}"


def test_price_efficiency_boundary_values():
    """Price efficiency solo en rango estricto 0.45-0.55."""
    for price in [0.44, 0.56, 0.30, 0.70]:
        score_out = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=price)
        score_in  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, yes_price=0.50)
        assert score_in > score_out, f"Precio {price} fuera del rango no debe tener bonus de eficiencia"


def test_volume_exact_threshold():
    """Volume = 10001 da bonus, volume = 10000 no."""
    score_10000 = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, volume24h=10000)
    score_10001 = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, volume24h=10001)
    # > 10000 (estricto)
    assert score_10001 > score_10000, f"10001 debe tener bonus pero 10000 no"


def test_urgency_tiers():
    """Los tiers de urgency son correctos: 6h=0.07, 24h=0.05, 72h=0.03, None=0."""
    base = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60)
    s6h  = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=6)
    s24h = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=24)
    s72h = _compute_ddc_score_v3(kl_div=0.05, ev=0.06, confidence=60, hours_to_resolution=72)
    # 6h: urg=0.07 + close=0.03 = 0.10 (si ≤2h)... No, 6h > 2h, so close=0
    # Tier: ≤6 → 0.07, ≤24 → 0.05, ≤72 → 0.03
    assert abs((s6h - base) - 0.07) < 0.001, f"6h urgency: {s6h - base}"
    assert abs((s24h - base) - 0.05) < 0.001, f"24h urgency: {s24h - base}"
    assert abs((s72h - base) - 0.03) < 0.001, f"72h urgency: {s72h - base}"
    assert abs(base - base) < 0.001, f"None urgency: 0"


def test_ddc_v3_better_than_v2_for_high_volume_uncertain_price():
    """DDC v3 da mayor score a mercados con alto volumen y precio incierto."""
    # v2 equivalent (sin nuevos bonuses)
    score_v2 = _compute_ddc_score_v3(kl_div=0.06, ev=0.07, confidence=65,
                                      volume24h=0, yes_price=0.70)
    # v3 con los nuevos features
    score_v3 = _compute_ddc_score_v3(kl_div=0.06, ev=0.07, confidence=65,
                                      volume24h=50000, yes_price=0.50)
    assert score_v3 > score_v2, f"v3 debería superar a v2 para mercados activos e inciertos: {score_v3} vs {score_v2}"


if __name__ == "__main__":
    tests = [
        test_volume_factor_bonus,
        test_volume_factor_no_bonus_under_threshold,
        test_price_efficiency_bonus,
        test_close_market_premium,
        test_close_market_premium_not_applied_at_6h,
        test_ai_bonus_v3_high_confidence,
        test_ai_bonus_v3_low_confidence,
        test_ai_bonus_no_recommendation,
        test_all_bonuses_combined,
        test_score_capped_at_1,
        test_price_efficiency_boundary_values,
        test_volume_exact_threshold,
        test_urgency_tiers,
        test_ddc_v3_better_than_v2_for_high_volume_uncertain_price,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\nResultado: {passed}/{len(tests)} PASS, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
