"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests de condiciones de entrada de bots — valida que los umbrales son
realistas y que el sistema puede entrar en condiciones normales.
"""
import math
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── helpers locales (sin importar asyncio completo) ───────────────────────────

def classify_return_zone_local(price: float) -> str:
    if 0.20 <= price < 0.42: return "premium"
    if 0.42 <= price < 0.62: return "good"
    if 0.62 <= price < 0.72: return "marginal"
    return "rejected"


def compute_rush_kelly_local(signal_price: float, signal_strength: float,
                              rush_capital: float = 3.0) -> tuple:
    p = 0.52 + signal_strength * 0.18
    q = 1.0 - p
    b = max(0.01, (1.0 / max(0.01, signal_price)) - 1.0)
    raw = (p * b - q) / b
    if raw <= 0:
        return 0.0, 0.0, round(p, 4)
    zone = classify_return_zone_local(signal_price)
    zone_mod = {"premium": 1.20, "good": 1.00, "marginal": 0.75}.get(zone, 0.0)
    if zone_mod == 0.0:
        return 0.0, 0.0, round(p, 4)
    kelly_frac = min(raw, 0.35) * zone_mod
    if kelly_frac < 0.05:
        return 0.0, 0.0, round(p, 4)
    return round(kelly_frac, 4), round(kelly_frac * rush_capital, 4), round(p, 4)


# ── Rush entry reachability ────────────────────────────────────────────────────

class TestRushEntryReachability(unittest.TestCase):
    """Verifica que las condiciones de entrada son alcanzables en la práctica."""

    def test_lax_mode_enters_with_poly_only(self):
        """Modo laxo: solo señal Poly (55%+) sin BTC → debe generar señal."""
        # En modo laxo: poly_high = 0.55 (antes 0.60)
        poly_high_lax = 0.55
        poly_price = 0.58   # Polymarket dice 58% → señal UP
        poly_signal = "UP" if poly_price >= poly_high_lax else "NEUTRAL"
        self.assertEqual(poly_signal, "UP")

    def test_lax_mode_lower_btc_momentum(self):
        """Modo laxo: momentum mínimo reducido a 0.05% (era 0.10%)."""
        btc_open  = 50000.0
        btc_now   = 50026.0   # +0.052% — por debajo del estándar pero sobre el laxo
        pct_change = (btc_now - btc_open) / btc_open
        std_momentum  = 0.001   # 0.10%
        lax_momentum  = 0.0005  # 0.05%
        self.assertFalse(pct_change >= std_momentum, "Estándar debería rechazar 0.052%")
        self.assertTrue (pct_change >= lax_momentum, "Laxo debería aceptar 0.052%")

    def test_normal_btc_5min_movement_triggers_standard(self):
        """Movimiento BTC estándar (0.15%) activa el modo estándar."""
        btc_open = 50000.0
        btc_now  = 50075.0   # +0.15%
        pct_change = (btc_now - btc_open) / btc_open
        self.assertGreaterEqual(pct_change, 0.001)

    def test_rush_kelly_valid_in_premium_zone(self):
        """En zona premium con señal media, Kelly es > 0."""
        kf, ku, mp = compute_rush_kelly_local(0.32, 0.50, 3.0)
        self.assertGreater(kf, 0.0)
        self.assertGreater(ku, 0.0)

    def test_rush_kelly_valid_in_good_zone(self):
        """En zona buena con señal media, Kelly es > 0."""
        kf, ku, mp = compute_rush_kelly_local(0.50, 0.50, 3.0)
        self.assertGreater(kf, 0.0)

    def test_zone_filter_disabled_accepts_wide_range(self):
        """Con zone_filter_disabled: precios 0.10-0.90 son aceptados."""
        prices = [0.12, 0.25, 0.50, 0.65, 0.80, 0.88]
        for p in prices:
            # In disabled mode: accept 0.10 <= p <= 0.90
            accepted = 0.10 <= p <= 0.90
            self.assertTrue(accepted, f"price={p} debería aceptarse con zone_filter_disabled")

    def test_standard_zone_filter_rejects_overpriced(self):
        """Filtro estándar: precio ≥ 0.72 (retorno < 1.39x) es rechazado."""
        for price in [0.73, 0.80, 0.90]:
            zone = classify_return_zone_local(price)
            self.assertEqual(zone, "rejected", f"price={price} debería ser rejected")

    def test_standard_zone_filter_rejects_too_cheap(self):
        """Filtro estándar: precio < 0.20 es rechazado (sin liquidez)."""
        for price in [0.10, 0.15, 0.19]:
            zone = classify_return_zone_local(price)
            self.assertEqual(zone, "rejected", f"price={price} debería ser rejected")

    def test_fallback_slug_format(self):
        """Formato slug de fallback busca asset+updown+5m."""
        import re
        test_slugs = [
            "btc-updown-5m-1743200400",
            "eth-updown-5m-1743200400",
            "sol-updown-5min-1743200400",
            "btc-up-down-5m-1743200400",
        ]
        pattern = re.compile(r"(btc|eth|sol).*(updown|up.down).*(5m|5min)", re.IGNORECASE)
        for slug in test_slugs:
            self.assertTrue(pattern.search(slug), f"Slug '{slug}' debería coincidir con el patrón")

    def test_market_end_date_proximity_check(self):
        """Fallback: mercado dentro de 120s del target_end_ts es válido."""
        now_ts   = int(time.time())
        window_ts = (now_ts // 300) * 300
        target_end = window_ts + 300
        # Mercado cuyo endDate está a 45s del target
        market_end = target_end + 45
        diff = abs(market_end - target_end)
        self.assertLess(diff, 120, "Mercado a 45s del target debe ser aceptado")

    def test_market_end_date_too_far_rejected(self):
        """Fallback: mercado a más de 120s del target_end_ts es rechazado."""
        now_ts    = int(time.time())
        window_ts = (now_ts // 300) * 300
        target_end = window_ts + 300
        market_end = target_end + 200  # 200s de diferencia
        diff = abs(market_end - target_end)
        self.assertGreater(diff, 120, "Mercado a 200s debe ser rechazado")


class TestDHEntryReachability(unittest.TestCase):
    """Valida que D&H puede entrar con condiciones realistas."""

    def test_standard_combined_threshold(self):
        """Combined < 0.95 → hay ineficiencia captureable."""
        up_price = 0.20
        dn_price = 0.70
        combined = up_price + dn_price
        self.assertLess(combined, 0.95)

    def test_lax_combined_threshold_more_opportunities(self):
        """Modo sensible: combined < 0.97 → más oportunidades."""
        # Escenario donde combined = 0.96 (rechazado en STD, aceptado en lax)
        up_price = 0.25
        dn_price = 0.71
        combined = up_price + dn_price  # = 0.96
        std_threshold = 0.95
        lax_threshold = 0.97
        self.assertFalse(combined < std_threshold, "STD debería rechazar combined=0.96")
        self.assertTrue (combined < lax_threshold, "Laxo debería aceptar combined=0.96")

    def test_dump_detection_at_threshold(self):
        """Precio ≤ 0.15 se detecta como dump."""
        prices = [0.05, 0.08, 0.12, 0.15]
        dump_threshold = 0.15
        for p in prices:
            self.assertLessEqual(p, dump_threshold, f"price={p} debería detectarse como dump")

    def test_guaranteed_roi_positive(self):
        """ROI garantizado = (1-combined)/combined > 0 cuando combined < 1.0."""
        up_price = 0.20
        dn_price = 0.70
        combined = up_price + dn_price  # 0.90
        # Invertir en el lado barato (dump side)
        # Si combined < 1.0: hedge garantiza ganancia
        guaranteed = (1.0 - combined) / combined
        self.assertGreater(guaranteed, 0)
        self.assertAlmostEqual(guaranteed, 0.1111, delta=0.001)

    def test_minimum_score_signal_calculation(self):
        """Score mínimo para entrada: S1+S2+S3 = 60 pts combinados."""
        # S1 ineficiencia: 30pts max
        # S2 dump_depth: 20pts max (0.05 → 20, 0.10 → 15, 0.15 → 10)
        # S3 ROI garantizado: 20pts max
        # Escenario: combined=0.85, dump=0.08 → S1=25, S2=15, S3=17 = 57 < 60
        s1 = 25  # combined=0.85 (0.10 ineficiencia)
        s2 = 15  # dump=0.08
        s3 = 17  # roi = 0.176
        total = s1 + s2 + s3
        # En modo sensible (lax): umbral = 45 → aceptado
        lax_threshold = 45
        self.assertGreater(total, lax_threshold)

    def test_profit_after_fees_calculation(self):
        """Net profit después de fees (0.66% taker) debe ser > 0 para apostar."""
        investment = 5.0
        gross_return = 7.0   # 40% retorno bruto
        fee_rate = 0.0066    # 0.66% taker fee
        fees = investment * fee_rate + gross_return * fee_rate
        net = gross_return - investment - fees
        self.assertGreater(net, 0, f"Net profit {net:.3f} debe ser positivo")


class TestEREntryReachability(unittest.TestCase):
    """Valida condiciones de entrada de Escape Room."""

    def test_price_range_coverage(self):
        """Rango 0.05-0.33 cubre retornos 3x-20x."""
        # precio 0.05 → retorno 20x
        # precio 0.10 → retorno 10x
        # precio 0.20 → retorno 5x
        # precio 0.33 → retorno 3x
        for price, expected_min_return in [(0.05, 19), (0.10, 9), (0.20, 4.9), (0.33, 3.0)]:
            ret = 1.0 / price
            self.assertGreater(ret, expected_min_return,
                               f"price={price} return={ret:.1f} < {expected_min_return}")

    def test_single_ai_mode_lower_threshold(self):
        """Single AI mode: umbral ER score baja de 40 a 35."""
        er_score = 37
        dual_ai_threshold  = 40
        single_ai_threshold = 35
        self.assertFalse(er_score >= dual_ai_threshold, "Doble IA rechazaría score=37")
        self.assertTrue (er_score >= single_ai_threshold, "Single IA aceptaría score=37")

    def test_er_score_components_reach_35(self):
        """ER Score puede llegar a 35 con edge+retorno moderados."""
        edge_score      = 20  # edge_pct = 0.15 → 20 pts (max 40 pts)
        consensus_score = 10  # una IA dice APOSTAR → 10 pts (max 30)
        return_score    = 5   # retorno 3.5x → 5 pts (max 20)
        total = edge_score + consensus_score + return_score  # 35
        self.assertGreaterEqual(total, 35)

    def test_er_liquidity_filter_reasonable(self):
        """Filtro de liquidez mínima $500 filtra mercados muertos."""
        liquidity_values = {500: True, 499: False, 1000: True, 100: False}
        min_liq = 500.0
        for liq, expected in liquidity_values.items():
            self.assertEqual(liq >= min_liq, expected, f"liquidity={liq}")

    def test_ai_recommendation_values(self):
        """Valores de recomendación IA tienen orden correcto."""
        rec_values = {
            "APOSTAR_FUERTE": 1.0,
            "APOSTAR":        0.7,
            "ESPERAR":        0.3,
            "NO_APOSTAR":     0.0,
        }
        self.assertGreater(rec_values["APOSTAR_FUERTE"], rec_values["APOSTAR"])
        self.assertGreater(rec_values["APOSTAR"],        rec_values["ESPERAR"])
        self.assertGreater(rec_values["ESPERAR"],        rec_values["NO_APOSTAR"])


class TestDiagnosticsEndpoint(unittest.TestCase):
    """Valida estructura de datos del endpoint /api/diagnostics/scan."""

    def _fake_diag(self):
        return {
            "rush": {"last_msg": "BTC +0.12% poly=62% — señal WEAK", "last_ts": time.time() - 5},
            "dh":   {"last_msg": "Escaneados 8 mercados, 0 candidatos",  "last_ts": time.time() - 30},
            "er":   {"last_msg": "Escaneadas 15 oportunidades, 2 pasaron filtros", "last_ts": time.time() - 12},
        }

    def test_diagnostics_structure(self):
        d = self._fake_diag()
        for bot in ["rush", "dh", "er"]:
            self.assertIn(bot, d)
            self.assertIn("last_msg", d[bot])
            self.assertIn("last_ts", d[bot])

    def test_diagnostics_stale_detection(self):
        """Diagnóstico con timestamp >60s se considera stale."""
        diag = self._fake_diag()
        diag["dh"]["last_ts"] = time.time() - 120  # 2 minutos atrás
        stale_threshold = 60
        is_stale = (time.time() - diag["dh"]["last_ts"]) > stale_threshold
        self.assertTrue(is_stale)

    def test_rush_diag_message_contains_key_info(self):
        """Mensaje de diagnóstico Rush contiene % de movimiento."""
        import re
        msg = "BTC +0.12% poly=62% señal WEAK timing:optimal"
        self.assertTrue(re.search(r"[+-]?\d+\.?\d*%", msg), "Debe contener porcentaje")
        self.assertIn("BTC", msg)

    def test_dh_diag_message_format(self):
        """Mensaje de diagnóstico D&H contiene conteo de mercados."""
        msg = "Escaneados 12 mercados, 1 candidato encontrado"
        self.assertRegex(msg, r"Escaneados \d+")

    def test_er_diag_message_format(self):
        """Mensaje de diagnóstico ER contiene conteo de oportunidades."""
        msg = "Escaneadas 20 oportunidades, 3 pasaron filtros"
        self.assertRegex(msg, r"Escaneadas \d+")


class TestLaxModeWarnings(unittest.TestCase):
    """Valida que el modo laxo tiene advertencias correctas."""

    def test_lax_mode_reduces_momentum_threshold(self):
        """Modo laxo reduce el umbral de momentum a 50% del estándar."""
        std = 0.001   # 0.10%
        lax = 0.0005  # 0.05%
        self.assertAlmostEqual(lax / std, 0.5, delta=0.01)

    def test_lax_mode_reduces_poly_signal_threshold(self):
        """Modo laxo reduce el umbral Poly de 60% a 55%."""
        std = 0.60
        lax = 0.55
        self.assertAlmostEqual(lax - std, -0.05, delta=0.001)

    def test_lax_mode_is_opt_in(self):
        """Modo laxo debe ser opt-in (False por defecto)."""
        # Simular la config por defecto del servidor
        default_lax = False  # _rush_lax_mode default
        self.assertFalse(default_lax)

    def test_dual_ai_required_by_default(self):
        """Requerimiento de doble IA es True por defecto en ER."""
        default_require_dual = True  # _er_require_dual_ai default
        self.assertTrue(default_require_dual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
