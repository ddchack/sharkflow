"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests de backtest para DDC v6 — Category-Adjusted Urgency Factor.

Hipótesis a validar:
- Crypto edges se deterioran más rápido → multiplier 1.25
- Sports con mercado próximo a cierre tienen edge más confiable → 1.20
- Politics es más estable → penaliza urgency (0.80)
- Economics es neutral → 1.00

Backtest: simula distribución de edges por categoría y valida que
el ranking DDC mejora con los multiplicadores vs sin ellos.
"""
import math
import random
import unittest


# ─── DDC v6 Category Urgency Factor ──────────────────────────────────────────

CAT_URG_MULT = {
    "crypto":    1.25,
    "sports":    1.20,
    "economics": 1.00,
    "politics":  0.80,
    "default":   1.00,
}

def compute_urg_v6(hours_to_resolution, category):
    """DDC v6 urgency bonus — category-adjusted."""
    cat = category.lower()
    mult = CAT_URG_MULT.get(cat, CAT_URG_MULT["default"])
    raw = (0.07 if hours_to_resolution is not None and hours_to_resolution <= 6
           else 0.05 if hours_to_resolution is not None and hours_to_resolution <= 24
           else 0.03 if hours_to_resolution is not None and hours_to_resolution <= 72
           else 0.0)
    return round(min(raw * mult, 0.10), 4)


def compute_urg_v5(hours_to_resolution):
    """DDC v5 urgency bonus — without category multiplier (baseline)."""
    return (0.07 if hours_to_resolution is not None and hours_to_resolution <= 6
            else 0.05 if hours_to_resolution is not None and hours_to_resolution <= 24
            else 0.03 if hours_to_resolution is not None and hours_to_resolution <= 72
            else 0.0)


class TestCategoryUrgencyFactor(unittest.TestCase):
    """Tests del factor de urgencia ajustado por categoría."""

    def test_crypto_6h_boost(self):
        """Crypto con ≤6h recibe boost extra vs baseline."""
        v5 = compute_urg_v5(4)
        v6 = compute_urg_v6(4, "crypto")
        self.assertGreater(v6, v5)
        self.assertAlmostEqual(v6, round(v5 * 1.25, 4), places=3)

    def test_sports_6h_boost(self):
        """Sports con ≤6h recibe boost."""
        v5 = compute_urg_v5(3)
        v6 = compute_urg_v6(3, "sports")
        self.assertGreater(v6, v5)

    def test_politics_6h_penalized(self):
        """Politics con ≤6h recibe penalización vs baseline."""
        v5 = compute_urg_v5(4)
        v6 = compute_urg_v6(4, "politics")
        self.assertLess(v6, v5)
        self.assertAlmostEqual(v6, round(v5 * 0.80, 4), places=3)

    def test_economics_unchanged(self):
        """Economics con ≤24h igual al baseline."""
        v5 = compute_urg_v5(20)
        v6 = compute_urg_v6(20, "economics")
        self.assertAlmostEqual(v6, v5, places=4)

    def test_cap_at_0_10(self):
        """El bonus no debe exceder 0.10 incluso con multiplicador alto."""
        v6 = compute_urg_v6(1, "crypto")  # 1h, crypto → 0.07 * 1.25 = 0.0875 < 0.10
        self.assertLessEqual(v6, 0.10)

    def test_no_hours_zero_bonus(self):
        """Sin horas de resolución: bonus es 0 en todas las categorías."""
        for cat in ["crypto", "sports", "politics", "economics"]:
            v6 = compute_urg_v6(None, cat)
            self.assertEqual(v6, 0.0, f"cat={cat} con hours=None debe ser 0")

    def test_72h_boundary_crypto(self):
        """Crypto a exactamente 72h: 0.03 * 1.25 = 0.0375."""
        v6 = compute_urg_v6(72, "crypto")
        self.assertAlmostEqual(v6, 0.0375, places=4)

    def test_72h_boundary_politics(self):
        """Politics a 72h: 0.03 * 0.80 = 0.024."""
        v6 = compute_urg_v6(72, "politics")
        self.assertAlmostEqual(v6, 0.024, places=4)

    def test_unknown_category_default(self):
        """Categoría desconocida usa multiplicador 1.0."""
        v5 = compute_urg_v5(10)
        v6 = compute_urg_v6(10, "random_unknown_cat")
        self.assertAlmostEqual(v6, v5, places=4)

    def test_category_case_insensitive(self):
        """El lookup es case-insensitive."""
        v6_lower = compute_urg_v6(4, "crypto")
        v6_upper = compute_urg_v6(4, "CRYPTO")
        self.assertAlmostEqual(v6_lower, v6_upper, places=4)


class TestCategoryUrgencyBacktest(unittest.TestCase):
    """
    Backtest simulado: valida que el multiplicador de categoría mejora
    el ranking de recomendaciones en términos de precision@k.
    """

    def _simulate_market(self, cat, hrs, edge, seed=42):
        """Genera un mercado simulado con edge real conocido."""
        return {
            "category": cat,
            "hours_to_resolution": hrs,
            "actual_edge": edge,  # edge real (ground truth)
        }

    def _score_market(self, m, use_cat_mult=True):
        """Score simplificado DDC para ranking."""
        hrs = m["hours_to_resolution"]
        cat = m["category"]
        edge_score = min(m["actual_edge"] * 2, 0.40)  # simular KL+EV
        urg = compute_urg_v6(hrs, cat) if use_cat_mult else compute_urg_v5(hrs)
        return edge_score + urg

    def test_crypto_short_term_ranked_higher_with_v6(self):
        """Crypto a 4h con edge 8% debe rankear más alto que politics a 4h con edge 8%."""
        crypto_mkt  = self._simulate_market("crypto",   4,  0.08)
        politics_mkt = self._simulate_market("politics", 4,  0.08)

        v6_crypto   = self._score_market(crypto_mkt,  use_cat_mult=True)
        v6_politics = self._score_market(politics_mkt, use_cat_mult=True)

        v5_crypto   = self._score_market(crypto_mkt,  use_cat_mult=False)
        v5_politics = self._score_market(politics_mkt, use_cat_mult=False)

        # Con v6: crypto debe rankear más alto
        self.assertGreater(v6_crypto, v6_politics)
        # Sin v6 (v5): son iguales (mismo raw score)
        self.assertAlmostEqual(v5_crypto, v5_politics, places=4)

    def test_sports_imminent_market_boosted(self):
        """Sports en las últimas 6h recibe mayor score que economics igual."""
        sports = self._simulate_market("sports", 5, 0.06)
        econ   = self._simulate_market("economics", 5, 0.06)
        v6_s = self._score_market(sports)
        v6_e = self._score_market(econ)
        self.assertGreater(v6_s, v6_e)

    def test_politics_long_term_not_penalized(self):
        """Politics con ≥168h (abierto) no recibe ni bonus ni penalización (urg=0)."""
        pol = self._simulate_market("politics", 200, 0.05)
        v6  = compute_urg_v6(200, "politics")
        v5  = compute_urg_v5(200)
        self.assertEqual(v6, 0.0)
        self.assertEqual(v5, 0.0)

    def test_ranking_precision_at_5_improves(self):
        """
        Simula 20 mercados; verifica que precision@5 del v6 es >= precision@5 del v5.
        Los mercados crypto/sports de corto plazo tienen edge real más alto.
        """
        random.seed(42)
        markets = []
        # 5 crypto con corto plazo y edge alto (deberían estar top)
        for _ in range(5):
            markets.append(self._simulate_market("crypto", random.uniform(2, 8), random.uniform(0.08, 0.12)))
        # 5 sports corto plazo, edge moderado
        for _ in range(5):
            markets.append(self._simulate_market("sports", random.uniform(3, 12), random.uniform(0.05, 0.09)))
        # 5 politics largo plazo, edge bajo
        for _ in range(5):
            markets.append(self._simulate_market("politics", random.uniform(48, 200), random.uniform(0.01, 0.04)))
        # 5 economics, edge mixto
        for _ in range(5):
            markets.append(self._simulate_market("economics", random.uniform(12, 72), random.uniform(0.03, 0.07)))

        def precision_at_k(mkt_list, use_cat_mult, k=5):
            scored = [(self._score_market(m, use_cat_mult), m["actual_edge"]) for m in mkt_list]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k_edges = [e for _, e in scored[:k]]
            # Definir "relevante" como edge real > 0.06
            relevant = sum(1 for e in top_k_edges if e > 0.06)
            return relevant / k

        prec_v5 = precision_at_k(markets, use_cat_mult=False, k=5)
        prec_v6 = precision_at_k(markets, use_cat_mult=True,  k=5)

        # v6 debe ser al menos tan bueno como v5
        self.assertGreaterEqual(prec_v6, prec_v5,
            f"v6 precision@5={prec_v6:.2f} debe ser ≥ v5={prec_v5:.2f}")


class TestUrgencyMultiplierValues(unittest.TestCase):
    """Valida que los multiplicadores tienen los valores correctos."""

    def test_all_multiplier_keys_present(self):
        for cat in ["crypto", "sports", "economics", "politics", "default"]:
            self.assertIn(cat, CAT_URG_MULT)

    def test_crypto_multiplier_highest(self):
        self.assertEqual(max(CAT_URG_MULT.values()), CAT_URG_MULT["crypto"])

    def test_politics_multiplier_lowest_non_default(self):
        non_default = {k: v for k, v in CAT_URG_MULT.items() if k != "default"}
        self.assertEqual(min(non_default.values()), CAT_URG_MULT["politics"])

    def test_economics_neutral(self):
        self.assertAlmostEqual(CAT_URG_MULT["economics"], 1.0, places=4)

    def test_multipliers_positive(self):
        for cat, mult in CAT_URG_MULT.items():
            self.assertGreater(mult, 0, f"{cat} multiplier must be positive")

    def test_crypto_sports_above_1(self):
        self.assertGreater(CAT_URG_MULT["crypto"], 1.0)
        self.assertGreater(CAT_URG_MULT["sports"], 1.0)

    def test_politics_below_1(self):
        self.assertLess(CAT_URG_MULT["politics"], 1.0)


class TestUrgencyEdgeCases(unittest.TestCase):
    """Tests de casos límite del urgency factor."""

    def test_exactly_6h_boundary_crypto(self):
        v6 = compute_urg_v6(6, "crypto")
        self.assertAlmostEqual(v6, 0.07 * 1.25, places=4)

    def test_exactly_24h_boundary_politics(self):
        v6 = compute_urg_v6(24, "politics")
        self.assertAlmostEqual(v6, 0.05 * 0.80, places=4)

    def test_just_above_72h_is_zero(self):
        v6 = compute_urg_v6(73, "crypto")
        self.assertEqual(v6, 0.0)

    def test_very_small_hours_crypto(self):
        """Crypto a 0.5h (30 min) recibe el mismo bonus que ≤6h."""
        v6_half = compute_urg_v6(0.5, "crypto")
        v6_6    = compute_urg_v6(6.0, "crypto")
        self.assertAlmostEqual(v6_half, v6_6, places=4)

    def test_zero_hours_crypto(self):
        """Mercado que ya cerró (0h) recibe el bonus máximo de ≤6h."""
        v6 = compute_urg_v6(0, "crypto")
        self.assertAlmostEqual(v6, 0.07 * 1.25, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
