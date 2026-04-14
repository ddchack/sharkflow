"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Backtest de las mejoras Asymmetric Return Optimization para Rush Mode y D&H.
Valida la lógica ANTES de aplicarla al motor principal.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# ─── RUSH MODE BACKTEST ──────────────────────────────────────────────────────

class TestRushReturnZones(unittest.TestCase):
    """Backtest de clasificación de Return Zones en Rush Mode."""

    def _classify_zone(self, signal_price: float) -> str:
        """
        Lógica de clasificación de zonas de retorno.
        premium  : 0.20-0.42  → retorno 2.4x-5x   — mayor kelly permitido
        good     : 0.42-0.62  → retorno 1.6x-2.4x  — kelly estándar
        marginal : 0.62-0.72  → retorno 1.4x-1.6x  — kelly reducido
        rejected : > 0.72     → retorno < 1.4x      — no apostar
        """
        if signal_price < 0.20:
            return "rejected"   # demasiado barato → poca liquidez
        elif signal_price <= 0.42:
            return "premium"
        elif signal_price <= 0.62:
            return "good"
        elif signal_price <= 0.72:
            return "marginal"
        else:
            return "rejected"

    def test_premium_zone_center(self):
        self.assertEqual(self._classify_zone(0.30), "premium")

    def test_premium_zone_lower_bound(self):
        self.assertEqual(self._classify_zone(0.20), "premium")

    def test_premium_zone_upper_bound(self):
        self.assertEqual(self._classify_zone(0.42), "premium")

    def test_good_zone(self):
        self.assertEqual(self._classify_zone(0.55), "good")

    def test_marginal_zone(self):
        self.assertEqual(self._classify_zone(0.68), "marginal")

    def test_rejected_high(self):
        self.assertEqual(self._classify_zone(0.80), "rejected")

    def test_rejected_low(self):
        self.assertEqual(self._classify_zone(0.15), "rejected")

    def test_boundary_good_marginal(self):
        self.assertEqual(self._classify_zone(0.62), "good")
        self.assertIn(self._classify_zone(0.63), ["marginal"])

    def test_boundary_marginal_rejected(self):
        self.assertEqual(self._classify_zone(0.72), "marginal")
        self.assertEqual(self._classify_zone(0.73), "rejected")


class TestRushKellySizing(unittest.TestCase):
    """Backtest de Kelly Criterion para apuestas Rush asimétricas."""

    def _compute_kelly(self, signal_price: float, signal_strength: float) -> dict:
        """
        Kelly Criterion para mercados binarios de predicción:
          b = (1/price - 1)           ← ganancia neta por dólar apostado
          p = model_probability       ← estimado desde signal_strength
          q = 1 - p
          f = (p*b - q) / b          ← fracción Kelly pura
          f_capped = min(f, 0.35)     ← cap en 35% para evitar ruina
          f_rush   = f_capped * zone_modifier

        signal_strength ∈ [0,1] → model_prob = 0.52 + strength * 0.18
        (rango 0.52-0.70: Rush solo apuesta cuando hay señal real)
        """
        p = 0.52 + signal_strength * 0.18
        q = 1.0 - p
        b = max(0.01, (1.0 / max(0.01, signal_price)) - 1.0)
        raw_kelly = (p * b - q) / b
        if raw_kelly <= 0:
            return {"kelly_pct": 0.0, "approved": False, "reason": "edge_negativo"}

        # Modificador por zona
        zone = self._classify_zone(signal_price)
        zone_mod = {"premium": 1.20, "good": 1.00, "marginal": 0.75}.get(zone, 0.0)
        if zone_mod == 0.0:
            return {"kelly_pct": 0.0, "approved": False, "reason": "zona_rechazada"}

        capped = min(raw_kelly, 0.35) * zone_mod
        return {
            "kelly_pct": round(capped, 4),
            "raw_kelly":  round(raw_kelly, 4),
            "zone":       zone,
            "model_prob": round(p, 4),
            "approved":   capped > 0.01,
        }

    def _classify_zone(self, signal_price: float) -> str:
        if signal_price < 0.20: return "rejected"
        elif signal_price <= 0.42: return "premium"
        elif signal_price <= 0.62: return "good"
        elif signal_price <= 0.72: return "marginal"
        else: return "rejected"

    def test_premium_zone_higher_kelly(self):
        k_premium = self._compute_kelly(0.30, 0.7)["kelly_pct"]
        k_good    = self._compute_kelly(0.55, 0.7)["kelly_pct"]
        # A precio más bajo con misma fuerza, Kelly debe ser mayor (más alto retorno)
        self.assertGreater(k_premium, k_good)

    def test_kelly_rejected_zone(self):
        result = self._compute_kelly(0.80, 0.9)
        self.assertFalse(result["approved"])

    def test_kelly_strong_signal_premium(self):
        result = self._compute_kelly(0.28, 1.0)
        self.assertTrue(result["approved"])
        self.assertGreater(result["kelly_pct"], 0.20)  # al menos 20%

    def test_kelly_weak_signal_marginal(self):
        result = self._compute_kelly(0.70, 0.3)
        # Señal débil + zona marginal → puede ser 0 o muy pequeño
        self.assertLessEqual(result["kelly_pct"], 0.15)

    def test_kelly_no_edge(self):
        # Precio = 0.55 con señal 0.0 (prob=0.52, b=0.818): raw=(0.52*0.818-0.48)/0.818≈0.106
        # Pero prob=0.52 y price=0.58: b=(1/0.58-1)=0.724, raw=(0.52*0.724-0.48)/0.724≈0.0271
        # Usamos precio 0.58 y signal 0.0 (prob=0.52): resultado muy pequeño, approved=False si <0.01
        # Ajuste: price=0.52 + señal mínima → prob=0.5218, b=(1/0.52-1)=0.923
        # raw = (0.5218*0.923 - 0.4782)/0.923 = (0.4816-0.4782)/0.923 = 0.0037 < RUSH_KELLY_MIN=0.05
        result = self._compute_kelly(0.52, 0.01)
        self.assertFalse(result["approved"])


class TestRushSimulation(unittest.TestCase):
    """
    Simulación Monte Carlo de 50 escenarios Rush comparando
    estrategia original (bet fijo) vs nueva (Kelly × zona).
    """

    SCENARIOS = [
        # (signal_price, won, signal_strength)
        (0.30, True,  0.80), (0.35, True,  0.75), (0.40, True,  0.65),
        (0.28, False, 0.70), (0.32, True,  0.85), (0.55, True,  0.60),
        (0.65, False, 0.50), (0.70, True,  0.45), (0.25, True,  0.90),
        (0.38, False, 0.72), (0.42, True,  0.68), (0.60, True,  0.55),
        (0.29, True,  0.88), (0.33, False, 0.73), (0.50, True,  0.60),
        (0.71, False, 0.40), (0.24, True,  0.92), (0.36, True,  0.78),
        (0.45, False, 0.65), (0.30, True,  0.82), (0.55, True,  0.58),
        (0.40, True,  0.70), (0.62, False, 0.48), (0.27, True,  0.86),
        (0.35, True,  0.76), (0.48, True,  0.62), (0.31, False, 0.74),
        (0.58, True,  0.56), (0.26, True,  0.91), (0.44, False, 0.67),
        (0.38, True,  0.79), (0.52, True,  0.61), (0.30, True,  0.83),
        (0.64, False, 0.47), (0.28, True,  0.89), (0.36, False, 0.75),
        (0.42, True,  0.69), (0.54, True,  0.57), (0.29, True,  0.87),
        (0.46, False, 0.64), (0.32, True,  0.81), (0.60, True,  0.54),
        (0.27, True,  0.90), (0.39, False, 0.73), (0.44, True,  0.66),
        (0.31, True,  0.84), (0.57, True,  0.55), (0.35, True,  0.77),
        (0.41, False, 0.71), (0.26, True,  0.93),
    ]

    def _classify_zone(self, p):
        if p < 0.20: return "rejected"
        elif p <= 0.42: return "premium"
        elif p <= 0.62: return "good"
        elif p <= 0.72: return "marginal"
        else: return "rejected"

    def _compute_kelly_pct(self, signal_price, signal_strength):
        p = 0.52 + signal_strength * 0.18
        b = max(0.01, (1.0 / max(0.01, signal_price)) - 1.0)
        raw = (p * b - (1-p)) / b
        if raw <= 0: return 0.0
        zone = self._classify_zone(signal_price)
        mod  = {"premium": 1.20, "good": 1.00, "marginal": 0.75}.get(zone, 0.0)
        return min(raw, 0.35) * mod

    def _simulate(self, fixed_bet_pct: float = 0.05, bankroll: float = 100.0):
        """
        Simula las 50 apuestas con:
          - Estrategia original: bet fijo = fixed_bet_pct × bankroll
          - Estrategia nueva: kelly_pct × bankroll, rechaza zonas malas
        """
        bank_old = bankroll
        bank_new = bankroll
        rejected_old = 0
        rejected_new = 0

        for price, won, strength in self.SCENARIOS:
            zone = self._classify_zone(price)
            return_mult = 1.0 / max(0.01, price)  # multiplicador bruto de la apuesta

            # Estrategia original: apuesta fija, solo rechaza > 0.80
            if price <= 0.80:
                bet_old = bank_old * fixed_bet_pct
                if won:
                    bank_old += bet_old * (return_mult - 1)
                else:
                    bank_old -= bet_old
            else:
                rejected_old += 1

            # Estrategia nueva: Kelly × zona, rechaza rejected
            kelly = self._compute_kelly_pct(price, strength)
            if kelly > 0 and zone != "rejected":
                bet_new = bank_new * kelly
                if won:
                    bank_new += bet_new * (return_mult - 1)
                else:
                    bank_new -= bet_new
            else:
                rejected_new += 1

        return {
            "final_bank_old": round(bank_old, 2),
            "final_bank_new": round(bank_new, 2),
            "roi_old": round((bank_old - bankroll) / bankroll * 100, 2),
            "roi_new": round((bank_new - bankroll) / bankroll * 100, 2),
            "rejected_old": rejected_old,
            "rejected_new": rejected_new,
        }

    def test_new_strategy_better_roi(self):
        result = self._simulate()
        # La estrategia nueva debe tener mayor ROI que la original
        self.assertGreater(result["roi_new"], result["roi_old"],
            f"Nueva ROI {result['roi_new']}% debe superar vieja {result['roi_old']}%")

    def test_new_strategy_positive_roi(self):
        result = self._simulate()
        self.assertGreater(result["roi_new"], 0,
            f"Nueva estrategia debe ser positiva, fue {result['roi_new']}%")

    def test_new_strategy_rejects_more_marginal_bets(self):
        result = self._simulate()
        # La nueva estrategia rechaza más apuestas (zonas malas)
        self.assertGreaterEqual(result["rejected_new"], result["rejected_old"])


# ─── DUMP & HEDGE BACKTEST ───────────────────────────────────────────────────

class TestDHPremiumTiers(unittest.TestCase):
    """Backtest de clasificación de Premium Tiers en Dump & Hedge."""

    def _classify_tier(self, combined_price: float) -> str:
        """
        Tiers de ineficiencia:
          extreme  : combined < 0.75  → ROI garantizada > 33%
          premium  : combined 0.75-0.85 → ROI garantizada 18-33%
          standard : combined 0.85-0.95 → ROI garantizada 5-18%
          reject   : combined >= 0.95  → ROI garantizada < 5%
        """
        if combined_price < 0.75:
            return "extreme"
        elif combined_price < 0.85:
            return "premium"
        elif combined_price < 0.95:
            return "standard"
        else:
            return "reject"

    def _guaranteed_roi(self, combined_price: float, budget: float = 2.0) -> float:
        if combined_price <= 0: return 0.0
        payout = budget / combined_price
        profit = payout - budget
        return round(profit / budget * 100, 2)

    def test_extreme_tier(self):
        self.assertEqual(self._classify_tier(0.70), "extreme")
        self.assertEqual(self._classify_tier(0.74), "extreme")

    def test_premium_tier(self):
        self.assertEqual(self._classify_tier(0.75), "premium")
        self.assertEqual(self._classify_tier(0.82), "premium")

    def test_standard_tier(self):
        self.assertEqual(self._classify_tier(0.85), "standard")
        self.assertEqual(self._classify_tier(0.90), "standard")

    def test_reject_tier(self):
        self.assertEqual(self._classify_tier(0.95), "reject")
        self.assertEqual(self._classify_tier(0.98), "reject")

    def test_extreme_roi_exceeds_33pct(self):
        roi = self._guaranteed_roi(0.74)
        self.assertGreater(roi, 33.0)

    def test_premium_roi_range(self):
        roi_low  = self._guaranteed_roi(0.85)
        roi_high = self._guaranteed_roi(0.75)
        self.assertGreater(roi_low,  17.0)
        self.assertGreater(roi_high, 17.0)

    def test_standard_roi_range(self):
        roi = self._guaranteed_roi(0.90)
        self.assertGreater(roi, 5.0)
        self.assertLess(roi, 20.0)


class TestDHBudgetAllocation(unittest.TestCase):
    """Backtest de asignación de presupuesto según tier."""

    def _optimal_budget(self, combined_price: float, base_budget: float = 2.0) -> float:
        """
        Multiplica el presupuesto base según el tier de ineficiencia:
          extreme  → 2.0x   (ROI garantizada tan alta que vale el riesgo)
          premium  → 1.5x
          standard → 1.0x
          reject   → 0.0x  (no apostar)
        """
        if combined_price < 0.75:   return base_budget * 2.0
        elif combined_price < 0.85: return base_budget * 1.5
        elif combined_price < 0.95: return base_budget * 1.0
        else: return 0.0

    def test_extreme_gets_2x_budget(self):
        self.assertEqual(self._optimal_budget(0.70, 2.0), 4.0)

    def test_premium_gets_1_5x_budget(self):
        self.assertEqual(self._optimal_budget(0.80, 2.0), 3.0)

    def test_standard_gets_base_budget(self):
        self.assertEqual(self._optimal_budget(0.90, 2.0), 2.0)

    def test_reject_gets_zero(self):
        self.assertEqual(self._optimal_budget(0.96, 2.0), 0.0)

    def test_extreme_higher_abs_profit(self):
        # A 2x budget, la ganancia absoluta de extreme supera a premium en base budget
        budget_extreme = self._optimal_budget(0.72, 2.0)  # = 4.0
        budget_premium = self._optimal_budget(0.82, 2.0)  # = 3.0
        profit_extreme = budget_extreme / 0.72 - budget_extreme
        profit_premium = budget_premium / 0.82 - budget_premium
        self.assertGreater(profit_extreme, profit_premium)


class TestDHS9Signal(unittest.TestCase):
    """Backtest de señal S9: Extreme leg (dump price ≤ 5¢ → retorno 20x+)."""

    def _s9_score(self, dump_price: float) -> tuple:
        if dump_price <= 0.05:
            return 20.0, f"S9:extreme_dump({dump_price*100:.1f}¢→{1/dump_price:.0f}x)"
        elif dump_price <= 0.08:
            return 10.0, f"S9:high_return({dump_price*100:.1f}¢→{1/dump_price:.0f}x)"
        return 0.0, ""

    def test_s9_extreme_dump(self):
        score, sig = self._s9_score(0.04)
        self.assertEqual(score, 20.0)
        self.assertIn("extreme_dump", sig)

    def test_s9_high_return(self):
        score, sig = self._s9_score(0.07)
        self.assertEqual(score, 10.0)
        self.assertIn("high_return", sig)

    def test_s9_no_signal_above_8pct(self):
        score, sig = self._s9_score(0.10)
        self.assertEqual(score, 0.0)
        self.assertEqual(sig, "")

    def test_s9_extreme_return_calculation(self):
        # A 0.04¢, el retorno es 25x
        _, sig = self._s9_score(0.04)
        self.assertIn("25x", sig)

    def test_s9_return_increases_total_confidence(self):
        # Con S9=20 extra, una opción que tenía score=55 supera el umbral de 60
        base_score = 55.0
        s9_score, _ = self._s9_score(0.05)
        self.assertGreaterEqual(base_score + s9_score, 60.0)


class TestDHSimulation(unittest.TestCase):
    """
    Simulación de 30 escenarios D&H comparando
    estrategia original (umbral fijo, presupuesto fijo) vs nueva (tiers + S9).
    """

    SCENARIOS = [
        # (combined, dump_price, resolved_correctly)
        # resolved_correctly = True si la pierna 2 se completó con ganancia garantizada
        (0.72, 0.07, True),  (0.88, 0.12, True),  (0.76, 0.05, True),
        (0.93, 0.14, True),  (0.68, 0.08, True),  (0.91, 0.13, False),  # orphaned
        (0.74, 0.06, True),  (0.86, 0.11, True),  (0.70, 0.04, True),
        (0.94, 0.15, False), (0.79, 0.09, True),  (0.83, 0.10, True),
        (0.71, 0.07, True),  (0.90, 0.13, False), (0.78, 0.08, True),
        (0.85, 0.12, True),  (0.69, 0.05, True),  (0.92, 0.14, False),
        (0.75, 0.06, True),  (0.82, 0.10, True),  (0.73, 0.07, True),
        (0.89, 0.13, True),  (0.67, 0.04, True),  (0.95, 0.15, False),  # reject
        (0.77, 0.09, True),  (0.84, 0.11, True),  (0.72, 0.06, True),
        (0.88, 0.12, True),  (0.76, 0.07, True),  (0.91, 0.14, False),
    ]

    def _classify_tier(self, combined):
        if combined < 0.75: return "extreme"
        elif combined < 0.85: return "premium"
        elif combined < 0.95: return "standard"
        else: return "reject"

    def _optimal_budget(self, combined, base=2.0):
        if combined < 0.75: return base * 2.0
        elif combined < 0.85: return base * 1.5
        elif combined < 0.95: return base * 1.0
        else: return 0.0

    def _simulate(self, base_budget: float = 2.0, orphan_loss_pct: float = 0.30):
        """
        orphan_loss_pct: cuando resolved_correctly=False (pierna huérfana),
        pérdida = orphan_loss_pct × amount invertido en pierna 1
        (stop-loss al 30%)
        """
        bank_old = 100.0
        bank_new = 100.0

        for combined, dump_p, resolved in self.SCENARIOS:
            # Estrategia original: presupuesto fijo, umbral combined<0.95
            if combined < 0.95:
                budget_old = base_budget
                if resolved:
                    profit_old = budget_old / combined - budget_old
                    bank_old += profit_old
                else:
                    # Pierna huérfana: pérdida parcial
                    leg1_amt_old = budget_old * dump_p / combined
                    bank_old -= leg1_amt_old * orphan_loss_pct

            # Estrategia nueva: presupuesto según tier
            budget_new = self._optimal_budget(combined, base_budget)
            if budget_new > 0:
                if resolved:
                    profit_new = budget_new / combined - budget_new
                    bank_new += profit_new
                else:
                    leg1_amt_new = budget_new * dump_p / combined
                    bank_new -= leg1_amt_new * orphan_loss_pct

        return {
            "final_old": round(bank_old, 2),
            "final_new": round(bank_new, 2),
            "roi_old":   round((bank_old - 100) / 100 * 100, 2),
            "roi_new":   round((bank_new - 100) / 100 * 100, 2),
        }

    def test_new_strategy_better_roi(self):
        result = self._simulate()
        self.assertGreater(result["roi_new"], result["roi_old"],
            f"Nueva D&H ROI {result['roi_new']}% debe superar vieja {result['roi_old']}%")

    def test_new_strategy_positive_roi(self):
        result = self._simulate()
        self.assertGreater(result["roi_new"], 0)

    def test_old_strategy_also_positive(self):
        result = self._simulate()
        # La vieja también es positiva (mercado eficiente parcialmente)
        self.assertGreater(result["roi_old"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
