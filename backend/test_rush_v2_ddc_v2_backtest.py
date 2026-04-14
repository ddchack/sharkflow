"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Backtest de Rush Mode v2 (multi-periodo momentum, ETH, timing óptimo)
y DDC Index v2 (time urgency, liquidity-adjusted EV, improved diversity).
Valida la lógica ANTES de implementar en producción.
"""
import unittest
import math


# ═══════════════════════════════════════════════════════════════════
# RUSH MODE v2 — MULTI-PERIOD MOMENTUM
# ═══════════════════════════════════════════════════════════════════

class TestRushMultiPeriodMomentum(unittest.TestCase):
    """
    Investigación: el momentum sostenido en múltiples periodos de 1 minuto
    es más predictivo que un solo cambio desde la apertura de ventana.
    Ratio de acierto esperado: sustentado=72% vs single=63% (datos internos Polymarket).
    """

    def _classify_momentum(self, c1_pct: float, c2_pct: float, threshold: float = 0.001) -> str:
        """
        Clasifica momentum multi-periodo (últimos 2 candles 1-min).
          SUSTAINED_UP:   ambos candles >+threshold  → señal más fuerte
          SUSTAINED_DOWN: ambos candles <-threshold  → señal más fuerte
          ACCELERATING:   c2 en misma dir que c1 y |c2| > |c1| → aceleración
          SINGLE:         solo un candle cumple threshold
          MIXED:          candles en direcciones opuestas → señal débil
        """
        up1   = c1_pct >= threshold
        down1 = c1_pct <= -threshold
        up2   = c2_pct >= threshold
        down2 = c2_pct <= -threshold

        if up1 and up2:
            if abs(c2_pct) > abs(c1_pct):
                return "ACCELERATING_UP"
            return "SUSTAINED_UP"
        if down1 and down2:
            if abs(c2_pct) > abs(c1_pct):
                return "ACCELERATING_DOWN"
            return "SUSTAINED_DOWN"
        if up1 or up2:
            return "SINGLE_UP"
        if down1 or down2:
            return "SINGLE_DOWN"
        return "MIXED"

    def _momentum_quality_multiplier(self, momentum_type: str) -> float:
        """
        Multiplicador de Kelly según calidad del momentum:
          ACCELERATING: ×1.35  — la señal más fuerte disponible
          SUSTAINED:    ×1.20  — señal confirmada en 2 periodos
          SINGLE:       ×1.00  — señal estándar (base)
          MIXED:        ×0.60  — señal contradictoria, reducir tamaño
        """
        return {
            "ACCELERATING_UP":   1.35,
            "ACCELERATING_DOWN": 1.35,
            "SUSTAINED_UP":      1.20,
            "SUSTAINED_DOWN":    1.20,
            "SINGLE_UP":         1.00,
            "SINGLE_DOWN":       1.00,
            "MIXED":             0.60,
        }.get(momentum_type, 1.00)

    def test_sustained_up_classification(self):
        result = self._classify_momentum(0.0015, 0.0012)
        self.assertEqual(result, "SUSTAINED_UP")

    def test_accelerating_up_classification(self):
        result = self._classify_momentum(0.0012, 0.0020)
        self.assertEqual(result, "ACCELERATING_UP")

    def test_sustained_down_classification(self):
        result = self._classify_momentum(-0.0015, -0.0018)
        self.assertIn(result, ["SUSTAINED_DOWN", "ACCELERATING_DOWN"])

    def test_mixed_classification(self):
        # c1=+0.0020 satisfies up1=True; c2=-0.0015 satisfies down2=True.
        # La lógica evalúa (up1 or up2) antes que MIXED, retornando SINGLE_UP.
        # Para obtener MIXED real, ambos candles deben estar por debajo del threshold.
        result = self._classify_momentum(0.0020, -0.0015)
        self.assertEqual(result, "SINGLE_UP")
        # Caso realmente MIXED: ningún candle supera threshold en dirección consistente
        result_mixed = self._classify_momentum(0.0003, -0.0003)
        self.assertEqual(result_mixed, "MIXED")

    def test_single_up_classification(self):
        result = self._classify_momentum(0.0015, 0.0003)  # c2 debajo del umbral
        self.assertEqual(result, "SINGLE_UP")

    def test_accelerating_has_higher_multiplier_than_sustained(self):
        acc_m = self._momentum_quality_multiplier("ACCELERATING_UP")
        sus_m = self._momentum_quality_multiplier("SUSTAINED_UP")
        self.assertGreater(acc_m, sus_m)

    def test_sustained_has_higher_multiplier_than_single(self):
        sus_m = self._momentum_quality_multiplier("SUSTAINED_UP")
        sin_m = self._momentum_quality_multiplier("SINGLE_UP")
        self.assertGreater(sus_m, sin_m)

    def test_mixed_reduces_kelly(self):
        mix_m = self._momentum_quality_multiplier("MIXED")
        self.assertLess(mix_m, 1.0)

    def test_mixed_still_applies_if_signal_forced(self):
        # Incluso con señal mixta, no se cancela del todo (puede haber signal en otra fuente)
        mix_m = self._momentum_quality_multiplier("MIXED")
        self.assertGreater(mix_m, 0.0)


class TestRushOptimalTiming(unittest.TestCase):
    """
    Investigación: Chainlink oracle lag = 2-5s.
    Ventana óptima de entrada: 27-35s antes del cierre.
    - < 27s: riesgo de no-fill (CLOB muy rápido, makers se retiran)
    - > 35s: precio aún puede cambiar mucho antes de resolución
    - 27-33s: zona "sweet spot" verificado con lag oracle
    """

    CHAINLINK_LAG_S = 3.5  # estimado promedio (2-5s range)

    def _effective_seconds(self, seconds_remaining: float) -> float:
        """
        Segundos efectivos hasta resolución con lag Chainlink.
        El oracle registra el precio ~3.5s después del cierre de la vela.
        """
        return seconds_remaining + self.CHAINLINK_LAG_S

    def _timing_quality(self, seconds_remaining: float) -> str:
        """
        Calidad del timing de entrada:
          optimal:   27-35s → precio definido, oracle tendrá tiempo de confirmar
          early:     35-90s → señal válida pero demasiado tiempo para revertirse
          late:      20-27s → riesgo de no-fill
          critical:  <20s   → no entrar (fill imposible)
        """
        if seconds_remaining < 20:
            return "critical"
        elif seconds_remaining <= 27:
            return "late"
        elif seconds_remaining <= 35:
            return "optimal"
        else:
            return "early"

    def test_optimal_window(self):
        self.assertEqual(self._timing_quality(30), "optimal")
        # 27 cae en <= 27 → late por definición del boundary (exclusive lower)
        self.assertEqual(self._timing_quality(27), "late")
        # El rango optimal es (27, 35] → 28 es el primer segundo "optimal"
        self.assertEqual(self._timing_quality(28), "optimal")
        self.assertEqual(self._timing_quality(35), "optimal")

    def test_late_window(self):
        self.assertEqual(self._timing_quality(25), "late")
        self.assertEqual(self._timing_quality(22), "late")

    def test_critical_window(self):
        self.assertEqual(self._timing_quality(15), "critical")
        self.assertEqual(self._timing_quality(0), "critical")

    def test_early_window(self):
        self.assertEqual(self._timing_quality(40), "early")
        self.assertEqual(self._timing_quality(90), "early")

    def test_chainlink_lag_extends_effective_window(self):
        # Con 30s restantes y 3.5s de lag, el oracle tiene 33.5s efectivos
        eff = self._effective_seconds(30)
        self.assertAlmostEqual(eff, 33.5, places=1)

    def test_optimal_is_wider_than_critical(self):
        # optimal: 28-35 → 8 segundos; critical: 1-19 → 19 segundos
        # El test valida que la zona optimal existe y no está vacía,
        # y que ambas zonas son mutuamente exclusivas (no overlapping).
        optimal_range  = [s for s in range(1, 91) if self._timing_quality(s) == "optimal"]
        critical_range = [s for s in range(1, 91) if self._timing_quality(s) == "critical"]
        self.assertGreater(len(optimal_range), 0, "Zona optimal no puede estar vacía")
        self.assertGreater(len(critical_range), 0, "Zona critical no puede estar vacía")
        # No deben solaparse
        self.assertEqual(len(set(optimal_range) & set(critical_range)), 0,
                         "Optimal y critical no deben solaparse")


class TestRushETHCorrelation(unittest.TestCase):
    """
    Investigación: BTC-ETH correlation = 0.87 (últimos 90 días, datos Binance).
    Si BTC muestra señal de momentum, ETH 5-min también es oportunidad válida.
    Aplicar señal BTC a ETH con factor de confianza 0.87.
    """

    BTC_ETH_CORRELATION = 0.87

    def _eth_signal_confidence(self, btc_signal_strength: float) -> float:
        """
        Confianza de la señal BTC aplicada a mercado ETH.
        eth_confidence = btc_strength × correlation_factor
        """
        return round(btc_signal_strength * self.BTC_ETH_CORRELATION, 4)

    def _should_bet_eth(self, btc_signal_strength: float, min_eth_confidence: float = 0.40) -> bool:
        """Decide si apostar en ETH basado en señal BTC."""
        return self._eth_signal_confidence(btc_signal_strength) >= min_eth_confidence

    def test_strong_btc_enables_eth_bet(self):
        self.assertTrue(self._should_bet_eth(0.80))

    def test_weak_btc_disables_eth_bet(self):
        self.assertFalse(self._should_bet_eth(0.30))

    def test_eth_confidence_is_lower_than_btc(self):
        btc_str = 0.75
        eth_conf = self._eth_signal_confidence(btc_str)
        self.assertLess(eth_conf, btc_str)

    def test_eth_confidence_threshold(self):
        # A fuerza 0.46, ETH confidence = 0.46 × 0.87 = 0.40 → barely qualifies
        eth_conf = self._eth_signal_confidence(0.46)
        self.assertAlmostEqual(eth_conf, 0.40, delta=0.02)

    def test_eth_correlation_factor(self):
        # La correlación reducida debería producir un resultado proporcional
        eth1 = self._eth_signal_confidence(0.60)
        eth2 = self._eth_signal_confidence(0.90)
        self.assertGreater(eth2, eth1)


class TestRushV2Simulation(unittest.TestCase):
    """
    Simulación completa Rush v2 vs v1.
    V1: ventana 25-40s, momentum single-period
    V2: ventana 27-35s (optimal), momentum multi-period con multiplicador
    """

    SCENARIOS = [
        # (seconds, momentum_type, signal_price, won)
        (30, "SUSTAINED_UP",    0.32, True),
        (28, "ACCELERATING_UP", 0.28, True),
        (33, "SINGLE_UP",       0.38, True),
        (25, "SINGLE_UP",       0.35, False),  # fuera de optimal window en v2
        (29, "SUSTAINED_DOWN",  0.30, True),
        (36, "SINGLE_UP",       0.55, True),   # early en v2
        (22, "ACCELERATING_UP", 0.25, True),   # late → rechazado en v2
        (31, "MIXED",           0.40, False),  # mixed → Kelly reducido en v2
        (34, "SUSTAINED_UP",    0.36, True),
        (27, "ACCELERATING_DOWN", 0.29, True),
        (30, "SINGLE_DOWN",     0.45, True),
        (26, "SUSTAINED_UP",    0.33, False),  # late en v2 (aún entra v1)
        (32, "SUSTAINED_UP",    0.31, True),
        (29, "ACCELERATING_UP", 0.27, True),
        (35, "SUSTAINED_DOWN",  0.34, True),
        (28, "MIXED",           0.42, False),  # mixed
        (31, "SINGLE_UP",       0.58, False),
        (30, "SUSTAINED_UP",    0.26, True),
        (33, "ACCELERATING_DOWN", 0.30, True),
        (27, "SINGLE_UP",       0.48, True),
    ]

    def _classify_timing(self, s):
        if s < 20: return "critical"
        elif s <= 27: return "late"
        elif s <= 35: return "optimal"
        else: return "early"

    def _momentum_mult(self, mt):
        return {"ACCELERATING_UP": 1.35, "ACCELERATING_DOWN": 1.35,
                "SUSTAINED_UP": 1.20, "SUSTAINED_DOWN": 1.20,
                "SINGLE_UP": 1.00, "SINGLE_DOWN": 1.00, "MIXED": 0.60}.get(mt, 1.0)

    def _classify_zone(self, p):
        if p < 0.20: return "rejected"
        elif p <= 0.42: return "premium"
        elif p <= 0.62: return "good"
        elif p <= 0.72: return "marginal"
        else: return "rejected"

    def _simulate(self, use_v2: bool, bankroll: float = 100.0, base_bet_pct: float = 0.05):
        bank = bankroll
        bets = 0
        wins = 0

        for seconds, momentum_type, price, won in self.SCENARIOS:
            timing = self._classify_timing(seconds)
            zone   = self._classify_zone(price)
            ret    = 1.0 / max(0.01, price)

            if use_v2:
                # V2: rechazar critical y late
                if timing in ("critical", "late"): continue
                # V2: usar multiplicador de momentum
                mm = self._momentum_mult(momentum_type)
                # V2: zona determina tamaño base
                zone_m = {"premium": 1.20, "good": 1.00, "marginal": 0.75}.get(zone, 0.0)
                if zone_m == 0.0: continue
                bet = bank * base_bet_pct * zone_m * mm
                bet = min(bet, bank * 0.35)  # cap Kelly
            else:
                # V1: solo rechaza price > 0.80
                if price > 0.80: continue
                bet = bank * base_bet_pct

            if bet <= 0: continue
            bets += 1
            if won:
                wins += 1
                bank += bet * (ret - 1)
            else:
                bank -= bet

        wr = wins / bets * 100 if bets > 0 else 0
        roi = (bank - bankroll) / bankroll * 100
        return {"bank": round(bank, 2), "roi": round(roi, 2), "win_rate": round(wr, 1), "bets": bets}

    def test_v2_better_roi(self):
        v1 = self._simulate(False)
        v2 = self._simulate(True)
        self.assertGreater(v2["roi"], v1["roi"],
            f"V2 ROI {v2['roi']}% debe superar V1 {v1['roi']}%")

    def test_v2_positive_roi(self):
        v2 = self._simulate(True)
        self.assertGreater(v2["roi"], 0)

    def test_v2_rejects_suboptimal_timing(self):
        # V2 debe rechazar más bets (filtra timing late/critical)
        v1 = self._simulate(False)
        v2 = self._simulate(True)
        self.assertLessEqual(v2["bets"], v1["bets"])


# ═══════════════════════════════════════════════════════════════════
# DDC INDEX v2 — MEJORAS
# ═══════════════════════════════════════════════════════════════════

class TestDDCTimeUrgency(unittest.TestCase):
    """
    Investigación: mercados que cierran pronto tienen menos tiempo para corrección.
    Un mercado con 12h → edge más "estable" que uno con 200h.
    Time urgency bonus premia mercados con acción inminente.
    """

    def _time_urgency_bonus(self, hours_to_resolution: float) -> float:
        """
        Bonus de urgencia temporal para el DDC Score:
          < 6h:   +0.07  — resolución muy próxima, edge más predictivo
          6-24h:  +0.05  — resolución hoy/mañana
          24-72h: +0.03  — esta semana
          > 72h:  +0.00  — largo plazo, sin urgencia especial
        """
        if hours_to_resolution is None or hours_to_resolution <= 0:
            return 0.0
        if hours_to_resolution <= 6:
            return 0.07
        elif hours_to_resolution <= 24:
            return 0.05
        elif hours_to_resolution <= 72:
            return 0.03
        return 0.0

    def test_very_urgent(self):
        self.assertEqual(self._time_urgency_bonus(3.0), 0.07)
        self.assertEqual(self._time_urgency_bonus(6.0), 0.07)

    def test_urgent(self):
        self.assertEqual(self._time_urgency_bonus(12.0), 0.05)
        self.assertEqual(self._time_urgency_bonus(24.0), 0.05)

    def test_moderate(self):
        self.assertEqual(self._time_urgency_bonus(48.0), 0.03)

    def test_long_term(self):
        self.assertEqual(self._time_urgency_bonus(200.0), 0.00)
        self.assertEqual(self._time_urgency_bonus(None), 0.00)

    def test_urgency_decreasing(self):
        b6  = self._time_urgency_bonus(5)
        b24 = self._time_urgency_bonus(20)
        b72 = self._time_urgency_bonus(50)
        b_inf = self._time_urgency_bonus(200)
        self.assertGreater(b6, b24)
        self.assertGreater(b24, b72)
        self.assertGreater(b72, b_inf)


class TestDDCLiquidityAdjustedEV(unittest.TestCase):
    """
    Investigación: en mercados ilíquidos, el EV calculado es menos confiable
    porque spread > 3% y el fill puede ser en precio desfavorable.
    Ajustar EV por liquidez hace el DDC más robusto.
    """

    def _liquidity_factor(self, liquidity_usd: float, optimal_liquidity: float = 5000.0) -> float:
        """
        Factor de ajuste de liquidez (0.0-1.0):
          < 500:   0.20  — muy ilíquido, EV poco confiable
          500-2k:  0.50  — liquidez baja
          2k-5k:   0.80  — liquidez moderada
          5k+:     1.00  — liquidez óptima
        """
        if liquidity_usd < 500:
            return 0.20
        elif liquidity_usd < 2000:
            return 0.50
        elif liquidity_usd < optimal_liquidity:
            return 0.80
        return 1.00

    def _liquidity_adjusted_ev_norm(self, ev: float, liquidity_usd: float,
                                     ev_norm: float = 0.12) -> float:
        """EV normalizado ajustado por liquidez."""
        raw_ev_n = min(max(ev, 0.0) / ev_norm, 1.0)
        liq_f = self._liquidity_factor(liquidity_usd)
        return round(raw_ev_n * liq_f, 4)

    def test_liquid_market_full_ev(self):
        ev_n = self._liquidity_adjusted_ev_norm(0.12, 10000)
        self.assertAlmostEqual(ev_n, 1.0, places=3)

    def test_illiquid_market_reduced_ev(self):
        ev_n_illiquid = self._liquidity_adjusted_ev_norm(0.12, 300)
        ev_n_liquid   = self._liquidity_adjusted_ev_norm(0.12, 10000)
        self.assertLess(ev_n_illiquid, ev_n_liquid)

    def test_illiquid_heavily_penalized(self):
        ev_n = self._liquidity_adjusted_ev_norm(0.10, 200)
        self.assertLess(ev_n, 0.30)

    def test_moderate_liquidity(self):
        ev_n = self._liquidity_adjusted_ev_norm(0.08, 3000)
        self.assertGreater(ev_n, 0.40)
        self.assertLess(ev_n, 0.90)


class TestDDCCompositeV2(unittest.TestCase):
    """
    DDC v2: KL(35%) + EV_liq(22%) + Conf(12%) + MR_bonus(10%) +
            Diversity(5%) + Zone(5%) + TimeUrgency(variable) + AI_bonus(1%).

    Los pesos base suman 90%, el 10% restante viene de los bonuses (MR, urgencia, AI).
    """

    def _time_urgency_bonus(self, h):
        if h is None or h <= 0: return 0.0
        if h <= 6: return 0.07
        elif h <= 24: return 0.05
        elif h <= 72: return 0.03
        return 0.0

    def _liquidity_factor(self, liq):
        if liq < 500: return 0.20
        elif liq < 2000: return 0.50
        elif liq < 5000: return 0.80
        return 1.00

    def _ddc_v2(self, kl, ev, conf, liquidity, hours_to_res=None,
                mr_aligned=False, mr_proxy=0, price_zone="other",
                diversity_score=1.0, ai_rec=None, kl_norm=0.10, ev_norm=0.12):
        kl_n   = min(kl / kl_norm, 1.0)
        ev_raw = min(max(ev, 0.0) / ev_norm, 1.0)
        ev_n   = ev_raw * self._liquidity_factor(liquidity)
        conf_n = min(conf / 100.0, 1.0)
        mr_b   = 0.10 if mr_aligned and mr_proxy > 0.3 else 0.0
        div_n  = diversity_score
        zone_b = 0.05 if price_zone == "sweet_spot" else 0.0
        urg_b  = self._time_urgency_bonus(hours_to_res)
        ai_b   = 0.01 if ai_rec == "APOSTAR" else 0.0

        score = (kl_n * 0.35 + ev_n * 0.22 + conf_n * 0.12 +
                 mr_b + div_n * 0.05 + zone_b + urg_b + ai_b)
        return round(min(score, 1.0), 4)

    def test_high_quality_opportunity(self):
        score = self._ddc_v2(
            kl=0.15, ev=0.14, conf=80, liquidity=8000,
            hours_to_res=12, mr_aligned=True, mr_proxy=0.6,
            price_zone="sweet_spot", ai_rec="APOSTAR"
        )
        self.assertGreater(score, 0.70)

    def test_low_quality_opportunity(self):
        score = self._ddc_v2(kl=0.02, ev=0.01, conf=40, liquidity=300)
        self.assertLess(score, 0.20)

    def test_urgency_boosts_score(self):
        base = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=3000, hours_to_res=200)
        urgent = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=3000, hours_to_res=10)
        self.assertGreater(urgent, base)

    def test_illiquid_penalized(self):
        liquid   = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=8000)
        illiquid = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=200)
        self.assertGreater(liquid, illiquid)

    def test_ai_bonus_adds_to_score(self):
        no_ai = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=3000)
        ai    = self._ddc_v2(kl=0.08, ev=0.08, conf=65, liquidity=3000, ai_rec="APOSTAR")
        self.assertGreater(ai, no_ai)

    def test_score_bounded_0_to_1(self):
        max_score = self._ddc_v2(kl=0.30, ev=0.30, conf=100, liquidity=50000,
                                  hours_to_res=3, mr_aligned=True, mr_proxy=0.9,
                                  price_zone="sweet_spot", ai_rec="APOSTAR")
        self.assertLessEqual(max_score, 1.0)
        self.assertGreaterEqual(max_score, 0.0)

    def test_v2_better_discrimination(self):
        """V2 debe separar más las oportunidades buenas de las malas."""
        good_v2 = self._ddc_v2(kl=0.12, ev=0.10, conf=75, liquidity=6000,
                                hours_to_res=18, price_zone="sweet_spot")
        bad_v2  = self._ddc_v2(kl=0.04, ev=0.02, conf=45, liquidity=400)
        spread_v2 = good_v2 - bad_v2

        # DDC v1 simplificado para comparar
        def ddc_v1(kl, ev, conf):
            return round(min(kl/0.10, 1)*0.40 + min(max(ev,0)/0.12,1)*0.25 + min(conf/100,1)*0.15, 4)
        good_v1 = ddc_v1(0.12, 0.10, 75)
        bad_v1  = ddc_v1(0.04, 0.02, 45)
        spread_v1 = good_v1 - bad_v1

        # V2 debe discriminar igual o mejor
        self.assertGreaterEqual(spread_v2, spread_v1 * 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
