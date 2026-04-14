"""
Tests para TradeValidator — ejecutar con: python test_validator.py
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from trade_validator import TradeValidator, CIRCUIT_BREAKER_THRESHOLD


class TestTradeValidator(unittest.TestCase):

    def setUp(self):
        self.v = TradeValidator()

    # ── 1. Validación de precio ───────────────────────────────────────────

    def test_precio_0_1_cents_bloqueado(self):
        """Precio 0.1¢ (0.001) debe ser bloqueado — resultado ya decidido."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.001, "no_price": 0.999}
        )
        self.assertFalse(ok)
        self.assertIn("5¢", reason)

    def test_precio_99_9_cents_bloqueado(self):
        """Precio 99.9¢ (0.999) debe ser bloqueado — resultado ya decidido."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.999, "no_price": 0.001}
        )
        self.assertFalse(ok)
        self.assertIn("95¢", reason)

    def test_precio_3_cents_bloqueado(self):
        """Precio 3¢ (0.03) debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "NO", 1.0, 1.0,
            {"yes_price": 0.97, "no_price": 0.03}
        )
        self.assertFalse(ok)
        self.assertFalse(ok)

    def test_precio_96_cents_bloqueado(self):
        """Precio 96¢ (0.96) debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.96, "no_price": 0.04}
        )
        self.assertFalse(ok)
        self.assertIn("95¢", reason)

    def test_spread_extremo_bloqueado(self):
        """Spread > 90¢ debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            # YES=0.95, NO=0.04 → spread = 0.91 > MAX_SPREAD(0.90)
            {"yes_price": 0.95, "no_price": 0.04}
        )
        self.assertFalse(ok)

    def test_precio_45_cents_pasa(self):
        """Precio 45¢ con Kelly $1.50 debe pasar todas las validaciones."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.5, 1.5,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T23:59:59Z"}
        )
        self.assertTrue(ok, f"Debería pasar pero falló: {reason}")
        self.assertEqual(reason, "OK")

    # ── 2. Estado del evento ──────────────────────────────────────────────

    def test_evento_finalizado_bloqueado(self):
        """end_date pasado debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2020-01-01T00:00:00Z"}  # fecha pasada
        )
        self.assertFalse(ok)
        self.assertIn("finalizó", reason)

    def test_mercado_cerrado_bloqueado(self):
        """closed=True debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55, "closed": True,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertIn("cerrado", reason)

    def test_mercado_inactivo_bloqueado(self):
        """active=False debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55, "active": False,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertIn("inactivo", reason)

    # ── 3. Kelly vs mínimo ────────────────────────────────────────────────

    def test_kelly_0_02_bloqueado(self):
        """Kelly $0.02 (< $0.50 = mitad del mínimo) debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 0.02,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertIn("0.02", reason)
        self.assertIn("mitad", reason)

    def test_kelly_0_49_bloqueado(self):
        """Kelly $0.49 (< $0.50) debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 0.49,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok)

    def test_kelly_0_50_pasa(self):
        """Kelly $0.50 (= mínimo de override) debe pasar."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.0, 0.50,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertTrue(ok, f"Kelly $0.50 debería pasar: {reason}")

    def test_kelly_1_50_pasa(self):
        """Kelly $1.50 con precio 45¢ debe pasar."""
        ok, reason = self.v.validate_trade(
            "mkt_test", "YES", 1.5, 1.5,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertTrue(ok, f"Trade válida debería pasar: {reason}")

    # ── 4. Precio en mercado con score publicado ──────────────────────────

    def test_mercado_score_publicado_bloqueado(self):
        """Mercado deportivo con precio > 95¢ (resultado conocido) debe bloquearse.
        El bloqueo puede venir por la regla de precio general O la de deportes,
        lo importante es que sea False."""
        ok, reason = self.v.validate_trade(
            "mkt_sports", "YES", 1.0, 1.0,
            {"yes_price": 0.985, "no_price": 0.015,
             "category": "sports",
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok, "Precio 98.5¢ en mercado deportivo debe ser bloqueado")
        # El bloqueo puede ser por regla general (>95¢) o deportes-específica
        blocked_by_price_or_sports = ("95¢" in reason or "extremo" in reason or "decidido" in reason)
        self.assertTrue(blocked_by_price_or_sports, f"Razón inesperada: {reason}")

    def test_mercado_sports_cache_finalizado(self):
        """Market marcado como finalizado en cache debe ser bloqueado."""
        self.v.mark_sports_finished("mkt_sports_cache", "Score final: 2-0")
        ok, reason = self.v.validate_trade(
            "mkt_sports_cache", "YES", 1.0, 1.0,
            {"yes_price": 0.50, "no_price": 0.50,
             "category": "sports",
             "end_date": "2099-12-31T00:00:00Z"}
        )
        self.assertFalse(ok)
        self.assertIn("finalizado", reason)

    # ── 5. Circuit breaker ────────────────────────────────────────────────

    def test_circuit_breaker_activado(self):
        """Después de CIRCUIT_BREAKER_THRESHOLD bloqueos consecutivos, el CB debe activarse."""
        mkt = {"yes_price": 0.999, "no_price": 0.001}  # siempre bloquea
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            self.v.validate_trade("mkt_cb", "YES", 1.0, 1.0, mkt)
        self.assertTrue(self.v.check_circuit_breaker(),
                        "Circuit breaker debería estar activo")

    def test_circuit_breaker_reset(self):
        """Reset manual del circuit breaker debe funcionar."""
        mkt = {"yes_price": 0.999, "no_price": 0.001}
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            self.v.validate_trade("mkt_cb", "YES", 1.0, 1.0, mkt)
        self.assertTrue(self.v.check_circuit_breaker())
        self.v.reset_circuit_breaker()
        self.assertFalse(self.v.check_circuit_breaker(),
                         "Circuit breaker debería estar reseteado")

    def test_exito_resetea_contador_consecutivos(self):
        """Un trade válido debe resetear el contador de bloqueos consecutivos."""
        bad_mkt = {"yes_price": 0.999, "no_price": 0.001}
        self.v.validate_trade("mkt_cb", "YES", 1.0, 1.0, bad_mkt)  # bloquear 1
        self.assertEqual(self.v._consecutive_blocked, 1)
        good_mkt = {"yes_price": 0.45, "no_price": 0.55, "end_date": "2099-01-01T00:00:00Z"}
        self.v.validate_trade("mkt_ok", "YES", 1.0, 1.0, good_mkt)  # pasar
        self.assertEqual(self.v._consecutive_blocked, 0)

    # ── 6. Logging ────────────────────────────────────────────────────────

    def test_bloqueo_guardado_en_historial(self):
        """Cada bloqueo debe guardarse en el historial."""
        before = len(self.v._blocked_history)
        self.v.validate_trade("mkt_log", "YES", 1.0, 1.0,
                              {"yes_price": 0.01, "no_price": 0.99})
        self.assertEqual(len(self.v._blocked_history), before + 1)

    def test_historial_tiene_campos_correctos(self):
        """El historial debe contener todos los campos requeridos."""
        self.v.validate_trade("mkt_fields", "YES", 2.0, 0.02,
                              {"yes_price": 0.45, "no_price": 0.55,
                               "end_date": "2099-01-01T00:00:00Z"})
        entry = self.v._blocked_history[-1]
        for campo in ("timestamp", "market_id", "outcome", "amount", "kelly", "price", "reason"):
            self.assertIn(campo, entry, f"Falta campo: {campo}")

    # ── 7. Trade válida completa ──────────────────────────────────────────

    def test_trade_valida_completa(self):
        """Trade con todos los campos correctos debe pasar SIN aparecer en historial."""
        before = len(self.v._blocked_history)
        ok, reason = self.v.validate_trade(
            "mkt_valid_full", "YES", 2.50, 2.50,
            {
                "yes_price":  0.45,
                "no_price":   0.55,
                "end_date":   "2099-06-30T20:00:00Z",
                "closed":     False,
                "active":     True,
                "category":   "crypto",
                "volume_24h": 50000.0,
            }
        )
        self.assertTrue(ok, f"Trade válida falló: {reason}")
        self.assertEqual(reason, "OK")
        self.assertEqual(len(self.v._blocked_history), before,
                         "Trade válida NO debe aparecer en historial de bloqueados")


    # ── 8. Checks 6-9 (liquidez, volumen, tiempo, dedup) ─────────────────

    def test_liquidez_insuficiente_bloqueada(self):
        """Liquidez < 500 USD debe ser bloqueada."""
        ok, reason = self.v.validate_trade(
            "mkt_liq", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z",
             "liquidity": 200.0}
        )
        self.assertFalse(ok)
        self.assertIn("Liquidez", reason)

    def test_liquidez_cero_pasa(self):
        """Liquidez = 0 (sin dato) no debe bloquear (fail-open)."""
        ok, reason = self.v.validate_trade(
            "mkt_liq0", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z",
             "liquidity": 0.0}
        )
        self.assertTrue(ok, f"Liquidez=0 no debe bloquear: {reason}")

    def test_volumen_bajo_bloqueado(self):
        """Volumen 24h < 100 USD debe ser bloqueado."""
        ok, reason = self.v.validate_trade(
            "mkt_vol", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55,
             "end_date": "2099-12-31T00:00:00Z",
             "volume_24h": 50.0}
        )
        self.assertFalse(ok)
        self.assertIn("Volumen", reason)

    def test_resolucion_inminente_bloqueada(self):
        """Mercado que cierra en menos de 10 minutos debe ser bloqueado."""
        from datetime import datetime, timezone, timedelta
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok, reason = self.v.validate_trade(
            "mkt_time", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55, "end_date": soon}
        )
        self.assertFalse(ok)
        self.assertIn("inminente", reason)

    def test_resolucion_horas_pasa(self):
        """Mercado que cierra en horas debe pasar el check de tiempo."""
        from datetime import datetime, timezone, timedelta
        later = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok, reason = self.v.validate_trade(
            "mkt_time_ok", "YES", 1.0, 1.0,
            {"yes_price": 0.45, "no_price": 0.55, "end_date": later}
        )
        self.assertTrue(ok, f"Mercado con horas restantes debe pasar: {reason}")

    def test_antidup_mismo_mercado_bloqueado(self):
        """El mismo mercado en menos de 5 minutos debe ser bloqueado."""
        mkt = {"yes_price": 0.45, "no_price": 0.55, "end_date": "2099-12-31T00:00:00Z"}
        # Primera vez: debe pasar
        ok1, r1 = self.v.validate_trade("mkt_dup", "YES", 1.0, 1.0, mkt)
        self.assertTrue(ok1, f"Primera vez debe pasar: {r1}")
        # Segunda vez inmediata: debe bloquear
        ok2, r2 = self.v.validate_trade("mkt_dup", "YES", 1.0, 1.0, mkt)
        self.assertFalse(ok2)
        self.assertIn("duplicado", r2)

    def test_antidup_diferente_mercado_pasa(self):
        """Mercados distintos no deben bloquearse mutuamente por anti-dup."""
        mkt = {"yes_price": 0.45, "no_price": 0.55, "end_date": "2099-12-31T00:00:00Z"}
        self.v.validate_trade("mkt_a", "YES", 1.0, 1.0, mkt)
        ok, reason = self.v.validate_trade("mkt_b", "YES", 1.0, 1.0, mkt)
        self.assertTrue(ok, f"Mercado distinto debe pasar: {reason}")

    def test_9_checks_activos(self):
        """get_status() debe reportar 9 checks activos."""
        status = self.v.get_status()
        self.assertEqual(status["checks_active"], 9)


if __name__ == "__main__":
    print("=" * 60)
    print("SharkFlow — TradeValidator Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestTradeValidator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
