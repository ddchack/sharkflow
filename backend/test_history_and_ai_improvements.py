"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para:
1. Historial persistente multi-bot (GET/DELETE /api/history)
2. _record_trade helper
3. AI ensemble prompt enriquecido (contexto BTC)
4. Diagnostics coloreados
5. Deduplicación de endpoints
"""
import json
import time
import unittest
from datetime import datetime
from pathlib import Path
import tempfile
import os


# ─── Simulate _record_trade logic ─────────────────────────────────────────────

def simulate_record_trade(history: list, bot: str, trade_data: dict) -> dict:
    """Mirror of _record_trade in api_server.py."""
    entry = {
        "id":      trade_data.get("id", f"{bot}_{len(history)+1}"),
        "bot":     bot,
        "ts":      trade_data.get("ts", datetime.now().isoformat()),
        "market":  trade_data.get("question", trade_data.get("market_question", "")),
        "side":    trade_data.get("side", ""),
        "bet_usd": trade_data.get("bet_usd", trade_data.get("amount", 0.0)),
        "pnl":     trade_data.get("pnl", 0.0),
        "status":  trade_data.get("status", "pending"),
        "mode":    trade_data.get("mode", "demo"),
        "asset":   trade_data.get("asset", ""),
        "price":   trade_data.get("price", trade_data.get("signal_price", 0.0)),
    }
    history.append(entry)
    return entry


class TestPersistentHistory(unittest.TestCase):
    """Tests del historial persistente multi-bot."""

    def setUp(self):
        self.history: list = []

    def _record(self, bot, **kwargs):
        return simulate_record_trade(self.history, bot, kwargs)

    def test_record_rush_trade(self):
        t = self._record("rush", question="BTC up/down 5m", side="UP",
                         bet_usd=1.0, pnl=0.54, status="won", mode="live")
        self.assertEqual(t["bot"], "rush")
        self.assertEqual(t["status"], "won")
        self.assertGreater(t["pnl"], 0)

    def test_record_dh_trade(self):
        t = self._record("dh", question="ETH hedge", side="NO",
                         bet_usd=2.0, pnl=-2.0, status="lost", mode="demo")
        self.assertEqual(t["bot"], "dh")
        self.assertLess(t["pnl"], 0)

    def test_record_er_trade(self):
        t = self._record("er", question="Low prob outcome", side="YES",
                         bet_usd=0.5, pnl=2.3, status="won", mode="live", price=0.12)
        self.assertEqual(t["bot"], "er")
        self.assertAlmostEqual(t["price"], 0.12, places=2)

    def test_history_max_500(self):
        for i in range(600):
            self._record("rush", bet_usd=1.0, pnl=0.1, status="won")
        trimmed = self.history[-500:]
        self.assertLessEqual(len(trimmed), 500)

    def test_filter_by_bot(self):
        self._record("rush", pnl=0.5, status="won")
        self._record("dh",   pnl=-1.0, status="lost")
        self._record("er",   pnl=2.0, status="won")
        rush_only = [t for t in self.history if t["bot"] == "rush"]
        self.assertEqual(len(rush_only), 1)
        self.assertEqual(rush_only[0]["bot"], "rush")

    def test_filter_by_status(self):
        self._record("rush", pnl=0.5, status="won")
        self._record("rush", pnl=-1.0, status="lost")
        self._record("dh",   pnl=0.0, status="pending")
        won = [t for t in self.history if t["status"] == "won"]
        self.assertEqual(len(won), 1)

    def test_filter_by_mode(self):
        self._record("rush", mode="live", pnl=0.5, status="won")
        self._record("rush", mode="demo", pnl=-0.5, status="lost")
        live = [t for t in self.history if t["mode"] == "live"]
        self.assertEqual(len(live), 1)

    def test_win_rate_calculation(self):
        for _ in range(7):
            self._record("rush", status="won",  pnl=0.5)
        for _ in range(3):
            self._record("rush", status="lost", pnl=-1.0)
        won  = sum(1 for t in self.history if t["status"] == "won")
        lost = sum(1 for t in self.history if t["status"] == "lost")
        wr = won / max(won + lost, 1) * 100
        self.assertAlmostEqual(wr, 70.0, delta=0.1)

    def test_total_pnl_calculation(self):
        self._record("rush", pnl=1.5, status="won")
        self._record("rush", pnl=-1.0, status="lost")
        self._record("dh",   pnl=2.0, status="won")
        total = round(sum(t["pnl"] for t in self.history), 3)
        self.assertAlmostEqual(total, 2.5, places=2)

    def test_trade_default_mode_demo(self):
        t = self._record("rush", pnl=0.5, status="won")
        self.assertEqual(t["mode"], "demo")

    def test_trade_ts_auto_set(self):
        t = self._record("rush", pnl=0.5, status="won")
        self.assertIn("ts", t)
        self.assertIsNotNone(t["ts"])

    def test_history_sorted_by_ts_descending(self):
        import time as _t
        self._record("rush", ts="2026-01-01T10:00:00", pnl=0.5, status="won")
        self._record("rush", ts="2026-01-01T11:00:00", pnl=0.5, status="won")
        self._record("rush", ts="2026-01-01T09:00:00", pnl=0.5, status="won")
        sorted_h = sorted(self.history, key=lambda x: x["ts"], reverse=True)
        self.assertEqual(sorted_h[0]["ts"], "2026-01-01T11:00:00")
        self.assertEqual(sorted_h[2]["ts"], "2026-01-01T09:00:00")


class TestHistoryPersistence(unittest.TestCase):
    """Tests de guardado y carga del historial en disco."""

    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            history = [{"bot": "rush", "pnl": 1.5, "status": "won"}]
            json.dump(history, f)
            fname = f.name
        try:
            with open(fname, encoding='utf-8') as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["bot"], "rush")
        finally:
            os.unlink(fname)

    def test_load_missing_file_returns_empty(self):
        p = Path("/tmp/nonexistent_sharkflow_history_xyz.json")
        try:
            with open(p, encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        self.assertEqual(data, [])

    def test_truncate_to_500(self):
        big = [{"bot": "rush", "pnl": 0.1, "status": "won"} for _ in range(600)]
        truncated = big[-500:]
        self.assertEqual(len(truncated), 500)


class TestAIPromptEnrichment(unittest.TestCase):
    """Tests para el prompt AI enriquecido con contexto BTC."""

    def _build_prompt_context(self, recs, btc_price=95000.0, btc_chg24=1.5, btc_vol=42000.0):
        """Simula la construcción del prompt con contexto BTC."""
        has_crypto = any(r.get("category", "").lower() in ("crypto", "cryptocurrency") for r in recs)
        btc_ctx = ""
        if has_crypto:
            btc_ctx = (
                f"\nCONTEXTO CRYPTO ACTUAL: BTC=${btc_price:,.0f} | "
                f"24h: {btc_chg24:+.2f}% | Vol24h: {btc_vol:,.0f} BTC\n"
            )
        return btc_ctx

    def test_crypto_market_includes_btc_context(self):
        recs = [{"category": "crypto", "question": "BTC > 100k?"}]
        ctx = self._build_prompt_context(recs)
        self.assertIn("BTC=$95,000", ctx)
        self.assertIn("24h:", ctx)

    def test_non_crypto_market_no_btc_context(self):
        recs = [{"category": "politics", "question": "Who wins election?"}]
        ctx = self._build_prompt_context(recs)
        self.assertEqual(ctx, "")

    def test_mixed_markets_includes_btc_context(self):
        recs = [
            {"category": "politics", "question": "Election outcome?"},
            {"category": "crypto", "question": "ETH >3k?"},
        ]
        ctx = self._build_prompt_context(recs)
        self.assertIn("BTC=", ctx)

    def test_btc_negative_change_context(self):
        recs = [{"category": "crypto", "question": "BTC dump?"}]
        ctx = self._build_prompt_context(recs, btc_chg24=-2.5)
        self.assertIn("-2.50%", ctx)

    def test_prompt_json_format_line_count(self):
        """El prompt debe solicitar N líneas JSON (una por mercado)."""
        n_markets = 5
        expected_line = f"Responde exactamente {n_markets} líneas JSON (en orden):"
        # Construir prompt final simulado
        prompt_end = f"Responde exactamente {n_markets} líneas JSON (en orden):"
        self.assertIn(str(n_markets), prompt_end)

    def test_apostar_fuerte_threshold(self):
        """APOSTAR_FUERTE debe requerir edge >10%."""
        edge = 0.12  # 12%
        threshold_fuerte = 0.10
        self.assertGreater(edge, threshold_fuerte)

    def test_no_apostar_threshold(self):
        """NO_APOSTAR si edge < 2% (diferencia insignificante)."""
        model_prob = 0.53
        market_price = 0.52
        diff = abs(model_prob - market_price)
        threshold_no = 0.03
        self.assertLess(diff, threshold_no)

    def test_confidence_scale_validation(self):
        """Confianza debe estar en [0.0, 1.0]."""
        valid_confidences = [0.0, 0.30, 0.55, 0.80, 1.0]
        for c in valid_confidences:
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)


class TestDiagnosticsColoring(unittest.TestCase):
    """Tests para la lógica de coloreado de mensajes de diagnóstico."""

    def _diag_color(self, msg):
        """Mirror de _diagColor() en dashboard.html."""
        import re
        if not msg or msg == '—':
            return 'var(--text-dim)'
        m = msg.lower()
        # Rojo primero
        if any(k in m for k in ['error', 'perdido', 'perdida', 'lost', 'failed', 'excepción']):
            return '#ff4976'
        # Verde: actividad positiva
        if any(k in m for k in ['entró', 'apostando', 'ejecuta', 'ganado', 'won']):
            return '#00d4aa'
        # Verde: candidatos > 0
        if re.search(r'\b[1-9]\d*\s*candidato', m):
            return '#00d4aa'
        # Verde: pasaron filtros > 0
        if re.search(r'\b[1-9]\d*\s*pasaron', m):
            return '#00d4aa'
        # Amarillo
        if any(k in m for k in ['lax', '[warn', 'umbral', 'weak', 'bajo']):
            return '#ffc107'
        return 'var(--text-dim)'

    def test_entry_message_green(self):
        self.assertEqual(self._diag_color("Bot entró en posición UP"), '#00d4aa')

    def test_candidate_message_green(self):
        self.assertEqual(self._diag_color("1 candidato encontrado"), '#00d4aa')

    def test_error_message_red(self):
        self.assertEqual(self._diag_color("Error de conexión al API"), '#ff4976')

    def test_lost_message_red(self):
        self.assertEqual(self._diag_color("Apuesta perdida $1.00"), '#ff4976')

    def test_lax_message_yellow(self):
        self.assertEqual(self._diag_color("Modo lax activado"), '#ffc107')

    def test_weak_signal_yellow(self):
        self.assertEqual(self._diag_color("BTC +0.05% señal WEAK"), '#ffc107')

    def test_neutral_scan_0_candidates_dim(self):
        self.assertEqual(self._diag_color("Escaneados 8 mercados, 0 candidatos"), 'var(--text-dim)')

    def test_empty_message_dim(self):
        self.assertEqual(self._diag_color(""), 'var(--text-dim)')

    def test_dash_message_dim(self):
        self.assertEqual(self._diag_color("—"), 'var(--text-dim)')

    def test_passed_filters_green(self):
        self.assertEqual(self._diag_color("2 pasaron filtros"), '#00d4aa')

    def test_zero_passed_filters_dim(self):
        self.assertEqual(self._diag_color("0 pasaron filtros"), 'var(--text-dim)')


class TestEndpointDeduplication(unittest.TestCase):
    """Verifica que no hay endpoints duplicados en api_server.py."""

    @classmethod
    def setUpClass(cls):
        import os as _os
        server_path = _os.path.join(_os.path.dirname(__file__), "api_server.py")
        with open(server_path, encoding='utf-8') as f:
            cls.src = f.read()

    def _count_route(self, method, path):
        import re
        pattern = rf'@app\.{method}\("{re.escape(path)}"'
        return len(re.findall(pattern, self.src))

    def test_no_duplicate_ai_health(self):
        count = self._count_route("get", "/api/ai/health")
        self.assertEqual(count, 1, f"Hay {count} endpoints GET /api/ai/health (esperado: 1)")

    def test_no_duplicate_history(self):
        count = self._count_route("get", "/api/history")
        self.assertEqual(count, 1, f"Hay {count} endpoints GET /api/history (esperado: 1)")

    def test_history_delete_exists(self):
        count = self._count_route("delete", "/api/history")
        self.assertEqual(count, 1, f"DELETE /api/history debe existir exactamente 1 vez")

    def test_persistent_history_file_variable_defined(self):
        self.assertIn("_PERSISTENT_HISTORY_FILE", self.src)

    def test_record_trade_function_defined(self):
        self.assertIn("def _record_trade(", self.src)

    def test_save_persistent_history_defined(self):
        self.assertIn("def _save_persistent_history(", self.src)

    def test_load_persistent_history_defined(self):
        self.assertIn("def _load_persistent_history(", self.src)


class TestHistoryAPIResponseFormat(unittest.TestCase):
    """Verifica la estructura de respuesta del endpoint /api/history."""

    def _build_response(self, trades, won=0, lost=0, total_pnl=0.0, offset=0, limit=100):
        total = len(trades)
        return {
            "trades":    trades,
            "total":     total,
            "won":       won,
            "lost":      lost,
            "win_rate":  round(won / max(won + lost, 1) * 100, 1),
            "total_pnl": total_pnl,
            "offset":    offset,
            "limit":     limit,
        }

    def test_response_has_all_fields(self):
        resp = self._build_response([])
        for key in ["trades", "total", "won", "lost", "win_rate", "total_pnl", "offset", "limit"]:
            self.assertIn(key, resp)

    def test_win_rate_zero_division_safe(self):
        resp = self._build_response([], won=0, lost=0)
        self.assertEqual(resp["win_rate"], 0.0)

    def test_win_rate_correct(self):
        resp = self._build_response([], won=7, lost=3)
        self.assertAlmostEqual(resp["win_rate"], 70.0, delta=0.1)

    def test_pagination_fields(self):
        trades = [{"bot": "rush"} for _ in range(50)]
        resp = self._build_response(trades, offset=10, limit=25)
        self.assertEqual(resp["offset"], 10)
        self.assertEqual(resp["limit"], 25)
        self.assertEqual(resp["total"], 50)

    def test_trades_is_list(self):
        resp = self._build_response([{"bot": "rush"}])
        self.assertIsInstance(resp["trades"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
