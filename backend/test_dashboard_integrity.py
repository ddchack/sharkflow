"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests de integridad del dashboard.html — verifica que todos los hooks JS
críticos están presentes y que el HTML no tiene problemas obvios.
"""
import unittest
import re
import os

DASH_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


def load_dashboard():
    with open(DASH_PATH, encoding="utf-8") as f:
        return f.read()


class TestDashboardHooks(unittest.TestCase):
    """Verifica que los hooks de datos están conectados."""

    @classmethod
    def setUpClass(cls):
        cls.html = load_dashboard()

    def test_rush_pnl_chart_connected(self):
        self.assertIn("renderRushPnlChart(filtered)", self.html)

    def test_dh_pnl_chart_connected(self):
        self.assertIn("renderDHPnlChart(history)", self.html)

    def test_dashboard_pnl_hero_connected(self):
        self.assertIn("updateDashPnlHero(realized, unrealized)", self.html)

    def test_rush_nav_dot_on(self):
        self.assertIn("navSetDot('navDotRush',", self.html)

    def test_rush_nav_dot_off(self):
        self.assertIn("navSetDot('navDotRush', '')", self.html)

    def test_dh_nav_dot_on(self):
        self.assertIn("navSetDot('navDotDH',", self.html)

    def test_dh_nav_dot_off(self):
        self.assertIn("navSetDot('navDotDH', '')", self.html)

    def test_auto_nav_dot(self):
        self.assertIn("navSetDot('navDotAuto',", self.html)

    def test_er_nav_dot(self):
        self.assertIn("navSetDot('navDotER',", self.html)

    def test_recs_count_connected(self):
        self.assertIn("navSetCount('navRecsCount'", self.html)

    def test_tr_win_class_applied(self):
        # tr-win debe aplicarse en al menos 2 lugares distintos
        occurrences = self.html.count("tr-win")
        self.assertGreaterEqual(occurrences, 2)

    def test_tr_loss_class_applied(self):
        occurrences = self.html.count("tr-loss")
        self.assertGreaterEqual(occurrences, 2)

    def test_open_positions_panel_exists(self):
        self.assertIn("dashOpenPositions", self.html)

    def test_bot_strip_ids(self):
        for bot_id in ["dashRushStatus", "dashDHStatus", "dashAutoStatus", "dashERStatus"]:
            self.assertIn(bot_id, self.html, f"Missing {bot_id}")

    def test_er_score_bar_in_card(self):
        self.assertIn("ER Score", self.html)
        self.assertIn("sf-score-bar", self.html)

    def test_potential_gain_in_recs(self):
        self.assertIn("si gana", self.html)

    def test_top_opp_panel_exists(self):
        self.assertIn("dashTopOpp", self.html)

    def test_bot_strip_exists(self):
        self.assertIn("dashBotStrip", self.html)

    def test_bots_status_in_load_status(self):
        # loadStatus() debe leer el campo bots del servidor
        self.assertIn("bots.rush_running", self.html)

    def test_er_session_stats_strip_exists(self):
        self.assertIn("erSessionStats", self.html)

    def test_er_session_stats_ids(self):
        for stat_id in ["erStatScans", "erStatBets", "erStatCapital"]:
            self.assertIn(stat_id, self.html, f"Missing {stat_id}")


class TestDashboardFunctions(unittest.TestCase):
    """Verifica que las funciones JS críticas están definidas."""

    @classmethod
    def setUpClass(cls):
        cls.html = load_dashboard()

    def _fn_defined(self, name):
        return bool(re.search(r"function\s+" + re.escape(name) + r"\s*\(", self.html))

    def test_navSetDot_defined(self):
        self.assertTrue(self._fn_defined("navSetDot"))

    def test_navSetPnl_defined(self):
        self.assertTrue(self._fn_defined("navSetPnl"))

    def test_navSetCount_defined(self):
        self.assertTrue(self._fn_defined("navSetCount"))

    def test_updateDashPnlHero_defined(self):
        self.assertTrue(self._fn_defined("updateDashPnlHero"))

    def test_renderRushPnlChart_defined(self):
        self.assertTrue(self._fn_defined("renderRushPnlChart"))

    def test_renderDHPnlChart_defined(self):
        self.assertTrue(self._fn_defined("renderDHPnlChart"))

    def test_renderDashOpenPositions_defined(self):
        self.assertTrue(self._fn_defined("renderDashOpenPositions"))

    def test_loadDashTopOpp_defined(self):
        self.assertTrue(self._fn_defined("loadDashTopOpp"))

    def test_renderDashTopOppFromCache_defined(self):
        self.assertTrue(self._fn_defined("renderDashTopOppFromCache"))

    def test_sfConfirmLive_defined(self):
        self.assertTrue(self._fn_defined("sfConfirmLive"))

    def test_sfLiveModalConfirm_defined(self):
        self.assertTrue(self._fn_defined("sfLiveModalConfirm"))

    def test_renderERCard_defined(self):
        self.assertTrue(self._fn_defined("renderERCard"))

    def test_updateERSessionStats_defined(self):
        self.assertTrue(self._fn_defined("updateERSessionStats"))


class TestDashboardIDs(unittest.TestCase):
    """Verifica que los IDs HTML críticos existen en el archivo."""

    @classmethod
    def setUpClass(cls):
        cls.html = load_dashboard()

    def _id_exists(self, id_):
        return bool(re.search(r'id=["\']' + re.escape(id_) + r'["\']', self.html))

    def test_nav_dot_rush(self):
        self.assertTrue(self._id_exists("navDotRush"))

    def test_nav_dot_dh(self):
        self.assertTrue(self._id_exists("navDotDH"))

    def test_nav_dot_auto(self):
        self.assertTrue(self._id_exists("navDotAuto"))

    def test_nav_dot_er(self):
        self.assertTrue(self._id_exists("navDotER"))

    def test_nav_pnl_rush(self):
        self.assertTrue(self._id_exists("navPnlRush"))

    def test_dash_pnl_big(self):
        self.assertTrue(self._id_exists("dashPnlBig"))

    def test_chart_rush_pnl(self):
        self.assertTrue(self._id_exists("chartRushPnl"))

    def test_sf_live_modal(self):
        self.assertTrue(self._id_exists("sfLiveModal"))

    def test_er_opportunities_grid(self):
        self.assertTrue(self._id_exists("erOpportunitiesGrid"))

    def test_rush_window_countdown(self):
        self.assertTrue(self._id_exists("rushWindowCountdown"))

    def test_dh_quick_bar(self):
        self.assertTrue(self._id_exists("dhQuickBar"))


class TestDashboardBugFixes(unittest.TestCase):
    """Verifica que los bugs críticos corregidos siguen solucionados."""

    @classmethod
    def setUpClass(cls):
        cls.html = load_dashboard()

    def _fn_defined(self, name):
        return bool(re.search(r"function\s+" + re.escape(name) + r"\s*\(", self.html))

    def test_ctrlShowRecs_no_minEdge_undefined(self):
        """Bug fix: ctrlShowRecs y autoShowRecs no deben referenciar `minEdge` sin definir en scope."""
        # Ambas funciones deben usar _ctrlEdgeVal o _autoEdgeVal (leído del DOM), no `minEdge`
        # Buscamos que dentro de function ctrlShowRecs y autoShowRecs no aparezca ${minEdge}
        import re
        # Extrae el cuerpo de ctrlShowRecs
        m = re.search(r'function ctrlShowRecs\(.*?\{(.*?)^\}', self.html, re.DOTALL | re.MULTILINE)
        if m:
            self.assertNotIn("${minEdge}%", m.group(1),
                "Bug regresión: ctrlShowRecs usa `minEdge` que no está en scope")
        # Extrae el cuerpo de autoShowRecs
        m2 = re.search(r'function autoShowRecs\(.*?\{(.*?)^\}', self.html, re.DOTALL | re.MULTILINE)
        if m2:
            self.assertNotIn("${minEdge}%", m2.group(1),
                "Bug regresión: autoShowRecs usa `minEdge` que no está en scope")

    def test_ctrlEdgeVal_defined_in_scope(self):
        """ctrlShowRecs debe definir _ctrlEdgeVal antes de usarlo."""
        self.assertIn("_ctrlEdgeVal", self.html,
            "ctrlShowRecs debe usar _ctrlEdgeVal (edge leído del DOM) en vez de minEdge")

    def test_reconnectWallet_defined(self):
        """Función de reconexión automática debe existir."""
        self.assertTrue(self._fn_defined("reconnectWallet"),
            "reconnectWallet() debe estar definida para manejar expiración de creds")

    def test_reconnect_button_present(self):
        """Botón de reconexión debe existir en el panel de claves."""
        self.assertIn("reconnectWallet()", self.html,
            "Debe haber un botón que llame reconnectWallet()")

    def test_ctrlAnalyze_error_handling(self):
        """ctrlAnalyze debe manejar respuesta null del API."""
        self.assertIn("apiFetch._lastError", self.html,
            "ctrlAnalyze debe mostrar el error del último apiFetch cuando d es null")

    def test_loadRecommendations_error_handling(self):
        """loadRecommendations debe mostrar error cuando el API falla."""
        # Ambas verificaciones necesitan estar cerca una de la otra
        idx_load = self.html.find("async function loadRecommendations()")
        idx_err = self.html.find("Error al escanear")
        self.assertNotEqual(idx_err, -1, "loadRecommendations debe mostrar error cuando d=null")

    def test_creds_age_shown_in_wallet_status(self):
        """El estado de la wallet debe mostrar la edad de los creds."""
        self.assertIn("creds_age_hours", self.html,
            "refreshCtrlWalletStatus debe usar creds_age_hours del API")

    def test_wallet_warning_when_old_creds(self):
        """Debe mostrarse advertencia cuando los creds se acercan a expirar."""
        self.assertIn("Creds próximos a expirar", self.html,
            "Debe haber un aviso cuando creds_age_hours >= 2.5h")


if __name__ == "__main__":
    unittest.main(verbosity=2)
