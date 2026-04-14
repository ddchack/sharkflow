"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para el clasificador de mercados — category_models.classify_market()
Cubre casos edge, word boundaries, y desambiguación.
"""
import unittest
from category_models import classify_market, MarketCategory


class TestCategoryClassifier(unittest.TestCase):

    # ── Crypto ──────────────────────────────────────────────
    def test_bitcoin_crypto(self):
        self.assertEqual(classify_market("Will Bitcoin reach $100K?"), MarketCategory.CRYPTO)

    def test_btc_abbrev_crypto(self):
        self.assertEqual(classify_market("Will BTC price exceed $80K?"), MarketCategory.CRYPTO)

    def test_eth_crypto(self):
        self.assertEqual(classify_market("Will ETH price hit $10K?"), MarketCategory.CRYPTO)

    def test_eth_not_in_method(self):
        """'eth' en 'method' no debe clasificar como crypto"""
        result = classify_market("Will the new payment method be accepted?")
        self.assertNotEqual(result, MarketCategory.CRYPTO)

    def test_crypto_price_not_economics(self):
        """'price' con señal crypto no debe ir a economics"""
        result = classify_market("Will the ETH price exceed $10K?")
        self.assertEqual(result, MarketCategory.CRYPTO)

    def test_btc_usd_crypto(self):
        result = classify_market("Will BTC/USD close above $95K on Dec 31?")
        self.assertEqual(result, MarketCategory.CRYPTO)

    # ── Sports ──────────────────────────────────────────────
    def test_nba_sports(self):
        self.assertEqual(classify_market("Will the NBA Finals go to Game 7?"), MarketCategory.SPORTS)

    def test_spain_euro_sports(self):
        """'spain' contenía 'ai' causando falso positivo TECH — word boundary fix"""
        result = classify_market("Will Spain win Euro 2026?")
        self.assertEqual(result, MarketCategory.SPORTS)

    def test_copa_america_sports(self):
        result = classify_market("Will Lionel Messi win Copa America MVP?")
        self.assertEqual(result, MarketCategory.SPORTS)

    def test_india_pakistan_sports(self):
        result = classify_market("Will India beat Pakistan in cricket World Cup?")
        self.assertEqual(result, MarketCategory.SPORTS)

    def test_ai_in_country_not_tech(self):
        """'ai' en nombre de país no debe ser TECH"""
        result = classify_market("Will Spain qualify for the World Cup?")
        self.assertNotEqual(result, MarketCategory.TECH)

    # ── Tech ────────────────────────────────────────────────
    def test_openai_tech(self):
        self.assertEqual(classify_market("Will OpenAI release GPT-5 in 2026?"), MarketCategory.TECH)

    def test_claude_tech(self):
        self.assertEqual(classify_market("Will Claude AI be ranked #1?"), MarketCategory.TECH)

    def test_ai_explicit_tech(self):
        result = classify_market("Will the new AI model pass the bar exam?")
        self.assertEqual(result, MarketCategory.TECH)

    # ── Geopolitics ─────────────────────────────────────────
    def test_netanyahu_geopolitics(self):
        result = classify_market("Will Netanyahu resign before July 2026?")
        self.assertEqual(result, MarketCategory.GEOPOLITICS)

    def test_hamas_geopolitics(self):
        result = classify_market("Will Hamas accept ceasefire deal?")
        self.assertEqual(result, MarketCategory.GEOPOLITICS)

    def test_zelensky_geopolitics(self):
        result = classify_market("Will Zelensky remain president through 2026?")
        self.assertEqual(result, MarketCategory.GEOPOLITICS)

    def test_taiwan_geopolitics(self):
        result = classify_market("Will China invade Taiwan in 2026?")
        self.assertEqual(result, MarketCategory.GEOPOLITICS)

    # ── Politics ────────────────────────────────────────────
    def test_election_politics(self):
        result = classify_market("Will the Democrat win the 2026 Senate race?")
        self.assertEqual(result, MarketCategory.POLITICS)

    def test_trump_politics(self):
        result = classify_market("Will Trump sign the tax bill before December?")
        self.assertEqual(result, MarketCategory.POLITICS)

    # ── Economics ───────────────────────────────────────────
    def test_tesla_stock_economics(self):
        self.assertEqual(classify_market("Will Tesla stock exceed $500?"), MarketCategory.ECONOMICS)

    def test_apple_stock_economics(self):
        result = classify_market("Will Apple stock exceed $250?")
        self.assertEqual(result, MarketCategory.ECONOMICS)

    def test_fed_rates_economics(self):
        result = classify_market("Will the Fed cut rates in March?")
        self.assertEqual(result, MarketCategory.ECONOMICS)

    def test_gdp_economics(self):
        result = classify_market("Will the US GDP grow more than 2%?")
        self.assertEqual(result, MarketCategory.ECONOMICS)

    # ── Edge cases ──────────────────────────────────────────
    def test_empty_question_returns_default(self):
        """Pregunta vacía no debe crash"""
        result = classify_market("")
        self.assertIsInstance(result, MarketCategory)

    def test_numbers_only(self):
        """Solo números no debe crash"""
        result = classify_market("12345")
        self.assertIsInstance(result, MarketCategory)

    def test_word_boundary_sol_not_in_console(self):
        """'sol' en 'console' no debe ser crypto"""
        result = classify_market("Will the gaming console sell 10M units?")
        self.assertNotEqual(result, MarketCategory.CRYPTO)

    def test_btc_word_boundary(self):
        """'btc' debe ser detectado con word boundary"""
        result = classify_market("Will btc close above 90k this weekend?")
        self.assertEqual(result, MarketCategory.CRYPTO)


if __name__ == "__main__":
    unittest.main(verbosity=2)
