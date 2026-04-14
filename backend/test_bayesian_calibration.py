"""
# SharkFlow by Carlos David Donoso Cordero (ddchack)
Tests para BayesianProbabilityEngine y CalibrationSuite.
Cubre: prior/posterior, time-decay, señales, calibradores sin datos,
BetaCalibrator fit mínimo, TemperatureScaler, CalibrationSuite routing.
"""
import unittest
import math
from bayesian_engine import BayesianProbabilityEngine, Signal, BayesianEstimate
from calibration_v2 import BetaCalibrator, IsotonicCalibrator, TemperatureScaler, CalibrationSuite


class TestBayesianPrior(unittest.TestCase):

    def setUp(self):
        self.engine = BayesianProbabilityEngine(prior_strength=10.0)

    def test_prior_matches_market_price(self):
        alpha, beta = self.engine.market_to_prior(0.6)
        self.assertAlmostEqual(alpha / (alpha + beta), 0.6, places=5)

    def test_prior_at_boundaries(self):
        alpha, beta = self.engine.market_to_prior(0.01)
        mean = alpha / (alpha + beta)
        self.assertGreater(mean, 0.01)  # clamped
        alpha, beta = self.engine.market_to_prior(0.99)
        mean = alpha / (alpha + beta)
        self.assertLess(mean, 0.99)  # clamped

    def test_beta_mean_formula(self):
        self.assertAlmostEqual(BayesianProbabilityEngine.beta_mean(6, 4), 0.6)

    def test_no_signals_posterior_equals_prior(self):
        est = self.engine.estimate_probability(0.55)
        self.assertAlmostEqual(est.prior, est.posterior, delta=0.02)


class TestBayesianSignals(unittest.TestCase):

    def setUp(self):
        self.engine = BayesianProbabilityEngine(prior_strength=10.0)

    def test_positive_sentiment_raises_posterior(self):
        base = self.engine.estimate_probability(0.50)
        with_sent = self.engine.estimate_with_raw_data(
            market_price=0.50, sentiment_score=0.8, num_articles=5)
        self.assertGreater(with_sent.posterior, base.posterior)

    def test_negative_sentiment_lowers_posterior(self):
        base = self.engine.estimate_probability(0.50)
        with_sent = self.engine.estimate_with_raw_data(
            market_price=0.50, sentiment_score=-0.8, num_articles=5)
        self.assertLess(with_sent.posterior, base.posterior)

    def test_posterior_clamped_between_02_and_98(self):
        est = self.engine.estimate_with_raw_data(
            market_price=0.99, sentiment_score=1.0, num_articles=100)
        self.assertLessEqual(est.posterior, 0.98)
        est2 = self.engine.estimate_with_raw_data(
            market_price=0.01, sentiment_score=-1.0, num_articles=100)
        self.assertGreaterEqual(est2.posterior, 0.02)

    def test_signals_used_populated(self):
        est = self.engine.estimate_with_raw_data(
            market_price=0.60, sentiment_score=0.5, num_articles=3,
            volume_24h=1000, avg_volume_7d=500)
        self.assertGreater(len(est.signals_used), 0)

    def test_no_signals_no_crash(self):
        est = self.engine.estimate_with_raw_data(market_price=0.45)
        self.assertIsInstance(est, BayesianEstimate)


class TestBayesianTimeDecay(unittest.TestCase):

    def setUp(self):
        self.engine = BayesianProbabilityEngine()

    def test_6h_pulls_heavily_toward_market(self):
        market_p = 0.30
        est = self.engine.estimate_with_raw_data(
            market_price=market_p, sentiment_score=0.9, num_articles=10,
            hours_to_resolution=3.0)
        # With 80% blend, posterior should be close to market_p
        self.assertAlmostEqual(est.posterior, market_p, delta=0.15)

    def test_no_decay_far_future(self):
        """>7 días: blend=0, posterior libre de moverse por señales"""
        est_no_decay = self.engine.estimate_with_raw_data(
            market_price=0.50, sentiment_score=0.9, num_articles=10,
            hours_to_resolution=500)
        est_baseline = self.engine.estimate_with_raw_data(
            market_price=0.50, sentiment_score=0.9, num_articles=10)
        self.assertAlmostEqual(est_no_decay.posterior, est_baseline.posterior, places=4)

    def test_null_hours_no_crash(self):
        est = self.engine.estimate_with_raw_data(
            market_price=0.60, hours_to_resolution=None)
        self.assertIsInstance(est, BayesianEstimate)

    def test_decay_ordering(self):
        """Mayor cercanía → posterior más cercano al precio de mercado"""
        market_p = 0.70
        est_far  = self.engine.estimate_with_raw_data(market_price=market_p, hours_to_resolution=200)
        est_week = self.engine.estimate_with_raw_data(market_price=market_p, hours_to_resolution=50)
        est_day  = self.engine.estimate_with_raw_data(market_price=market_p, hours_to_resolution=12)
        est_imm  = self.engine.estimate_with_raw_data(market_price=market_p, hours_to_resolution=2)
        # All posteriors should be ≤ market_p (assuming no signals push above)
        # The key property: more imminent → closer to market
        dist_far  = abs(est_far.posterior  - market_p)
        dist_week = abs(est_week.posterior - market_p)
        dist_day  = abs(est_day.posterior  - market_p)
        dist_imm  = abs(est_imm.posterior  - market_p)
        self.assertGreaterEqual(dist_far, dist_week)
        self.assertGreaterEqual(dist_week, dist_day)
        self.assertGreaterEqual(dist_day,  dist_imm)


class TestBetaCalibrator(unittest.TestCase):

    def test_unfitted_returns_identity(self):
        cal = BetaCalibrator()
        self.assertAlmostEqual(cal.calibrate(0.7), 0.7, places=5)

    def test_fit_requires_30_obs(self):
        cal = BetaCalibrator()
        for i in range(29):
            cal.add(0.6, 1)
        fitted = cal.fit()
        self.assertFalse(fitted)

    def test_fit_with_enough_obs(self):
        cal = BetaCalibrator()
        import random
        random.seed(42)
        for _ in range(60):
            p = random.uniform(0.2, 0.8)
            outcome = 1 if random.random() < p else 0
            cal.add(p, outcome)
        result = cal.fit()
        # May succeed or fail depending on optimizer; just no crash
        self.assertIsInstance(result, bool)

    def test_calibrate_output_in_range(self):
        cal = BetaCalibrator()
        import random; random.seed(1)
        for _ in range(60):
            p = random.uniform(0.1, 0.9)
            cal.add(p, 1 if random.random() < p else 0)
        cal.fit()
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            out = cal.calibrate(p)
            self.assertGreater(out, 0)
            self.assertLess(out, 1)


class TestTemperatureScaler(unittest.TestCase):

    def test_unfitted_identity(self):
        ts = TemperatureScaler()
        self.assertAlmostEqual(ts.calibrate(0.65), 0.65, places=5)

    def test_fit_requires_30_obs(self):
        ts = TemperatureScaler()
        for i in range(29):
            ts.add(0.5, 1)
        self.assertFalse(ts.fit())

    def test_fit_with_enough(self):
        ts = TemperatureScaler()
        import random; random.seed(7)
        for _ in range(50):
            p = random.uniform(0.2, 0.8)
            ts.add(p, 1 if random.random() < p else 0)
        ts.fit()  # should not crash
        out = ts.calibrate(0.6)
        self.assertGreater(out, 0)
        self.assertLess(out, 1)


class TestCalibrationSuite(unittest.TestCase):

    def test_add_observation_increments_count(self):
        suite = CalibrationSuite()
        n0 = suite.get_status().get("n_observations", 0)
        suite.add_observation(0.7, 1)
        n1 = suite.get_status().get("n_observations", 0)
        self.assertEqual(n1, n0 + 1)

    def test_calibrate_returns_valid_prediction(self):
        suite = CalibrationSuite()
        pred = suite.calibrate(0.65)
        self.assertGreater(pred.calibrated, 0)
        self.assertLess(pred.calibrated, 1)
        self.assertGreater(pred.uncertainty, 0)

    def test_calibrate_identity_unfitted(self):
        suite = CalibrationSuite()
        pred = suite.calibrate(0.40)
        self.assertAlmostEqual(pred.calibrated, 0.40, delta=0.05)

    def test_fit_all_no_crash(self):
        suite = CalibrationSuite()
        import random; random.seed(3)
        for _ in range(80):
            p = random.uniform(0.1, 0.9)
            suite.add_observation(p, 1 if random.random() < p else 0)
        suite.fit_all()  # should not crash

    def test_get_status_keys(self):
        suite = CalibrationSuite()
        status = suite.get_status()
        self.assertIn("method", status)
        self.assertIn("n_observations", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
