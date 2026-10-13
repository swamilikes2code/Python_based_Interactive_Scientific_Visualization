import unittest
from unittest.mock import patch

import numpy as np

from engine.sindy_model import SINDyEngine


class _ZeroPredictModel:
    def predict(self, X):
        return np.zeros_like(X)


class _ZeroSimulationModel:
    def simulate(self, x0, t):
        return np.zeros((len(t), len(x0)))


class _FailingSimulationModel:
    def simulate(self, x0, t):
        raise ValueError("solver exploded")


class _ProbeSINDy:
    def fit(self, *args, **kwargs):
        return self

    def get_feature_names(self):
        return ["x"]


class _BootstrapSINDy:
    def __init__(self, coefficient=None, fail=False):
        self.coefficient = coefficient
        self.fail = fail

    def fit(self, *args, **kwargs):
        if self.fail:
            raise ValueError("intentional bootstrap failure")
        return self

    def coefficients(self):
        return np.array([[self.coefficient]], dtype=float)


class SINDyEngineTests(unittest.TestCase):
    def test_pool_trajectories_differentiates_each_trajectory_separately(self):
        engine = SINDyEngine()
        calls = []

        def fake_derivatives(X, t):
            calls.append(np.asarray(t).copy())
            return np.zeros_like(X, dtype=float)

        engine.compute_derivatives = fake_derivatives
        trajectory_1 = (np.ones((5, 1)), np.arange(5, dtype=float))
        trajectory_2 = (np.ones((7, 1)), np.arange(7, dtype=float))
        X_pool, dX_pool, t_pool = engine._pool_trajectories(
            [trajectory_1, trajectory_2])

        self.assertEqual([len(t) for t in calls], [5, 7])
        self.assertEqual(X_pool.shape, (12, 1))
        self.assertEqual(dX_pool.shape, (12, 1))
        self.assertEqual(len(t_pool), 12)

    def test_multi_trajectory_diagnostics_align_different_fft_grids(self):
        engine = SINDyEngine()
        engine.model = _ZeroPredictModel()
        engine.feature_names = ["x"]

        t1 = np.arange(1000) * 0.01
        t2 = np.arange(1000) * 0.02
        X1 = np.sin(2 * np.pi * t1)[:, None]
        X2 = np.sin(2 * np.pi * t2)[:, None]
        diagnostics = engine.compute_diagnostics_multi([(X1, t1), (X2, t2)])

        freqs = diagnostics["fft_freqs"]
        amplitudes = diagnostics["fft_amps"]["x"]
        peak = freqs[1:][np.argmax(amplitudes[1:])]
        self.assertAlmostEqual(freqs[1] - freqs[0], 0.1)
        self.assertAlmostEqual(freqs[-1], 25.0)
        self.assertAlmostEqual(peak, 1.0)
        self.assertEqual(len(diagnostics["residual_segments"]), 2)
        self.assertEqual(
            [segment["label"] for segment in diagnostics["residual_segments"]],
            ["IC1", "IC2"],
        )

    def test_ensemble_inclusion_uses_only_successful_fits(self):
        engine = SINDyEngine()
        t = np.linspace(0, 2, 40)
        X = np.sin(t)[:, None]
        fake_models = [
            _ProbeSINDy(),
            _BootstrapSINDy(coefficient=1.0),
            _BootstrapSINDy(fail=True),
            _BootstrapSINDy(coefficient=0.0),
        ]

        with patch("engine.sindy_model.ps.SINDy", side_effect=fake_models), \
                patch("builtins.print"):
            result = engine.fit_ensemble(
                X, t, poly_degree=1, threshold=0.1, names=["x"],
                n_bootstrap=3,
            )

        self.assertEqual(result["n_successful_bootstrap"], 2)
        self.assertEqual(result["n_failed_bootstrap"], 1)
        self.assertEqual(result["per_state"]["x"]["inclusion_pct"]["x"], 0.5)

    def test_ensemble_raises_when_every_bootstrap_fails(self):
        engine = SINDyEngine()
        t = np.linspace(0, 2, 40)
        X = np.sin(t)[:, None]
        fake_models = [
            _ProbeSINDy(),
            _BootstrapSINDy(fail=True),
            _BootstrapSINDy(fail=True),
        ]

        with patch("engine.sindy_model.ps.SINDy", side_effect=fake_models), \
                patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "All ensemble"):
                engine.fit_ensemble(
                    X, t, poly_degree=1, threshold=0.1, names=["x"],
                    n_bootstrap=2,
                )

    def test_zero_trajectory_is_a_valid_prediction(self):
        engine = SINDyEngine()
        t = np.linspace(0, 10, 50)
        result = engine.simulate_with_model(_ZeroSimulationModel(), [0.0, 0.0], t)
        self.assertTrue(np.all(result == 0.0))

    def test_simulation_failure_raises_instead_of_returning_fake_zeros(self):
        engine = SINDyEngine()
        with self.assertRaisesRegex(RuntimeError, "solver exploded"):
            engine.simulate_with_model(
                _FailingSimulationModel(), [1.0], np.linspace(0, 1, 5))


if __name__ == "__main__":
    unittest.main()
