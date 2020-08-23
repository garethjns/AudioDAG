import copy
import unittest

import numpy as np

from audiodag.signal.components.noise_component import NoiseComponent
from tests.unit.signal.components.test_component import SHOW


class TestNoiseComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, cache=True)

    def test_invalid_noise_dist_raises_error(self):
        self.assertRaises(ValueError, lambda: NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, dist='invalid'))

    def test_expected_signal_statistics_between_normal_and_uniform_dists(self):
        ev_norm = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, dist='normal')

        ev_uniform = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, dist='uniform')

        ev_norm.plot(show=SHOW)
        ev_uniform.plot(show=SHOW)

        self.assertAlmostEqual(float(np.mean(ev_norm.y)), 0.0, -1)
        self.assertLess(float(np.mean(ev_norm.y)), float(np.mean(ev_uniform.y)), 0)
        self.assertLess(float(np.std(ev_uniform.y)), float(np.std(ev_norm.y)))

    def test_consistent_y_each_regen(self):
        y_1 = copy.deepcopy(self._sut)
        y_1.clear()

        y_2 = copy.deepcopy(self._sut)

        self.assertTrue(np.all(y_1.y == y_2.y))

    def test_normalise_limits_range(self):
        ev_1 = NoiseComponent(fs=10000, seed=123, duration=20, mag=1, clip=np.inf, normalise=False)
        ev_2 = NoiseComponent(fs=10000, seed=123, duration=20, mag=1, clip=np.inf, normalise=True)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))
        self.assertLessEqual(ev_2.y.max(), 1.0)
        self.assertGreaterEqual(ev_2.y.min(), 0.0)

    def test_no_seed_different_y(self):
        ev_1 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf)

        ev_2 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_same_seed_same_y(self):
        ev_1 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, seed=1)

        ev_2 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, seed=1)

        self.assertTrue(ev_1 == ev_2)
        self.assertTrue(np.all(ev_1.y == ev_2.y))

    def test_diff_seed_diff_y(self):
        ev_1 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, seed=1)

        ev_2 = NoiseComponent(fs=10000, duration=20, mag=1, clip=np.inf, seed=2)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_plot_raises_no_errors(self):
        self._sut.plot(show=SHOW)

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)

    def test_cache_keeps_generated_array(self):
        # Arrange
        ev1 = NoiseComponent(fs=200, duration=20, mag=1, clip=10.0, cache=False)
        ev2 = NoiseComponent(fs=200, duration=20, mag=1, clip=10.0, cache=True)

        # Act
        ev1_y = ev1.y
        ev2_y = ev2.y

        # Assert
        self.assertIsInstance(ev1_y, np.ndarray)
        self.assertIsInstance(ev2_y, np.ndarray)
        self.assertIsNone(ev1._y)
        self.assertIsNotNone(ev2._y)
