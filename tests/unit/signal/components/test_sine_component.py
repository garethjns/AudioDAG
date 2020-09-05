import unittest

import numpy as np

from audiodag.signal.components.sine_component import SineComponent
from tests.unit.signal.components.test_component import SHOW


class TestSineComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = SineComponent(fs=10000, duration=20, mag=1, clip=np.inf)

    def test_plot(self):
        self._sut.plot(show=SHOW)

    def test_normalise_limits_range(self):
        ev_1 = SineComponent(fs=10000, normalise=False)
        ev_2 = SineComponent(fs=10000, normalise=True)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))
        self.assertLessEqual(ev_2.y.max(), 1.0)
        self.assertGreaterEqual(ev_2.y.min(), 0.0)

    def test_cache_keeps_generated_array(self):
        # Arrange
        ev1 = SineComponent(fs=200, duration=20, mag=1, clip=10.0, cache=False)
        ev2 = SineComponent(fs=200, duration=20, mag=1, clip=10.0, cache=True)

        # Act
        ev1_y = ev1.y
        ev2_y = ev2.y

        # Assert
        self.assertIsInstance(ev1_y, np.ndarray)
        self.assertIsInstance(ev2_y, np.ndarray)
        self.assertIsNone(ev1._y)
        self.assertIsNotNone(ev2._y)
