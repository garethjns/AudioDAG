import copy
import os
import unittest

import numpy as np

from audiodag.signal.components.component import Component, CompoundComponent

# Use to show plots when debugging
SHOW = os.environ.get('SHOW_DEBUG_PLOTS') is not None


class TestComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = Component(fs=10000, duration=20, mag=1, clip=10.0)

    def test_length_as_expected(self):
        self.assertEqual(len(self._sut.y), 200)

    def test_constant_envelope_has_no_effect_on_energy_in_y(self):
        ev = copy.deepcopy(self._sut)
        ev.envelope = lambda x: x

        self.assertListEqual(list(ev.y), list(np.ones(shape=(200,))))

    def test_plot_raises_no_exceptions(self):
        self._sut.plot(show=SHOW)

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)

    def test_sine_noise_multiply_inconsistent_fs_raises_error(self):
        other_event = Component(fs=200, duration=20, mag=1, clip=10.0)

        self.assertRaises(ValueError, lambda: self._sut * other_event)

    def test_construct_from_list_of_2(self):
        compound_event = CompoundComponent(events=[self._sut, self._sut])

        self.assertAlmostEqual(float(np.mean(self._sut.y)), float(np.mean(compound_event.y)), 0)
        # This should be same as weighting is equal proportions by default
        self.assertAlmostEqual(float(np.std(compound_event.y)), float(np.std(self._sut.y)))

    def test_construct_from_list_of_2_inconsistent_fs_raises_error(self):
        other_event = Component(fs=200, duration=20, mag=1, clip=10.0)
        self.assertRaises(ValueError, lambda: CompoundComponent(events=[self._sut, other_event]))
