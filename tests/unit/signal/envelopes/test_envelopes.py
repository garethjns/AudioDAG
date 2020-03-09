import unittest

import numpy as np

from audiodag.signal.envelopes.templates import ConstantEnvelope, CosEnvelope, CosRiseEnvelope

# Use to show plots when debugging
SHOW = False

ONES = np.ones(shape=(1000,))
ZEROS = np.zeros(shape=(800,))


class TestConstantEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = ConstantEnvelope()

    def test_envelope_no_effect_on_energy_in_constant_signal(self) -> None:
        self.assertTrue(np.all(self._sut(ONES) == ONES))

    def test_envelope_no_effect_on_energy_in_empty_signal(self) -> None:
        self.assertTrue(np.all(self._sut(ZEROS) == ZEROS))

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)


class TestCosEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = CosEnvelope()

    def test_envelope_reduces_energy_of_constant_signal(self) -> None:
        self.assertLess(np.sum(self._sut(ONES)), np.sum(ONES))

    def test_envelope_no_effect_on_energy_in_empty_signal(self) -> None:
        self.assertTrue(np.all(np.round(self._sut(ZEROS)) == ZEROS))

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)


class TestCosRiseEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = CosRiseEnvelope(fs=1000,
                                   rise=200)

    def test_envelope_reduces_energy_of_constant_signal(self) -> None:
        self.assertLess(np.sum(self._sut(ONES)), np.sum(ONES))

    def test_envelope_no_effect_on_energy_in_empty_signal(self) -> None:
        self.assertTrue(np.all(self._sut(ZEROS) == ZEROS))

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)
