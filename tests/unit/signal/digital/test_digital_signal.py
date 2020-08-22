import unittest
from unittest.mock import patch

import numpy as np

from audiodag.signal.digital.conversion import pts_to_ms, ms_to_pts
from audiodag.signal.digital.digital_siginal import DigitalSignal
from unittest.mock import MagicMock

# Use to show plots when debugging
SHOW = False


class TestFunctions(unittest.TestCase):
    def test_ms_to_pts(self):
        self.assertEqual(ms_to_pts(t_ms=1000,
                                   fs=1000), 1000)

        self.assertEqual(ms_to_pts(t_ms=500,
                                   fs=1000), 500)

        self.assertEqual(ms_to_pts(t_ms=1000,
                                   fs=500), 500)

        self.assertEqual(ms_to_pts(t_ms=20,
                                   fs=800), 16)

    def test_pts_to_ms(self):
        self.assertEqual(pts_to_ms(t_pts=1000,
                                   fs=1000), 1000)

        self.assertEqual(pts_to_ms(t_pts=500,
                                   fs=1000), 500)


class TestDigitalSignal(unittest.TestCase):

    @classmethod
    @patch.multiple(DigitalSignal, __abstractmethods__=set())
    def setUpClass(cls) -> None:
        cls._sut_1 = DigitalSignal(fs=100,
                                   duration=1000)
        cls._sut_2 = DigitalSignal(fs=100,
                                   duration=500)

    def test_equality_between_equals(self):
        self.assertEqual(self._sut_1, self._sut_1)

    def test_equality_between_unequals(self):
        self.assertNotEqual(self._sut_1, self._sut_2)

    def test_unique_works_as_expected(self):
        self.assertEqual(len(np.unique([self._sut_1, self._sut_1])), 1)
        self.assertEqual(len(np.unique([self._sut_1, self._sut_2])), 2)

    def test_duration_in_pts_property_converts_as_expected(self):
        self.assertEqual(self._sut_1.duration_pts, 100)
        self.assertEqual(self._sut_2.duration_pts, 50)

    @patch.multiple(DigitalSignal, __abstractmethods__=set())
    def test_init_with_defaults_sets_correct_duration(self):
        dt = DigitalSignal()

        self.assertEqual(len(dt.x), 20)
        self.assertEqual(len(dt.x_pts), 20)

    def test_no_exceptions_on_plot_call(self):
        self._sut_1.plot(show=SHOW)

    @patch.multiple(DigitalSignal, __abstractmethods__=set())
    def test_eval_repr_equality(self):
        """Note that __repr__ is currently used to define equality for DigitalSignal, making this test currently
        somewhat pointless, but this may change in the future."""

        # Used by eval
        # noinspection PyPep8Naming
        ConstantEnvelope = MagicMock()

        clone = eval(self._sut_1.__repr__())
        self.assertEqual(self._sut_1, clone)
