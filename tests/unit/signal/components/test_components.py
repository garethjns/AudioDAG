import copy
import unittest
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from audiodag.signal.components.component import Component, CompoundComponent
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent
from audiodag.signal.envelopes.templates import CosRiseEnvelope

# Use to show plots when debugging
SHOW = False


class TestComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = Component(fs=10000,
                             duration=20,
                             mag=1,
                             clip=10.0)

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


class TestNoiseComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = NoiseComponent(fs=10000,
                                  duration=20,
                                  mag=1,
                                  clip=np.inf,
                                  cache=True)

    def test_invalid_noise_dist_raises_error(self):
        self.assertRaises(ValueError, lambda: NoiseComponent(fs=10000,
                                                             duration=20,
                                                             mag=1,
                                                             clip=np.inf,
                                                             dist='invalid'))

    def test_expected_signal_statistics_between_normal_and_uniform_dists(self):
        ev_norm = NoiseComponent(fs=10000,
                                 duration=20,
                                 mag=1,
                                 clip=np.inf,
                                 dist='normal')

        ev_uniform = NoiseComponent(fs=10000,
                                    duration=20,
                                    mag=1,
                                    clip=np.inf,
                                    dist='uniform')

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

    def test_no_seed_different_y(self):
        ev_1 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf)

        ev_2 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_same_seed_same_y(self):
        ev_1 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf,
                              seed=1)

        ev_2 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf,
                              seed=1)

        self.assertTrue(ev_1 == ev_2)
        self.assertTrue(np.all(ev_1.y == ev_2.y))

    def test_diff_seed_diff_y(self):
        ev_1 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf,
                              seed=1)

        ev_2 = NoiseComponent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf,
                              seed=2)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_plot_raises_no_errors(self):
        self._sut.plot(show=SHOW)

    def test_eval_repr_equality(self):
        clone = eval(self._sut.__repr__())
        self.assertEqual(self._sut, clone)


class TestSineComponent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._sut = SineComponent(fs=10000,
                                 duration=20,
                                 mag=1,
                                 clip=np.inf)

    def test_plot(self):
        self._sut.plot(show=SHOW)


class TestCompoundComponent(unittest.TestCase):

    @staticmethod
    def _mock_compound_event(fs, start, duration_pts) -> MagicMock:
        return MagicMock(fs=fs,
                         start=start,
                         duration_pts=duration_pts,
                         x=np.linspace(start, duration_pts, duration_pts))

    def test_sine_noise_multiply(self):
        sine_event = SineComponent(fs=10000)
        noise_event = NoiseComponent(mag=0.3,
                                     fs=10000)

        compound_event = sine_event * noise_event

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 1)

    def test_construct_from_list_of_2(self):
        sine_event = SineComponent()
        compound_event = CompoundComponent(events=[sine_event, sine_event])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        # This should be same as weighting is equal proportions by default
        self.assertAlmostEqual(float(np.std(compound_event.y)), float(np.std(sine_event.y)))

    def test_construct_from_list_of_2_set_weights_that_sum_to_one(self):
        sine_event = SineComponent(weight=0.5,
                                   seed=123)
        sine_event_2 = SineComponent(weight=0.5,
                                     seed=231)
        compound_event = CompoundComponent(events=[sine_event, sine_event_2])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)

        self.assertAlmostEqual(compound_event.events[0].weight, 0.5)
        self.assertAlmostEqual(compound_event.events[1].weight, 0.5)

    def test_construct_from_list_of_2_set_weights_that_dont_sum_to_one(self):
        sine_event = SineComponent(weight=1.0,
                                   seed=123)
        sine_event_2 = SineComponent(weight=0.5,
                                     seed=231)

        self.assertAlmostEqual(sine_event.weight, 1.0, 3)
        self.assertAlmostEqual(sine_event_2.weight, 0.5, 3)

        compound_event = CompoundComponent(events=[sine_event, sine_event_2])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)

        self.assertAlmostEqual(compound_event.events[0].weight, 0.667, 3)
        self.assertAlmostEqual(compound_event.events[1].weight, 0.333, 3)

        # Check weight of original objects is unchanged
        self.assertAlmostEqual(sine_event.weight, 1.0, 3)
        self.assertAlmostEqual(sine_event_2.weight, 0.5, 3)

    def test_construct_from_list_of_3(self):
        sine_event = SineComponent()
        noise_event = NoiseComponent()
        compound_event = CompoundComponent(events=[sine_event, sine_event, noise_event])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)

    def test_construct_from_offset_list_of_3(self):
        sine_event_1 = SineComponent(start=200,
                                     duration=1000)
        sine_event_2 = SineComponent(start=300,
                                     duration=1000)
        sine_event_3 = SineComponent(start=500,
                                     duration=1000)

        compound_event = CompoundComponent(events=[sine_event_1, sine_event_2, sine_event_3])

        # TODO: Finish
        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=SHOW)
        compound_event.plot_subplots(show=SHOW)

    def test_construct_from_offset_list_of_3_adjusted_start(self):
        sine_event_1 = SineComponent(start=200,
                                     duration=1000)
        sine_event_2 = SineComponent(start=300,
                                     duration=1000)
        sine_event_3 = SineComponent(start=500,
                                     duration=1000)

        compound_event = CompoundComponent(events=[sine_event_1, sine_event_2, sine_event_3],
                                           start=0)

        # TODO: Finish
        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=SHOW)
        compound_event.plot_subplots(show=SHOW)

    def test_incompatible_events_fs_raises_error(self):
        sine_event = SineComponent()
        noise_event = NoiseComponent(fs=100)

        self.assertRaises(ValueError, lambda: CompoundComponent(events=[sine_event, sine_event, noise_event]))

    def test_create_from_fully_overlapping_long_list(self):
        rng = np.random.RandomState(123)

        ev_kwargs = {'fs': 2000,
                     'duration': 100,
                     'envelope': partial(CosRiseEnvelope,
                                         rise=10)}

        n = 40
        evs = [SineComponent(freq=e_i, **ev_kwargs) for e_i in rng.randint(5, 10, n)]

        compound_event = CompoundComponent(evs)
        compound_event.plot(channels=True,
                            show=SHOW)

        self.assertEqual(compound_event.y.shape[0], 200)
        self.assertEqual(compound_event.channels().shape[0], n)
        self.assertEqual(compound_event.channels().shape[1], 200)

    def test_create_from_non_overlapping_long_list(self):
        rng = np.random.RandomState(123)

        ev_kwargs = {'fs': 2000,
                     'duration': 100}

        n = 6
        evs = [SineComponent(freq=f,
                             start=s_i * 100,
                             **ev_kwargs) for s_i, f in enumerate(rng.randint(5, 10, 6))]

        compound_event = CompoundComponent(evs)
        compound_event.plot(channels=True,
                            show=SHOW)

        self.assertEqual(compound_event.y.shape[0], 200 * n)
        self.assertEqual(compound_event.channels().shape[0], n)
        self.assertEqual(compound_event.channels().shape[1], 200 * n)

    def test_create_from_partially_overlapping_long_list(self):
        ev_kwargs = {'fs': 2000,
                     'duration': 100}

        n = 3
        start_step = 80
        evs = [SineComponent(freq=10,
                             start=s_i * start_step,
                             **ev_kwargs) for s_i in range(n)]

        compound_event = CompoundComponent(evs)
        compound_event.plot(channels=True,
                            show=SHOW)

        self.assertEqual(compound_event.y.shape[0], 200 + 200 + 120)
        self.assertEqual(compound_event.channels().shape[0], n)
        self.assertEqual(compound_event.channels().shape[1], 200 + 200 + 120)

    def test_eval_repr_equality(self):
        ev_kwargs = {'fs': 2000,
                     'duration': 100}

        n = 3
        start_step = 80
        evs = [SineComponent(freq=10,
                             start=s_i * start_step,
                             **ev_kwargs) for s_i in range(n)]

        compound_event = CompoundComponent(evs)
        compound_event.plot(channels=True,
                            show=SHOW)

        clone = eval(compound_event.__repr__())
        self.assertEqual(compound_event, clone)
