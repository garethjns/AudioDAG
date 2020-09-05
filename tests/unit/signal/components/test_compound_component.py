import unittest
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent
from audiodag.signal.envelopes.templates import CosRiseEnvelope
from tests.unit.signal.components.test_component import SHOW


class TestCompoundComponent(unittest.TestCase):
    _sut = CompoundComponent

    @staticmethod
    def _mock_compound_event(fs, start, duration_pts) -> MagicMock:
        return MagicMock(fs=fs,
                         start=start,
                         duration_pts=duration_pts,
                         x=np.linspace(start, duration_pts, duration_pts))

    def test_sine_noise_multiply(self):
        # Arrange
        sine_event = SineComponent(fs=800, duration=20)
        noise_event = NoiseComponent(mag=0.3, fs=800, duration=20)

        # Act
        compound_event = sine_event * noise_event

        # Assert
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(16, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)
        self.assertEqual(noise_event.fs, compound_event.fs)
        self.assertEqual(sine_event.fs, compound_event.fs)

    def test_construct_from_list_of_2(self):
        # Arrange
        sine_event = SineComponent()

        # Act
        compound_event = self._sut(components=[sine_event, sine_event])

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(20, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)
        self.assertEqual(2, len(compound_event.events))
        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        self.assertAlmostEqual(float(np.std(sine_event.y)) * 2, float(np.std(compound_event.y)))

    def test_construct_from_list_of_2_with_normalised_weights(self):
        # Arrange
        sine_event = SineComponent()

        # Act
        compound_event = self._sut(components=[sine_event, sine_event], normalise_weights=True)

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(20, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)
        self.assertEqual(2, len(compound_event.events))
        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        # This should be same as weighting is equal proportions by default
        self.assertAlmostEqual(float(np.std(sine_event.y)), float(np.std(compound_event.y)))

    def test_construct_from_list_of_2_set_weights_that_sum_to_one(self):
        # Arrange
        sine_event = SineComponent(weight=0.5, seed=123)
        sine_event_2 = SineComponent(weight=0.5, seed=231)

        # Act
        compound_event = self._sut(components=[sine_event, sine_event_2])

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(20, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)
        self.assertEqual(2, len(compound_event.events))
        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        self.assertAlmostEqual(compound_event.events[0].weight, 0.5)
        self.assertAlmostEqual(compound_event.events[1].weight, 0.5)

    def test_construct_from_list_of_2_set_weights_that_dont_sum_to_one_with_normalise_weights(self):
        sine_event = SineComponent(weight=1.0, seed=123)
        sine_event_2 = SineComponent(weight=0.5, seed=231)

        compound_event = self._sut(components=[sine_event, sine_event_2], normalise_weights=True)

        # Assert
        # Check weight of original objects is unchanged but compound is normalised
        self.assertAlmostEqual(sine_event.weight, 1.0, 3)
        self.assertAlmostEqual(sine_event_2.weight, 0.5, 3)
        self.assertAlmostEqual(compound_event.events[0].weight, 0.667, 3)
        self.assertAlmostEqual(compound_event.events[1].weight, 0.333, 3)
        self.assertEqual(2, len(compound_event.events))
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(20, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

    def test_construct_from_list_of_3(self):
        # Arrange
        sine_event = SineComponent()
        noise_event = NoiseComponent()

        # Act
        compound_event = self._sut(components=[sine_event, sine_event, noise_event])

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(3, len(compound_event.events))
        self.assertEqual(20, compound_event.duration)
        self.assertEqual(20, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

    def test_construct_from_offset_list_of_3_aligned_ms_and_pts(self):
        # Arrange
        sine_event_1 = SineComponent(start=200, duration=1000)
        sine_event_2 = SineComponent(start=300, duration=1000)
        sine_event_3 = SineComponent(start=500, duration=1000)

        # Act
        compound_event = CompoundComponent(components=[sine_event_1, sine_event_2, sine_event_3])

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(3, len(compound_event.events))
        self.assertEqual(1300, compound_event.duration)
        self.assertEqual(1300, compound_event.duration_pts)
        self.assertEqual(200, compound_event.start)
        self.assertEqual(200, compound_event.start_pts)

        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=SHOW)
        compound_event.plot_subplots(show=SHOW)

    def test_construct_from_offset_list_of_3_with_varied_fs(self):
        # Arrange
        sine_event_1 = SineComponent(start=200, duration=1000, fs=150)
        sine_event_2 = SineComponent(start=300, duration=1000, fs=150)
        sine_event_3 = SineComponent(start=500, duration=1000, fs=150)

        # Act
        compound_event = self._sut(components=[sine_event_1, sine_event_2, sine_event_3])

        # Assert
        self.assertEqual(150, compound_event.fs)
        self.assertEqual(3, len(compound_event.events))
        self.assertEqual(1300, compound_event.duration)
        self.assertEqual(195, compound_event.duration_pts)
        self.assertEqual(200, compound_event.start)
        self.assertEqual(30, compound_event.start_pts)

        # TODO: Finish
        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=SHOW)
        compound_event.plot_subplots(show=SHOW)

    def test_construct_from_offset_list_of_3_adjusted_start(self):
        # Arrange
        sine_event_1 = SineComponent(start=200, duration=1000)
        sine_event_2 = SineComponent(start=300, duration=1000)
        sine_event_3 = SineComponent(start=500, duration=1000)

        # Act
        compound_event = self._sut(components=[sine_event_1, sine_event_2, sine_event_3],
                                   start=0)

        # Assert
        self.assertEqual(1000, compound_event.fs)
        self.assertEqual(3, len(compound_event.events))
        self.assertEqual(1300, compound_event.duration)
        self.assertEqual(1300, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

        # TODO: Finish
        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=SHOW)
        compound_event.plot_subplots(show=SHOW)

    def test_incompatible_events_fs_raises_error(self):
        # Arrange
        sine_event = SineComponent()
        noise_event = NoiseComponent(fs=100)

        # Acert
        self.assertRaises(ValueError, lambda: self._sut(components=[sine_event, sine_event, noise_event]))

    def test_create_from_fully_overlapping_long_list(self):
        # Arrange
        rng = np.random.RandomState(123)
        ev_kwargs = {'fs': 2000, 'duration': 100, 'envelope': partial(CosRiseEnvelope, rise=10)}
        n = 40
        evs = [SineComponent(freq=e_i, **ev_kwargs) for e_i in rng.randint(5, 10, n)]

        # Act
        compound_event = self._sut(evs)

        # Assert
        self.assertEqual(200, compound_event.y.shape[0])
        self.assertEqual(n, compound_event.channels().shape[0])
        self.assertEqual(200, compound_event.channels().shape[1])
        self.assertEqual(2000, compound_event.fs)
        self.assertEqual(n, len(compound_event.events))
        self.assertEqual(100, compound_event.duration)
        self.assertEqual(200, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

        compound_event.plot(channels=True,
                            show=SHOW)

    def test_create_from_non_overlapping_long_list(self):
        # Arrange
        rng = np.random.RandomState(123)
        ev_kwargs = {'fs': 2000, 'duration': 100}
        n = 6
        evs = [SineComponent(freq=f, start=s_i * 100,
                             **ev_kwargs) for s_i, f in enumerate(rng.randint(5, 10, 6))]

        # Act
        compound_event = self._sut(evs)

        # Assert
        self.assertEqual(200 * n, compound_event.y.shape[0])
        self.assertEqual(n, compound_event.channels().shape[0])
        self.assertEqual(200 * n, compound_event.channels().shape[1])
        self.assertEqual(2000, compound_event.fs)
        self.assertEqual(n, len(compound_event.events))
        self.assertEqual(100 * n, compound_event.duration)
        self.assertEqual(200 * n, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

        compound_event.plot(channels=True, show=SHOW)

    def test_create_from_partially_overlapping_long_list(self):
        # Arrange
        ev_kwargs = {'fs': 2000, 'duration': 100}
        n = 3
        start_step = 80
        evs = [SineComponent(freq=10, start=s_i * start_step,
                             **ev_kwargs) for s_i in range(n)]

        # Act
        compound_event = self._sut(evs)

        # Assert
        self.assertEqual(n, compound_event.channels().shape[0])
        self.assertEqual(200 + 200 + 120, compound_event.channels().shape[1])
        self.assertEqual(2000, compound_event.fs)
        self.assertEqual(n, len(compound_event.events))
        self.assertEqual(260, compound_event.duration)
        self.assertEqual(520, compound_event.duration_pts)
        self.assertEqual(0, compound_event.start)
        self.assertEqual(0, compound_event.start_pts)

        compound_event.plot(channels=True, show=SHOW)

    def test_eval_repr_equality(self):
        # Arrange
        ev_kwargs = {'fs': 2000,
                     'duration': 100}
        n = 3
        start_step = 80
        evs = [SineComponent(freq=10, start=s_i * start_step, **ev_kwargs) for s_i in range(n)]
        compound_event = CompoundComponent(evs)

        # Act
        clone = eval(compound_event.__repr__())

        # Assert
        self.assertEqual(compound_event, clone)
        self.assertEqual(clone.channels().shape[0], compound_event.channels().shape[0])
        self.assertEqual(clone.channels().shape[1], compound_event.channels().shape[1])
        self.assertEqual(clone.fs, compound_event.fs)
        self.assertEqual(len(clone.events), len(compound_event.events))
        self.assertListEqual(clone.events, compound_event.events)
        self.assertEqual(clone.duration, compound_event.duration)
        self.assertEqual(clone.duration_pts, compound_event.duration_pts)
        self.assertEqual(clone.start, compound_event.start)
        self.assertEqual(clone.start_pts, compound_event.start_pts)

        compound_event.plot(channels=True, show=SHOW)

    def test_previous_start_bug(self):
        """
        This stim raised error when trying to add second event due to subtracting self.start rather than
        self.start_pts in CompoundComponent.channels. Make sure it doesn't come back...
        """
        # Arrange
        ev_kwargs = {'fs': 1600, 'duration': 200}
        noise_events = [NoiseComponent(mag=0.1, start=100, **ev_kwargs),
                        NoiseComponent(mag=0.06, start=300, **ev_kwargs)]

        # Act
        compound_event = self._sut(noise_events)

        compound_event.plot_subplots(show=SHOW)

        # Assert
        try:
            _ = compound_event.y  # Previous error
        except IndexError:
            self.fail("argh, bug.")

        self.assertEqual(2, compound_event.channels().shape[0])
        self.assertEqual(0.4 * 1600, compound_event.channels().shape[1])
        self.assertEqual(1600, compound_event.fs)
        self.assertEqual(2, len(compound_event.events))
        self.assertEqual(500 - 100, compound_event.duration)
        self.assertEqual(0.4 * 1600, compound_event.duration_pts)
        self.assertEqual(100, compound_event.start)
        self.assertEqual(160, compound_event.start_pts)

    def test_normalise_limits_range(self):
        # Arrange
        ev_kwargs = {'fs': 1600, 'duration': 200}
        noise_events = [NoiseComponent(mag=0.1, start=100, **ev_kwargs),
                        NoiseComponent(mag=0.06, start=300, **ev_kwargs)]

        compound_1 = self._sut(noise_events, normalise=False)
        compound_2 = self._sut(noise_events, normalise=True)

        self.assertFalse(compound_1 == compound_2)
        self.assertFalse(np.all(compound_1.y == compound_2.y))
        self.assertLessEqual(compound_2.y.max(), 1.0)
        self.assertGreaterEqual(compound_2.y.min(), 0.0)

    def test_cache_keeps_generated_array(self):
        # Arrange
        ev_kwargs = {'fs': 1600, 'duration': 200}
        noise_events = [NoiseComponent(mag=0.1, start=100, **ev_kwargs),
                        NoiseComponent(mag=0.06, start=300, **ev_kwargs)]

        compound_1 = self._sut(noise_events, cache=False)
        compound_2 = self._sut(noise_events, cache=True)

        # Act
        ev1_y = compound_1.y
        ev2_y = compound_2.y

        # Assert
        self.assertIsInstance(ev1_y, np.ndarray)
        self.assertIsInstance(ev2_y, np.ndarray)
        self.assertIsNone(compound_1._y)
        self.assertIsNotNone(compound_2._y)
