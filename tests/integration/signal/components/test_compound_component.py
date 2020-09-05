import os
import unittest

from audiodag.signal.components.component import CompoundComponent, Component
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent

# Use to show plots when debugging
SHOW = os.environ.get('SHOW_DEBUG_PLOTS') is not None


class TestCompoundComponent(unittest.TestCase):
    _sut = CompoundComponent

    def test_compound_plus_normal_event(self):
        # Arrange
        ev_kwargs = {'fs': 1500, 'duration': 100}
        n = 3
        start_step = 80
        evs = [SineComponent(freq=10, start=s_i * start_step, **ev_kwargs) for s_i in range(n)]
        compound_event_1 = CompoundComponent(evs)
        noise_event = NoiseComponent(mag=0.06, fs=ev_kwargs['fs'], duration=400)

        # Act
        compound_event_2 = CompoundComponent([compound_event_1, noise_event])

        # Assert
        self.assertEqual(600, compound_event_2.y.shape[0])
        self.assertEqual(600, compound_event_2.channels().shape[1])
        self.assertEqual(600, compound_event_2.duration_pts)
        self.assertEqual(400, compound_event_2.duration)
        self.assertEqual(2, compound_event_2.channels().shape[0])
        self.assertEqual(2, len(compound_event_2.events))
        self.assertEqual(1500, compound_event_2.fs)
        self.assertEqual(0, compound_event_2.start)
        self.assertEqual(0, compound_event_2.start_pts)

        compound_event_2.plot(show=SHOW, channels=True)
        compound_event_1.plot_subplots(show=SHOW)
        compound_event_2.plot_subplots(show=SHOW)

    def test_compound_plus_compound_event(self):
        # Arrange
        ev_kwargs = {'fs': 1600, 'duration': 200}
        n = 3
        start_step = 80
        evs = [SineComponent(freq=10, start=s_i * start_step, **ev_kwargs) for s_i in range(n)]
        compound_event_1 = CompoundComponent(evs)
        noise_events = [NoiseComponent(mag=0.1, start=100, **ev_kwargs),
                        NoiseComponent(mag=0.06, start=300, **ev_kwargs)]
        compound_event_2 = CompoundComponent(noise_events)

        # Act
        compound_event_3 = self._sut([compound_event_1, compound_event_2])

        # Assert
        self.assertEqual(640, compound_event_2.y.shape[0])
        self.assertEqual(640, compound_event_2.channels().shape[1])
        self.assertEqual(640, compound_event_2.duration_pts)
        self.assertEqual(400, compound_event_2.duration)
        self.assertEqual(2, compound_event_2.channels().shape[0])
        self.assertEqual(2, len(compound_event_2.events))
        self.assertEqual(1600, compound_event_2.fs)
        self.assertEqual(100, compound_event_2.start)
        self.assertEqual(160, compound_event_2.start_pts)

        compound_event_1.plot_subplots(show=SHOW)
        compound_event_2.plot(show=SHOW, channels=True)
        compound_event_2.plot_subplots(show=SHOW)
        compound_event_3.plot(show=SHOW, channels=True)
        compound_event_3.plot_subplots(show=SHOW)

    def test_eval_or_repr_equal_for_complex_event(self):
        # Arrange
        ev_kwargs = {'fs': 1600, 'duration': 200}
        n = 3
        start_step = 80
        evs = [SineComponent(freq=10, start=s_i * start_step, **ev_kwargs) for s_i in range(n)]
        compound_event_1 = CompoundComponent(evs)
        noise_events = [NoiseComponent(mag=0.1, start=100, **ev_kwargs),
                        NoiseComponent(mag=0.06, start=300, **ev_kwargs)]
        compound_event_2 = CompoundComponent(noise_events)
        compound_event_3 = CompoundComponent([compound_event_1, compound_event_2])

        # Act
        clone = eval(compound_event_3.__repr__())

        # Assert
        self.assertEqual(compound_event_3, clone)
        self.assertListEqual(compound_event_3.events, clone.events)
        self.assertEqual(clone.channels().shape[0], compound_event_3.channels().shape[0])
        self.assertEqual(clone.channels().shape[1], compound_event_3.channels().shape[1])
        self.assertEqual(clone.fs, compound_event_3.fs)
        self.assertEqual(len(clone.events), len(compound_event_3.events))
        self.assertListEqual(clone.events, compound_event_3.events)
        self.assertEqual(clone.duration, compound_event_3.duration)
        self.assertEqual(clone.duration_pts, compound_event_3.duration_pts)
        self.assertEqual(clone.start, compound_event_3.start)
        self.assertEqual(clone.start_pts, compound_event_3.start_pts)

    def test_specific_msi_models_two_gap_bug_working_condition(self):
        """Testing specific bug from use in msi models stim."""
        stim = CompoundComponent(components=[
            Component(start=0, duration=2000, mag=0.02, fs=1000, seed=1283778829, cache=False, clip=2.0,
                      weight=0.08695652173913043),
            Component(start=1205, duration=50, mag=0.02, fs=1000, seed=604704893, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1255, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1255, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1280, duration=25, mag=0.02, fs=1000, seed=265799168, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1305, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1305, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1330, duration=50, mag=0.02, fs=1000, seed=335768414, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1380, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1380, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1405, duration=50, mag=0.02, fs=1000, seed=1914910913, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1455, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1455, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1480, duration=25, mag=0.02, fs=1000, seed=98974688, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1505, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1505, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1530, duration=25, mag=0.02, fs=1000, seed=1642792235, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1555, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1555, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1580, duration=50, mag=0.02, fs=1000, seed=246687274, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1630, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1630, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1655, duration=50, mag=0.02, fs=1000, seed=449561071, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1705, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1705, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1730, duration=25, mag=0.02, fs=1000, seed=1339302259, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1755, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1755, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1780, duration=50, mag=0.02, fs=1000, seed=700116608, cache=False, clip=2.0,
                      weight=0.043478260869565216), CompoundComponent(components=[
                Component(start=1830, duration=25, mag=2, fs=1000, seed=885587907, cache=False, clip=2.0, weight=0.5),
                Component(start=1830, duration=25, mag=0.02, fs=1000, seed=1242102769, cache=False, clip=2.0,
                          weight=0.5)]),
            Component(start=1855, duration=50, mag=0.02, fs=1000, seed=850417960, cache=False, clip=2.0,
                      weight=0.043478260869565216)])

        self.assertEqual(2000, len(stim.y))
