"""
Prototype tests for event combinations.

TODO: Tests incomplete.
"""

import unittest

from audiodag.signal.components.component import CompoundComponent, Component
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent

# Use to show plots when debugging
SHOW = False


class TestCompoundComponent(unittest.TestCase):
    def test_compound_plus_normal_event(self):
        ev_kwargs = {'fs': 2000,
                     'duration': 100}

        n = 3
        start_step = 80
        evs = [SineComponent(freq=10,
                             start=s_i * start_step,
                             **ev_kwargs) for s_i in range(n)]

        compound_event_1 = CompoundComponent(evs)

        noise_event = NoiseComponent(mag=0.06,
                                     fs=2000,
                                     duration=400)

        compound_event_2 = CompoundComponent([compound_event_1, noise_event])
        compound_event_2.plot(show=SHOW,
                              channels=True)

        compound_event_2.channels()

    def test_compound_plus_compound_event(self):
        ev_kwargs = {'fs': 2000,
                     'duration': 100}

        n = 3
        start_step = 80
        evs = [SineComponent(freq=10,
                             start=s_i * start_step,
                             **ev_kwargs) for s_i in range(n)]

        compound_event_1 = CompoundComponent(evs)

        noise_events = [NoiseComponent(mag=0.1,
                                       start=100,
                                       **ev_kwargs),
                        NoiseComponent(mag=0.06,
                                       start=300,
                                       **ev_kwargs)]

        compound_event_2 = CompoundComponent(noise_events)

        compound_event_3 = CompoundComponent([compound_event_1, compound_event_2])
        compound_event_3.plot(show=SHOW,
                              channels=True)

        compound_event_3.channels()

    def test_specific_bug(self):
        """TODO: Testing specific bug from use in msi models stim. Remove or complete."""
        from functools import partial
        event_sine_component = SineComponent(duration=25, freq=8, fs=1200, mag=2)
        event_noise_component = NoiseComponent(duration=25, fs=1200, mag=0.02)
        event = partial(CompoundComponent,
                        events=[event_sine_component, event_noise_component])

        evs = [Component(fs=1200, start=0, duration=900, weight=1.0, seed=1229919515),
               Component(fs=1200, start=114, duration=25, weight=1.0, seed=748261090),
               CompoundComponent(events=[Component(fs=1200, start=139, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=139, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=164, duration=50, weight=1.0, seed=1367662983),
               CompoundComponent(events=[Component(fs=1200, start=214, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=214, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=239, duration=25, weight=1.0, seed=1797495637),
               CompoundComponent(events=[Component(fs=1200, start=264, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=264, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=289, duration=50, weight=1.0, seed=2043303123),
               CompoundComponent(events=[Component(fs=1200, start=339, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=339, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=364, duration=50, weight=1.0, seed=1834626168),
               CompoundComponent(events=[Component(fs=1200, start=414, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=414, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=439, duration=50, weight=1.0, seed=2036979866),
               CompoundComponent(events=[Component(fs=1200, start=489, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=489, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=514, duration=25, weight=1.0, seed=380782414),
               CompoundComponent(events=[Component(fs=1200, start=539, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=539, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=564, duration=50, weight=1.0, seed=775591264),
               CompoundComponent(events=[Component(fs=1200, start=614, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=614, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=639, duration=50, weight=1.0, seed=521866500),
               CompoundComponent(events=[Component(fs=1200, start=689, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=689, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=714, duration=50, weight=1.0, seed=1555919082),
               CompoundComponent(events=[Component(fs=1200, start=764, duration=25, weight=0.5, seed=2094416880),
                                         Component(fs=1200, start=764, duration=25, weight=0.5, seed=371521246)]),
               Component(fs=1200, start=789, duration=50, weight=1.0, seed=1050702081)]

        ev = CompoundComponent(evs)
