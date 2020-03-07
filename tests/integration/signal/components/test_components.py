"""
Prototype tests for event combinations.

TODO: Tests incomplete.
"""

import unittest

from audiodag.signal.components.component import CompoundComponent
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
