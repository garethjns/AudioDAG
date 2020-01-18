import numpy as np

from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.envelopes.envelope import Envelope
from audiodag.signal.events.event import CompoundEvent
from audiodag.signal.events.noise import NoiseEvent
from audiodag.signal.events.tonal import SineEvent


def template_complex():
    """
    Sine waves of increasing complexity in 3 steps:
    4 Hz -> 4 + (6 and 2) ->  4 + (6 and 2) + (8 and 10)

    Overlayed with increasing noise.
    :return:
    """
    start = 0
    sine_4 = SineEvent(start=start,
                       duration=1400,
                       freq=4)

    start = 200
    duration = 600
    sine_2_6 = CompoundEvent([SineEvent(start=start,
                                        duration=duration,
                                        freq=2),
                              SineEvent(start=start,
                                        duration=duration,
                                        freq=6)])

    start = 600
    duration = 1000
    sine_2_12 = CompoundEvent([SineEvent(start=start,
                                         duration=duration,
                                         freq=2),
                               SineEvent(start=start,
                                         duration=duration,
                                         freq=12)])
    sine_8_10 = CompoundEvent([SineEvent(start=start,
                                         duration=duration,
                                         freq=8),
                               SineEvent(start=start,
                                         duration=duration,
                                         freq=10),
                               sine_2_12])

    class IncreasingEnvelope(Envelope):
        def f(self, y):
            return y * np.linspace(0, 1, len(y))

    noise = NoiseEvent(start=0,
                       duration=1000,
                       envelope=IncreasingEnvelope,
                       mag=db_to_lin(ref=1,
                                     db_change=-120))

    return CompoundEvent([sine_4, sine_2_6, sine_8_10, noise])


if __name__ == "__main__":
    ev = template_complex()
    ev.plot(show=True)
    ev.plot_subplots(show=True)

    ev.to_list()
