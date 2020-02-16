import numpy as np

from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.envelopes.templates import IncreasingEnvelope
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent


def template_complex():
    """
    Sine waves of increasing complexity in 3 steps:
    4 Hz -> 4 + (6 and 2) ->  4 + (6 and 2) + (8 and 10)

    Overlayed with increasing noise.
    :return:
    """
    start = 0
    sine_4 = SineComponent(start=start,
                           duration=1400,
                           freq=4)

    start = 200
    duration = 600
    sine_2_6 = CompoundComponent([SineComponent(start=start,
                                                duration=duration,
                                                freq=2),
                                  SineComponent(start=start,
                                                duration=duration,
                                                freq=6)])

    start = 600
    duration = 1000
    sine_2_12 = CompoundComponent([SineComponent(start=start,
                                                 duration=duration,
                                                 freq=2),
                                   SineComponent(start=start,
                                                 duration=duration,
                                                 freq=12)])
    sine_8_10 = CompoundComponent([SineComponent(start=start,
                                                 duration=duration,
                                                 freq=8),
                                   SineComponent(start=start,
                                                 duration=duration,
                                                 freq=10),
                                   sine_2_12])

    noise = NoiseComponent(start=0,
                           duration=1000,
                           envelope=IncreasingEnvelope,
                           mag=db_to_lin(ref=1,
                                     db_change=-120))

    return CompoundComponent([sine_4, sine_2_6, sine_8_10, noise])


if __name__ == "__main__":
    ev = template_complex()

    ev.to_list()

    # Plot final signal
    ev.plot(show=True)
    # Plot components of compound components
    ev.plot_subplots(show=True)
    ev.events[1].plot_subplots(show=True)
    ev.events[2].plot_subplots(show=True)
    ev.events[2].events[2].plot_subplots(show=True)

    ev.to_list()
