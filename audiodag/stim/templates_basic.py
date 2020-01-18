from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.events.tonal import SineEvent
from audiodag.signal.events.noise import NoiseEvent
from audiodag.signal.events.event import CompoundEvent
from audiodag.stim.templates_complex import template_complex


def template_noisy_sine(duration: int = 1000,
                        fs: int = 5000,
                        freq: int = 12,
                        noise_mag: float = -80) -> CompoundEvent:
    """

    :param: fs: Sampling rate, Hz.
    :param duration: Duration in ms.
    :param freq: Frequency of sine wave, Hz.
    :param noise_mag: Noise mag in db (relative to sine mag of 1v)
    :return:
    """
    sin = SineEvent(freq=freq,
                    mag=1,
                    fs=fs,
                    duration=duration)
    noise = NoiseEvent(fs=fs,
                       duration=duration,
                       mag=db_to_lin(ref=1,
                                     db_change=noise_mag))

    return CompoundEvent([sin, noise])


if __name__ == "__main__":

    ev = template_noisy_sine()
    ev.plot(show=True,
            channels=True)
