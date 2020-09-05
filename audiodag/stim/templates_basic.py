from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent
from audiodag.signal.digital.conversion import db_to_lin


def template_noisy_sine(duration: int = 1000,
                        fs: int = 5000,
                        freq: int = 12,
                        noise_mag: float = -80) -> CompoundComponent:
    """

    :param: fs: Sampling rate, Hz.
    :param duration: Duration in ms.
    :param freq: Frequency of sine wave, Hz.
    :param noise_mag: Noise mag in db (relative to sine mag of 1v)
    :return:
    """
    sin = SineComponent(freq=freq,
                        mag=1,
                        start=100,
                        fs=fs,
                        duration=duration)
    noise = NoiseComponent(fs=fs,
                           duration=duration,
                           mag=db_to_lin(ref=1,
                                         db_change=noise_mag))

    return CompoundComponent([sin, noise])


if __name__ == "__main__":
    ev = template_noisy_sine()
    ev.plot(show=True,
            channels=True)

    ev.plot_subplots(show=True)
