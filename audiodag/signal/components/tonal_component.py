import numpy as np

from audiodag.signal.components.component import Component


class SineComponent(Component):
    """Class specifically for tonal components."""

    def __init__(self,
                 freq: int = 2000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.freq = freq

    def __repr__(self):
        return f"SineComponent(start={self.start}, duration={self.duration}, mag={self.mag}, fs={self.fs}, " \
               f"seed={self.seed}, cache={self.cache}, clip={'np.inf' if np.isinf(self.clip) else self.clip}, " \
               f"normalise={self.normalise}, weight={self.weight}, freq={self.freq})"

    def _generate_f(self) -> np.ndarray:
        """Generate vector for components"""
        return np.sin(np.linspace(0, 4 * np.pi * self.freq, self.duration_pts)) * 0.5 * self.mag
