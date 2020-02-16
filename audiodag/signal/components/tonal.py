import numpy as np

from audiodag.signal.components.component import Component


class SineComponent(Component):
    """Class specifically for tonal components."""
    def __init__(self,
                 freq: int = 2000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.freq = freq

    def _generate_f(self) -> np.ndarray:
        """Generate vector for components"""
        return np.sin(np.linspace(0, 4 * np.pi * self.freq, self.duration_pts)) * 0.5 * self.mag
