import numpy as np

from audiodag.signal.components.component import Component


class NoiseComponent(Component):
    """Class specifically for noisy components."""

    def __init__(self,
                 dist: str = 'normal',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist = dist

    def __repr__(self):
        return f"NoiseComponent(start={self.start}, duration={self.duration}, mag={self.mag}, fs={self.fs}, " \
               f"seed={self.seed}, cache={self.cache}, clip={'np.inf' if np.isinf(self.clip) else self.clip}, " \
               f"weight={self.weight}, normalise={self.normalise}, dist='{self.dist}')"

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist: str) -> None:
        if dist.lower() not in ['normal', 'uniform']:
            raise ValueError(f"Dist {dist} is not valid.")

        self._dist = dist

    def _generate_f(self) -> np.ndarray:
        """Sample noise from the RandomState."""
        return getattr(self.state, self.dist)(size=(self.duration_pts,)) * self.mag
