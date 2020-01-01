from functools import reduce, partial
from typing import List, Tuple, Callable, Union

import numpy as np
import matplotlib.pyplot as plt

from signal.digital.conversion import pts_to_ms
from signal.digital.digital_siginal import DigitalSignal
from signal.envelopes.templates import ConstantEnvelope


class Event(DigitalSignal):
    def __repr__(self):
        return f"Event(fs={self.fs}, duration={self.duration}, seed={self.seed})"

    def _generate_f(self) -> np.ndarray:
        """Default events is constant 1s * mag"""
        return np.ones(shape=(self.duration_pts,)) * self.mag

    def __mul__(self, other):
        """
        Multiplying events generates a CompoundEvent object with a generator for the combined signals.

        Weighting is even.
        """

        return CompoundEvent(events=[self, other])


class CompoundEvent(Event):
    """
    Object for combining events, for example adding noise to another events.

    Supports combination of multiple events, but only with equal weighting and same durations for now.
    """
    def __init__(self, events: List[Event],
                 weights: List[float] = None,
                 envelope = ConstantEnvelope):

        self._verify_event_list(events)
        start, _, duration = self._new_duration(events)

        super().__init__(fs=events[0].fs,
                         start=start,
                         duration=pts_to_ms(duration,
                                            fs=events[0].fs))

        if weights is None:
            weights = [1 / len(events) for _ in range(len(events))]
        self.weights = weights
        self.events = events

        self._generate_f = self._make_generate_f()

    @property


    @staticmethod
    def _verify_event_list(events: List[Event]):
        check_params = ['fs']

        for p in check_params:
            param_values = [getattr(e, p) for e in events]
            if len(np.unique(param_values)) > 1:
                raise ValueError(f"Param {p} is inconsistent across events: {param_values}")

    @staticmethod
    def _new_duration(events: List[Event]):
        start = reduce(lambda ev_a, ev_b: min(ev_a, ev_b), [e.x_pts.min() for e in events])
        end = reduce(lambda ev_a, ev_b: max(ev_a, ev_b), [e.x_pts.max() for e in events])
        return start, end, end - start + 1

    def channels(self) -> np.ndarray:

        y = np.zeros(shape=(len(self.events), self.duration_pts))
        for e_i, (e, w) in enumerate(zip(self.events, self.weights)):
            y[e_i, e.x_pts - self.start] = e.y * w

        return y

    def _combiner(self) -> Tuple[np.ndarray, np.ndarray]:

        y = self.channels()
        y = y.sum(axis=0)

        return y

    def _make_generate_f(self) -> Callable:
        """
        Make the generator function.

        Returns as Callable that does the combination using the .y properties on each Event. This callable can be
        assigned to ._generate_f which maintains the Event API.
        """
        return self._combiner

    def plot(self,
             channels: bool = False,
             *args, **kwargs):

        if channels:
            plt.plot(self.x, self.channels().T)

        super().plot(*args, **kwargs)

    def plot_subplots(self,
                      show: bool = False):
        fig, ax = plt.subplots(nrows=len(self.events) + 1,
                               ncols=1)

        for e_i, e in enumerate(self.events):
            ax[e_i].plot(e.x, e.y)
            ax[e_i].set_xlim([self.x[0], self.x[-1]])

        ax[-1].plot(self.x, self.y)

        if show:
            plt.show()

    def to_list(self) -> List[Event]:
        evs = []
        for ev, w in zip(self.events, self.weights):
            evs.append(self.recursive_transverse(ev,
                                                 last_weight=1,
                                                 next_weight=w,
                                                 depth=0))

        return evs

    def recursive_transverse(self, ev: Union[List[Event], Event],
                             next_weight: Union[List[float], float],
                             last_weight: float = 1,
                             depth: int = 0) -> Tuple[int, float, float, Event]:

        if not isinstance(ev, CompoundEvent) and not isinstance(ev, list):
            return depth, next_weight, last_weight * next_weight, ev

        else:
            return self.recursive_transverse(ev.events,
                                             last_weight=last_weight,
                                             next_weight=ev.weights,
                                             depth=depth + 1)
