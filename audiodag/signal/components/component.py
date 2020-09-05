import copy
from functools import reduce
from typing import List, Tuple, Callable, Iterable

import numpy as np
from matplotlib import pyplot as plt

from audiodag.signal.digital.digital_siginal import DigitalSignal
from audiodag.signal.envelopes.envelope import Envelope
from audiodag.signal.envelopes.templates import ConstantEnvelope


class Component(DigitalSignal):
    def __init__(self, weight: float = 1.0,
                 *args, **kwargs):
        """

        :param weight: Relative event weight, used when combined with other components, for example.
        """

        self.weight = weight
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"Component(start={self.start}, duration={self.duration}, mag={self.mag}, fs={self.fs}, " \
               f"seed={self.seed}, cache={self.cache}, clip={'np.inf' if np.isinf(self.clip) else self.clip}, " \
               f"weight={self.weight}, normalise={self.normalise})"

    def _generate_f(self) -> np.ndarray:
        """Default components is constant 1s * mag"""

        return np.ones(shape=(self.duration_pts,)) * self.mag

    def __mul__(self, other):
        """Multiplying components generates a CompoundEvent object with a generator for the combined signals."""

        return CompoundComponent(components=[self, other])


class CompoundComponent(Component):
    """Object for combining components, for example adding noise to another components."""

    def __init__(self, components: List[Component], *args,
                 weights: List[float] = None,
                 normalise_weights: bool = False,
                 start: int = None,
                 envelope: Envelope = ConstantEnvelope, **kwargs):
        """

        :param components: List of components to combine.
        :param weights: List of weights for each component. Default assumes equal for all.
        :param normalise_weights: Whether or not to normalise weights to sum to 1. Convenient for constraining
                                  overlapping signals.
                                  However, bear in mind that if this is on, non-overlapping signals will shrink
                                  proportionally with the number of components, even if all weights are 1.
        :param start: New start time for the component. if None, start is inherited from combined components.
        """

        self.normalise_weights = normalise_weights
        self._verify_event_list(components)
        start_sub, _, duration = self._new_duration(components)

        super().__init__(*args, fs=components[0].fs,
                         start=start_sub,
                         duration=duration,
                         envelope=envelope, **kwargs)

        self.events = []

        self._assign_events(components)
        self._assign_weights(weights)

        # Adjust start of self and all sub components, if a new start is specified.
        if start is not None:
            self._adjust_start(start - start_sub)
            self.start = start

        self._generate_f = self._make_generate_f()

    def __repr__(self) -> str:
        return f"CompoundComponent(components={self.events}, start={self.start}, weights={self.weights}," \
               f"normalise={self.normalise})"

    def _adjust_start(self, start_delta: int):
        """
        Given a delta, recursively adjust start of all parent components.
        TODO: Recursiveness not tested....
        """

        for ev in self.events:
            if isinstance(ev, CompoundComponent):
                ev._adjust_start(start_delta)
            else:
                ev.start = ev.start + start_delta

    def _assign_events(self, events: List[Component]) -> None:
        # Events need to be copied here as weights may need to be updated
        # Shouldn't need to be a deep copy, though. Can also clear any arrays, may want to parameterize this.
        for ev in events:
            ev = copy.copy(ev)
            ev.clear()
            self.events.append(ev)

    @staticmethod
    def _normalise_to_sum_1(y: np.ndarray) -> np.ndarray:
        """Normalise vector y to a sum to 1."""
        return y / y.sum()

    def _assign_weights(self,
                        weights: List[float] = None) -> None:
        """For the provided components, reset their normalised, relative weights. Or override them with supplied."""
        if weights is None:
            # Set using values in components
            weights = [ev.weight for ev in self.events]

        # Normalise
        if self.normalise_weights:
            weights = self._normalise_to_sum_1(np.array(weights))

        # Update event weights with normalised
        for ev, w in zip(self.events, weights):
            ev.weight = w

        self.weights = weights

    @staticmethod
    def _verify_event_list(events: List[Component]):
        check_params = ['fs']

        for p in check_params:
            param_values = [getattr(e, p) for e in events]
            if len(np.unique(param_values)) > 1:
                raise ValueError(f"Param {p} is inconsistent across components: {param_values}")

    @staticmethod
    def _new_duration(events: List[Component]) -> Tuple[int, int, int]:
        start = int(reduce(lambda ev_a, ev_b: min(ev_a, ev_b), [e.x.min() for e in events]))
        end = int(reduce(lambda ev_a, ev_b: max(ev_a, ev_b), [e.x.max() for e in events]))
        return start, end, end - start

    def channels(self) -> np.ndarray:
        y = np.zeros(shape=(len(self.events), self.duration_pts))
        for e_i, e in enumerate(self.events):
            y[e_i, e.x_pts - self.start_pts] = e.y * e.weight

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

    def plot(self, channels: bool = False, show: bool = False, **kwargs):

        if channels:
            plt.plot(self.x, self.channels().T, **kwargs)
            plt.xlabel('Time')
            plt.ylabel('Mag.')
            if show:
                plt.show()
        else:
            super().plot(show=show, **kwargs)

    def plot_subplots(self, show: bool = False):
        fig, ax = plt.subplots(nrows=len(self.events) + 1,
                               ncols=1)

        for e_i, e in enumerate(self.events):
            ax[e_i].plot(e.x, e.y,
                         color='k' if isinstance(e, CompoundComponent) else 'r')
            ax[e_i].set_xlim([self.x[0], self.x[-1]])
            ax[e_i].set_title(f"Signal component {e_i}")

        ax[-1].plot(self.x, self.y)
        ax[-1].set_xlim([self.x[0], self.x[-1]])
        ax[-1].set_title('Combined signal')

        if show:
            plt.show()

    def to_list(self) -> List[Tuple[int, Component]]:
        return self.recursive_transverse(self)

    def recursive_transverse(self, ev: Component,
                             depth: int = 0,
                             path: int = 0,
                             previous_node: str = '|') -> Iterable[Tuple[int, Component]]:
        """Currently just prints a very crude representation of the graph to the console."""

        node = f"{previous_node} <- {path}({str(depth)})"

        if isinstance(ev, CompoundComponent):
            return [self.recursive_transverse(e,
                                              depth=depth + 1,
                                              path=p,
                                              previous_node=node) for p, e in enumerate(ev.events)]
        else:
            return node, ev
