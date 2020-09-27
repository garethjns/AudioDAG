# AudioDAG
![CI](https://github.com/garethjns/AudioDAG/workflows/CI/badge.svg?branch=master) [![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=garethjns_AudioDAG&metric=alert_status)](https://sonarcloud.io/dashboard?id=garethjns_AudioDAG)

Construct digital audio signals from individual components, designed for psychophysics stimulus generations. Supports lazy construction or in-memory caching,

# Install
````bash
pip install audiodag
````
# Usage

## Signal components

The audio_dag.signal.component.Component class is designed to handle a function describing ones aspect of a digital signaly. It handles specification of signal properties such as magnitude, duration, and hardware properties such as sampling rate, clipping, and enveloping, etc.

Individual signal component are defined by inheriting from the base component class. The child should define the ._generate_f method and and handle any additional parameters. For example, to create a sine wave:

````python
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

sin = SineComponent(freq=12, mag=1, start=100, fs=5000, duration=1000)
sin.plot(show=True)
````
![example_sine](https://raw.githubusercontent.com/garethjns/AudioDAG/master/images/example_sine.PNG)

The generation function is called when Component.y is accessed, and can be optionally cached with cache=True, or generated on the fly on each call. The envelope is applied immediately after signal generation (default = constant).

Some predefined tonal and noise signals are defined in audio_dag.signal.components.noise and .tonal.

## Compound signal components

Compound components handle combining components (and/or other compound components). The CompoundComponent class automatically creates a generation function that combines the generation functions of the supplied components. Similar to the individual events, this is only evaluated when the top objects .y is called (with optional caching). Meaning a whole DAG (or tree) of components can be combined before allocating any time or memory.

In simple cases, these can be constructed by multiplying components together. For example, to add noise to a sine wave:


### Simple - mul
````python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.components.sine_component import SineComponent
from audiodag.signal.components.noise_component import NoiseComponent

sin = SineComponent(freq=12, mag=1, fs=5000, duration=1000)
noise = NoiseComponent(fs=5000, duration=1000, mag=db_to_lin(ref=1, db_change=-80))

compound_component = sin * noise
compound_component.plot_subplots(show=True)
````
![example_mul](https://raw.githubusercontent.com/garethjns/AudioDAG/master/images/example_mul.PNG)

### Simple - from list
In more complex cases, for example where unequal weighting or a new envelope is required, components can be specified in a list.

````python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.components.sine_component import SineComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.envelopes.templates import CosEnvelope

sin = SineComponent(freq=12, mag=1, start=100, fs=5000, duration=1000)
noise = NoiseComponent(fs=5000, duration=1000, mag=db_to_lin(ref=1, db_change=-80))

compound_component  = CompoundComponent([sin, noise],
                                        envelope=CosEnvelope)
compound_component.plot_subplots(show=True)
````
![example_simple](https://raw.githubusercontent.com/garethjns/AudioDAG/master/images/example_simple.PNG)

## Complex
````Python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.envelopes.templates import IncreasingEnvelope
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent

start = 0
sine_4 = SineComponent(start=start, duration=1400, freq=4)

start = 200
duration = 600
sine_2_6 = CompoundComponent([SineComponent(start=start, duration=duration, freq=2),
                              SineComponent(start=start, duration=duration, freq=6)])

start = 600
duration = 1000
sine_2_12 = CompoundComponent([SineComponent(start=start, duration=duration, freq=2),
                               SineComponent(start=start, duration=duration, freq=12)])
sine_8_10 = CompoundComponent([SineComponent(start=start, duration=duration, freq=8),
                               SineComponent(start=start, duration=duration, freq=10),
                               sine_2_12])

noise = NoiseComponent(start=0, duration=1000, envelope=IncreasingEnvelope,
                       mag=db_to_lin(ref=1, db_change=-120))

signal = CompoundComponent([sine_4, sine_2_6, sine_8_10, noise])
signal.plot_subplots(show=True)
````
![example_complex](https://raw.githubusercontent.com/garethjns/AudioDAG/master/images/example_complex.png)


## Very complex example
As generated by TwoGapStim used in [MSIModels project](https://github.com/garethjns/MSIModels).

![example_complex](https://raw.githubusercontent.com/garethjns/AudioDAG/master/images/example_very_complex.png)

```python
import matplotlib.pyplot as plt
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise_component import NoiseComponent
from audiodag.signal.components.sine_component import SineComponent

signal = CompoundComponent(components=[
    NoiseComponent(start=0, duration=1300, mag=0.1, fs=1000, seed=583566848, cache=True, clip=2.0, weight=2.0,
                   normalise=False, dist='normal'),
    NoiseComponent(start=65, duration=25, mag=0.02, fs=1000, seed=2068698735, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=90, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=90, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=90, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=115, duration=25, mag=0.02, fs=1000, seed=618014710, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=140, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=140, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=140, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=165, duration=25, mag=0.02, fs=1000, seed=1220412928, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=190, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=190, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=190, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=215, duration=25, mag=0.02, fs=1000, seed=735073669, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=240, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=240, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=240, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=265, duration=25, mag=0.02, fs=1000, seed=1126067363, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=290, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=290, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=290, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=315, duration=25, mag=0.02, fs=1000, seed=819904825, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=340, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=340, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=340, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=365, duration=25, mag=0.02, fs=1000, seed=1613153861, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=390, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=390, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=390, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=415, duration=25, mag=0.02, fs=1000, seed=88620281, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=440, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=440, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=440, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=465, duration=25, mag=0.02, fs=1000, seed=987359829, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=490, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=490, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=490, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=515, duration=25, mag=0.02, fs=1000, seed=1256982798, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=540, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=540, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=540, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=565, duration=25, mag=0.02, fs=1000, seed=775602058, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=590, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=590, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=590, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=615, duration=50, mag=0.02, fs=1000, seed=963415820, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=665, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=665, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=665, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=690, duration=25, mag=0.02, fs=1000, seed=104274552, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=715, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=715, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=715, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=740, duration=50, mag=0.02, fs=1000, seed=187413258, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=790, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=790, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=790, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=815, duration=25, mag=0.02, fs=1000, seed=2068769826, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=840, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=840, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=840, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=865, duration=25, mag=0.02, fs=1000, seed=1581641620, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal'), CompoundComponent(components=[
        SineComponent(start=890, duration=25, mag=2, fs=1000, seed=1505166658, cache=True, clip=2.0, normalise=False,
                      weight=1.0, freq=8),
        NoiseComponent(start=890, duration=25, mag=0.02, fs=1000, seed=1914494741, cache=True, clip=2.0, weight=1.0,
                       normalise=False, dist='normal')], start=890, weights=[1.0, 1.0], normalise=False),
    NoiseComponent(start=915, duration=25, mag=0.02, fs=1000, seed=1803778047, cache=True, clip=2.0, weight=1,
                   normalise=False, dist='normal')], start=0,
    weights=[2.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1], normalise=False)

signal.plot(show=False)
plt.xlabel('pts')
plt.ylabel('Mag.')
plt.xlim([400, 800])
plt.title('Combined signal')
plt.show()
```
