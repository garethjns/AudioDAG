# AudioDAG

![CI](https://github.com/garethjns/AudioDAG/workflows/CI/badge.svg?branch=master)

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
![example_sine](https://github.com/garethjns/AudioDAG/blob/master/images/example_sine.PNG) 

The generation function is called when Component.y is accessed, and can be optionally cached with cache=True, or generated on the fly on each call. The envelope is applied immediately after signal generation (default = constant).

Some predefined tonal and noise signals are defined in audio_dag.signal.components.noise and .tonal.

## Compound signal components

Compound components handle combining components (and/or other compound components). The CompoundComponent class automatically creates a generation function that combines the generation functions of the supplied components. Similar to the individual events, this is only evaluated when the top objects .y is called (with optional caching). Meaning a whole DAG (or tree) of components can be combined before allocating any time or memory.

In simple cases, these can be constructed by multiplying components together. For example, to add noise to a sine wave:


### Simple - mul
````python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.components.tonal import SineComponent
from audiodag.signal.components.noise import NoiseComponent

sin = SineComponent(freq=12, mag=1, fs=5000, duration=1000)
noise = NoiseComponent(fs=5000, duration=1000, mag=db_to_lin(ref=1, db_change=-80))

compound_component = sin * noise
compound_component.plot_subplots(show=True)
````
![example_mul](https://github.com/garethjns/AudioDAG/blob/master/images/example_mul.PNG) 

### Simple - from list
In more complex cases, for example where unequal weighting or a new envelope is required, components can be specified in a list.

````python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.components.tonal import SineComponent
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.envelopes.templates import CosEnvelope

sin = SineComponent(freq=12, mag=1, start=100, fs=5000, duration=1000)
noise = NoiseComponent(fs=5000, duration=1000, mag=db_to_lin(ref=1, db_change=-80))

compound_component  = CompoundComponent([sin, noise],
                                        envelope=CosEnvelope)
compound_component.plot_subplots(show=True)
````
![example_simple](https://github.com/garethjns/AudioDAG/blob/master/images/example_simple.PNG) 

## Complex
````Python
from audiodag.signal.digital.conversion import db_to_lin
from audiodag.signal.envelopes.templates import IncreasingEnvelope
from audiodag.signal.components.component import CompoundComponent
from audiodag.signal.components.noise import NoiseComponent
from audiodag.signal.components.tonal import SineComponent

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
````
![example_complex](https://github.com/garethjns/AudioDAG/blob/master/images/example_complex.png) 
