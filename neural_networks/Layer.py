import numpy as np
import numpy.typing as npt
from Neuron import Neuron
from Activations import Activation

class Layer:
    def __init__(self, input_size: int, neuron_count: int, activation: Activation,
                 rng: np.random.Generator, low=None, high=None):
        self.neuron_count = neuron_count
        self.activation_function:Activation = activation
        self.neurons: list[Neuron] = [
            Neuron(np.zeros(input_size, dtype=float), 0.0, activation)
            for _ in range(neuron_count)
        ]
        self.initialize_random(rng, low, high)

    def initialize_random(self, rng, low=None, high=None):
        # Explicit bounds override the activation-aware defaults.
        limit = self.activation_function.weight_limit(
            len(self.neurons[0].weights), self.neuron_count
        )
        low = -limit if low is None else low
        high = limit if high is None else high
        for neuron in self.neurons:
            neuron.weights = rng.uniform(
                low,
                high,
                size=len(neuron.weights)
            )
            neuron.bias = 0.0

    def forward(self, inputs: npt.ArrayLike) -> np.ndarray:
        return np.array([
            neuron.activate(inputs)
            for neuron in self.neurons
        ])
