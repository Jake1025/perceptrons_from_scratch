import numpy as np
from Layer import Layer
from Neuron import Neuron


class Network:
    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.layers = []

    def initialize_random(self, low=-1, high=1, seed=None):
        rng = np.random.default_rng(seed)
        self.layers = []

        for input_count, neuron_count in zip(self.sizes[:-1], self.sizes[1:]):
            neurons = [
                Neuron(
                    weights=rng.uniform(low, high, size=input_count),
                    bias=0.0,
                )
                for _ in range(neuron_count)
            ]
            self.layers.append(Layer(neurons))

    def forward(self, inputs):
        outputs = np.asarray(inputs, dtype=float)

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs
