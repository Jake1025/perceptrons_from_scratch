import numpy as np
import numpy.typing as npt
from Neuron import Neuron

class Layer:
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def forward(self, inputs: npt.ArrayLike) -> np.ndarray:
        return np.array([
            neuron.activation(inputs)
            for neuron in self.neurons
        ])
