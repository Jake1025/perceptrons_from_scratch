import numpy as np
import numpy.typing as npt
from Activations import Activation

class Neuron:
    def __init__(self, weights: npt.ArrayLike, bias: float, activation_function: Activation):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.activation_function: Activation = activation_function

        self.last_inputs = None
        self.last_weighted_sum = None
        self.last_activation = None
        self.delta = None

    def weighted_sum(self, inputs: npt.ArrayLike):
        self.last_weighted_sum = float(np.dot(inputs, self.weights) + self.bias)
        return self.last_weighted_sum

    def activate(self, inputs: npt.ArrayLike):
        self.last_inputs = np.asarray(inputs, dtype=float)
        self.last_activation = self.activation_function.apply(self.weighted_sum(self.last_inputs))
        return self.last_activation

    def activation_derivative(self):
        return self.activation_function.derivative(self.last_weighted_sum, self.last_activation)
