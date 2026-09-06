import math
import numpy as np

class Neuron:
    def __init__(self, weights:np.typing.ArrayLike, bias:float):
        self.weights = weights
        self.bias = bias
        self.last_inputs = None
        self.last_weighted_sum = None
        self.last_activation = None
        self.delta = None

    def weighted_sum(self, inputs: np.typing.ArrayLike):
        self.last_weighted_sum = float(np.sum(inputs * self.weights) + self.bias)
        return self.last_weighted_sum

    def activation(self, inputs:float):
        self.last_inputs = inputs
        self.last_activation = 1 / (1 + math.exp(-self.weighted_sum(inputs)))
        return self.last_activation
