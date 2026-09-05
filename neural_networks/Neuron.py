import math
import numpy as np

class Neuron:
    def __init__(self, weights:np.typing.ArrayLike, bias:float):
        self.weights = weights
        self.bias = bias

    def weighted_sum(self, inputs: np.typing.ArrayLike):
        return float(np.sum(inputs * self.weights) + self.bias)

    def activation(self, inputs:float):
        return 1 / (1 + math.exp(-self.weighted_sum(inputs)))
